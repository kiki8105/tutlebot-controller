#!/usr/bin/env python3
"""ROS 2 Humble / PyQt5 hold-to-drive controller (no colcon build required).

source /opt/ros/humble/setup.bash
ROS_DOMAIN_ID=64 python3 cmd_vel_controller.py
Optional ROS remap: --ros-args -r cmd_vel:=/robot/cmd_vel
Requires a desktop display or SSH X11 forwarding. No obstacle avoidance.
"""
import argparse
import os
import signal
import sys
import time
import threading

from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy

from PyQt5 import QtCore, QtGui, QtWidgets
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from navigation_panels import RosBridge, MapPanel, ParameterPanel


SETTINGS_PATH = os.path.expanduser('~/.config/turtlebot3/cmd_vel_controller.ini')


def domain_settings(path=SETTINGS_PATH):
    return QtCore.QSettings(path, QtCore.QSettings.IniFormat)


def initial_domain(settings):
    # A saved UI selection takes precedence over the terminal's environment.
    for candidate in (settings.value('ros/domain_id'), os.getenv('ROS_DOMAIN_ID'), 64):
        try:
            value = int(candidate)
            if 0 <= value <= 232:
                return value
        except (TypeError, ValueError):
            pass
    return 64


def decode_camera_image(msg):
    """Use Qt's built-in image decoder; avoid importing OpenCV into the GUI."""
    if isinstance(msg, CompressedImage):
        frame = QtGui.QImage.fromData(bytes(msg.data))
        if frame.isNull():
            raise ValueError('JPEG/PNG 압축 영상을 해석할 수 없습니다.')
        return frame.convertToFormat(QtGui.QImage.Format_RGB888)
    formats = {
        'rgb8': (QtGui.QImage.Format_RGB888, 3, False),
        'bgr8': (QtGui.QImage.Format_RGB888, 3, True),
        'mono8': (QtGui.QImage.Format_Grayscale8, 1, False),
        'rgba8': (QtGui.QImage.Format_RGBA8888, 4, False),
        'bgra8': (QtGui.QImage.Format_RGBA8888, 4, True),
    }
    if msg.encoding not in formats:
        raise ValueError(f'지원하지 않는 영상 형식: {msg.encoding}. rgb8 또는 압축 영상을 선택하세요.')
    fmt, channels, swap = formats[msg.encoding]
    if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width*channels or len(msg.data) < msg.step*msg.height:
        raise ValueError('영상 크기/stride/데이터 길이가 잘못되었습니다.')
    frame = QtGui.QImage(bytes(msg.data), msg.width, msg.height, msg.step, fmt).copy()
    return frame.rgbSwapped() if swap else frame


class CameraFeed:
    """Decode in a dedicated ROS thread; GUI only consumes the latest frame."""
    def __init__(self):
        self.request = None
        self.latest = None
        self.status = '카메라 연결 대기'
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def connect(self, topic, compressed):
        self.latest = None
        self.request = (topic, compressed)
        self.status = '영상 수신 대기'

    def run(self):
        node = Node('turtlebot3_controller_camera')
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        sub = None
        active = None
        try:
            while not self.closed.is_set() and rclpy.ok():
                if self.request != active:
                    if sub is not None:
                        node.destroy_subscription(sub)
                        sub = None
                    active = self.request
                    if active:
                        topic, compressed = active
                        def receive(msg, token=active):
                            try:
                                frame = decode_camera_image(msg)
                                w, h = frame.width(), frame.height()
                                if token == self.request:
                                    self.latest = (frame, time.monotonic())
                                    self.status = f'{w} × {h}'
                            except Exception as exc:
                                self.status = f'영상 변환 오류: {exc}'
                        try:
                            sub = node.create_subscription(
                                CompressedImage if compressed else Image, topic, receive,
                                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
                        except Exception as exc:
                            self.status = f'연결 오류: {exc}'
                executor.spin_once(timeout_sec=.1)
        except Exception as exc:
            self.status = f'카메라 수신 오류: {exc}'
        finally:
            executor.shutdown()
            node.destroy_node()

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2)


class Controller(QtWidgets.QWidget):
    vehicleRequested = QtCore.pyqtSignal(str)
    def __init__(self, publisher, domain_id=None, settings=None, camera=None,
                 node=None, camera_topic=None):
        super().__init__()
        self.publisher = publisher
        self.camera = camera
        self.ros_node = node
        self.vehicle_namespace = node.get_namespace().rstrip('/') if node is not None else ''
        self.bridge = RosBridge(node) if node is not None else None
        self.settings = settings if settings is not None else domain_settings()
        self.domain_id = initial_domain(self.settings) if domain_id is None else domain_id
        self.keys = set()
        self.active = None
        self.manual_override_active = False
        self.direction = (0, 0)
        self.stop_ticks = 0
        self.setWindowTitle('TurtleBot3 · 카메라 / 파라미터 / 지도 / 조종')
        # Keep the application usable on smaller displays.  The dashboard below
        # owns its natural minimum size, so a compact window gets scroll bars
        # instead of forcing buttons and input fields below their minimum size.
        self.setMinimumSize(640, 480)
        self.setStyleSheet('''
            QWidget { background: white; color: black; font-size: 13px; }
            QPushButton { background: white; border: 1px solid black;
                          border-radius: 0px; padding: 3px; }
            QPushButton:pressed { background: #e5e5e5; }
            QDoubleSpinBox, QSpinBox, QComboBox { background: white; color: black;
                border: 1px solid black; padding: 7px; border-radius: 0px; }
            QToolTip { background: white; color: black; border: 1px solid black; padding: 5px; }
        ''')
        window_layout = QtWidgets.QVBoxLayout(self)
        window_layout.setContentsMargins(0, 0, 0, 0)
        self.dashboard_scroll = QtWidgets.QScrollArea()
        self.dashboard_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.dashboard_scroll.setWidgetResizable(True)
        self.dashboard = QtWidgets.QWidget()
        outer = QtWidgets.QGridLayout(self.dashboard)
        self.dashboard_scroll.setWidget(self.dashboard)
        window_layout.addWidget(self.dashboard_scroll)
        controls = QtWidgets.QGroupBox('4. 수동 조종기')
        self.controls = controls
        layout = QtWidgets.QVBoxLayout(controls)
        outer.addWidget(controls, 1, 1)
        camera_box = QtWidgets.QGroupBox('1. 카메라')
        panel = QtWidgets.QVBoxLayout(camera_box)
        outer.addWidget(camera_box, 0, 0)
        self.parameters = ParameterPanel(self.bridge, self.domain_id, self.stop)
        self.map_panel = MapPanel(
            self.bridge, self.stop, domain=self.domain_id)
        outer.addWidget(self.parameters, 0, 1)
        outer.addWidget(self.map_panel, 1, 0)
        for index in range(2):
            outer.setRowStretch(index, 1)
            outer.setColumnStretch(index, 1)
        panel.addWidget(QtWidgets.QLabel('실시간 카메라'))
        row = QtWidgets.QHBoxLayout()
        self.camera_topic = QtWidgets.QComboBox()
        self.camera_topic.setEditable(True)
        self.camera_settings_prefix = (f'vehicles/{self.domain_id}/{self.vehicle_namespace.strip(chr(47))}/camera/'
                                       if self.vehicle_namespace else 'camera/')
        saved_camera_topic = self.settings.value(
            self.camera_settings_prefix + 'topic',
            self.vehicle_namespace + '/camera/image_raw',
        )
        self.camera_topic.addItem(camera_topic or str(saved_camera_topic))
        row.addWidget(self.camera_topic, 1)
        refresh = QtWidgets.QPushButton('토픽 검색')
        refresh.setStyleSheet('min-height: 32px; padding: 4px;')
        refresh.clicked.connect(self.refresh_camera_topics)
        row.addWidget(refresh)
        panel.addLayout(row)
        self.compressed = QtWidgets.QCheckBox('압축 영상 (CompressedImage)')
        self.compressed.setChecked(str(self.settings.value(self.camera_settings_prefix+'compressed', 'false')).lower() == 'true')
        panel.addWidget(self.compressed)
        connect = QtWidgets.QPushButton('카메라 연결')
        connect.setStyleSheet('min-height: 32px; padding: 4px;')
        connect.clicked.connect(self.connect_camera)
        panel.addWidget(connect)
        self.video = QtWidgets.QLabel('카메라 영상 대기')
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setMinimumSize(320, 180)
        self.video.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.video.setStyleSheet('background: white; color: black; border: 1px solid black;')
        panel.addWidget(self.video, 1)
        self.camera_status = QtWidgets.QLabel('카메라 연결 대기')
        self.camera_status.setWordWrap(True)
        panel.addWidget(self.camera_status)
        self.resize(1280, 960)
        self.video_timer = QtCore.QTimer(self)
        self.video_timer.timeout.connect(self.render_camera)
        self.video_timer.start(66)
        if camera is not None:
            QtCore.QTimer.singleShot(0, self.connect_camera)
        title = QtWidgets.QLabel('TurtleBot3 수동 조종')
        title.setStyleSheet('font-size: 18px; font-weight: bold;')
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel('버튼 / 화살표 키를 누르는 동안 이동 · 놓으면 정지'))
        layout.addWidget(QtWidgets.QLabel(
            f'Topic: {publisher.topic_name}  |  연결된 Domain: {self.domain_id}'))
        vehicle_row = QtWidgets.QHBoxLayout()
        vehicle_row.addWidget(QtWidgets.QLabel('차량 namespace'))
        self.vehicle_box = QtWidgets.QComboBox()
        self.vehicle_box.setEditable(True)
        self.vehicle_box.addItem(self.vehicle_namespace or '/')
        vehicle_row.addWidget(self.vehicle_box, 1)
        discover = QtWidgets.QPushButton('차량 검색')
        discover.clicked.connect(self.discover_vehicles)
        vehicle_row.addWidget(discover)
        reconnect = QtWidgets.QPushButton('차량 연결')
        reconnect.setToolTip('조종기를 재시작하여 대상을 전환합니다. 경유점·미저장 파라미터 목록은 초기화됩니다.')
        reconnect.clicked.connect(self.select_vehicle)
        vehicle_row.addWidget(reconnect)
        layout.addLayout(vehicle_row)
        domain_row = QtWidgets.QHBoxLayout()
        domain_row.addWidget(QtWidgets.QLabel('ROS Domain ID'))
        self.domain_input = QtWidgets.QSpinBox()
        self.domain_input.setRange(0, 232)
        self.domain_input.setValue(self.domain_id)
        domain_row.addWidget(self.domain_input)
        self.domain_save = QtWidgets.QPushButton('저장')
        self.domain_save.setStyleSheet('min-height: 36px; padding: 4px 12px;')
        self.domain_save.clicked.connect(self.save_domain)
        domain_row.addWidget(self.domain_save)
        layout.addLayout(domain_row)
        domain_hint = QtWidgets.QLabel('실행파일의 --domain-id가 저장값보다 우선합니다. 현재 연결 ID를 확인하세요.')
        domain_hint.setStyleSheet('font-size: 12px; color: black;')
        layout.addWidget(domain_hint)
        speed_row = QtWidgets.QFormLayout()
        self.linear_speed = QtWidgets.QDoubleSpinBox()
        self.linear_speed.setRange(0.01, 0.22)
        self.linear_speed.setSingleStep(0.01)
        self.linear_speed.setValue(0.10)
        self.linear_speed.setSuffix(' m/s')
        self.angular_speed = QtWidgets.QDoubleSpinBox()
        self.angular_speed.setRange(0.10, 1.00)
        self.angular_speed.setSingleStep(0.05)
        self.angular_speed.setValue(0.50)
        self.angular_speed.setSuffix(' rad/s')
        speed_row.addRow('전진 / 후진 속도', self.linear_speed)
        speed_row.addRow('회전 / 선회 속도', self.angular_speed)
        layout.addLayout(speed_row)
        grid = QtWidgets.QGridLayout()
        self.buttons = {}
        for name, text, row, col, direction in [
                ('forward', '▲ 전진', 0, 1, (1, 0)),
                ('forward_left', '◤ 좌선회', 0, 0, (1, 1)),
                ('forward_right', '우선회 ◥', 0, 2, (1, -1)),
                ('left', '◀ 좌회전', 1, 0, (0, 1)),
                ('right', '우회전 ▶', 1, 2, (0, -1)),
                ('backward_left', '◣ 후진 좌', 2, 0, (-1, 1)),
                ('backward_right', '후진 우 ◢', 2, 2, (-1, -1)),
                ('backward', '▼ 후진', 2, 1, (-1, 0))]:
            button = QtWidgets.QPushButton(text)
            button.setMinimumHeight(46)
            button.setFocusPolicy(QtCore.Qt.NoFocus)
            button.pressed.connect(lambda b=button, d=direction: self.start(b, d))
            button.released.connect(
                lambda: self.stop(resume_autonomous=True))
            grid.addWidget(button, row, col)
            self.buttons[name] = button
        stop_button = QtWidgets.QPushButton('■ 정지')
        stop_button.setStyleSheet('QPushButton { background: #e53935; color: white; } '
                                 'QPushButton:pressed { background: #b71c1c; }')
        stop_button.setMinimumHeight(46)
        stop_button.pressed.connect(self.emergency_stop)
        grid.addWidget(stop_button, 1, 1)
        layout.addLayout(grid)
        self.status = QtWidgets.QLabel('정지')
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status)
        hint = QtWidgets.QLabel('좌·우: 제자리 회전 | ↑+←/→ 또는 모서리 버튼: 선회 | Space / Esc: 정지 + Nav2 취소\n자동주행 중 수동 이동 잠금. 수동 조종에는 장애물 회피가 없습니다.')
        hint.setWordWrap(True)
        hint.setStyleSheet('font-size: 13px; color: black;')
        layout.addWidget(hint)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)  # 20 Hz, using wall time rather than simulation time.
        # QGridLayout may otherwise compress its children when the top-level
        # window becomes smaller than the complete 2 x 2 dashboard.
        outer.activate()
        self.dashboard.setMinimumSize(self.dashboard.minimumSizeHint())

    def eventFilter(self, obj, event):
        if not isinstance(obj, QtWidgets.QWidget) or obj.window() is not self:
            return False
        kind = event.type()
        if kind in (QtCore.QEvent.WindowDeactivate, QtCore.QEvent.FocusOut):
            self.stop()
        if kind not in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease,
                        QtCore.QEvent.ShortcutOverride):
            return False
        key = event.key()
        directions = {QtCore.Qt.Key_Up: (1, 0), QtCore.Qt.Key_Down: (-1, 0),
                      QtCore.Qt.Key_Left: (0, 1), QtCore.Qt.Key_Right: (0, -1)}
        if key in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Escape):
            if kind == QtCore.QEvent.KeyPress:
                self.emergency_stop()
            event.accept()
            return True
        if key not in directions:
            return False
        # Releases must stop even if focus moved into an input field.
        if kind == QtCore.QEvent.KeyRelease:
            if not event.isAutoRepeat() and key in self.keys:
                self.keys.discard(key)
                if self.keys:
                    self.direction = self.combine(directions[k] for k in self.keys)
                else:
                    self.stop(resume_autonomous=True)
            return False
        editing = isinstance(obj, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox,
                                   QtWidgets.QComboBox))
        if editing or event.modifiers() != QtCore.Qt.NoModifier:
            return False
        if kind == QtCore.QEvent.ShortcutOverride:
            event.accept()
            return True
        if not self.manual_drive_ready():
            event.accept()
            return True
        if not event.isAutoRepeat():
            if not self.keys:
                self.stop()
            # Holding ↑/↓ and pressing ←/→ combines into an arc turn.
            self.keys.add(key)
            self.direction = self.combine(directions[k] for k in self.keys)
            self.manual_override_active = True
            self.stop_ticks = 0
            self.tick()
        return True

    @staticmethod
    def combine(directions):
        linear = angular = 0
        for d in directions:
            linear += d[0]
            angular += d[1]
        return (max(-1, min(1, linear)), max(-1, min(1, angular)))

    def discover_vehicles(self):
        self.stop()
        selected = self.vehicle_box.currentText()
        namespaces = {'/', self.vehicle_namespace or '/'}
        if self.ros_node:
            for topic, types in self.ros_node.get_topic_names_and_types():
                if topic.endswith('/cmd_vel') and 'geometry_msgs/msg/Twist' in types:
                    namespaces.add(topic[:-len('/cmd_vel')] or '/')
        self.vehicle_box.clear()
        self.vehicle_box.addItems(sorted(namespaces))
        self.vehicle_box.setCurrentText(selected)
        self.status.setText('차량 namespace를 선택하고 연결하세요. 같은 Domain의 /cmd_vel 기준 검색')

    def select_vehicle(self):
        if self.bridge and self.bridge.busy:
            self.status.setText('주행/리셋 종료 후 차량을 전환하세요.')
            return
        if self.parameters.write_pending:
            self.status.setText('파라미터 적용 완료 후 차량을 전환하세요.')
            return
        namespace = '/'+self.vehicle_box.currentText().strip('/ ')
        try:
            from rclpy.validate_namespace import validate_namespace
            validate_namespace(namespace)
        except Exception as exc:
            self.status.setText('namespace 오류: '+str(exc))
            return
        if namespace.rstrip('/') == self.vehicle_namespace:
            self.status.setText('이미 연결된 차량입니다.')
            return
        self.stop()
        self.settings.setValue(f'vehicles/domain_{self.domain_id}/namespace', namespace)
        self.settings.sync()
        self.vehicleRequested.emit(namespace)

    def refresh_camera_topics(self):
        self.stop()
        if self.ros_node is None:
            return
        selected = self.camera_topic.currentText()
        topics = [name for name, types in self.ros_node.get_topic_names_and_types()
                  if name.startswith(self.vehicle_namespace+'/') and any(t in ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage') for t in types)]
        self.camera_topic.clear()
        self.camera_topic.addItems(sorted(set(topics + [selected])))
        self.camera_topic.setCurrentText(selected)
        self.camera_status.setText(f'영상 토픽 {len(topics)}개 발견 · 선택 후 연결')

    def connect_camera(self):
        self.stop()
        topic = self.camera_topic.currentText().strip()
        if not topic or self.camera is None:
            return
        compressed = self.compressed.isChecked() or topic.endswith('/compressed')
        self.compressed.setChecked(compressed)
        self.settings.setValue(self.camera_settings_prefix+'topic', topic)
        self.settings.setValue(self.camera_settings_prefix+'compressed', compressed)
        self.settings.sync()
        self.camera.connect(topic, compressed)
        self.video.clear()
        self.video.setText('영상 수신 대기')

    def render_camera(self):
        if self.camera is None:
            return
        latest = self.camera.latest
        if latest is not None and time.monotonic()-latest[1] < 2:
            self.video.setPixmap(QtGui.QPixmap.fromImage(latest[0]).scaled(
                self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            self.camera_status.setText('수신 중 · ' + self.camera.status)
        else:
            self.video.clear()
            self.video.setText('영상 미수신' if latest is None else '영상 수신 끊김')
            self.camera_status.setText(self.camera.status + ' · 토픽 / Domain / 카메라 실행 확인')

    def save_domain(self):
        self.stop()
        value = self.domain_input.value()
        self.settings.setValue('ros/domain_id', value)
        self.settings.sync()
        if self.settings.status() != QtCore.QSettings.NoError:
            QtWidgets.QMessageBox.warning(self, '저장 실패',
                                          '설정 파일을 저장하지 못했습니다. 기존 ID를 유지합니다.')
            return
        if value == self.domain_id:
            self.status.setText(f'Domain {value} 저장 완료 · 정지')
        else:
            self.status.setText(f'Domain {value} 저장 완료 · 조종기를 다시 실행하면 적용됩니다')

    def publish(self, linear=0.0, angular=0.0):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.publisher.publish(msg)

    def emergency_stop(self):
        if self.bridge:
            cancel = getattr(self.bridge, 'cancel', None)
            if callable(cancel):
                cancel()
            disable = getattr(self.bridge, 'disable_autonomous_drive', None)
            if callable(disable):
                disable()
        self.manual_override_active = False
        self.stop()
        self.publish()

    def manual_drive_ready(self):
        """Reserve /cmd_vel for manual control before accepting one input."""
        if not self.bridge:
            return True
        refresh = getattr(self.bridge, 'refresh_autonomous_drivers', None)
        if callable(refresh):
            refresh()
        if getattr(self.bridge, 'custom_drive_busy', False) is True:
            disable = getattr(self.bridge, 'disable_autonomous_drive', None)
            if callable(disable):
                disable()
            self.keys.clear()
            self.active = None
            self.direction = (0, 0)
            self.stop_ticks = 0
            self.publish()
            self.status.setText(
                '자동주행 정지 요청 중: 완료 후 다시 조작하세요.')
            return False
        if getattr(self.bridge, 'busy', False):
            self.status.setText('Nav2 이동 중: 먼저 정지/취소하세요.')
            return False
        return True

    def start(self, button, direction):
        if not self.manual_drive_ready():
            return
        self.keys.clear()
        self.active, self.direction = button, direction
        self.manual_override_active = True
        self.stop_ticks = 0
        self.tick()

    def stop(self, resume_autonomous=False):
        was_manual = bool(self.keys or self.active is not None)
        self.keys.clear()
        self.active = None
        self.direction = (0, 0)
        if was_manual or not self.bridge or not self.bridge.busy:
            self.publish()
        self.stop_ticks = 5  # Short stop burst; do not continuously compete when idle.
        self.status.setText('정지')
        if resume_autonomous and self.manual_override_active:
            self.manual_override_active = False
            QtCore.QTimer.singleShot(300, self.resume_autonomous)

    def resume_autonomous(self):
        if self.bridge:
            enable = getattr(self.bridge, 'enable_autonomous_drive', None)
            if callable(enable):
                enable()

    def tick(self):
        if self.bridge and self.bridge.busy:
            was_manual = bool(self.keys or self.active is not None)
            self.keys.clear()
            self.active = None
            self.stop_ticks = 0
            if was_manual:
                self.publish()
            return
        if self.keys or self.active is not None:
            if not self.keys and not self.active.isDown():
                self.stop(resume_autonomous=True)
                return
            linear = self.direction[0] * self.linear_speed.value()
            angular = self.direction[1] * self.angular_speed.value()
            self.publish(linear, angular)
            self.status.setText(f'명령 전송: {linear:+.2f} m/s · {angular:+.2f} rad/s')
        elif self.stop_ticks:
            self.publish()
            self.stop_ticks -= 1

    def application_state_changed(self, state):
        if state != QtCore.Qt.ApplicationActive:
            self.stop(resume_autonomous=True)

    def hideEvent(self, event):
        self.stop(resume_autonomous=True)
        super().hideEvent(event)

    def closeEvent(self, event):
        # Discover and stop a custom driver before deciding that the window is
        # idle.  It may not have appeared in the periodic graph poll yet.
        self.emergency_stop()
        if self.bridge and self.bridge.busy:
            self.status.setText(
                '자동/Nav2 주행 정지 요청 중입니다. 완료 후 창을 닫으세요.')
            event.ignore()
            return
        QtWidgets.QApplication.instance().removeEventFilter(self)
        self.timer.stop()
        event.accept()


def main():
    if not (os.getenv('DISPLAY') or os.getenv('WAYLAND_DISPLAY')):
        print('GUI 화면이 없습니다. PC 데스크탑에서 실행하거나 ssh -X로 접속하세요.',
              file=sys.stderr)
        return 1
    # Own SIGINT/SIGTERM so a stop can be sent before the ROS context is destroyed.
    from rclpy.signals import SignalHandlerOptions
    settings = domain_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-id', type=int, choices=range(233), metavar='0..232',
                        help='저장된 ID보다 우선하여 이번 실행의 ROS Domain 지정')
    parser.add_argument('--robot-namespace', help='연결할 차량 namespace (예: /robot1)')
    parser.add_argument('--camera-topic', help='시작 시 연결할 ROS Image 토픽')
    options, ros_args = parser.parse_known_args()
    domain_id = initial_domain(settings) if options.domain_id is None else options.domain_id
    os.environ['ROS_DOMAIN_ID'] = str(domain_id)
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    namespace = options.robot_namespace or str(settings.value(f'vehicles/domain_{domain_id}/namespace', '/'))
    namespace = '/'+namespace.strip('/')
    tf_args = ['--ros-args', '-r', '/tf:='+namespace.rstrip('/')+'/tf',
               '-r', '/tf_static:='+namespace.rstrip('/')+'/tf_static',
               '-r', '/controller_shared_tf:=/tf', '-r', '/controller_shared_tf_static:=/tf_static']
    node = Node('turtlebot3_pyqt_controller', namespace=namespace, cli_args=tf_args,
                parameter_overrides=[Parameter('use_sim_time', value=(domain_id == 64))])
    publisher = node.create_publisher(Twist, 'cmd_vel', 1)
    app = QtWidgets.QApplication([sys.argv[0]])
    app.setQuitOnLastWindowClosed(False)
    camera = CameraFeed()
    window = Controller(
        publisher,
        domain_id,
        settings,
        camera,
        node,
        camera_topic=options.camera_topic,
    )
    # RosBridge가 /map 등의 구독과 Qt signal 연결을 모두 만든 뒤 executor를
    # 시작한다. 먼저 spin하면 durable /map이 패널 연결 전에 한 번 도착해
    # 화면에서 유실될 수 있다.
    ros_executor = SingleThreadedExecutor()
    ros_executor.add_node(node)
    ros_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    ros_thread.start()
    app.applicationStateChanged.connect(window.application_state_changed)
    shutting_down = False
    restart_namespace = None

    def shutdown(*_):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        window.emergency_stop()
        started = time.monotonic()
        def finish_when_idle():
            nonlocal shutting_down
            if window.bridge and window.bridge.busy:
                if time.monotonic()-started > 8:
                    window.show()
                    window.status.setText('Nav2 취소 확인 실패. 연결/목표 상태를 확인하고 다시 종료하세요.')
                    shutting_down = False
                    return
                QtCore.QTimer.singleShot(100, finish_when_idle)
                return
            window.hide()
            QtCore.QTimer.singleShot(300, app.quit)
        finish_when_idle()

    def reconnect(namespace):
        nonlocal restart_namespace
        restart_namespace = namespace
        shutdown()
    window.vehicleRequested.connect(reconnect)
    app.lastWindowClosed.connect(shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    window.show()
    try:
        result = app.exec_()
    finally:
        window.timer.stop()
        window.video_timer.stop()
        if window.bridge:
            window.bridge.timer.stop()
        camera.close()
        if rclpy.ok():
            window.publish()
        ros_executor.shutdown(timeout_sec=2.0)
        ros_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()
    if restart_namespace is not None:
        restart_args = [
            sys.executable,
            os.path.abspath(__file__),
            '--domain-id',
            str(domain_id),
            '--robot-namespace',
            restart_namespace,
        ]
        if options.camera_topic:
            restart_args.extend(['--camera-topic', options.camera_topic])
        os.execv(sys.executable, restart_args + ros_args)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
