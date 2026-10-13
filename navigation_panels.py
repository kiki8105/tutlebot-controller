"""PyQt map/parameter panels and non-blocking ROS interfaces for the controller."""
import math
from array import array
import os
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image as PILImage
from PyQt5 import QtCore, QtGui, QtWidgets

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from nav2_msgs.action import FollowWaypoints
from nav2_msgs.srv import LoadMap, ManageLifecycleNodes
from rcl_interfaces.srv import GetParameters, SetParametersAtomically, ListParameters, DescribeParameters
from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterEvent
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_msgs.msg import TFMessage


def yaw_of(q):
    return math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))


class MapData:
    def __init__(self, cells, resolution, origin, frame='map'):
        self.cells = np.asarray(cells, dtype=np.int16)
        self.resolution = float(resolution)
        self.origin = tuple(float(v) for v in origin)
        self.frame = frame
        if self.cells.ndim != 2 or not self.cells.size or self.resolution <= 0:
            raise ValueError('지도 크기 또는 해상도가 잘못되었습니다.')
        if not all(math.isfinite(v) for v in (*self.origin, self.resolution)):
            raise ValueError('지도 좌표는 유한한 값이어야 합니다.')
        self.h, self.w = self.cells.shape

    @classmethod
    def from_msg(cls, msg):
        return cls(np.array(msg.data).reshape(msg.info.height, msg.info.width),
                   msg.info.resolution,
                   (msg.info.origin.position.x, msg.info.origin.position.y,
                    yaw_of(msg.info.origin.orientation)), msg.header.frame_id)

    @classmethod
    def from_yaml(cls, filename):
        meta = yaml.safe_load(Path(filename).read_text())
        if meta.get('mode', 'trinary') != 'trinary':
            raise ValueError('현재 파일 미리보기는 trinary 지도만 지원합니다.')
        path = Path(meta['image'])
        if not path.is_absolute():
            path = Path(filename).parent / path
        with PILImage.open(path) as image:
            rgba = np.asarray(image.convert('RGBA'))
        shade = rgba[:, :, :3].mean(axis=2) / 255.0
        occ = shade if int(meta.get('negate', 0)) else 1.0-shade
        free, occupied = float(meta['free_thresh']), float(meta['occupied_thresh'])
        if not 0 <= free < occupied <= 1:
            raise ValueError('지도 free_thresh / occupied_thresh가 잘못되었습니다.')
        cells = np.full(occ.shape, -1, dtype=np.int16)
        cells[occ < free] = 0
        cells[occ > occupied] = 100
        cells[rgba[:, :, 3] < 255] = -1
        return cls(np.flipud(cells).copy(), meta['resolution'], meta['origin'])

    def pixel_to_world(self, px, py):
        # Scene's y points down; OccupancyGrid row zero is at the bottom.
        lx, ly = px*self.resolution, (self.h-py)*self.resolution
        x, y, yaw = self.origin
        return x+math.cos(yaw)*lx-math.sin(yaw)*ly, y+math.sin(yaw)*lx+math.cos(yaw)*ly

    def world_to_pixel(self, x, y):
        dx, dy = x-self.origin[0], y-self.origin[1]
        c, s = math.cos(self.origin[2]), math.sin(self.origin[2])
        return (c*dx+s*dy)/self.resolution, self.h-(-s*dx+c*dy)/self.resolution

    def free_at(self, x, y):
        px, py = self.world_to_pixel(x, y)
        col, row = math.floor(px), math.floor(self.h-py)
        return 0 <= row < self.h and 0 <= col < self.w and self.cells[row, col] == 0


class MapView(QtWidgets.QGraphicsView):
    posePicked = QtCore.pyqtSignal(float, float, float)

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setMinimumSize(360, 220)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setDragMode(self.ScrollHandDrag)
        self.mode = 'pan'
        self.data = None
        self.waypoints = []
        self.robot = None
        self.plan = []
        self.press_point = None
        self.background = None
        self.overlays = []

    def set_map(self, data):
        reset = self.data is None or (self.data.w, self.data.h, self.data.origin, self.data.resolution) != (data.w, data.h, data.origin, data.resolution)
        self.data = data
        gray = np.where(data.cells < 0, 180, 255-data.cells.clip(0, 100)*255//100).astype(np.uint8)
        gray = np.ascontiguousarray(np.flipud(gray))
        image = QtGui.QImage(gray.data, data.w, data.h, gray.strides[0], QtGui.QImage.Format_Grayscale8).copy()
        if self.background is None:
            self.background = self.scene().addPixmap(QtGui.QPixmap.fromImage(image))
        else:
            self.background.setPixmap(QtGui.QPixmap.fromImage(image))
        self.scene().setSceneRect(0, 0, data.w, data.h)
        self.draw_overlays()
        if reset:
            self.fit_map()

    def fit_map(self):
        if self.data:
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def draw_overlays(self):
        for item in self.overlays:
            self.scene().removeItem(item)
        self.overlays = []
        if self.data is None:
            return
        def line(points, color, dashed=False):
            if len(points) < 2:
                return
            path = QtGui.QPainterPath()
            path.moveTo(*self.data.world_to_pixel(*points[0][:2]))
            for p in points[1:]:
                path.lineTo(*self.data.world_to_pixel(*p[:2]))
            pen = QtGui.QPen(QtGui.QColor(color), 2)
            pen.setCosmetic(True)
            if dashed:
                pen.setStyle(QtCore.Qt.DashLine)
            self.overlays.append(self.scene().addPath(path, pen))
        line(self.waypoints, '#1767b0', True)
        line(self.plan, '#009040')
        for i, (x, y, yaw) in enumerate(self.waypoints):
            px, py = self.data.world_to_pixel(x, y)
            item = self.scene().addEllipse(px-2, py-2, 4, 4, QtGui.QPen(QtCore.Qt.blue), QtGui.QBrush(QtCore.Qt.blue))
            self.overlays.append(item)
            label = self.scene().addSimpleText(str(i+1))
            label.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
            label.setPos(px+3, py+3)
            self.overlays.append(label)
        if self.robot:
            x, y, yaw = self.robot
            px, py = self.data.world_to_pixel(x, y)
            r = .22/self.data.resolution
            self.overlays.append(self.scene().addEllipse(px-r, py-r, 2*r, 2*r, QtGui.QPen(QtCore.Qt.red)))
            end = self.data.world_to_pixel(x+.35*math.cos(yaw), y+.35*math.sin(yaw))
            self.overlays.append(self.scene().addLine(px, py, *end, QtGui.QPen(QtCore.Qt.red)))

    def wheelEvent(self, event):
        self.scale(1.2 if event.angleDelta().y() > 0 else 1/1.2,
                   1.2 if event.angleDelta().y() > 0 else 1/1.2)

    def mousePressEvent(self, event):
        if self.mode != 'pan' and self.data and event.button() == QtCore.Qt.LeftButton:
            self.press_point = self.mapToScene(event.pos())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.press_point is not None and event.button() == QtCore.Qt.LeftButton:
            start = self.press_point
            self.press_point = None
            end = self.mapToScene(event.pos())
            x, y = self.data.pixel_to_world(start.x(), start.y())
            ex, ey = self.data.pixel_to_world(end.x(), end.y())
            yaw = math.atan2(ey-y, ex-x) if math.hypot(ex-x, ey-y) > .03 else 0.0
            self.posePicked.emit(x, y, yaw)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class RosBridge(QtCore.QObject):
    mapReceived = QtCore.pyqtSignal(object)
    poseReceived = QtCore.pyqtSignal(object)
    planReceived = QtCore.pyqtSignal(object)
    navStatus = QtCore.pyqtSignal(str)
    driverStateReceived = QtCore.pyqtSignal(str, object)

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.namespace = node.get_namespace().rstrip('/')
        self.calls = []
        self.goal = None
        self.pending_goal = False
        self.cancel_requested = False
        self.maintenance = False
        self.external_active = {}
        self.autonomous_drivers = {}
        # Nodes publishing the selected cmd_vel endpoint are authoritative.
        # Their node namespace may differ from the topic namespace when they
        # publish an absolute topic (for example /robot1/pure_pursuit ->
        # /cmd_vel), so retain their exact fully-qualified node names.
        self.publisher_drivers = set()
        self.driver_queries = set()
        self.driver_stop_pending = set()
        self.driver_generations = {}
        self.driver_discovery_due = 0.0
        self.pose = None
        # /map은 TRANSIENT_LOCAL로 보통 한 번만 발행된다. 조종기가 나중에
        # 연결되면 MapPanel의 signal 연결보다 먼저 수신될 수 있으므로 보관한다.
        self.latest_map = None
        self.frame = 'map'
        self.action = ActionClient(node, FollowWaypoints, self.endpoint('follow_waypoints'))
        self.initial_pub = node.create_publisher(PoseWithCovarianceStamped, self.endpoint('initialpose'), 1)
        self.goal_pose_pub = node.create_publisher(
            PoseStamped, self.endpoint('goal_pose'), 1)
        self.buffer = Buffer(node=node)
        self.listener = TransformListener(self.buffer, node, spin_thread=False)
        durable = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.subs = [node.create_subscription(OccupancyGrid, self.endpoint('map'), self.on_map, durable),
                     node.create_subscription(NavPath, self.endpoint('plan'), self.on_plan, 1),
                     node.create_subscription(ParameterEvent, '/parameter_events',
                                              self.on_parameter_event, 100)]
        if self.namespace:
            # Some multi-robot setups share /tf but prefix frame IDs. Only ingest
            # transforms whose child belongs to the selected vehicle.
            self.subs.append(node.create_subscription(TFMessage, '/controller_shared_tf',
                             lambda msg: self.shared_tf(msg, False), 100))
            self.subs.append(node.create_subscription(TFMessage, '/controller_shared_tf_static',
                             lambda msg: self.shared_tf(msg, True), durable))
        for name in ('navigate_to_pose', 'navigate_through_poses', 'follow_waypoints'):
            self.subs.append(node.create_subscription(GoalStatusArray, self.endpoint(name+'/_action/status'),
                lambda m, name=name: self.on_action_status(name, m), durable))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        # Parameter events arrive on the ROS executor thread.  Route state
        # mutations through Qt so GUI reads and writes stay on one thread.
        self.driverStateReceived.connect(self.set_autonomous_driver_state)
        self.timer.start(100)

    def shared_tf(self, msg, static):
        prefix = self.namespace.strip('/')+'/'
        for transform in msg.transforms:
            if transform.child_frame_id.startswith(prefix):
                if static:
                    self.buffer.set_transform_static(transform, 'selected_vehicle_shared_tf')
                else:
                    self.buffer.set_transform(transform, 'selected_vehicle_shared_tf')

    def endpoint(self, name):
        return self.namespace+'/'+name.lstrip('/')

    @property
    def nav_busy(self):
        return self.maintenance or self.pending_goal or self.goal is not None or any(self.external_active.values())

    @property
    def custom_drive_busy(self):
        # Unknown is treated as active until the parameter query completes.
        # This prevents manual /cmd_vel from competing with a newly discovered
        # autonomous controller.
        return bool(self.driver_queries or self.driver_stop_pending) or any(
            enabled is not False for enabled in self.autonomous_drivers.values())

    @property
    def busy(self):
        return self.nav_busy or self.custom_drive_busy

    @staticmethod
    def is_autonomous_driver(node_name):
        return 'pure_pursuit' in node_name.rsplit('/', 1)[-1]

    def owns_autonomous_driver(self, node_name):
        """Return whether a driver belongs to the selected robot namespace."""
        normalized = '/' + node_name.strip('/')
        if not self.is_autonomous_driver(normalized):
            return False
        if self.namespace:
            return normalized.startswith(self.namespace + '/')
        # Selecting '/' must not control /robot1, /robot2, ... drivers.
        return normalized.count('/') == 1

    def tracks_autonomous_driver(self, node_name):
        """Return whether parameter events belong to the active driver set."""
        normalized = '/' + node_name.strip('/')
        if not self.is_autonomous_driver(normalized):
            return False
        if self.publisher_drivers:
            return normalized in self.publisher_drivers
        return self.owns_autonomous_driver(normalized)

    @QtCore.pyqtSlot(str, object)
    def set_autonomous_driver_state(self, node_name, enabled):
        node_name = '/' + node_name.strip('/')
        if (node_name in self.autonomous_drivers
                and self.tracks_autonomous_driver(node_name)):
            self.autonomous_drivers[node_name] = enabled

    def on_parameter_event(self, msg):
        node_name = '/' + msg.node.strip('/')
        if not self.tracks_autonomous_driver(node_name):
            return
        for parameter in list(msg.new_parameters) + list(msg.changed_parameters):
            if parameter.name == 'drive_enabled':
                value = Parameter.from_parameter_msg(parameter).value
                self.driverStateReceived.emit(node_name, bool(value))
        if any(parameter.name == 'drive_enabled'
               for parameter in msg.deleted_parameters):
            self.driverStateReceived.emit(node_name, None)

    def refresh_autonomous_drivers(self):
        """Discover Pure Pursuit nodes and query their drive lock."""
        services = self.node.get_service_names_and_types()
        suffix = '/set_parameters_atomically'
        service_type = 'rcl_interfaces/srv/SetParametersAtomically'
        service_nodes = {
            name.removesuffix(suffix)
            for name, types in services
            if name.endswith(suffix) and service_type in types
            and self.owns_autonomous_driver(name.removesuffix(suffix))
        }
        # A publisher can become visible just before its parameter services.
        # Treat it as unknown/busy during that graph propagation window.
        publisher_nodes = set()
        try:
            for info in self.node.get_publishers_info_by_topic(
                    self.endpoint('cmd_vel')):
                node_name = '/'.join((
                    info.node_namespace.rstrip('/'), info.node_name))
                node_name = '/' + node_name.strip('/')
                if self.is_autonomous_driver(node_name):
                    publisher_nodes.add(node_name)
        except (AttributeError, RuntimeError, TypeError):
            pass
        # Exact publishers of the selected velocity topic are the strongest
        # ownership signal.  Fall back to namespace-scoped parameter services
        # only while no matching publisher endpoint is visible.
        if publisher_nodes:
            self.publisher_drivers = publisher_nodes
            discovered = publisher_nodes
        else:
            self.publisher_drivers = set()
            discovered = service_nodes
        for node_name in set(self.autonomous_drivers) - discovered:
            if node_name in self.driver_stop_pending:
                continue
            self.driver_queries.discard(node_name)
            self.autonomous_drivers.pop(node_name, None)
            self.driver_generations.pop(node_name, None)
        for node_name in discovered:
            if node_name not in self.autonomous_drivers:
                self.autonomous_drivers[node_name] = None
                self.driver_generations.setdefault(node_name, 0)
            if (self.autonomous_drivers[node_name] is None
                    and node_name not in self.driver_queries
                    and node_name not in self.driver_stop_pending):
                self.query_autonomous_driver(node_name)

    def query_autonomous_driver(self, node_name):
        request = GetParameters.Request()
        request.names = ['drive_enabled']
        self.driver_queries.add(node_name)
        generation = self.driver_generations.get(node_name, 0)

        def done(result, error):
            self.driver_queries.discard(node_name)
            if node_name not in self.autonomous_drivers:
                return
            # A stop request has priority over an older read even when its Set
            # response happens to arrive before this Get response.
            if generation != self.driver_generations.get(node_name, 0):
                return
            if error or not result or not result.values:
                self.autonomous_drivers[node_name] = None
                return
            if result.values[0].type == Parameter.Type.NOT_SET.value:
                self.autonomous_drivers[node_name] = None
                return
            value = Parameter.from_parameter_msg(ParameterMsg(
                name='drive_enabled', value=result.values[0])).value
            self.autonomous_drivers[node_name] = bool(value)

        self.call(GetParameters, node_name + '/get_parameters', request, done,
                  timeout=3)

    def disable_autonomous_drive(self):
        """Atomically disable every discovered Pure Pursuit cmd_vel source."""
        self.refresh_autonomous_drivers()
        had_pending = bool(self.driver_stop_pending)
        targets = [name for name, enabled in self.autonomous_drivers.items()
                   if enabled is not False
                   and name not in self.driver_stop_pending]
        if not targets:
            return had_pending
        for node_name in targets:
            request = SetParametersAtomically.Request()
            request.parameters = [Parameter(
                'drive_enabled', value=False).to_parameter_msg()]
            self.driver_generations[node_name] = (
                self.driver_generations.get(node_name, 0) + 1)
            self.autonomous_drivers[node_name] = None
            self.driver_stop_pending.add(node_name)

            def done(result, error, node_name=node_name):
                self.driver_stop_pending.discard(node_name)
                successful = bool(
                    result and result.result and result.result.successful)
                self.autonomous_drivers[node_name] = False if successful else None
                if not successful:
                    reason = error or (
                        result.result.reason if result and result.result else
                        '응답 없음')
                    self.navStatus.emit(
                        f'{node_name} 자동주행 정지 실패: {reason}')
                elif (not self.driver_stop_pending
                      and self.autonomous_drivers
                      and all(enabled is False for enabled
                              in self.autonomous_drivers.values())):
                    self.navStatus.emit('자동주행 정지 완료: drive_enabled=false')

            self.call(SetParametersAtomically,
                      node_name + '/set_parameters_atomically', request, done,
                      timeout=3)
        self.navStatus.emit(
            f'자동주행 정지 요청 중: {len(targets)}개 제어 노드')
        return True

    def on_action_status(self, name, msg):
        self.external_active[name] = any(s.status in (1, 2, 3) for s in msg.status_list)

    def on_map(self, msg):
        if msg.info.width and msg.info.height:
            self.frame = msg.header.frame_id
            self.latest_map = MapData.from_msg(msg)
            self.mapReceived.emit(self.latest_map)

    def on_plan(self, msg):
        if msg.header.frame_id == self.frame:
            self.planReceived.emit([(p.pose.position.x, p.pose.position.y) for p in msg.poses])

    def poll(self):
        for call in self.calls[:]:
            client, request, future, end, callback = call
            if future is None and client.service_is_ready():
                future = client.call_async(request)
                call[2] = future
            if (future is not None and future.done()) or time.monotonic() > end:
                self.calls.remove(call)
                try:
                    if future is None:
                        callback(None, '서비스 없음: '+client.srv_name)
                    elif not future.done():
                        future.cancel()
                        callback(None, '응답 시간 초과 — 적용 여부는 다시 조회하세요.')
                    else:
                        try:
                            result = future.result()
                        except Exception as exc:
                            callback(None, str(exc))
                        else:
                            callback(result, None)
                finally:
                    self.node.destroy_client(client)
        self.pose = None
        for base in ((self.namespace.strip('/')+'/base_footprint', self.namespace.strip('/')+'/base_link') if self.namespace else ()) + ('base_footprint', 'base_link'):
            try:
                tf = self.buffer.lookup_transform(self.frame, base, Time(), timeout=Duration())
                age = (self.node.get_clock().now()-Time.from_msg(tf.header.stamp)).nanoseconds/1e9
                if not -.5 <= age <= 2:
                    continue
                self.pose = (tf.transform.translation.x, tf.transform.translation.y, yaw_of(tf.transform.rotation))
                break
            except TransformException:
                pass
        self.poseReceived.emit(self.pose)
        now = time.monotonic()
        if now >= self.driver_discovery_due:
            self.driver_discovery_due = now + 1.0
            self.refresh_autonomous_drivers()

    def call(self, kind, name, request, callback, timeout=8):
        client = self.node.create_client(kind, name)
        self.calls.append([client, request, None, time.monotonic()+timeout, callback])

    def initial_pose(self, x, y, yaw):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
        msg.pose.covariance[0] = msg.pose.covariance[7] = .25
        msg.pose.covariance[35] = .06854
        self.initial_pub.publish(msg)

    def publish_goal_pose(self, point):
        """Publish one map goal for controllers that subscribe /goal_pose."""
        x, y, yaw = point
        msg = PoseStamped()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = float(x), float(y)
        msg.pose.orientation.z = math.sin(float(yaw) / 2.0)
        msg.pose.orientation.w = math.cos(float(yaw) / 2.0)
        if self.node.count_subscribers(self.goal_pose_pub.topic_name) < 1:
            self.navStatus.emit(
                'Goal 수신 노드가 없습니다. /goal_pose를 구독하는 주행 '
                '노드를 먼저 실행하세요.')
            return False
        self.goal_pose_pub.publish(msg)
        self.navStatus.emit(
            f'Goal 전송: x={x:.2f}, y={y:.2f}, '
            f'yaw={math.degrees(yaw):.0f}°. 경로 확인 후 drive_enabled=true')
        return True

    def follow(self, points):
        if self.busy:
            self.navStatus.emit('이미 이동 중입니다. 취소 완료 후 다시 실행하세요.')
            return
        if self.pose is None:
            self.navStatus.emit('현재 위치 TF 없음/지연: 초기 위치와 Nav2 상태를 확인하세요.')
            return
        if not self.action.server_is_ready():
            self.navStatus.emit('FollowWaypoints 서버 없음: Nav2를 먼저 실행하세요.')
            return
        request = FollowWaypoints.Goal()
        for x, y, yaw in points:
            p = PoseStamped()
            p.header.frame_id = self.frame
            p.header.stamp = self.node.get_clock().now().to_msg()
            p.pose.position.x, p.pose.position.y = x, y
            p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
            request.poses.append(p)
        self.pending_goal = True
        self.cancel_requested = False
        self.navStatus.emit(f'{len(points)}개 경유점 요청 중')
        future = self.action.send_goal_async(request, feedback_callback=lambda m: self.navStatus.emit(f'경유점 {m.feedback.current_waypoint+1}/{len(points)} 이동 중'))
        future.add_done_callback(self.goal_response)

    def goal_response(self, future):
        self.pending_goal = False
        try:
            self.goal = future.result()
            if not self.goal.accepted:
                self.goal = None
                self.navStatus.emit('Nav2가 목표를 거부했습니다.')
                return
            self.goal.get_result_async().add_done_callback(self.goal_result)
            if self.cancel_requested:
                self.goal.cancel_goal_async()
        except Exception as exc:
            self.goal = None
            self.navStatus.emit('목표 전송 오류: '+str(exc))

    def goal_result(self, future):
        self.goal = None
        try:
            result = future.result()
            missed = list(result.result.missed_waypoints)
            if result.status == GoalStatus.STATUS_SUCCEEDED and not missed:
                self.navStatus.emit('모든 경유점 도착 완료')
            elif result.status == GoalStatus.STATUS_CANCELED:
                self.navStatus.emit('이동 취소 완료')
            else:
                self.navStatus.emit(f'이동 실패: status={result.status}, 미도착 번호={[v+1 for v in missed]}')
        except Exception as exc:
            self.navStatus.emit('결과 수신 오류: '+str(exc))

    def cancel(self):
        driver_requested = self.disable_autonomous_drive()
        if not self.nav_busy:
            if not driver_requested:
                self.navStatus.emit('활성 이동 명령이 없습니다.')
            return
        self.cancel_requested = True
        self.navStatus.emit(
            'Nav2 취소 요청 중 — 종료 확인 전 수동 이동 잠금')
        for name in ('navigate_to_pose', 'navigate_through_poses', 'follow_waypoints'):
            self.call(CancelGoal, self.endpoint(name+'/_action/cancel_goal'), CancelGoal.Request(),
                      lambda result, error: self.navStatus.emit('취소 응답: '+error) if error else None)


def make_button(text, callback):
    button = QtWidgets.QPushButton(text)
    button.clicked.connect(callback)
    return button


class MapPanel(QtWidgets.QGroupBox):
    def __init__(self, bridge, stop_manual, domain=None):
        super().__init__('3. 지도 · 목표 / 웨이포인트')
        self.bridge = bridge
        self.stop_manual = stop_manual
        self.domain = domain
        self.console_label = (
            f'control_{domain}' if domain is not None else 'control')
        self.last_reported_map = None
        self.import_path = None
        self.live = False
        self.loading = False
        self.require_initial = False
        # HOME is the map-frame pose at which the current run starts.  It is
        # set explicitly with Initial Pose, or lazily from TF when the first
        # V3 goal is sent.  Keep it in the controller so a HOME request is
        # still available after the V3 node has discarded its current path.
        self.home_pose = None
        self.view = MapView()
        self.status = QtWidgets.QLabel('Nav2를 실행하면 /map을 표시합니다.')
        self.status.setWordWrap(True)
        self.pose_status = QtWidgets.QLabel('로봇 위치: TF 대기')
        self.count_label = QtWidgets.QLabel('경유점 0개')
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_button('지도 YAML 열기', self.import_map))
        row.addWidget(make_button('Nav2에 지도 적용', self.apply_map))
        row.addWidget(make_button('실시간 /map', self.use_live))
        row.addWidget(make_button('화면 맞춤', self.view.fit_map))
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['화면 이동', '초기 위치 지정', '목표 / 경유점 추가'])
        self.mode.currentIndexChanged.connect(self.set_mode)
        row.addWidget(self.mode)
        row.addWidget(self.count_label)
        row.addWidget(make_button('마지막 삭제', self.undo))
        row.addWidget(make_button('전체 삭제', self.clear))
        layout.addLayout(row)
        layout.addWidget(self.view, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_button('Goal 전송', self.start_v3_goal))
        self.home_button = make_button('HOME 복귀', self.return_home)
        row.addWidget(self.home_button)
        row.addWidget(make_button('경로점 순회 시작', self.start_route))
        row.addWidget(make_button('이동 취소', self.cancel))
        layout.addLayout(row)
        layout.addWidget(self.pose_status)
        layout.addWidget(self.status)
        hint = QtWidgets.QLabel('클릭→드래그: 위치·방향 지정 | 휠: 확대 | 파란 점선: 순서 / 초록선: Nav2 경로')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.view.posePicked.connect(self.pick)
        self.latest_map = None
        if bridge:
            bridge.mapReceived.connect(self.on_map)
            bridge.poseReceived.connect(self.on_pose)
            bridge.planReceived.connect(self.on_plan)
            bridge.navStatus.connect(self.status.setText)
            # Durable /map이 signal 연결 전에 도착했어도 즉시 복원한다.
            cached_map = getattr(bridge, 'latest_map', None)
            if cached_map is not None:
                self.on_map(cached_map)

    def set_mode(self, index):
        self.stop_manual()
        self.view.mode = ['pan', 'initial', 'goal'][index]
        self.view.setDragMode(self.view.ScrollHandDrag if index == 0 else self.view.NoDrag)

    def editable(self):
        if self.loading or (self.bridge and self.bridge.busy):
            self.status.setText('이동/지도 변경 중입니다. 완료 또는 취소 후 편집하세요.')
            return False
        return True

    def on_map(self, data):
        self.latest_map = data
        if self.import_path is None:
            previous = self.view.data
            map_changed = previous and (
                previous.frame != data.frame
                or previous.resolution != data.resolution)
            if map_changed:
                self.view.waypoints = []
                self.home_pose = None
                self.require_initial = True
            self.live = True
            self.view.set_map(data)
            self.count_label.setText(f'경유점 {len(self.view.waypoints)}개')
            self.status.setText(
                f'실시간 /map 수신: {data.w}×{data.h}, 해상도 {data.resolution:.3f} m/cell')
            report = (
                data.w, data.h, data.resolution, data.origin, data.frame)
            if report != self.last_reported_map:
                print(
                    f'[{self.console_label}] /map 표시: {data.w}x{data.h}, '
                    f'{data.resolution:.3f} m/cell', flush=True)
                self.last_reported_map = report

    def on_pose(self, pose):
        self.view.robot = pose if self.live else None
        self.view.draw_overlays()
        self.pose_status.setText('로봇 위치: TF 없음/지연' if pose is None else f'로봇: x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={math.degrees(pose[2]):.0f}°')

    def on_plan(self, plan):
        self.view.plan = plan if self.live else []
        self.view.draw_overlays()

    def import_map(self):
        self.stop_manual()
        if not self.editable():
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, '지도 YAML 선택', '/home/kiki', 'Map YAML (*.yaml *.yml)')
        if not filename:
            return
        try:
            data = MapData.from_yaml(filename)
            self.import_path, self.live = filename, False
            self.view.waypoints, self.view.plan, self.view.robot = [], [], None
            self.view.set_map(data)
            self.count_label.setText('경유점 0개')
            self.status.setText('미리보기: Nav2에 지도 적용 후 초기 위치를 지정하세요. '+filename)
        except Exception as exc:
            self.status.setText('지도 열기 실패: '+str(exc))

    def apply_map(self):
        self.stop_manual()
        if not self.editable() or not self.bridge or not self.import_path:
            return
        request = LoadMap.Request()
        request.map_url = self.import_path
        self.loading = True
        self.status.setText('Nav2 map_server에 지도 적용 중 (파일은 map_server PC에도 있어야 합니다)')
        def done(result, error):
            self.loading = False
            if error or result.result != 0:
                self.status.setText('지도 적용 실패: '+(error or str(result.result)))
                return
            self.import_path = None
            self.view.waypoints = []
            self.home_pose = None
            self.require_initial = True
            self.on_map(MapData.from_msg(result.map))
            self.status.setText('지도 적용 완료. 초기 위치를 다시 지정하세요.')
            self.mode.setCurrentIndex(1)
        self.bridge.call(LoadMap, self.bridge.endpoint('map_server/load_map'), request, done)

    def use_live(self):
        self.stop_manual()
        if not self.editable():
            return
        self.import_path = None
        self.view.waypoints = []
        if self.latest_map:
            self.on_map(self.latest_map)
            self.status.setText('현재 Nav2 /map 사용')
        else:
            self.live = False
            self.status.setText('/map 대기: Nav2 또는 SLAM을 실행하세요.')

    def pick(self, x, y, yaw):
        self.stop_manual()
        if not self.editable() or not self.live:
            if not self.live:
                self.status.setText('미리보기 지도에서는 이동 불가. Nav2에 지도를 먼저 적용하세요.')
            return
        if not self.view.data.free_at(x, y):
            self.status.setText('장애물/미확인/지도 밖입니다. 빈 공간을 선택하세요.')
            return
        if self.view.mode == 'initial' and self.bridge:
            self.bridge.initial_pose(x, y, yaw)
            self.home_pose = (float(x), float(y), float(yaw))
            self.require_initial = False
            self.status.setText(
                f'초기 위치/HOME 전송: ({x:.2f}, {y:.2f}). '
                '로봇 표시를 확인하세요.')
        elif self.view.mode == 'goal':
            self.view.waypoints.append((x, y, yaw))
            self.count_label.setText(f'경유점 {len(self.view.waypoints)}개')
            self.status.setText(f'추가: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.0f}°')
            self.view.draw_overlays()

    def undo(self):
        if self.editable() and self.view.waypoints:
            self.view.waypoints.pop()
            self.count_label.setText(f'경유점 {len(self.view.waypoints)}개')
            self.view.draw_overlays()

    def clear(self):
        if self.editable():
            self.view.waypoints = []
            self.count_label.setText('경유점 0개')
            self.view.draw_overlays()

    def start_route(self):
        self.stop_manual()
        if self.require_initial:
            self.status.setText('지도를 변경했습니다. 초기 위치를 먼저 지정하세요.')
            return
        if self.editable() and self.bridge and self.live and self.view.waypoints:
            self.bridge.follow(list(self.view.waypoints))
        elif not self.view.waypoints:
            self.status.setText('경유점을 먼저 추가하세요.')

    def start_v3_goal(self):
        """Send the most recently selected map point on /goal_pose."""
        self.stop_manual()
        if self.require_initial:
            self.status.setText('지도를 변경했습니다. 초기 위치를 먼저 지정하세요.')
            return
        if self.editable() and self.bridge and self.live and self.view.waypoints:
            if self.home_pose is None:
                pose = getattr(self.bridge, 'pose', None)
                if pose is None:
                    self.status.setText(
                        'HOME 위치를 정할 수 없습니다. 로봇 TF 수신 후 다시 '
                        'Goal을 전송하거나 초기 위치를 지정하세요.')
                    return
                self.home_pose = tuple(float(value) for value in pose)
            self.bridge.publish_goal_pose(self.view.waypoints[-1])
        elif not self.view.waypoints:
            self.status.setText('목표점을 먼저 추가하세요.')

    def return_home(self):
        """Replace the current V3 goal with the recorded start pose."""
        self.stop_manual()
        if self.require_initial:
            self.status.setText('지도를 변경했습니다. 초기 위치를 먼저 지정하세요.')
            return
        if not self.editable() or not self.bridge or not self.live:
            return
        if self.home_pose is None:
            self.status.setText(
                'HOME 위치 없음: 초기 위치를 지정하거나 로봇 TF 수신 후 '
                '첫 Goal을 전송하세요.')
            return
        if self.bridge.publish_goal_pose(self.home_pose) is False:
            return
        x, y, _ = self.home_pose
        self.status.setText(f'HOME 복귀 목표 전송: x={x:.2f}, y={y:.2f}')

    def cancel(self):
        self.stop_manual()
        if self.bridge:
            self.bridge.cancel()


DEFAULT_CONFIG = '/home/kiki/turtlebot3/src/turtlebot3/turtlebot3_navigation2/param/humble/waffle_pi.yaml'


def export_parameters(source, target, changes):
    """Preserve full Nav2 config and overlay confirmed node-specific edits."""
    data = yaml.safe_load(Path(source).read_text())
    for (node, name), value in changes.items():
        components = node.strip('/').split('/')
        section = data
        for c in components:
            if not isinstance(section, dict) or c not in section:
                raise ValueError(f'기준 YAML에 노드 {node} 없음. 해당 노드 설정이 있는 파일을 선택하세요.')
            section = section[c]
        params = section['ros__parameters']
        if name in params:
            params[name] = value
        else:
            parts = name.split('.')
            cursor = params
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
    target = Path(target)
    temp = target.with_name(target.name+'.tmp')
    temp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    os.replace(temp, target)


PARAMETER_TYPES = {0: '미설정', 1: 'bool', 2: 'int', 3: 'double', 4: 'string',
                   5: 'byte[]', 6: 'bool[]', 7: 'int[]', 8: 'double[]', 9: 'string[]'}


# ROS ParameterDescriptor는 설명이 비어 있는 경우가 많다. 자주 조절하는
# TurtleBot3/Nav2 값은 조종기에서 바로 판단할 수 있도록 짧은 운용 도움말을
# 보완한다. 키는 ros2 param list에 표시되는 실제 파라미터 이름이다.
PARAMETER_GUIDES = {
    'drive_enabled': (
        'Pure Pursuit 자동주행을 명시적으로 시작하거나 정지합니다.',
        'false는 즉시 정지하며, 초기 위치와 경로를 확인한 뒤 true로 바꿔야 출발합니다.',
        '초기 위치 지정 중: false · 출발할 때: true', ''),
    'lookahead_distance': (
        'Pure Pursuit가 목표를 바라보는 전방 주시 거리입니다.',
        '높이면 선회가 완만해지고, 낮추면 경유점 방향에 더 빠르게 반응합니다.',
        'House 실습 시작값: 0.4~0.7', 'm'),
    'linear_velocity': (
        'Pure Pursuit가 추종할 목표 전진 속도입니다.',
        '높이면 이동은 빨라지지만 경로 오차와 제동거리가 커질 수 있습니다.',
        'Waffle Pi 실내: 0.08~0.15', 'm/s'),
    'max_angular_velocity': (
        'Pure Pursuit 출력에 적용하는 최대 회전 속도입니다.',
        '높이면 급한 방향 오차를 빨리 보정하고, 낮추면 선회가 부드러워집니다.',
        'House 실습: 0.6~1.0', 'rad/s'),
    'linear_acceleration_limit': (
        'Pure Pursuit 선속도가 가속·감속할 때 한 초에 변할 수 있는 최대량입니다.',
        '낮추면 출발과 속도 변경이 부드러워지고, 너무 낮으면 반응이 느립니다.',
        '부드러운 출발: 0.10~0.30', 'm/s²'),
    'angular_acceleration_limit': (
        'Pure Pursuit 각속도가 가속·감속할 때 한 초에 변할 수 있는 최대량입니다.',
        '낮추면 급선회가 줄고, 너무 낮으면 코너에서 경로를 크게 벗어날 수 있습니다.',
        '부드러운 선회: 0.5~1.2', 'rad/s²'),
    'goal_tolerance': (
        'Pure Pursuit가 경유점 도착으로 인정하는 반경입니다.',
        '높이면 쉽게 다음 점으로 넘어가고, 낮추면 더 정확히 접근합니다.',
        'House 실습: 0.15~0.25', 'm'),
    'goal_topic': (
        '지도에서 선택한 단일 Goal/HOME PoseStamped를 받을 토픽입니다.',
        '기본값 /goal_pose를 바꾸면 조종기와 주행 노드가 같은 토픽을 사용해야 합니다.',
        'TurtleBot3 실습 기본값: /goal_pose', ''),
    'pose_timeout': (
        '위치 입력이 끊긴 뒤 정지할 때까지 기다리는 시간입니다.',
        'V3는 AMCL 갱신 사이를 odom으로 보간하므로 odom까지 끊겼을 때 이 제한이 작동합니다.',
        'V2: 0(검사 안 함) · Goal: 1.0', 's'),
    'odom_topic': (
        'AMCL 갱신 사이의 상대 이동을 받을 Odometry 토픽입니다.',
        '실제 로봇과 Gazebo 모두 보통 /odom을 사용하며 변경 후 노드를 재시작해야 합니다.',
        'TurtleBot3 기본값: /odom', ''),
    'odom_anchor_max_age': (
        'AMCL 측정 시각과 기준으로 사용할 odom 표본 사이의 최대 시간 차이입니다.',
        '너무 크면 오래된 odom을 중복 적용할 수 있고, 너무 작으면 AMCL 보정 직후 한 표본을 기다립니다.',
        'Gazebo/TurtleBot3 시작값: 0.20', 's'),
    'odom_reanchor_timeout': (
        'odom이 끊겼다가 돌아왔을 때 기존 좌표 차이를 버리고 기준을 다시 잡는 시간입니다.',
        '통신 복구나 시뮬레이션 일시정지 후 위치가 순간 이동하는 것을 막습니다.',
        'pose_timeout과 같은 값 권장: 1.0', 's'),
    'odom_jump_distance': (
        '연속 odom 표본 사이에서 좌표계 리셋으로 판단할 위치 변화량입니다.',
        'Gazebo 리셋이나 odom 드라이버 재시작의 큰 좌표 점프를 실제 이동으로 적용하지 않습니다.',
        'Waffle Pi 시작값: 0.50', 'm'),
    'odom_jump_angle': (
        '연속 odom 표본 사이에서 좌표계 리셋으로 판단할 방향 변화량입니다.',
        '이 값보다 큰 순간 방향 점프가 오면 현재 지도 위치를 유지한 채 odom 기준만 다시 잡습니다.',
        '시작값: 1.57', 'rad'),
    'control_period': (
        'Pure Pursuit 속도 명령의 계산·발행 간격입니다.',
        '짧으면 제어가 촘촘하지만 계산과 통신량이 늘어납니다.',
        '10 Hz: 0.10', 's'),
    'scan_timeout': (
        '마지막 LiDAR 데이터 이후 수신 끊김으로 판정하는 시간입니다.',
        '낮추면 통신 이상에 빨리 정지하고, 너무 낮으면 순간 지연에도 멈출 수 있습니다.',
        '실내 주행: 0.5~1.0', 's'),
    'emergency_stop_distance': (
        '전방 장애물 때문에 전진 속도를 즉시 0으로 만드는 긴급 거리입니다.',
        '높이면 일찍 멈추고, 낮추면 충돌 여유가 줄어듭니다.',
        'Goal 시작값: 0.12 (차체 앞면 기준)', 'm'),
    'front_stop_distance': (
        '전방 장애물을 만나 회피 선회를 시작하는 거리입니다.',
        '높이면 넓게 피하고, 낮추면 장애물에 더 가까이 접근한 뒤 회피합니다.',
        'Goal 시작값: 0.30 (차체 앞면 기준)', 'm'),
    'front_clear_distance': (
        '제자리 선회를 끝낼 만큼 전방이 다시 열렸다고 판단하는 거리입니다.',
        '높이면 더 넓은 공간을 확보한 뒤 전진하고, 낮추면 선회를 일찍 끝냅니다.',
        'front_stop_distance보다 크게 설정', 'm'),
    'front_slow_distance': (
        '전방 장애물에 접근할 때 감속을 시작하는 거리입니다.',
        '높이면 일찍 감속해 안정적이고, 낮추면 장애물 가까이까지 정상 속도를 유지합니다.',
        'Goal 시작값: 0.60 (차체 앞면 기준)', 'm'),
    'front_corridor_half_width': (
        '로봇 중심선 양옆에서 전방 장애물을 검사할 가상 통로의 반폭입니다.',
        '높이면 차체 옆 장애물에도 민감하고, 낮추면 좁은 통로를 지나기 쉬워집니다.',
        '차체 반폭과 안전 여유를 포함해 설정', 'm'),
    'scan_min_hits': (
        '장애물로 확정할 때 필요한 LiDAR 측정점의 최소 개수입니다.',
        '높이면 단발 잡음을 거르지만 얇은 장애물을 놓칠 수 있고, 낮추면 더 민감합니다.',
        'LDS 실내 시작값: 2~4', '개'),
    'spin_clearance_radius': (
        '제자리 선회 전에 로봇 주변 360도에서 확보해야 하는 최소 공간입니다.',
        '높이면 선회 중 접촉 위험은 줄지만 좁은 공간에서 회피를 시작하기 어렵습니다.',
        '차체 반경보다 약간 크게 설정', 'm'),
    'side_clearance': (
        '회피 중 장애물 측면과 유지하려는 목표 간격입니다.',
        '높이면 넓게 우회하고, 낮추면 좁은 통로를 지나기 쉽지만 충돌 여유가 줄어듭니다.',
        'Waffle Pi 실내: 0.30~0.45', 'm'),
    'avoidance_linear_velocity': (
        '장애물 측면을 따라 우회할 때의 전진 속도입니다.',
        '높이면 회피가 빨라지지만 라이다 반응과 제동 여유가 줄어듭니다.',
        '안전한 시작값: 0.05~0.10', 'm/s'),
    'avoidance_angular_velocity': (
        '전방이 막혔을 때 공간이 열린 쪽으로 도는 회전 속도입니다.',
        '높이면 빠르게 방향을 바꾸고, 낮추면 선회가 완만하지만 오래 걸립니다.',
        '실내 시작값: 0.4~0.7', 'rad/s'),
    'minimum_turn_angle': (
        '회피 선회를 시작한 뒤 반드시 돌아야 하는 최소 각도입니다.',
        '높이면 같은 장애물을 다시 향하는 현상을 줄이지만 경로 이탈이 커질 수 있습니다.',
        '실내 시작값: 0.35~0.8', 'rad'),
    'clear_hold_time': (
        '전방이 열린 상태가 연속으로 유지되어야 선회를 끝내는 시간입니다.',
        '높이면 센서 한두 프레임의 빈틈에 반응하지 않고, 낮추면 전진 복귀가 빨라집니다.',
        '실내 시작값: 0.3~0.8', 's'),
    'wall_follow_gain': (
        '회피 중 측면 목표 간격 오차를 회전 명령으로 바꾸는 게인입니다.',
        '높이면 간격을 강하게 보정하지만 좌우 흔들림이 커질 수 있습니다.',
        '실내 시작값: 1.2~2.2', ''),
    'obstacle_bias_gain': (
        '일반 경로 추종 중 가까운 장애물의 반대쪽으로 조향하는 보정 게인입니다.',
        '높이면 일찍 비켜가지만 원래 경로에서 크게 벗어나거나 흔들릴 수 있습니다.',
        '실내 시작값: 0.4~1.0', ''),
    'avoidance_min_duration': (
        '장애물을 만난 뒤 원래 경로로 복귀하기 전 유지하는 최소 회피 시간입니다.',
        '높이면 장애물을 충분히 지난 뒤 복귀하고, 낮추면 같은 장애물을 다시 만날 수 있습니다.',
        '실내 시작값: 1.0~2.5', 's'),
    'avoidance_max_duration': (
        '한 방향 회피에서 진전이 없다고 판단해 반대 방향을 시도하는 시간입니다.',
        '낮추면 방향을 자주 바꾸고, 높이면 막힌 방향을 오래 고집할 수 있습니다.',
        'House 실습: 12~25', 's'),
    'avoidance_retry_limit': (
        '회피 제한 시간을 넘겼을 때 반대 방향으로 다시 시도할 최대 횟수입니다.',
        '높이면 복구 기회가 늘지만 막힌 곳에서 오래 반복할 수 있습니다.',
        'Goal 시작값: 1', '회'),
    'rejoin_heading_tolerance': (
        '회피 후 목표 방향 추종으로 복귀할 때 허용하는 방향 오차입니다.',
        '높이면 빨리 복귀하고, 낮추면 목표 쪽으로 충분히 정렬한 뒤 복귀합니다.',
        '실내 시작값: 0.4~0.7', 'rad'),
    'avoidance_progress_distance': (
        '회피 시작점보다 목표에 이만큼 가까워져야 경로 복귀를 허용하는 진행량입니다.',
        '높이면 확실한 우회 후 복귀하지만 복귀가 늦어질 수 있습니다.',
        '실내 시작값: 0.05~0.20', 'm'),
    'waypoints_x': (
        '순서대로 방문할 경유점의 지도 x 좌표 배열입니다.',
        'waypoints_y와 같은 개수로 작성해야 하며 실행 중에는 변경할 수 없습니다.',
        'YAML 수정 후 노드 재시작', 'm'),
    'waypoints_y': (
        '순서대로 방문할 경유점의 지도 y 좌표 배열입니다.',
        'waypoints_x와 같은 개수로 작성해야 하며 실행 중에는 변경할 수 없습니다.',
        'YAML 수정 후 노드 재시작', 'm'),
    'use_sim_time': (
        '노드가 Gazebo의 /clock을 사용할지 정합니다.',
        'Gazebo에서는 true, 실제 로봇에서는 false로 맞춰야 TF 시간이 일치합니다.',
        'Gazebo: true · 실제 TurtleBot3: false', ''),
    'controller_frequency': (
        '로컬 경로 제어 명령을 계산하는 주기입니다.',
        '높이면 반응은 빨라지지만 CPU 부하와 제어 주기 누락이 늘어납니다.',
        'TurtleBot3: 10~15', 'Hz'),
    'failure_tolerance': (
        '컨트롤러 계산 실패를 복구 동작 전까지 허용하는 시간입니다.',
        '높이면 순간 오류를 견디고, 낮추면 빠르게 정지·복구합니다.',
        '실내 주행: 1.0~3.0', 's'),
    'progress_checker.required_movement_radius': (
        '진행 중이라고 인정하기 위해 움직여야 하는 최소 거리입니다.',
        '높이면 정체를 빨리 감지하지만 좁은 곳과 제자리 선회에서 오판할 수 있습니다.',
        'TurtleBot3 실내: 0.10~0.20', 'm'),
    'progress_checker.movement_time_allowance': (
        '최소 이동 거리를 달성할 때까지 기다리는 시간입니다.',
        '높이면 좁은 통로에서 여유가 생기고, 낮추면 정체를 빨리 판정합니다.',
        '실내 주행: 10~20', 's'),
    'general_goal_checker.xy_goal_tolerance': (
        '목표 도착으로 인정하는 위치 반경입니다.',
        '높이면 도착이 쉬워지고, 낮추면 목표점에 더 정확히 접근합니다.',
        '일반 주행: 0.15~0.25', 'm'),
    'general_goal_checker.yaw_goal_tolerance': (
        '목표 도착으로 인정하는 방향 오차입니다.',
        '높이면 불필요한 마지막 제자리 회전이 줄어듭니다.',
        '경유 주행: 0.3~0.6', 'rad'),
    'FollowPath.max_vel_x': (
        'DWB가 허용하는 최대 전진 선속도입니다.',
        '높이면 빨라지지만 제동거리와 경로 이탈 가능성이 커집니다.',
        'Waffle Pi 실내: 0.18~0.26', 'm/s'),
    'FollowPath.max_speed_xy': (
        '평면 이동 속도의 최대 크기입니다.',
        'max_vel_x와 함께 실제 최고속도를 제한합니다.',
        'Waffle Pi 실내: 0.18~0.26', 'm/s'),
    'FollowPath.max_vel_theta': (
        'DWB가 허용하는 최대 회전 속도입니다.',
        '높이면 선회가 빨라지고, 낮추면 코너가 부드러워집니다.',
        '실내 주행: 0.6~1.0', 'rad/s'),
    'FollowPath.acc_lim_x': (
        '전진 가속도 한계입니다.',
        '높이면 출발이 빠르고 거칠어지며, 낮추면 부드럽게 가속합니다.',
        '부드러운 주행: 0.3~0.8', 'm/s²'),
    'FollowPath.acc_lim_theta': (
        '회전 가속도 한계입니다.',
        '높이면 방향 전환이 민감해지고, 낮추면 선회가 완만해집니다.',
        '부드러운 주행: 0.8~1.8', 'rad/s²'),
    'FollowPath.decel_lim_x': (
        '전진 감속도 한계이며 보통 음수로 설정합니다.',
        '절댓값이 크면 급정지하고, 작으면 길게 감속합니다.',
        '부드러운 주행: -0.8~-0.3', 'm/s²'),
    'FollowPath.decel_lim_theta': (
        '회전 감속도 한계이며 보통 음수로 설정합니다.',
        '절댓값이 크면 회전을 급히 멈춥니다.',
        '부드러운 주행: -1.8~-0.8', 'rad/s²'),
    'FollowPath.vx_samples': (
        '평가할 선속도 후보 개수입니다.',
        '높이면 후보가 촘촘하지만 CPU 사용량이 증가합니다.',
        'TurtleBot3: 15~20', '개'),
    'FollowPath.vtheta_samples': (
        '평가할 회전 속도 후보 개수입니다.',
        '높이면 선회 선택은 촘촘해지지만 제어 계산이 느려집니다.',
        'TurtleBot3: 20~30', '개'),
    'FollowPath.sim_time': (
        '각 후보 궤적을 미래로 예측하는 시간입니다.',
        '높이면 멀리 내다보지만 둔해질 수 있고, 낮추면 민첩하지만 근시안적입니다.',
        '실내 주행: 1.5~2.0', 's'),
    'FollowPath.xy_goal_tolerance': (
        'DWB 내부에서 사용하는 목표 위치 허용 오차입니다.',
        'goal checker의 xy 허용 오차와 비슷하게 맞추면 도착 판정 충돌을 줄입니다.',
        '일반 주행: 0.15~0.25', 'm'),
    'FollowPath.trans_stopped_velocity': (
        '이 속도 아래를 정지 상태로 판단합니다.',
        '높이면 아직 움직여도 정지로 판단할 수 있습니다.',
        'TurtleBot3: 0.02~0.05', 'm/s'),
    'FollowPath.prune_plan': (
        '이미 지나온 전역 경로를 잘라낼지 정합니다.',
        'true를 사용하면 경로 추종 계산량을 줄일 수 있습니다.',
        '일반 주행: true', ''),
    'FollowPath.prune_distance': (
        '현재 위치를 찾기 위해 뒤쪽까지 검색할 전역 경로 거리입니다.',
        '높이면 긴 경로 재수신에 강하지만 검색량이 조금 늘어납니다.',
        '현재 House 지도: 20.0', 'm'),
    'FollowPath.forward_prune_distance': (
        '로봇 앞쪽에서 로컬 컨트롤러가 사용할 경로 길이입니다.',
        '높이면 멀리 내다보고, 낮추면 가까운 굴곡에 민감해집니다.',
        '실내 주행: 1.5~3.0', 'm'),
    'FollowPath.PathAlign.scale': (
        '로봇 방향을 전역 경로 방향에 맞추는 가중치입니다.',
        '높이면 경로 방향을 강하게 따르고 급한 보정이 늘 수 있습니다.',
        '기본 근처에서 소폭 조정', ''),
    'FollowPath.PathDist.scale': (
        '전역 경로와의 거리 오차 가중치입니다.',
        '높이면 선을 더 엄격히 따라가고 장애물 우회 여유가 줄 수 있습니다.',
        '기본 근처에서 소폭 조정', ''),
    'FollowPath.GoalAlign.scale': (
        '목표 방향을 향하려는 가중치입니다.',
        '높이면 일찍 목표 방향으로 선회합니다.',
        '기본 근처에서 소폭 조정', ''),
    'FollowPath.GoalDist.scale': (
        '목표까지 거리를 줄이려는 가중치입니다.',
        '높이면 목표로 직접 향하려는 성향이 강해집니다.',
        '기본 근처에서 소폭 조정', ''),
    'inflation_layer.inflation_radius': (
        '장애물 주위에 비용을 퍼뜨리는 최대 반경입니다.',
        '높이면 넓게 피하고, 낮추면 좁은 통로를 통과하기 쉬워집니다.',
        'Waffle Pi 실내: 0.30~0.45', 'm'),
    'inflation_layer.cost_scaling_factor': (
        '장애물에서 멀어질수록 비용이 감소하는 속도입니다.',
        '높이면 비용이 빨리 감소해 장애물 가까이 갈 수 있고, 낮추면 넓게 피합니다.',
        '실내 주행: 3.0~8.0', ''),
    'robot_radius': (
        '코스트맵에서 사용하는 원형 로봇 반경입니다.',
        '높이면 안전 여유가 커집니다. 실제 차체보다 작게 설정하면 충돌 위험이 생깁니다.',
        'Waffle Pi: 약 0.22', 'm'),
    'obstacle_max_range': (
        '라이다 측정값을 장애물로 등록할 최대 거리입니다.',
        '높이면 먼 장애물도 반영하지만 센서 잡음과 계산 범위가 늘어납니다.',
        'LDS 실내: 2.5~3.0', 'm'),
    'raytrace_max_range': (
        '라이다 광선을 따라 장애물이 없다고 지울 최대 거리입니다.',
        'obstacle_max_range보다 조금 크게 두면 지나간 장애물 흔적을 지우기 쉽습니다.',
        'LDS 실내: 3.0~3.5', 'm'),
    'max_obstacle_height': (
        '이 높이 이하의 센서 점을 장애물 후보로 사용합니다.',
        '2D 라이다에서는 보통 기본값을 유지합니다.',
        'TurtleBot3: 2.0', 'm'),
    'update_frequency': (
        '코스트맵을 센서 자료로 갱신하는 주기입니다.',
        '높이면 장애물 반영이 빨라지지만 CPU 부하가 증가합니다.',
        '로컬 5 · 글로벌 1', 'Hz'),
    'publish_frequency': (
        '코스트맵 토픽을 외부로 발행하는 주기입니다.',
        '높이면 화면 갱신은 빨라지지만 통신·CPU 부하가 증가합니다.',
        '로컬 2 · 글로벌 1', 'Hz'),
    'resolution': (
        '코스트맵 한 셀의 실제 크기입니다.',
        '작으면 정밀하지만 메모리와 계산량이 증가합니다.',
        'TurtleBot3 지도: 0.05', 'm/cell'),
    'track_unknown_space': (
        '미탐색 영역을 별도 unknown으로 유지할지 정합니다.',
        'true면 미탐색 영역을 자유 공간과 구분합니다.',
        '저장 지도 Navigation: true', ''),
    'always_send_full_costmap': (
        '갱신 때마다 전체 코스트맵을 발행할지 정합니다.',
        'false는 변경분 위주라 통신 부하를 줄일 수 있습니다.',
        '성능 우선: false', ''),
    'expected_planner_frequency': (
        '플래너가 기대하는 재계획 주기이며 주로 지연 경고 기준입니다.',
        '높인다고 실제 계산이 빨라지지는 않으며 경고가 늘 수 있습니다.',
        '정적 실내 지도: 5~10', 'Hz'),
    'GridBased.tolerance': (
        '정확한 목표 셀이 막혔을 때 주변에서 허용할 반경입니다.',
        '높이면 근처 대체점으로 경로를 만들기 쉬워집니다.',
        '일반 주행: 0.2~0.5', 'm'),
    'GridBased.use_astar': (
        'Navfn에서 A* 탐색을 사용할지 정합니다.',
        'true는 목표 지향 탐색, false는 기본 Dijkstra 방식입니다.',
        '둘 다 가능; 경로 비교 후 선택', ''),
    'GridBased.allow_unknown': (
        '플래너가 unknown 셀을 통과 후보로 허용할지 정합니다.',
        'true는 미탐색 공간도 계획하며 저장 지도 운용에서는 주의가 필요합니다.',
        '완성 지도: false · 탐색: true', ''),
    'loop_rate': (
        '웨이포인트 상태를 확인하는 반복 주기입니다.',
        '높이면 반응 차이는 작고 로그·CPU 부하가 증가합니다.',
        'Nav2 waypoint follower: 10~20', 'Hz'),
    'stop_on_failure': (
        '경유점 실패 시 전체 임무를 중단할지 정합니다.',
        'true는 실패를 명확히 알리고, false는 다음 점으로 건너뜁니다.',
        '안전한 임무: true', ''),
    'wait_at_waypoint.waypoint_pause_duration': (
        '각 경유점 도착 후 기다리는 시간입니다.',
        '높이면 매 경유점에서 오래 정지합니다.',
        '연속 주행: 0', 'ms'),
    'smoothing_frequency': (
        '속도 스무더가 cmd_vel을 보간해 발행하는 주기입니다.',
        '높이면 출력은 촘촘하지만 CPU와 통신 부하가 증가합니다.',
        'TurtleBot3: 10~20', 'Hz'),
    'max_velocity': (
        '속도 스무더의 [x, y, 회전] 최대속도 배열입니다.',
        'DWB 한계보다 크게 두면 DWB가, 작게 두면 스무더가 최종 속도를 제한합니다.',
        'Waffle Pi 예: [0.22, 0.0, 1.0]', '[m/s, m/s, rad/s]'),
    'max_accel': (
        '속도 스무더의 [x, y, 회전] 최대가속도 배열입니다.',
        '낮추면 출발과 선회가 부드러워지고 응답은 느려집니다.',
        '부드러운 시작 예: [0.5, 0.0, 1.2]', '[m/s², m/s², rad/s²]'),
    'max_decel': (
        '속도 스무더의 [x, y, 회전] 최대감속도 배열이며 보통 음수입니다.',
        '절댓값이 크면 더 급하게 정지합니다.',
        '부드러운 정지 예: [-0.7, 0.0, -1.5]', '[m/s², m/s², rad/s²]'),
    'feedback': (
        '속도 스무더가 명령값(OPEN_LOOP) 또는 측정 속도(CLOSED_LOOP)를 기준으로 동작합니다.',
        'CLOSED_LOOP는 신뢰할 수 있는 odom과 충분한 갱신률이 필요합니다.',
        'Gazebo 기본 실습: OPEN_LOOP', ''),
    'max_rotational_vel': (
        '복구 동작에서 허용하는 최대 회전 속도입니다.',
        '높이면 복구 선회가 빨라지고 급격해집니다.',
        '실내 복구: 0.5~0.8', 'rad/s'),
    'min_rotational_vel': (
        '복구 회전이 사용할 최소 회전 속도입니다.',
        '높이면 회전 시작이 거칠어질 수 있습니다.',
        '실내 복구: 0.2~0.4', 'rad/s'),
    'rotational_acc_lim': (
        '복구 회전의 각가속도 한계입니다.',
        '낮추면 복구 선회가 부드러워집니다.',
        '부드러운 복구: 0.8~1.5', 'rad/s²'),
    'update_min_d': (
        'AMCL이 이동 갱신을 수행하는 최소 이동 거리입니다.',
        '낮추면 위치가 자주 보정되지만 CPU 사용량이 증가합니다.',
        '실내 주행: 0.05~0.15', 'm'),
    'update_min_a': (
        'AMCL이 회전 갱신을 수행하는 최소 각도입니다.',
        '낮추면 회전 중 위치가 자주 보정됩니다.',
        '실내 주행: 0.05~0.15', 'rad'),
    'min_particles': (
        'AMCL이 유지할 최소 파티클 수입니다.',
        '높이면 위치 추정이 안정적일 수 있지만 CPU 부하가 증가합니다.',
        'TurtleBot3: 500 전후', '개'),
    'max_particles': (
        'AMCL이 사용할 최대 파티클 수입니다.',
        '높이면 어려운 위치 추정에 유리하지만 CPU 부하가 증가합니다.',
        'TurtleBot3: 1500~2000', '개'),
    'transform_tolerance': (
        '센서·제어에서 허용하는 TF 시간 여유입니다.',
        '높이면 지연에 관대하지만 오래된 변환을 허용할 수 있습니다.',
        '일반 주행: 0.2~0.5', 's'),
}


def parameter_guide(name):
    """Return curated help for a declared ROS parameter, if available."""
    return PARAMETER_GUIDES.get(name)


def parameter_tooltip(node, name, record):
    """Build compact hover help without adding permanent UI space."""
    value = parameter_text(record.get('value'))
    guide = parameter_guide(name)
    desc = record.get('descriptor')
    lines = [f'{name} = {value}']
    ros_description = desc.description.strip() if desc and desc.description else ''
    if ros_description:
        lines.append(ros_description)
    if guide:
        summary, effect, recommended, unit = guide
        if summary and summary != ros_description:
            lines.append(summary)
        if effect:
            lines.append('조절 영향: '+effect)
        if recommended:
            lines.append('권장: '+recommended+(f' {unit}' if unit and unit not in recommended else ''))
    elif not ros_description:
        lines.append('이 노드는 파라미터 설명을 제공하지 않습니다.')
    if desc:
        ranges = list(desc.floating_point_range)+list(desc.integer_range)
        if ranges:
            rng = ranges[0]
            lines.append(f'허용 범위: {rng.from_value} ~ {rng.to_value} · 간격 {rng.step}')
        if desc.read_only:
            lines.append('읽기 전용 · 설정 파일 수정 후 노드를 재시작해야 합니다.')
    if name in ('inflation_layer.inflation_radius', 'inflation_layer.cost_scaling_factor') and 'costmap' in node:
        lines.append('실행 중 적용 가능 · Nav2 리셋 불필요')
    return '\n'.join(lines)


def parameter_value(name, value):
    result = Parameter.from_parameter_msg(ParameterMsg(name=name, value=value)).value
    return result.tolist() if isinstance(result, array) else result


def parameter_text(value):
    return yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10000).split('\n...')[0].strip()


def parameter_group(name):
    return name.rsplit('.', 1)[0] if '.' in name else '(일반)'


class ParameterPanel(QtWidgets.QGroupBox):
    """QGC-like grouped browser for parameters actually exposed by ROS services."""
    def __init__(self, bridge, domain, stop_manual, autoload=True):
        super().__init__('2. 연결된 노드 파라미터')
        self.bridge = bridge
        self.domain = domain
        self.stop_manual = stop_manual
        self.changes = {}
        self.records = {}
        self.generation = 0
        self.refresh_id = 0
        self.loading = False
        self.write_pending = False
        self.resetting = False
        self.node_errors = []
        self.group_filter = None
        self.reset_baseline = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText('전체 검색: 이름 / 설명 / 노드')
        self.search.setClearButtonEnabled(True)
        row.addWidget(self.search, 1)
        self.changed_only = QtWidgets.QCheckBox('변경값')
        row.addWidget(self.changed_only)
        self.refresh_button = make_button('새로고침', self.refresh_all)
        row.addWidget(self.refresh_button)
        layout.addLayout(row)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.groups = QtWidgets.QTreeWidget()
        self.groups.setHeaderLabel('노드 / 그룹')
        self.groups.setMinimumWidth(110)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['노드', '파라미터', '현재값', '형식', '상태'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(110)
        self.table.setColumnWidth(0, 120)
        self.table.setColumnWidth(1, 210)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 65)
        self.table.setColumnWidth(4, 110)
        self.table.horizontalHeader().setSortIndicator(1, QtCore.Qt.AscendingOrder)
        self.splitter.addWidget(self.groups)
        self.splitter.addWidget(self.table)
        self.splitter.setSizes([150, 470])
        layout.addWidget(self.splitter, 1)

        # Hidden selectors keep a single identity for asynchronous reads/writes.
        self.node_box = QtWidgets.QComboBox(self)
        self.node_box.setEditable(True)
        self.node_box.addItem('/local_costmap/local_costmap')
        self.node_box.hide()
        self.name_box = QtWidgets.QComboBox(self)
        self.name_box.setEditable(True)
        self.name_box.addItem('inflation_layer.inflation_radius')
        self.name_box.hide()
        self.selected_label = QtWidgets.QLabel('목록에서 파라미터를 선택하세요.')
        self.selected_label.setWordWrap(True)
        self.selected_label.setMaximumHeight(38)
        layout.addWidget(self.selected_label)
        row = QtWidgets.QHBoxLayout()
        self.value = QtWidgets.QLineEdit()
        self.value.setPlaceholderText('선택한 값 수정 (YAML 형식)')
        row.addWidget(self.value, 1)
        row.addWidget(make_button('값 읽기', self.read_parameter))
        self.apply_button = make_button('값 적용', self.apply_parameter)
        self.apply_button.setEnabled(False)
        row.addWidget(self.apply_button)
        layout.addLayout(row)
        self.description = QtWidgets.QLabel('ROS 설명·범위·읽기 전용 여부를 표시합니다.')
        self.description.setWordWrap(True)
        self.description.setMaximumHeight(52)
        layout.addWidget(self.description)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_button('변경값 파일 저장', self.save_changes))
        row.addWidget(make_button('Nav2 YAML 저장', self.save_config))
        self.manager_box = QtWidgets.QComboBox()
        self.manager_box.setToolTip('리셋 대상 Nav2 lifecycle manager. 로봇 전원 재부팅이 아닙니다.')
        row.addWidget(self.manager_box, 1)
        self.reset_button = make_button('Nav2 리셋', self.reset_nav2)
        self.reset_button.setToolTip('선택한 관리자의 노드를 cleanup → configure → activate 합니다. 주행 중에는 사용할 수 없습니다.')
        self.reset_button.setEnabled(False)
        row.addWidget(self.reset_button)
        layout.addLayout(row)
        # The full Nav2 export uses this baseline; select it from the file dialog.
        self.config_path = QtWidgets.QLineEdit(DEFAULT_CONFIG, self)
        self.config_path.hide()
        self.status = QtWidgets.QLabel(f'Domain {domain}: 연결된 ROS 노드를 검색합니다. 저장 파일은 자동 재적용되지 않습니다.')
        self.status.setWordWrap(True)
        self.status.setMaximumHeight(44)
        layout.addWidget(self.status)
        self.search.textChanged.connect(self.filter_table)
        self.changed_only.toggled.connect(self.filter_table)
        self.groups.currentItemChanged.connect(self.group_selected)
        self.table.itemSelectionChanged.connect(self.row_selected)
        self.node_box.currentTextChanged.connect(self.selection_changed)
        self.name_box.currentTextChanged.connect(self.selection_changed)
        self.rebuild_groups()
        if bridge and autoload:
            QtCore.QTimer.singleShot(1200, self.refresh_all)

    def selection_changed(self):
        self.generation += 1
        self.value.clear()

    def selection(self):
        return '/'+self.node_box.currentText().strip('/ '), self.name_box.currentText().strip()

    def discover_managers(self, services):
        selected = self.manager_box.currentData()
        self.manager_box.clear()
        names = sorted(name for name, types in services
                       if name.endswith('/manage_nodes') and 'nav2_msgs/srv/ManageLifecycleNodes' in types)
        for name in names:
            self.manager_box.addItem(name.removesuffix('/manage_nodes'), name)
        index = self.manager_box.findData(selected)
        if index < 0:
            index = next((i for i, name in enumerate(names) if 'navigation' in name), 0)
        if names:
            self.manager_box.setCurrentIndex(index)
        self.reset_button.setEnabled(bool(names) and not self.resetting)

    def refresh_all(self):
        if not self.bridge or self.resetting or self.write_pending:
            return
        self.refresh_id += 1
        token = self.refresh_id
        self.loading = True
        self.refresh_button.setEnabled(False)
        self.node_errors = []
        self.records = {}
        self.value.clear()
        self.apply_button.setEnabled(False)
        services = self.bridge.node.get_service_names_and_types()
        namespace = getattr(self.bridge, 'namespace', '')
        if namespace:
            services = [(name, types) for name, types in services if name.startswith(namespace+'/')]
        self.discover_managers(services)
        nodes = sorted(set(name.removesuffix('/list_parameters') for name, types in services
                           if name.endswith('/list_parameters') and 'rcl_interfaces/srv/ListParameters' in types))
        self.rebuild_groups()
        self.status.setText(f'Domain {self.domain}: {len(nodes)}개 노드의 파라미터 조회 중…')
        # Process one node and 64 parameters at a time, never block the Qt loop.
        def next_node():
            if token != self.refresh_id:
                return
            if not nodes:
                self.loading = False
                self.refresh_button.setEnabled(True)
                self.rebuild_groups()
                suffix = f' / 조회 실패 {len(self.node_errors)}개: '+ '; '.join(self.node_errors) if self.node_errors else ''
                self.status.setText(f'{len(self.records)}개 파라미터 조회 완료'+suffix if self.records else '파라미터가 없습니다. 로봇/Nav2 실행과 Domain을 확인하고 새로고침하세요.'+suffix)
                self.status.setToolTip(self.status.text())
                return
            node = nodes.pop(0)
            req = ListParameters.Request()
            req.depth = 0
            def listed(result, error):
                if token != self.refresh_id:
                    return
                if error:
                    self.node_errors.append(node+': '+error)
                    next_node()
                    return
                names = sorted(result.result.names)
                self.fetch_chunks(node, names, token, next_node)
            self.bridge.call(ListParameters, node+'/list_parameters', req, listed, timeout=3)
        next_node()

    def fetch_chunks(self, node, names, token, finished):
        if token != self.refresh_id:
            return
        if not names:
            self.rebuild_groups()
            finished()
            return
        chunk, rest = names[:64], names[64:]
        values_req, desc_req = GetParameters.Request(), DescribeParameters.Request()
        values_req.names = desc_req.names = chunk
        parts = {}
        def got(which, result, error):
            if token != self.refresh_id:
                return
            parts[which] = (result, error)
            if len(parts) != 2:
                return
            vals, value_error = parts['values']
            descs, desc_error = parts['descriptors']
            if value_error:
                self.node_errors.append(node+': '+value_error)
                finished()
                return
            descriptions = {} if desc_error else {d.name:d for d in descs.descriptors}
            for name, value in zip(chunk, vals.values):
                self.records[(node, name)] = dict(value=parameter_value(name, value), type=value.type,
                                                 descriptor=descriptions.get(name))
            self.status.setText(f'조회 중: {node} / {len(self.records)}개 수신')
            self.fetch_chunks(node, rest, token, finished)
        self.bridge.call(GetParameters, node+'/get_parameters', values_req,
                         lambda result,error:got('values',result,error), timeout=3)
        self.bridge.call(DescribeParameters, node+'/describe_parameters', desc_req,
                         lambda result,error:got('descriptors',result,error), timeout=3)

    def rebuild_groups(self):
        wanted = self.group_filter
        self.groups.blockSignals(True)
        self.groups.clear()
        root = QtWidgets.QTreeWidgetItem([f'전체 ({len(self.records)})'])
        root.setData(0, QtCore.Qt.UserRole, None)
        self.groups.addTopLevelItem(root)
        chosen = root
        nodes = sorted(set(node for node, _ in self.records))
        for node in nodes:
            keys = [key for key in self.records if key[0] == node]
            parent = QtWidgets.QTreeWidgetItem([f'{node} ({len(keys)})'])
            parent.setData(0, QtCore.Qt.UserRole, (node, None))
            self.groups.addTopLevelItem(parent)
            if wanted == (node, None):
                chosen = parent
            groups = sorted(set(parameter_group(key[1]) for key in keys))
            for group in groups:
                count = sum(parameter_group(key[1]) == group for key in keys)
                child = QtWidgets.QTreeWidgetItem([f'{group} ({count})'])
                child.setData(0, QtCore.Qt.UserRole, (node, group))
                parent.addChild(child)
                if wanted == (node, group):
                    chosen = child
                    parent.setExpanded(True)
        self.groups.setCurrentItem(chosen)
        self.group_filter = chosen.data(0, QtCore.Qt.UserRole)
        self.groups.blockSignals(False)
        self.filter_table()

    def group_selected(self, item, previous):
        self.group_filter = item.data(0, QtCore.Qt.UserRole) if item else None
        self.filter_table()

    def filter_table(self):
        selected = self.selection()
        query = self.search.text().strip().lower()
        entries = []
        for key, record in self.records.items():
            node, name = key
            desc = record.get('descriptor')
            guide = parameter_guide(name)
            guide_text = ' '.join(part for part in guide if part) if guide else ''
            # Search spans all groups, like QGC; clearing it restores the group.
            if query:
                if query not in (node+' '+name+' '+(desc.description if desc else '')+' '+guide_text).lower():
                    continue
            elif self.group_filter:
                wanted_node, wanted_group = self.group_filter
                if node != wanted_node or (wanted_group and parameter_group(name) != wanted_group):
                    continue
            if self.changed_only.isChecked() and key not in self.changes:
                continue
            entries.append((key, record))
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(entries))
        selected_item = None
        for row, (key, record) in enumerate(sorted(entries)):
            desc = record.get('descriptor')
            state = '읽기 전용' if desc and desc.read_only else '편집 가능' if desc else '설명 미수신'
            if key in self.changes:
                state += ' · 변경'
            texts = [key[0], key[1], parameter_text(record['value']), PARAMETER_TYPES[record['type']], state]
            tooltip = parameter_tooltip(key[0], key[1], record)
            for col, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                item.setData(QtCore.Qt.UserRole, key)
                # Keep the compact QGC-like table; details appear only on hover.
                item.setToolTip(tooltip)
                self.table.setItem(row, col, item)
                if key == selected and col == 1:
                    selected_item = item
        self.table.setSortingEnabled(True)
        if selected_item:
            self.table.setCurrentItem(selected_item)
            self.table.selectRow(selected_item.row())
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)
        if selected_item:
            self.show_record(selected)
        else:
            self.apply_button.setEnabled(False)
            self.value.clear()
            self.selected_label.setText('목록에서 파라미터를 선택하세요.')

    def row_selected(self):
        items = self.table.selectedItems()
        if items:
            key = tuple(items[0].data(QtCore.Qt.UserRole))
            self.show_record(key)

    def show_record(self, key):
        record = self.records.get(key)
        if record is None:
            return
        node, name = key
        self.node_box.setCurrentText(node)
        self.name_box.setCurrentText(name)
        self.selected_label.setText(node+' / '+name)
        self.value.setText(parameter_text(record['value']))
        desc = record.get('descriptor')
        readonly = bool(desc and desc.read_only)
        self.value.setReadOnly(readonly)
        self.apply_button.setEnabled(not readonly and not self.write_pending and not self.resetting and record['type'] != 0)
        tooltip = parameter_tooltip(node, name, record)
        self.selected_label.setToolTip(tooltip)
        lines = tooltip.splitlines()[1:]
        if desc and desc.additional_constraints:
            lines.append(desc.additional_constraints)
        if readonly:
            lines.append('읽기 전용: 실행 중 변경 불가. 시작 설정 파일 수정 및 해당 프로세스 재실행 필요.')
        elif not parameter_guide(name) and not (desc and desc.description):
            lines.append('재시작 필요 여부: ROS 표준 정보 없음. 값 수락과 동작 반영은 노드 구현에 따릅니다.')
        # Only a short preview occupies the panel; hover shows the full guide.
        self.description.setText(' | '.join(lines[:2]))
        self.description.setToolTip(tooltip+'\n'+('\n'.join(lines[2:]) if len(lines) > 2 else ''))

    def list_parameters(self):
        self.refresh_all()

    def read_parameter(self):
        if not self.bridge:
            return
        node, name = self.selection()
        generation = self.generation
        request = GetParameters.Request()
        request.names = [name]
        def done(result, error):
            if error:
                self.status.setText(error)
                return
            if not result.values or result.values[0].type == 0:
                self.status.setText('선언되지 않은 파라미터입니다.')
                return
            prior = self.records.get((node,name), {})
            self.records[(node,name)] = dict(value=parameter_value(name,result.values[0]), type=result.values[0].type,
                                             descriptor=prior.get('descriptor'))
            if generation == self.generation:
                self.show_record((node,name))
            self.status.setText(f'조회 완료: {node} / {name}')
        self.bridge.call(GetParameters, node+'/get_parameters', request, done)

    def apply_parameter(self):
        self.stop_manual()
        if not self.bridge or self.write_pending or self.resetting:
            return
        if self.bridge.busy:
            self.status.setText('목표를 취소하고 정지한 뒤 파라미터를 적용하세요.')
            return
        node, name = self.selection()
        record = self.records.get((node,name), {})
        desc = record.get('descriptor')
        if desc and desc.read_only:
            self.status.setText('읽기 전용 파라미터입니다. 시작 설정 파일에서 변경하세요.')
            return
        try:
            value = yaml.safe_load(self.value.text())
            if value is None or isinstance(value, dict):
                raise ValueError('숫자, bool, 문자열 또는 배열을 입력하세요.')
            # Preserve declared numeric type when entering e.g. 5 instead of 5.0.
            declared = record.get('type')
            if declared == 3 and type(value) is int:
                value = float(value)
            if declared == 8 and isinstance(value,list) and all(type(v) in (int,float) for v in value):
                value = [float(v) for v in value]
            param = Parameter(name, value=value)
            if isinstance(value,list) and not value and declared in (5,6,7,8,9):
                param = Parameter(name,type_=Parameter.Type(declared),value=[])
            request = SetParametersAtomically.Request()
            request.parameters = [param.to_parameter_msg()]
        except Exception as exc:
            self.status.setText('값 오류: '+str(exc))
            return
        self.write_pending = True
        self.apply_button.setEnabled(False)
        self.status.setText(f'적용 요청: {node} / {name}')
        def done(result, error):
            self.write_pending = False
            if error:
                self.status.setText(error)
            elif result.result.successful:
                self.changes[(node,name)] = value
                self.records[(node,name)] = dict(value=value, type=request.parameters[0].value.type, descriptor=desc)
                self.filter_table()
                self.status.setText(f'값 수락: {name} = {parameter_text(value)}. 파일 저장은 별도입니다.')
            else:
                self.status.setText('적용 거부: '+result.result.reason)
            current = self.records.get(self.selection(), {})
            readonly = current.get('descriptor') and current['descriptor'].read_only
            self.apply_button.setEnabled(bool(current) and not readonly and not self.resetting)
        self.bridge.call(SetParametersAtomically, node+'/set_parameters_atomically', request, done)

    def save_changes(self):
        if not self.changes:
            self.status.setText('저장할 변경값이 없습니다.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '변경값 프로필 저장', f'/home/kiki/parameters_{self.domain}.yaml', 'YAML (*.yaml)')
        if not path:
            return
        try:
            data = {}
            for (node,name),value in sorted(self.changes.items()):
                data.setdefault(node, {'ros__parameters': {}})['ros__parameters'][name] = value
            target = Path(path)
            temp = target.with_name(target.name+'.tmp')
            temp.write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True))
            os.replace(temp,target)
            self.status.setText(f'저장 완료: {path} — 자동 재적용 안 됨. ros2 param load <노드> <파일>로 적용')
        except Exception as exc:
            self.status.setText('저장 실패: '+str(exc))

    def choose_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, '현재 사용하는 전체 Nav2 YAML 선택', self.config_path.text(), 'YAML (*.yaml *.yml)')
        if path:
            self.config_path.setText(path)
        return bool(path)

    def save_config(self):
        if not self.changes:
            self.status.setText('먼저 파라미터를 적용하세요. 적용 성공한 값만 저장합니다.')
            return
        if not self.choose_config():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '전체 Nav2 YAML 저장', f'/home/kiki/nav2_gui_{self.domain}.yaml', 'YAML (*.yaml)')
        if path:
            try:
                export_parameters(self.config_path.text(), path, self.changes)
                self.status.setText(f'저장 완료. 다음 Nav2 실행에 params_file:={path} 추가')
            except Exception as exc:
                self.status.setText('저장 실패: '+str(exc))

    def reset_nav2(self):
        self.stop_manual()
        if not self.bridge or self.resetting or self.write_pending:
            return
        if self.bridge.busy:
            self.status.setText('이동 취소 완료 후 Nav2를 리셋하세요.')
            return
        service = self.manager_box.currentData()
        if not service:
            self.status.setText('연결된 Nav2 lifecycle manager가 없습니다. 새로고침하세요.')
            return
        # Reset/reconfigure the selected manager's nodes, not the robot's power/OS.
        self.resetting = True
        self.bridge.maintenance = True
        self.refresh_id += 1
        self.loading = False
        self.refresh_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.status.setText('Nav2 리셋 중: cleanup → configure → activate. 수동/자동 이동 잠금')
        def finish(message):
            self.resetting = False
            self.bridge.maintenance = False
            self.reset_button.setEnabled(True)
            self.refresh_button.setEnabled(True)
            self.status.setText(message)
        def started(result,error):
            if error or not result.success:
                finish('Nav2 시작 실패: '+(error or '관리자가 실패를 반환했습니다.')+' 실행 터미널 로그를 확인하세요.')
            else:
                finish('Nav2 리셋 완료. 값 유지 여부는 새로고침으로 확인하세요. 초기 위치도 확인하세요.')
        def reset_done(result,error):
            if error or not result.success:
                finish('Nav2 리셋 실패: '+(error or '관리자가 실패를 반환했습니다.')+' 상태를 확인한 뒤 다시 시도하세요.')
                return
            req = ManageLifecycleNodes.Request()
            req.command = ManageLifecycleNodes.Request.STARTUP
            self.bridge.call(ManageLifecycleNodes,service,req,started,timeout=45)
        request = ManageLifecycleNodes.Request()
        request.command = ManageLifecycleNodes.Request.RESET
        self.bridge.call(ManageLifecycleNodes,service,request,reset_done,timeout=45)
