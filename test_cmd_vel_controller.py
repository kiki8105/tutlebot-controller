import copy
import unittest
import tempfile
from unittest.mock import patch
from PyQt5 import QtCore, QtWidgets
from cmd_vel_controller import Controller, domain_settings, initial_domain


class Publisher:
    topic_name = '/cmd_vel_test'
    def __init__(self):
        self.messages = []
    def publish(self, msg):
        self.messages.append(copy.deepcopy(msg))


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.pub = Publisher()
        self.temp = tempfile.TemporaryDirectory()
        self.settings = domain_settings(self.temp.name + '/controller.ini')
        self.window = Controller(self.pub, domain_id=64, settings=self.settings)
        self.window.timer.stop()
        self.window.video_timer.stop()

    def tearDown(self):
        self.window.close()
        self.temp.cleanup()

    def press(self, name):
        b = self.window.buttons[name]
        b.setDown(True)
        b.pressed.emit()

    def assert_stopped(self):
        m = self.pub.messages[-1]
        self.assertEqual((m.linear.x, m.angular.z), (0.0, 0.0))

    def test_direction_and_release(self):
        for name, expected in [('forward', (.1, 0)), ('backward', (-.1, 0)),
                               ('left', (0, .5)), ('right', (0, -.5))]:
            with self.subTest(name=name):
                self.press(name)
                m = self.pub.messages[-1]
                self.assertEqual((m.linear.x, m.angular.z), expected)
                self.window.buttons[name].setDown(False)
                self.window.buttons[name].released.emit()
                self.assert_stopped()

    def test_stop_does_not_restart_held_button(self):
        self.press('forward')
        self.window.stop()
        for _ in range(8):
            self.window.tick()
            self.assert_stopped()
        count = len(self.pub.messages)
        self.window.tick()
        self.assertEqual(len(self.pub.messages), count)

    def test_focus_loss_stops(self):
        self.press('left')
        self.window.application_state_changed(QtCore.Qt.ApplicationInactive)
        self.assert_stopped()
        self.window.tick()
        self.assert_stopped()

    def test_lost_release_and_close_stop(self):
        self.press('backward')
        self.window.buttons['backward'].setDown(False)
        self.window.tick()
        self.assert_stopped()
        self.press('forward')
        self.window.close()
        self.assert_stopped()

    def test_save_persists_for_next_launch_without_changing_active_domain(self):
        self.press('forward')
        self.window.domain_input.setValue(65)
        self.window.domain_save.click()
        self.assert_stopped()
        self.assertEqual(self.window.domain_id, 64)
        self.assertTrue(self.window.buttons['forward'].isEnabled())
        fresh = domain_settings(self.temp.name + '/controller.ini')
        with patch.dict('os.environ', {'ROS_DOMAIN_ID': '64'}):
            self.assertEqual(initial_domain(fresh), 65)

    def test_saving_current_domain(self):
        self.window.domain_save.click()
        self.assertEqual(self.window.domain_id, 64)
        self.assertEqual(initial_domain(self.settings), 64)

    def test_invalid_saved_domain_falls_back_to_environment(self):
        self.settings.setValue('ros/domain_id', 'not-an-id')
        with patch.dict('os.environ', {'ROS_DOMAIN_ID': '64'}):
            self.assertEqual(initial_domain(self.settings), 64)
        self.assertEqual(self.window.domain_input.minimum(), 0)
        self.assertEqual(self.window.domain_input.maximum(), 232)

    def test_compact_window_scrolls_without_crushing_drive_buttons(self):
        self.window.show()
        self.window.resize(700, 500)
        self.app.processEvents()

        self.assertTrue(
            self.window.dashboard_scroll.horizontalScrollBar().maximum() > 0
            or self.window.dashboard_scroll.verticalScrollBar().maximum() > 0
        )
        for button in self.window.buttons.values():
            with self.subTest(button=button.text()):
                self.assertGreaterEqual(button.height(), 46)


if __name__ == '__main__':
    unittest.main()


class CameraPanelTests(ControllerTests):
    def test_latest_frame_and_stale_frame(self):
        import time
        from types import SimpleNamespace
        from PyQt5 import QtGui
        frame=QtGui.QImage(16,12,QtGui.QImage.Format_RGB888)
        frame.fill(QtCore.Qt.red)
        self.window.camera=SimpleNamespace(latest=(frame,time.monotonic()),status='16 × 12')
        self.window.render_camera()
        self.assertFalse(self.window.video.pixmap().isNull())
        self.window.camera.latest=(frame,time.monotonic()-3)
        self.window.render_camera()
        self.assertEqual(self.window.video.text(),'영상 수신 끊김')

    def test_connect_saves_topic_and_stops_motion(self):
        from unittest.mock import Mock
        self.window.camera=Mock()
        self.press('forward')
        self.window.camera_topic.setCurrentText('/camera/image_raw/compressed')
        self.window.connect_camera()
        self.window.camera.connect.assert_called_once_with('/camera/image_raw/compressed',True)
        self.assertEqual(self.settings.value('camera/topic'),'/camera/image_raw/compressed')
        self.assert_stopped()


class KeyboardTests(ControllerTests):
    def send_key(self, key, release=False, repeat=False, target=None):
        from PyQt5 import QtGui
        event = QtGui.QKeyEvent(QtCore.QEvent.KeyRelease if release else QtCore.QEvent.KeyPress,
                               key, QtCore.Qt.NoModifier, '', repeat)
        self.app.sendEvent(target or self.window, event)

    def test_arrow_directions_release_and_space_latch(self):
        for key, expected in [(QtCore.Qt.Key_Up, (.1, 0)),
                              (QtCore.Qt.Key_Down, (-.1, 0)),
                              (QtCore.Qt.Key_Left, (0, .5)),
                              (QtCore.Qt.Key_Right, (0, -.5))]:
            self.send_key(key)
            self.window.tick()
            m = self.pub.messages[-1]
            self.assertEqual((m.linear.x, m.angular.z), expected)
            self.send_key(key, release=True, repeat=True)
            self.assertIn(key, self.window.keys)
            self.send_key(QtCore.Qt.Key_Space)
            self.assert_stopped()
            self.send_key(key, repeat=True)
            self.window.tick()
            self.assert_stopped()
            self.send_key(key, release=True)
            self.send_key(key)
            self.send_key(key, release=True)
            self.assert_stopped()

    def test_editing_does_not_drive_and_focus_loss_stops(self):
        self.send_key(QtCore.Qt.Key_Up, target=self.window.domain_input)
        self.assertFalse(self.window.keys)
        self.send_key(QtCore.Qt.Key_Up)
        self.app.sendEvent(self.window, QtCore.QEvent(QtCore.QEvent.WindowDeactivate))
        self.assert_stopped()
        self.assertFalse(self.window.keys)
