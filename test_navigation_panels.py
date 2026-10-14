import math
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import yaml
from PIL import Image
from PyQt5 import QtCore, QtWidgets
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter
from nav2_msgs.action import FollowWaypoints
from nav2_msgs.srv import LoadMap
from rcl_interfaces.srv import GetParameters
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from tf2_ros import TransformBroadcaster

from navigation_panels import (
    MapData, MapPanel, ParameterPanel, RosBridge, export_parameters, yaw_of)
from cmd_vel_controller import Controller


class MapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_rotated_map_pixel_world_roundtrip_and_cells(self):
        data = MapData([[0, 100], [-1, 0]], .5, (10, -2, math.pi/2))
        for px, py in [(.5, 1.5), (1.5, 1.5), (.5, .5), (1.5, .5)]:
            world = data.pixel_to_world(px, py)
            np.testing.assert_allclose(data.world_to_pixel(*world), (px, py))
        self.assertTrue(data.free_at(*data.pixel_to_world(.5, 1.5)))
        self.assertFalse(data.free_at(*data.pixel_to_world(1.5, 1.5)))
        self.assertFalse(data.free_at(*data.pixel_to_world(.5, .5)))
        self.assertFalse(data.free_at(*data.pixel_to_world(-.1, 1)))

    def test_yaml_import_flip_thresholds_and_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            Image.fromarray(np.array([[0, 205], [254, 254]], dtype=np.uint8)).save(d+'/map.pgm')
            Path(d+'/map.yaml').write_text(yaml.safe_dump(dict(image='map.pgm',resolution=.05,origin=[1,2,0],negate=0,free_thresh=.196,occupied_thresh=.65)))
            data=MapData.from_yaml(d+'/map.yaml')
            self.assertEqual(data.cells.tolist(), [[0,0],[100,-1]])

    def test_map_received_before_panel_creation_is_restored(self):
        class CachedBridge(QtCore.QObject):
            mapReceived = QtCore.pyqtSignal(object)
            poseReceived = QtCore.pyqtSignal(object)
            planReceived = QtCore.pyqtSignal(object)
            navStatus = QtCore.pyqtSignal(str)

            def __init__(self, data):
                super().__init__()
                self.latest_map = data
                self.busy = False

        data = MapData([[0, 100], [0, 0]], .05, (-1, -2, 0))
        panel = MapPanel(CachedBridge(data), Mock())
        self.assertIs(panel.view.data, data)
        self.assertIs(panel.latest_map, data)
        self.assertTrue(panel.live)

    def test_map_console_label_uses_domain_and_deduplicates(self):
        data = MapData([[0, 100], [0, 0]], .05, (-1, -2, 0))
        panel = MapPanel(None, Mock(), domain=226)

        with patch('builtins.print') as output:
            panel.on_map(data)
            panel.on_map(data)

        output.assert_called_once_with(
            '[control_226] /map 표시: 2x2, 0.050 m/cell',
            flush=True,
        )

    def test_export_preserves_full_config_and_nested_scope(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source.yaml';target=Path(d)/'out.yaml'
            original={'local_costmap':{'local_costmap':{'ros__parameters':{'robot_radius':.22,'inflation_layer':{'inflation_radius':.35}}}},'controller_server':{'ros__parameters':{'controller_frequency':20.0}}}
            source.write_text(yaml.safe_dump(original))
            export_parameters(source,target,{('/local_costmap/local_costmap','inflation_layer.inflation_radius'):.25})
            result=yaml.safe_load(target.read_text())
            self.assertEqual(result['controller_server'],original['controller_server'])
            self.assertEqual(result['local_costmap']['local_costmap']['ros__parameters']['robot_radius'],.22)
            self.assertEqual(result['local_costmap']['local_costmap']['ros__parameters']['inflation_layer']['inflation_radius'],.25)
            self.assertEqual(yaml.safe_load(source.read_text()),original)

    def test_preview_cannot_send_goal_and_obstacles_rejected(self):
        bridge=SimpleNamespace(
            busy=False, pose=(0.0, 0.0, 0.0), follow=Mock(),
            publish_goal_pose=Mock(),
            initial_pose=Mock(), mapReceived=Mock(), poseReceived=Mock(),
            planReceived=Mock(), navStatus=Mock())
        panel=MapPanel(bridge,Mock());panel.view.set_map(MapData([[0,100],[0,0]],1,(0,0,0)))
        panel.view.mode='goal';panel.pick(.5,.5,0)
        self.assertEqual(panel.view.waypoints,[])
        panel.live=True;panel.pick(1.5,.5,0)
        self.assertEqual(panel.view.waypoints,[])
        panel.pick(.5,.5,0);panel.start_route()
        bridge.follow.assert_called_once_with([(.5,.5,0)])
        panel.start_v3_goal()
        bridge.publish_goal_pose.assert_called_once_with((.5,.5,0))
        panel.require_initial=True;bridge.follow.reset_mock();panel.start_route()
        bridge.follow.assert_not_called()

    def test_home_button_republishes_recorded_initial_pose(self):
        bridge=SimpleNamespace(
            busy=False, pose=(4.0, 4.0, 0.0), follow=Mock(),
            publish_goal_pose=Mock(), initial_pose=Mock(),
            mapReceived=Mock(), poseReceived=Mock(), planReceived=Mock(),
            navStatus=Mock())
        panel=MapPanel(bridge, Mock())
        panel.view.set_map(MapData([[0, 0], [0, 0]], 1, (0, 0, 0)))
        panel.live=True
        panel.view.mode='initial'
        panel.pick(.5, .5, math.pi/4)

        bridge.initial_pose.assert_called_once_with(.5, .5, math.pi/4)
        self.assertEqual(panel.home_pose, (.5, .5, math.pi/4))
        self.assertEqual(panel.home_button.text(), 'HOME 복귀')

        panel.return_home()

        bridge.publish_goal_pose.assert_called_once_with(
            (.5, .5, math.pi/4))
        self.assertIn('HOME 복귀 목표 전송', panel.status.text())

    def test_first_v3_goal_uses_current_tf_as_home_fallback(self):
        bridge=SimpleNamespace(
            busy=False, pose=(.25, -.5, .1), follow=Mock(),
            publish_goal_pose=Mock(), initial_pose=Mock(),
            mapReceived=Mock(), poseReceived=Mock(), planReceived=Mock(),
            navStatus=Mock())
        panel=MapPanel(bridge, Mock())
        panel.view.set_map(MapData([[0, 0], [0, 0]], 1, (0, 0, 0)))
        panel.live=True
        panel.view.waypoints=[(1.5, .5, 0.0)]

        panel.start_v3_goal()

        self.assertEqual(panel.home_pose, (.25, -.5, .1))
        bridge.publish_goal_pose.assert_called_once_with((1.5, .5, 0.0))

    def test_custom_pure_pursuit_busy_allows_new_goal_pick_and_send(self):
        bridge=SimpleNamespace(
            busy=True, nav_busy=False, custom_drive_busy=True,
            pose=(.25, -.5, .1), follow=Mock(),
            publish_goal_pose=Mock(), initial_pose=Mock(),
            mapReceived=Mock(), poseReceived=Mock(), planReceived=Mock(),
            navStatus=Mock())
        panel=MapPanel(bridge, Mock())
        panel.view.set_map(MapData([[0, 0], [0, 0]], 1, (0, 0, 0)))
        panel.live=True
        panel.view.mode='goal'

        panel.pick(.5, .5, 0.0)
        panel.start_v3_goal()

        self.assertEqual(panel.view.waypoints, [(.5, .5, 0.0)])
        bridge.publish_goal_pose.assert_called_once_with((.5, .5, 0.0))

    def test_nav2_busy_still_blocks_v3_goal_pick_and_send(self):
        bridge=SimpleNamespace(
            busy=True, nav_busy=True, custom_drive_busy=False,
            pose=(.25, -.5, .1), follow=Mock(),
            publish_goal_pose=Mock(), initial_pose=Mock(),
            mapReceived=Mock(), poseReceived=Mock(), planReceived=Mock(),
            navStatus=Mock())
        panel=MapPanel(bridge, Mock())
        panel.view.set_map(MapData([[0, 0], [0, 0]], 1, (0, 0, 0)))
        panel.live=True
        panel.view.mode='goal'

        panel.pick(.5, .5, 0.0)
        self.assertEqual(panel.view.waypoints, [])

        panel.view.waypoints=[(.5, .5, 0.0)]
        panel.start_v3_goal()

        bridge.publish_goal_pose.assert_not_called()
        self.assertIn('Nav2', panel.status.text())

    def test_first_v3_goal_without_pose_is_blocked(self):
        bridge=SimpleNamespace(
            busy=False, pose=None, follow=Mock(), publish_goal_pose=Mock(),
            initial_pose=Mock(), mapReceived=Mock(), poseReceived=Mock(),
            planReceived=Mock(), navStatus=Mock())
        panel=MapPanel(bridge, Mock())
        panel.view.set_map(MapData([[0, 0], [0, 0]], 1, (0, 0, 0)))
        panel.live=True
        panel.view.waypoints=[(1.5, .5, 0.0)]

        panel.start_v3_goal()

        bridge.publish_goal_pose.assert_not_called()
        self.assertIsNone(panel.home_pose)
        self.assertIn('TF', panel.status.text())

    def test_map_frame_change_invalidates_home(self):
        bridge=SimpleNamespace(
            busy=False, pose=(0.0, 0.0, 0.0), follow=Mock(),
            publish_goal_pose=Mock(), initial_pose=Mock(),
            mapReceived=Mock(), poseReceived=Mock(), planReceived=Mock(),
            navStatus=Mock())
        panel=MapPanel(bridge, Mock())
        panel.on_map(MapData(
            [[0, 0], [0, 0]], 1, (0, 0, 0), frame='map'))
        panel.home_pose=(.5, .5, 0.0)
        panel.view.waypoints=[(1.5, .5, 0.0)]

        panel.on_map(MapData(
            [[0, 0], [0, 0]], 1, (0, 0, 0), frame='new_map'))

        self.assertIsNone(panel.home_pose)
        self.assertEqual(panel.view.waypoints, [])
        self.assertTrue(panel.require_initial)
        panel.return_home()
        bridge.publish_goal_pose.assert_not_called()

    def test_slam_origin_and_dimension_growth_keeps_home(self):
        bridge=SimpleNamespace(
            busy=False, pose=(0.0, 0.0, 0.0), follow=Mock(),
            publish_goal_pose=Mock(), initial_pose=Mock(),
            mapReceived=Mock(), poseReceived=Mock(), planReceived=Mock(),
            navStatus=Mock())
        panel=MapPanel(bridge, Mock())
        panel.on_map(MapData(
            [[0, 0], [0, 0]], .05, (0, 0, 0)))
        panel.home_pose=(.5, .5, 0.0)
        panel.view.waypoints=[(.7, .5, 0.0)]

        panel.on_map(MapData(
            [[0, 0, 0], [0, 0, 0]], .05, (-1, 0, 0)))

        self.assertEqual(panel.home_pose, (.5, .5, 0.0))
        self.assertEqual(panel.view.waypoints, [(.7, .5, 0.0)])
        self.assertFalse(panel.require_initial)

    def test_home_publish_failure_does_not_show_success(self):
        bridge=SimpleNamespace(
            busy=False, pose=(0.0, 0.0, 0.0), follow=Mock(),
            publish_goal_pose=Mock(return_value=False), initial_pose=Mock(),
            mapReceived=Mock(), poseReceived=Mock(), planReceived=Mock(),
            navStatus=Mock())
        panel=MapPanel(bridge, Mock())
        panel.view.set_map(MapData([[0, 0], [0, 0]], 1, (0, 0, 0)))
        panel.live=True
        panel.home_pose=(.5, .5, 0.0)
        panel.status.setText('/goal_pose 수신 노드가 없습니다.')

        panel.return_home()

        bridge.publish_goal_pose.assert_called_once_with((.5, .5, 0.0))
        self.assertEqual(panel.status.text(), '/goal_pose 수신 노드가 없습니다.')

    def test_home_button_does_not_capture_current_pose_as_home(self):
        bridge=SimpleNamespace(
            busy=False, pose=(3.0, 2.0, .4), follow=Mock(),
            publish_goal_pose=Mock(), initial_pose=Mock(),
            mapReceived=Mock(), poseReceived=Mock(), planReceived=Mock(),
            navStatus=Mock())
        panel=MapPanel(bridge, Mock())
        panel.view.set_map(MapData([[0, 0], [0, 0]], 1, (0, 0, 0)))
        panel.live=True

        panel.return_home()

        bridge.publish_goal_pose.assert_not_called()
        self.assertIsNone(panel.home_pose)
        self.assertIn('HOME 위치 없음', panel.status.text())

    def test_manual_locked_and_stop_cancels_navigation(self):
        pub=Mock(topic_name='/cmd_vel')
        with tempfile.TemporaryDirectory() as d:
            w=Controller(pub,domain_id=64,settings=QtCore.QSettings(d+'/settings.ini',QtCore.QSettings.IniFormat))
            w.bridge=SimpleNamespace(busy=True,cancel=Mock())
            w.buttons['forward'].setDown(True);w.start(w.buttons['forward'],(1,0));w.tick()
            pub.publish.assert_not_called()
            w.emergency_stop();w.bridge.cancel.assert_called_once()
            self.assertEqual(pub.publish.call_args.args[0].linear.x,0)
            w.bridge.busy=False
            w.close()


class RosIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.old_local = os.environ.get('ROS_LOCALHOST_ONLY')
        os.environ['ROS_LOCALHOST_ONLY']='1'
        rclpy.init(args=[],domain_id=213)
        cls.server=rclpy.create_node('panel_test_server')
        cls.server.declare_parameter('inflation_layer.inflation_radius',.35)
        cls.server.declare_parameter('max_velocity',[.12,0.,.35])
        cls.server_executor=MultiThreadedExecutor(num_threads=2)
        cls.server_executor.add_node(cls.server)
        cls.thread=threading.Thread(target=cls.server_executor.spin,daemon=True);cls.thread.start()
        cls.node=rclpy.create_node('panel_test_client')
        cls.executor=SingleThreadedExecutor();cls.executor.add_node(cls.node)
        cls.tf=TransformBroadcaster(cls.server)
        cls.requests=[]
        cls.maps=[]
        def load_map(request,response):
            cls.maps.append(request.map_url)
            response.result=0
            response.map.header.frame_id='map'
            response.map.info.width=response.map.info.height=2
            response.map.info.resolution=1.
            response.map.info.origin.orientation.w=1.
            response.map.data=[0,0,0,100]
            return response
        cls.map_service=cls.server.create_service(LoadMap,'/map_server/load_map',load_map)
        def execute(handle):
            cls.requests.append(handle.request)
            until=time.monotonic()+3
            while time.monotonic()<until:
                if handle.is_cancel_requested:
                    handle.canceled();return FollowWaypoints.Result()
                time.sleep(.02)
            handle.succeed();return FollowWaypoints.Result()
        cls.action=ActionServer(cls.server,FollowWaypoints,'/follow_waypoints',execute,
                               callback_group=ReentrantCallbackGroup(),
                               cancel_callback=lambda request:CancelResponse.ACCEPT,
                               goal_callback=lambda request:GoalResponse.ACCEPT)

    @classmethod
    def tearDownClass(cls):
        cls.action.destroy();cls.server_executor.shutdown();cls.thread.join(3)
        cls.node.destroy_node();cls.server.destroy_node();cls.executor.shutdown();rclpy.shutdown()
        if cls.old_local is None:os.environ.pop('ROS_LOCALHOST_ONLY',None)
        else:os.environ['ROS_LOCALHOST_ONLY']=cls.old_local

    def setUp(self):
        self.bridge=RosBridge(self.node)

    def tearDown(self):
        self.bridge.timer.stop()
        for sub in self.bridge.subs:self.node.destroy_subscription(sub)
        for call in self.bridge.calls:self.node.destroy_client(call[0])
        self.bridge.action.destroy();self.bridge.listener.unregister()
        self.node.destroy_publisher(self.bridge.initial_pub)
        self.node.destroy_publisher(self.bridge.goal_pose_pub)
        self.bridge.deleteLater()

    def until(self,predicate,timeout=4):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            self.executor.spin_once(timeout_sec=.01);self.app.processEvents()
            if predicate():return
        self.fail('Timed out')

    def test_parameter_read_apply_service_discovery_and_export_tracking(self):
        panel=ParameterPanel(self.bridge,213,Mock(),autoload=False)
        panel.node_box.setCurrentText('/panel_test_server')
        panel.read_parameter()
        self.until(lambda:panel.value.text()!='')
        self.assertEqual(float(panel.value.text()),.35)
        panel.value.setText('0.25');panel.apply_parameter()
        self.until(lambda:bool(panel.changes))
        self.assertEqual(self.server.get_parameter('inflation_layer.inflation_radius').value,.25)
        self.assertEqual(panel.changes[('/panel_test_server','inflation_layer.inflation_radius')],.25)
        panel.value.setText('"wrong type"');panel.apply_parameter()
        self.until(lambda:'거부' in panel.status.text())
        self.assertEqual(panel.changes[('/panel_test_server','inflation_layer.inflation_radius')],.25)
        panel.name_box.setCurrentText('max_velocity');panel.read_parameter()
        self.until(lambda:panel.value.text()!='')
        self.assertEqual(yaml.safe_load(panel.value.text()),[.12,0.,.35])

    def test_map_import_applies_to_server_and_requires_initial_pose(self):
        panel=MapPanel(self.bridge,Mock())
        panel.import_path='/tmp/isolated-test-map.yaml'
        panel.home_pose=(1.0, 2.0, 0.0)
        panel.view.waypoints=[(3.0, 4.0, 0.0)]
        panel.apply_map()
        self.until(lambda:panel.live)
        self.assertEqual(self.maps[-1],'/tmp/isolated-test-map.yaml')
        self.assertIsNone(panel.home_pose)
        self.assertEqual(panel.view.waypoints, [])
        self.assertTrue(panel.require_initial)
        panel.pick(.5,.5,0.)
        self.assertFalse(panel.require_initial)
        self.assertEqual(panel.mode.currentIndex(),1)

    def test_robot_namespace_isolates_topics_parameters_and_shared_tf(self):
        from tf2_msgs.msg import TFMessage
        from geometry_msgs.msg import TransformStamped
        robot1=rclpy.create_node('controller_server',namespace='/robot1')
        robot2=rclpy.create_node('controller_server',namespace='/robot2')
        robot1.declare_parameter('speed',.1);robot2.declare_parameter('speed',.2)
        self.server_executor.add_node(robot1);self.server_executor.add_node(robot2)
        client=rclpy.create_node('multi_vehicle_test',namespace='/robot1')
        self.executor.add_node(client)
        bridge=RosBridge(client)
        try:
            self.assertEqual(bridge.endpoint('follow_waypoints'),'/robot1/follow_waypoints')
            self.assertEqual(bridge.initial_pub.topic_name,'/robot1/initialpose')
            self.assertEqual(bridge.goal_pose_pub.topic_name,'/robot1/goal_pose')
            self.assertIn('/robot1/map',[sub.topic_name for sub in bridge.subs])
            self.assertNotIn('/robot2/map',[sub.topic_name for sub in bridge.subs])
            panel=ParameterPanel(bridge,213,Mock(),autoload=False)
            self.until(lambda:any(name=='/robot1/controller_server/list_parameters' for name,_ in client.get_service_names_and_types()))
            panel.refresh_all();self.until(lambda:not panel.loading)
            self.assertIn(('/robot1/controller_server','speed'),panel.records)
            self.assertFalse(any(key[0].startswith('/robot2/') for key in panel.records))
            panel.show_record(('/robot1/controller_server','speed'));panel.value.setText('0.15');panel.apply_parameter()
            self.until(lambda:bool(panel.changes))
            self.assertEqual(robot1.get_parameter('speed').value,.15)
            self.assertEqual(robot2.get_parameter('speed').value,.2)
            msg=TFMessage()
            for name in ('robot1/base_footprint','robot2/base_footprint'):
                tf=TransformStamped();tf.header.frame_id='map';tf.child_frame_id=name
                tf.transform.rotation.w=1.;tf.header.stamp=client.get_clock().now().to_msg();msg.transforms.append(tf)
            bridge.shared_tf(msg,True)
            from rclpy.time import Time
            self.assertTrue(bridge.buffer.can_transform('map','robot1/base_footprint',Time()))
            self.assertFalse(bridge.buffer.can_transform('map','robot2/base_footprint',Time()))
        finally:
            bridge.timer.stop();bridge.action.destroy();bridge.listener.unregister()
            for sub in bridge.subs:client.destroy_subscription(sub)
            client.destroy_publisher(bridge.initial_pub)
            client.destroy_publisher(bridge.goal_pose_pub)
            self.executor.remove_node(client);client.destroy_node()
            self.server_executor.remove_node(robot1);self.server_executor.remove_node(robot2)
            robot1.destroy_node();robot2.destroy_node()

    def test_camera_receiver_with_main_ros_executor(self):
        from sensor_msgs.msg import Image as RosImage
        from cmd_vel_controller import CameraFeed
        camera=CameraFeed()
        publisher=self.server.create_publisher(RosImage,'/isolated_test_camera',1)
        camera.connect('/isolated_test_camera',False)
        msg=RosImage(width=1,height=1,encoding='rgb8',step=3,data=bytes([255,0,0]))
        timer=self.server.create_timer(.05,lambda:publisher.publish(msg))
        try:
            self.until(lambda:camera.latest is not None)
            self.assertEqual(camera.latest[0].pixelColor(0,0).name(),'#ff0000')
        finally:
            camera.close();self.server.destroy_timer(timer);self.server.destroy_publisher(publisher)

    def test_follow_waypoints_order_and_cancel_without_robot(self):
        self.until(lambda:self.bridge.action.server_is_ready())
        transform=TransformStamped();transform.header.frame_id='map';transform.child_frame_id='base_footprint'
        transform.header.stamp=self.server.get_clock().now().to_msg();transform.transform.rotation.w=1.
        def publish_tf():
            transform.header.stamp=self.server.get_clock().now().to_msg()
            self.tf.sendTransform(transform)
        tf_timer=self.server.create_timer(.05,publish_tf)
        self.addCleanup(lambda:self.server.destroy_timer(tf_timer))
        self.until(lambda:self.bridge.pose is not None)
        before=len(self.requests);messages=[];self.bridge.navStatus.connect(messages.append)
        self.bridge.follow([(1.,2.,0.),(3.,4.,math.pi/2)])
        self.until(lambda:len(self.requests)>before)
        self.assertEqual([(p.pose.position.x,p.pose.position.y) for p in self.requests[-1].poses],[(1.,2.),(3.,4.)])
        self.assertTrue(self.bridge.busy)
        self.bridge.cancel()
        self.until(lambda:'이동 취소 완료' in messages)
        self.until(lambda:not self.bridge.busy)

    def test_goal_pose_without_subscriber_is_rejected(self):
        messages=[]
        self.bridge.navStatus.connect(messages.append)

        self.assertFalse(self.bridge.publish_goal_pose((1.0, 2.0, 0.0)))

        self.assertTrue(messages)
        self.assertIn('수신 노드가 없습니다', messages[-1])

    def test_v3_goal_pose_publisher_uses_last_selected_pose(self):
        received = []
        subscription = self.server.create_subscription(
            PoseStamped, '/goal_pose', received.append, 1)
        self.addCleanup(lambda: self.server.destroy_subscription(subscription))
        self.until(lambda: self.node.count_subscribers('/goal_pose') > 0)

        self.assertTrue(self.bridge.publish_goal_pose(
            (1.25, -0.75, math.pi / 2.0)))
        self.until(lambda: bool(received))

        message = received[-1]
        self.assertEqual(message.header.frame_id, 'map')
        self.assertAlmostEqual(message.pose.position.x, 1.25)
        self.assertAlmostEqual(message.pose.position.y, -0.75)
        self.assertAlmostEqual(yaw_of(message.pose.orientation), math.pi / 2.0)

    def test_cancel_disables_only_driver_publishing_selected_cmd_vel(self):
        selected_driver = rclpy.create_node(
            'pure_pursuit_v3_node', namespace='/robot1')
        other_driver = rclpy.create_node(
            'pure_pursuit_v3_node', namespace='/robot2')
        selected_driver.declare_parameter('drive_enabled', True)
        other_driver.declare_parameter('drive_enabled', True)
        selected_velocity_pub = selected_driver.create_publisher(
            Twist, '/cmd_vel', 1)
        other_velocity_pub = other_driver.create_publisher(
            Twist, '/robot2/cmd_vel', 1)
        self.server_executor.add_node(selected_driver)
        self.server_executor.add_node(other_driver)
        try:
            self.until(lambda: any(
                info.node_namespace == '/robot1'
                and info.node_name == 'pure_pursuit_v3_node'
                for info in self.node.get_publishers_info_by_topic('/cmd_vel')))
            self.bridge.refresh_autonomous_drivers()
            self.until(lambda: self.bridge.autonomous_drivers.get(
                '/robot1/pure_pursuit_v3_node') is True)
            self.assertNotIn(
                '/robot2/pure_pursuit_v3_node',
                self.bridge.autonomous_drivers)
            self.assertTrue(self.bridge.custom_drive_busy)

            # Parameter events from an exact topic publisher are accepted even
            # though its node namespace differs from the selected root topic.
            selected_driver.set_parameters([
                Parameter('drive_enabled', value=False)])
            self.until(lambda: self.bridge.autonomous_drivers.get(
                '/robot1/pure_pursuit_v3_node') is False)
            selected_driver.set_parameters([
                Parameter('drive_enabled', value=True)])
            self.until(lambda: self.bridge.autonomous_drivers.get(
                '/robot1/pure_pursuit_v3_node') is True)

            self.bridge.cancel()

            self.until(lambda: selected_driver.get_parameter(
                'drive_enabled').value is False)
            self.until(lambda: not self.bridge.custom_drive_busy)
            self.assertTrue(other_driver.get_parameter(
                'drive_enabled').value)
        finally:
            self.server_executor.remove_node(selected_driver)
            self.server_executor.remove_node(other_driver)
            selected_driver.destroy_publisher(selected_velocity_pub)
            other_driver.destroy_publisher(other_velocity_pub)
            selected_driver.destroy_node()
            other_driver.destroy_node()


class CameraDecodeTests(unittest.TestCase):
    def test_raw_stride_bgr_and_invalid_payload(self):
        from sensor_msgs.msg import Image as RosImage
        from cmd_vel_controller import decode_camera_image
        msg=RosImage(width=1,height=2,encoding='bgr8',step=4,data=bytes([0,0,255,0,0,255,0,0]))
        image=decode_camera_image(msg)
        self.assertEqual(image.pixelColor(0,0).name(),'#ff0000')
        self.assertEqual(image.pixelColor(0,1).name(),'#00ff00')
        msg.data=bytes([0])
        with self.assertRaises(ValueError):decode_camera_image(msg)

    def test_png_compressed_decodes(self):
        from PyQt5 import QtGui
        from sensor_msgs.msg import CompressedImage
        from cmd_vel_controller import decode_camera_image
        image=QtGui.QImage(8,6,QtGui.QImage.Format_RGB888);image.fill(QtCore.Qt.red)
        payload=QtCore.QByteArray();buffer=QtCore.QBuffer(payload);buffer.open(QtCore.QIODevice.WriteOnly)
        self.assertTrue(image.save(buffer,'PNG'))
        result=decode_camera_image(CompressedImage(format='png',data=bytes(payload)))
        self.assertEqual((result.width(),result.height()),(8,6))
        self.assertEqual(result.pixelColor(0,0).name(),'#ff0000')
