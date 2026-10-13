import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

from PyQt5 import QtWidgets, QtCore
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.parameter import Parameter
from nav2_msgs.srv import ManageLifecycleNodes
import yaml
from navigation_panels import ParameterPanel
from cmd_vel_controller import Controller


class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_panel(self, namespace=''):
        bridge=SimpleNamespace(busy=False,maintenance=False,namespace=namespace,node=Mock(),call=Mock())
        panel=ParameterPanel(bridge,64,Mock(),autoload=False)
        return panel,bridge

    def seed(self,panel):
        panel.records={
            ('/robot1/local_costmap/local_costmap','inflation_layer.inflation_radius'):
            dict(value=.35,type=3,descriptor=ParameterDescriptor(name='inflation_layer.inflation_radius',description='obstacle clearance')),
            ('/robot1/controller_server','use_sim_time'):
            dict(value=True,type=1,descriptor=ParameterDescriptor(name='use_sim_time',read_only=True)),
            ('/robot2/controller_server','FollowPath.max_vel_x'):
            dict(value=.2,type=3,descriptor=ParameterDescriptor(name='FollowPath.max_vel_x')),
        }
        panel.rebuild_groups()

    def test_group_search_sort_and_readonly_editor(self):
        panel,_=self.make_panel();self.seed(panel)
        self.assertEqual(panel.table.rowCount(),3)
        panel.group_filter=('/robot1/controller_server',None);panel.filter_table()
        self.assertEqual(panel.table.rowCount(),1)
        panel.search.setText('clearance')
        self.assertEqual(panel.table.rowCount(),1)
        self.assertIn('inflation',panel.table.item(0,1).text())
        panel.search.clear();panel.table.selectRow(0)
        self.assertTrue(panel.value.isReadOnly())
        self.assertFalse(panel.apply_button.isEnabled())
        panel.apply_parameter();self.assertIn('읽기 전용',panel.status.text())

    def test_curated_help_is_searchable_and_available_on_hover(self):
        panel,_=self.make_panel();self.seed(panel)
        original_columns=panel.table.columnCount()
        panel.search.setText('최대 전진')
        self.assertEqual(panel.table.rowCount(),1)
        self.assertEqual(panel.table.columnCount(),original_columns)
        tooltip=panel.table.item(0,1).toolTip()
        self.assertIn('FollowPath.max_vel_x = 0.2',tooltip)
        self.assertIn('조절 영향:',tooltip)
        self.assertIn('0.18~0.26',tooltip)
        panel.search.setText('좁은 통로')
        self.assertEqual(panel.table.rowCount(),1)
        self.assertIn('inflation_radius',panel.table.item(0,1).text())

    def test_discovery_scopes_nodes_and_managers_to_selected_vehicle(self):
        panel,bridge=self.make_panel('/robot1')
        bridge.node.get_service_names_and_types.return_value=[
            ('/robot1/controller_server/list_parameters',['rcl_interfaces/srv/ListParameters']),
            ('/robot2/controller_server/list_parameters',['rcl_interfaces/srv/ListParameters']),
            ('/robot1/lifecycle_manager_navigation/manage_nodes',['nav2_msgs/srv/ManageLifecycleNodes']),
            ('/robot2/lifecycle_manager_navigation/manage_nodes',['nav2_msgs/srv/ManageLifecycleNodes'])]
        def call(kind,name,request,callback,**kwargs):
            if name.endswith('/list_parameters'):
                callback(SimpleNamespace(result=SimpleNamespace(names=['a'])),None)
            elif name.endswith('/get_parameters'):
                callback(SimpleNamespace(values=[Parameter('a',value=1.0).to_parameter_msg().value]),None)
            else:
                callback(SimpleNamespace(descriptors=[ParameterDescriptor(name='a')]),None)
        bridge.call.side_effect=call
        panel.refresh_all()
        self.assertEqual(set(panel.records),{('/robot1/controller_server','a')})
        self.assertEqual(panel.manager_box.count(),1)
        self.assertTrue(panel.manager_box.currentData().startswith('/robot1/'))
        self.assertFalse(panel.loading)

    def test_reset_orders_reset_then_startup_and_releases_lock(self):
        panel,bridge=self.make_panel()
        panel.manager_box.addItem('navigation','/lifecycle_manager_navigation/manage_nodes')
        panel.reset_nav2()
        self.assertTrue(bridge.maintenance)
        first=bridge.call.call_args
        self.assertEqual(first.args[2].command,ManageLifecycleNodes.Request.RESET)
        first.args[3](SimpleNamespace(success=True),None)
        second=bridge.call.call_args
        self.assertEqual(second.args[2].command,ManageLifecycleNodes.Request.STARTUP)
        second.args[3](SimpleNamespace(success=True),None)
        self.assertFalse(bridge.maintenance)
        self.assertIn('완료',panel.status.text())

    def test_reset_not_sent_while_moving_and_failure_does_not_startup(self):
        panel,bridge=self.make_panel()
        panel.manager_box.addItem('navigation','/lifecycle_manager_navigation/manage_nodes')
        bridge.busy=True;panel.reset_nav2();bridge.call.assert_not_called()
        bridge.busy=False;panel.reset_nav2()
        bridge.call.call_args.args[3](SimpleNamespace(success=False),None)
        self.assertEqual(bridge.call.call_count,1)
        self.assertFalse(bridge.maintenance)
        self.assertIn('실패',panel.status.text())

    def test_changes_export_is_scoped_ros_yaml(self):
        panel,_=self.make_panel()
        panel.changes={('/robot1/controller_server','FollowPath.max_vel_x'):.12,
                       ('/robot2/controller_server','FollowPath.max_vel_x'):.08}
        with tempfile.TemporaryDirectory() as d:
            path=d+'/params.yaml'
            with patch.object(QtWidgets.QFileDialog,'getSaveFileName',return_value=(path,'')):
                panel.save_changes()
            data=yaml.safe_load(Path(path).read_text())
            self.assertEqual(data['/robot1/controller_server']['ros__parameters']['FollowPath.max_vel_x'],.12)
            self.assertEqual(data['/robot2/controller_server']['ros__parameters']['FollowPath.max_vel_x'],.08)

    def test_vehicle_selection_records_namespace_and_blocks_active_navigation(self):
        with tempfile.TemporaryDirectory() as d:
            settings=QtCore.QSettings(d+'/settings.ini',QtCore.QSettings.IniFormat)
            window=Controller(Mock(topic_name='/cmd_vel'),64,settings)
            result=[];window.vehicleRequested.connect(result.append)
            window.vehicle_box.setCurrentText('/robot2')
            window.select_vehicle()
            self.assertEqual(result,['/robot2'])
            self.assertEqual(settings.value('vehicles/domain_64/namespace'),'/robot2')
            window.bridge=SimpleNamespace(busy=True,cancel=Mock())
            window.vehicle_box.setCurrentText('/robot1');window.select_vehicle()
            self.assertEqual(result,['/robot2'])
            window.bridge=None;window.close()
