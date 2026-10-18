import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("paradox_perception_ros"),
                       "config", "perception.yaml")
    return LaunchDescription([
        Node(package="paradox_perception_ros", executable="perception_node",
             name="perception_node", output="screen", parameters=[cfg]),
    ])