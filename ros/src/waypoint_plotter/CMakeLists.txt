cmake_minimum_required(VERSION 2.8.3)
project(waypoint_plotter)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
  styx_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES waypoint_updater
#  CATKIN_DEPENDS geometry_msgs roscpp rospy sensor_msgs std_msgs styx_msgs styx_utils
#  DEPENDS system_lib
)

include_directories(${catkin_INCLUDE_DIRS})
