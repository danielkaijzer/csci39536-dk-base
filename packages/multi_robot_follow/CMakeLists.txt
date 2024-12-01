cmake_minimum_required(VERSION 3.0.2)
project(multi_robot_follow)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  cv_bridge
  tf
  tf2
  tf2_ros
  gazebo_ros
  urdf
  xacro
  turtlebot3_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
    nav_msgs
    cv_bridge
    tf
    tf2
    tf2_ros
    gazebo_ros
    urdf
    xacro
    turtlebot3_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# Make sure Python scripts are installed properly
catkin_install_python(PROGRAMS
  scripts/robot_follower.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

# Install URDF files
install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
  FILES_MATCHING PATTERN "*.urdf" PATTERN "*.xacro"
)

# Install world files if you have any
install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
  FILES_MATCHING PATTERN "*.world"
  PATTERN ".svn" EXCLUDE
)

# Add any C++ executables if needed
# add_executable(${PROJECT_NAME}_node src/your_cpp_file.cpp)
# target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})
