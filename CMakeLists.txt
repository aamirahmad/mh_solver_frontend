cmake_minimum_required(VERSION 2.8.3)
project(mh_solver_frontend)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  cv_bridge
  std_msgs
  image_transport
  message_generation
  geometry_msgs     
  nav_msgs
  read_omni_dataset
  tf
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  sensorPoolData.msg
  RobotState.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs     
  nav_msgs  
  read_omni_dataset
)

catkin_package(
CATKIN_DEPENDS roscpp rospy message_runtime read_omni_dataset
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})


include_directories(/home/aamir/local/include)
link_directories(/home/aamir/local/lib)
# link_libraries(csparse cxsparse g2o_csparse_extension g2o_cli g2o_core g2o_interface g2o_parser g2o_solver_cholmod g2o_solver_csparse g2o_solver_dense g2o_solver_pcg g2o_stuff g2o_types_icp g2o_types_sba g2o_types_sim3 g2o_types_slam2d g2o_types_slam3d)

### Eigen  #############################################
find_package(cmake_modules REQUIRED)

FIND_PACKAGE(Eigen REQUIRED)
INCLUDE_DIRECTORIES(${Eigen_INCLUDE_DIRS})

find_package(Boost COMPONENTS filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

### Eigen  #############################################


####### g2o ############################################

# SET (G2O_LIBS g2o_types_slam2d  g2o_cli g2o_core g2o_interface g2o_parser g2o_solver_cholmod
# g2o_solver_csparse g2o_solver_dense g2o_solver_pcg g2o_stuff g2o_types_icp
# g2o_types_sba g2o_types_sim3  g2o_types_slam3d csparse cxsparse g2o_csparse_extension)
SET (G2O_LIBS g2o_simulator
g2o_types_icp
g2o_types_sim3
g2o_incremental
g2o_types_slam3d_addons
g2o_types_sba
g2o_interactive
g2o_types_slam2d_addons
g2o_solver_slam2d_linear
g2o_types_sclam2d
g2o_types_data
g2o_types_slam3d
g2o_types_slam2d
g2o_viewer
g2o_solver_cholmod
g2o_solver_pcg
g2o_solver_dense
g2o_solver_csparse
#g2o_solver_eigen
g2o_solver_structure_only
g2o_cli
g2o_core
g2o_stuff
g2o_interface
g2o_parser
g2o_ext_freeglut_minimal
g2o_opengl_helper
g2o_csparse_extension)

add_library(G2O_LIBS STATIC IMPORTED)

# ADD_LIBRARY(G2O_LIBS libg2o /home/aamir/local/lib)

####### g2o custom types ###############################
#ADD_SUBDIRECTORY(src/custom_types)


add_executable(mh_solver_frontend src/mh_solver_frontend.cpp)
target_link_libraries(mh_solver_frontend ${G2O_LIBS} ${catkin_LIBRARIES} ${rosbag_LIBRARIES})

add_executable(single_robot_localization src/single_robot_localization.cpp)
target_link_libraries(single_robot_localization ${G2O_LIBS} ${catkin_LIBRARIES} ${rosbag_LIBRARIES})

