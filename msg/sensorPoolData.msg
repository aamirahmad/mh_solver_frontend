Header header

# total number of robots in the team
int8 numRobots

# One ball per robot
geometry_msgs/PoseWithCovarianceStamped[] targetPose

# 10 landmarks per robot in the team 
geometry_msgs/PoseWithCovarianceStamped[] landmarkPose

# 1 odometry message per robot
geometry_msgs/PoseWithCovarianceStamped[] robotPose
