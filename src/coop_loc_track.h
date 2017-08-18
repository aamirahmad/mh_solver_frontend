#include <cstdio>
#include <iostream>
#include <string>


#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <read_omni_dataset/BallData.h>
#include <read_omni_dataset/LRMLandmarksData.h>
#include <read_omni_dataset/LRMGTData.h>
#include <mh_solver_frontend/sensorPoolData.h>
#include <mh_solver_frontend/RobotState.h>
#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "g2o/config.h"
#include "g2o/core/estimate_propagator.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/hyper_dijkstra.h"
#include "g2o/core/hyper_graph_action.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/robust_kernel.h"
#include "g2o/core/robust_kernel_factory.h"
#include "g2o/core/optimization_algorithm.h"
#include "g2o/core/sparse_optimizer_terminate_action.h"


#include "g2o/stuff/macros.h"
#include "g2o/stuff/color_macros.h"
#include "g2o/stuff/command_args.h"
#include "g2o/stuff/filesys_tools.h"
#include "g2o/stuff/string_tools.h"
#include "g2o/stuff/timeutil.h"

#include "g2o/apps/g2o_cli/g2o_common.h"
#include "g2o/apps/g2o_cli/dl_wrapper.h"

#include "g2o/stuff/command_args.h"
#include "g2o/stuff/opengl_wrapper.h"
#include "g2o/types/slam2d/types_slam2d.h"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sys/time.h>

#define SAVE_GRAPHFILES
//#undef SAVE_GRAPHFILES 

// #define M_PI        3.141592653589793238462643383280    /* pi */


///These are just defaults. Must be set up from the launch file.

//////////////////////////////////////////
// const bool OMNI_ACTIVE[5] = {true,false,true,true,true};
int MY_ID;
int NUM_ROBOTS; // total number of robots in the team including self
int MAX_ROBOTS; // Just for convinience... not really a requirement
int NUM_TARGETS = 1;
int NUM_TARGETS_USED;
//////////////////////////////////////////


const std::size_t NUM_SENSORS_PER_ROBOT = 3;// SENSORS include odometry, each feature sensor like a ball detector, each landmark-set detector and so on. In this case for example the number of sensors are 3, 1-odometry, 1-orange ball, 1-landmarkset. Usually this must co-incide with the number of topics to which each robot is publishing its sensed data.



int NUM_POSES = 200;

//coefficients for landmark observation covariance
const std::size_t K1 = 2.0;
const std::size_t K2 = 0.5;

//coefficients for target observation covariance
// const std::size_t K3 = 0.2;
// const std::size_t K4 = 0.5;
// const std::size_t K5 = 0.5; 
const float K3 = 0.2, K4 = 0.5, K5 = 0.5; 

const std::size_t ROB_HT = 0.81; //fixed height above ground in meter


int MAX_VERTEX_COUNT;

//initial poses of the robot. Use the first one only for testing. second one is generic
//const double initArray[10] = {5.086676,-2.648978,0.0,0.0,1.688772,-2.095153,3.26839,-3.574936,4.058235,-0.127530};
const double initArray[10] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};


int MAX_INDIVIDUAL_STATES;

//these should be equal to MAX_VERTEX_COUNT in case of full graph optimization
int WINDOW_SIZE; //these two are not const anymore because they are set by the main is not 
int DECAY_LAMBDA;
int WINDOW_SIZE_TGT=10;
float avgComputationTime;


using namespace ros;
using namespace g2o;
using namespace std;

class Target
{
  
  public:
    Target(Eigen::Vector3d _initPose): initPose(_initPose), tgtVertexID(0), tgtVertexCounter(0) 
    {
    }
    
    //We use either this function to create a target node everytime a target is observed
    void addNewTGTVertexAndEdgesAround(double, double, double, double, int, int, Time, g2o::SparseOptimizer*);
    //Or we use this function assuming that the target nodes are created in sync with the Self-Robot Pose Nodes 
    void addObservationEdgesPoseToTarget(double, double, double, double, int, int, int, Time, g2o::SparseOptimizer*);
  
    Eigen::Vector3d initPose;
    long long unsigned curObservationTimestamp;
    long long unsigned prevObservationTimestamp;
    
    long long unsigned curNodeTimestamp;
    long long unsigned prevNodeTimestamp;
    
    int tgtVertexID;
    int tgtVertexID_prev;  
    int tgtVertexCounter;
    Eigen::Vector3d curPose;
    Eigen::Vector3d prevPose;
};


class SelfRobot
{
  geometry_msgs::PoseWithCovarianceStamped estimatedRobPose;  
  geometry_msgs::PoseStamped gtRobPose;  
  
  geometry_msgs::PointStamped estimatedTargetPose;  
  
  //One subscriber per sensor in the robot
  Subscriber sOdom_;
  Subscriber sBall_;
  Subscriber sLandmark_;
  Subscriber gtRob_sub_;  
  
  float computationTime[20000];

  int windowSolverInvokeCount;
  
  Eigen::Isometry2d initPose; // x y theta;
  Eigen::Isometry2d prevPose;
  int SE2vertexID_prev;
  int SE2vertexID; // current vertex id holder for the robot state
  //prev_id for target is only for the self robot
  int targetVertexID_prev;
  vector<bool> *ifRobotIsStarted;
  
  ///@TODO convert this to an array of pointers perhaps for multi-target case
  int *targetVertexID;  //current vertex id holder for the ball's state. it is passed as a pointer because the value is shared by all robots as it is for the same target
  int vertextCounter_; // counter of vertices for the individual robots
  int targetVertextCounter_;
  int *totalVertextCounter;
  int solverStep;
  mh_solver_frontend::RobotState msg;
  read_omni_dataset::BallData estimateBallState;
  Publisher selfState_publisher_generic, targetStatePublisher, targetStatePublisher_generic, virtualGTPublisher; // virtualGTPublisher because gt publisher is actually the rosbag but we need to subscribe to it and publish it synchronously with the estimated rbot state output.
  
  vector<Publisher> estimatedStatePublishers;
  
  vector<Target*> targetsToTrack;
  
  //New Variables for scaled arrival cost function in MHE (refer Rao 2013 for theoretical details)
  //double U_prior;
  //double U[MAX_INDIVIDUAL_STATES];
  double decayCoeff;
  vector<int> *currentPoseVertexIDs;
  
  FILE *mhls_g2o;
  

//   bool 
  
  public:
    SelfRobot(NodeHandle& nh, g2o::SparseOptimizer* graph, int robotNumber, int startCounter, int* totVertCount, int* tgtVertexID, Eigen::Isometry2d _initPose, vector<Target*> _targetsToTrack,vector<int>* _curPosVerID, vector<bool> * _ifRobotIsStarted): vertextCounter_(startCounter), totalVertextCounter(totVertCount), SE2vertexID_prev(0), SE2vertexID(0), initPose(_initPose), targetVertexID(tgtVertexID), targetVertextCounter_(0), targetsToTrack(_targetsToTrack), solverStep(0),currentPoseVertexIDs(_curPosVerID),ifRobotIsStarted(_ifRobotIsStarted)
    {
        
      (*ifRobotIsStarted)[robotNumber] = false;
      
      sOdom_ = nh.subscribe<nav_msgs::Odometry>("/omni"+boost::lexical_cast<string>(robotNumber+1)+"/odometry", 10, boost::bind(&SelfRobot::selfOdometryCallback,this, _1,robotNumber+1,graph));
      
      sBall_ = nh.subscribe<read_omni_dataset::BallData>("/omni"+boost::lexical_cast<string>(robotNumber+1)+"/orangeball3Dposition", 10, boost::bind(&SelfRobot::selfTargetDataCallback,this, _1,robotNumber+1,graph));
      
      sLandmark_ = nh.subscribe<read_omni_dataset::LRMLandmarksData>("/omni"+boost::lexical_cast<string>(robotNumber+1)+"/landmarkspositions", 10, boost::bind(&SelfRobot::selfLandmarkDataCallback,this, _1,robotNumber+1,graph));
      
      gtRob_sub_ =  nh.subscribe<geometry_msgs::PoseStamped>("/omni"+boost::lexical_cast<string>(robotNumber+1)+"/simPose", 10, boost::bind(&SelfRobot::selfGTDataCallback,this, _1,robotNumber+1,graph));
      
      ROS_INFO(" constructing robot object and called sensor subscribers for robot %d",robotNumber+1);
      
      selfState_publisher_generic = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("estimatedRobotPose_omni"+boost::lexical_cast<string>(robotNumber+1), 1000); //mhls is
    
      targetStatePublisher = nh.advertise<read_omni_dataset::BallData>("targetEstimatedState_by_omni"+boost::lexical_cast<string>(MY_ID), 1000);
      
      targetStatePublisher_generic = nh.advertise<geometry_msgs::PointStamped>("genericTargetEstimatedState_by_omni"+boost::lexical_cast<string>(MY_ID), 1000);

    
      estimatedStatePublishers.resize(NUM_ROBOTS);
      for(int i=0; i<NUM_ROBOTS; i++)
          estimatedStatePublishers[i] = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("estimatedRobotPose_omni"+boost::lexical_cast<string>(i+1), 1000); //mhls is
      
      mhls_g2o = fopen("mhls_g2o.g2o","w");
      
      fprintf(mhls_g2o,"VERTEX_XY 4 0 -1.5\n");
      fprintf(mhls_g2o,"FIX 4\n");
      fprintf(mhls_g2o,"VERTEX_XY 5 0 1.5\n");
      fprintf(mhls_g2o,"FIX 5\n");
      fprintf(mhls_g2o,"VERTEX_XY 6 3 -4.5\n");
      fprintf(mhls_g2o,"FIX 6\n");
      fprintf(mhls_g2o,"VERTEX_XY 7 3 4.5\n");
      fprintf(mhls_g2o,"FIX 7\n");
      fprintf(mhls_g2o,"VERTEX_XY 8 3.75 -2.25\n");
      fprintf(mhls_g2o,"FIX 8\n");
      fprintf(mhls_g2o,"VERTEX_XY 9 3.75 2.25\n");
      fprintf(mhls_g2o,"FIX 9\n");
      
        avgComputationTime = 0.0;
	windowSolverInvokeCount = 0; 
    }


    void selfOdometryCallback(const nav_msgs::Odometry::ConstPtr&, int, g2o::SparseOptimizer*);
    
    void selfTargetDataCallback(const read_omni_dataset::BallData::ConstPtr&, int, g2o::SparseOptimizer*);
     
    void selfLandmarkDataCallback(const read_omni_dataset::LRMLandmarksData::ConstPtr&, int, g2o::SparseOptimizer*);
  
    void selfGTDataCallback(const geometry_msgs::PoseStamped::ConstPtr&, int, g2o::SparseOptimizer*);

    /// Solve the sliding widnow graph
    void solveSlidingWindowGraph(g2o::SparseOptimizer*);
    
    /// publish the estimated state of all the robots and the target after solving the graph
    void publishSelfState(g2o::SparseOptimizer*);
  
    Eigen::Isometry2d curPose;
    Eigen::Isometry2d curCorrectedPose;
    Time curTime;
    Time prevTime;
    
};


class TeammateRobot
{
    
  geometry_msgs::PoseWithCovarianceStamped estimatedRobPose;  
  geometry_msgs::PoseStamped gtRobPose;  
  
  //One subscriber per sensor in the robot
  Subscriber sOdom_;
  Subscriber sBall_;
  Subscriber sLandmark_;
  
  Eigen::Isometry2d initPose; // x y theta;
  Eigen::Isometry2d prevPose;
  int SE2vertexID_prev;
  int SE2vertexID; // current vertex id holder for the robot state
  int *targetVertexID;  //current vertex id holder for the ball's state
  int vertextCounter_; // counter of vertices for the individual robots
  int *totalVertextCounter;
  mh_solver_frontend::RobotState msg;
  //Publisher robotState_publisher; // has no target state publisher... only selfRobot publishes target state
  vector<bool> *ifRobotIsStarted;
  
  vector<Target*> targetsToTrack;
  
  double decayCoeff;
  vector<int> *currentPoseVertexIDs;
  
  public:
    TeammateRobot(NodeHandle& nh, g2o::SparseOptimizer* graph, int robotNumber, int startCounter, int* totVertCount, int* tgtVertexID, Eigen::Isometry2d _initPose, vector<Target*> _targetsToTrack, vector<int> *  _curPosVerID, vector<bool> * _ifRobotIsStarted): vertextCounter_(startCounter), totalVertextCounter(totVertCount), SE2vertexID_prev(0), SE2vertexID(0), initPose(_initPose), targetVertexID(tgtVertexID), targetsToTrack(_targetsToTrack),currentPoseVertexIDs(_curPosVerID),ifRobotIsStarted(_ifRobotIsStarted)
    {
      
      (*ifRobotIsStarted)[robotNumber] = false;
       
      sOdom_ = nh.subscribe<nav_msgs::Odometry>("/omni"+boost::lexical_cast<string>(robotNumber+1)+"/odometry", 10, boost::bind(&TeammateRobot::teammateOdometryCallback,this, _1,robotNumber+1,graph));
      
      sBall_ = nh.subscribe<read_omni_dataset::BallData>("/omni"+boost::lexical_cast<string>(robotNumber+1)+"/orangeball3Dposition", 10, boost::bind(&TeammateRobot::teammateTargetDataCallback,this, _1,robotNumber+1,graph));
      
      sLandmark_ = nh.subscribe<read_omni_dataset::LRMLandmarksData>("/omni"+boost::lexical_cast<string>(robotNumber+1)+"/landmarkspositions", 10, boost::bind(&TeammateRobot::teammateLandmarkDataCallback,this, _1,robotNumber+1,graph));
      
      ROS_INFO(" constructing robot object and called sensor subscribers for robot %d",robotNumber+1);
      
      //robotState_publisher = nh.advertise<mh_solver_frontend::RobotState>("/omni"+boost::lexical_cast<string>(robotNumber+1)+"State", 1000);
      
    }


    void teammateOdometryCallback(const nav_msgs::Odometry::ConstPtr&, int, g2o::SparseOptimizer*);
    
    void teammateTargetDataCallback(const read_omni_dataset::BallData::ConstPtr&, int, g2o::SparseOptimizer*);
     
    void teammateLandmarkDataCallback(const read_omni_dataset::LRMLandmarksData::ConstPtr&, int, g2o::SparseOptimizer*);
  

    // publish the estimated state of all the teammate robot
    void publishState(g2o::SparseOptimizer*);
  
    Eigen::Isometry2d curPose;
    Time curTime;
    
};



class GenerateGraph
{
  NodeHandle nh_;
  Rate loop_rate_;
  
  
  SelfRobot* robot_;
  vector<TeammateRobot*> teammateRobots_;
  vector<Target*> targets_;
  
  g2o::SparseOptimizer* graph_;
  
  int totalVertextCounter_; // counter of vertices for the full graph (including all robots and the targets)
  int targetVertexID_;  //current vertex id holder for the ball's state
  vector<int> currentPoseVertexID;
  vector<bool> robotStarted; // to indicate whether a robot has started or not..
  
  
  public:
    GenerateGraph(NodeHandle &_nh, g2o::SparseOptimizer* _graph): nh_(_nh), graph_(_graph), totalVertextCounter_(0), loop_rate_(30),targetVertexID_(0)
    { 
        
        
      nh_.getParam("MY_ID", MY_ID);    
      nh_.getParam("NUM_ROBOTS", NUM_ROBOTS);      
      nh_.getParam("MAX_VERTEX_COUNT", MAX_VERTEX_COUNT);    
      nh_.getParam("MAX_INDIVIDUAL_STATES", MAX_INDIVIDUAL_STATES);    
      
      printf("MY_ID =%d\n",MY_ID);
      printf("NUM_ROBOTS =%d\n",NUM_ROBOTS);
        
      Eigen::Isometry2d initialRobotPose;
      Eigen::Vector3d initialTargetPosition;
      
      teammateRobots_.reserve(NUM_ROBOTS);
      targets_.reserve(NUM_TARGETS);
      currentPoseVertexID.reserve(NUM_ROBOTS);
      robotStarted.reserve(NUM_ROBOTS);      
      
      for(int j=0;j<NUM_TARGETS;j++)
      {
	initialTargetPosition = Eigen::Vector3d(0,0,0);
	Target *tempTarget = new Target(initialTargetPosition);
	targets_.push_back(tempTarget);
      }
	
      for(int i=0;i<NUM_ROBOTS;i++)
      {
	initialRobotPose = Eigen::Rotation2Dd(-M_PI).toRotationMatrix();
	initialRobotPose.translation() = Eigen::Vector2d(0,0); 
	
	if(i+1 == MY_ID)
	{
	  robot_ = new SelfRobot(nh_, graph_,i,0,&totalVertextCounter_,&targetVertexID_,initialRobotPose,targets_,&currentPoseVertexID,&robotStarted);
	}
	else
	{	  
	  TeammateRobot *tempRobot = new TeammateRobot(nh_, graph_,i,0,&totalVertextCounter_,&targetVertexID_,initialRobotPose, targets_,&currentPoseVertexID,&robotStarted);
	  teammateRobots_.push_back(tempRobot);
	}	
      }
   
      
    }
    
    void addFixedLandmarkNodes(g2o::SparseOptimizer*);

    
    
};






































