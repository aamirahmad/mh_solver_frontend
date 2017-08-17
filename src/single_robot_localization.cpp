
#include "single_robot_localization.h"


inline double normalizeAngle(double x)
{
    x = fmod(x + M_PI,2*M_PI);
    if (x < 0)
        x += 2*M_PI;
    return x - M_PI;
}

inline void addLandmark(int vertexId, double x, double y, g2o::SparseOptimizer* graph_ptr)
{
      Eigen::Vector2d landmark;
      landmark =  Eigen::Vector2d(x, y);
      VertexPointXY * v = new VertexPointXY();
      v->setId(vertexId);  
      v->setEstimate(landmark);
      v->setFixed(true); 
      graph_ptr->addVertex(v);
}


void addFixedLandmarkNodes(g2o::SparseOptimizer* graph_ptr)
{
    addLandmark(0,  6.0,  4.5, graph_ptr);
    addLandmark(1,  6.0, -4.5, graph_ptr);
    addLandmark(2,  0.0, -4.5, graph_ptr);
    addLandmark(3,  0.0,  4.5, graph_ptr);
    addLandmark(4,  0.0, -1.5, graph_ptr);
    addLandmark(5,  0.0,  1.5, graph_ptr);
    addLandmark(6,  3.0, -4.5, graph_ptr);
    addLandmark(7,  3.0,  4.5, graph_ptr);
    addLandmark(8,  3.75, -2.25, graph_ptr);
    addLandmark(9,  3.75,  2.25, graph_ptr);   
}




////////////////////////// METHOD DEFINITIONS OF THE SELFROBOT CLASS //////////////////////

void SelfRobot::selfOdometryCallback(const nav_msgs::Odometry::ConstPtr& odometry, int RobotNumber, g2o::SparseOptimizer* graph_ptr)
{
  //The robot has started
  (*ifRobotIsStarted)[RobotNumber-1]=true;
  uint seq = odometry->header.seq;
  prevTime = curTime;
  curTime = odometry->header.stamp;
  
  //ROS_WARN(" got odometry from robot %d at time %d",RobotNumber, odometry->header.stamp.sec);  
  //ROS_WARN(" value of vertextCount_ for robot %d is %d",RobotNumber,vertextCounter_);  
  //   ROS_WARN(" TotalvertextCount_ now is %d",*totalVertextCounter);  
  
  if((*totalVertextCounter)<MAX_VERTEX_COUNT)
  {
    
    double odomVector[3] = {odometry->pose.pose.position.x, odometry->pose.pose.position.y, tf::getYaw(odometry->pose.pose.orientation)};
    
    Eigen::Isometry2d odom; 
    odom = Eigen::Rotation2Dd(odomVector[2]).toRotationMatrix();
    odom.translation() = Eigen::Vector2d(odomVector[0], odomVector[1]);    
    
    if(vertextCounter_ == 0)
    {
      curPose = initPose;
      avgComputationTime = 0.0;
    }
    else
    {
      prevPose = curPose;
      curPose = prevPose*odom;
    }
    
    
    //Create Self Robot Pose Vertex
    // vertext id for the robots start at MAX_INDIVIDUAL_STATES times the robot number
    if(SE2vertexID == 0)
      SE2vertexID = (RobotNumber)*MAX_INDIVIDUAL_STATES;
    else
      SE2vertexID++;
    
    VertexSE2 * v = new VertexSE2();
    v->setId(SE2vertexID);  
    v->setEstimate(curPose);
    v->timestamp =  (unsigned long long) 1000000*curTime.sec + curTime.nsec/1000;
    v->agentID = RobotNumber;
    v->isOptimizedAtLeastOnce = false;
    v->odometryReading = odom;
    //if(vertextCounter_ == 0)
     //v->setFixed(true);
    graph_ptr->addVertex(v);
    (*currentPoseVertexIDs)[RobotNumber-1] = SE2vertexID;
    //ROS_WARN("SE2vertexID = %d",SE2vertexID);
    //cout<<"(*currentPoseVertexIDs)[RobotNumber] = "<<(*currentPoseVertexIDs)[RobotNumber-1]<<endl;
    
    //start adding self-robot pose-pose edges if seq is greater than 0
    if(vertextCounter_>0) 
    {
      SE2vertexID_prev = SE2vertexID - 1;
      v->setFixed(false);
      
      //Now add edge from the current node to the previous node
      EdgeSE2 * e = new EdgeSE2;
      // retrieve vertex pointers from graph with id's
      g2o::OptimizableGraph::Vertex * pose_a_vertex
	      = dynamic_cast<g2o::OptimizableGraph::Vertex*>
		(graph_ptr->vertices()[SE2vertexID_prev]);
      g2o::OptimizableGraph::Vertex * pose_b_vertex
	      = dynamic_cast<g2o::OptimizableGraph::Vertex*>
		(graph_ptr->vertices()[SE2vertexID]);			
		
      //std::cout <<"SE2 Edge from to \n";		
      //std::cout <<SE2vertexID_prev<<"\n"; 		
      //std::cout <<SE2vertexID<<"\n"; 			
      // error check vertices and associate them with the edge
      assert(pose_a_vertex!=NULL);
      assert(pose_a_vertex->dimension() == 3);
      e->vertices()[0] = pose_a_vertex;

      assert(pose_b_vertex!=NULL);
      assert(pose_b_vertex->dimension() == 3);
      e->vertices()[1] = pose_b_vertex;	  
            
      //cout<<"odometry values = "<<odomVector[0]<<" "<<odomVector[1]<<" "<<odomVector[2]<<endl;
      
      // add information matrix
      Eigen::Matrix<double, 3, 3> Lambda;      
      Lambda<<500, 0.0, 0.0, 0.0,500, 0.0,0.0,0.0, 5000;      
      
      // set the observation and imformation matrix
      const SE2 a(odom);
      e->setMeasurement(a);
      e->information() = Lambda;      
      
      // finally add the edge to the graph
      if(!graph_ptr->addEdge(e))
      {
	assert(false);
      } 
      
    }
    else ///First node is not set as a fixed node.. A prior node with 0 mean and high covariance is created and attached to the firt node with a very loose edge
    {
      v->setFixed(false); 
      
      Eigen::Isometry2d priorPose;
      VertexSE2* firstRobotnode = dynamic_cast<VertexSE2*>(graph_ptr->vertices()[SE2vertexID]);
      priorPose = Eigen::Rotation2Dd(0);
      priorPose.translation() = Eigen::Vector2d(0,0);   
      //Create the Prior SE2 Node now. This node will keep track of the past information.
      VertexSE2 * v_prior = new VertexSE2();
      v_prior->setId((RobotNumber)*MAX_INDIVIDUAL_STATES - 1); //The ID is one before the first node pose of this robot  
      v_prior->setEstimate(priorPose);
      v_prior->timestamp =  firstRobotnode->timestamp;
      v_prior->agentID = RobotNumber;
      v_prior->setFixed(true);
      graph_ptr->addVertex(v_prior);
      
      //Now connect the fixed prior node to the first node through a very "loose" edge! An edge that means we have poorinformation regarding the prior
      EdgeSE2 * edgeFirst_to_Prior = new EdgeSE2;	
      edgeFirst_to_Prior->vertices()[0] = v_prior;
      edgeFirst_to_Prior->vertices()[1] = firstRobotnode;	  

      // add information matrix
      Eigen::Matrix<double, 3, 3> Lambda;      
      Lambda<<0.001, 0.0, 0.0, 0.0,0.001, 0.0,0.0,0.0, 0.0001;      
      
      // set the observation and imformation matrix
      Eigen::Isometry2d tempMean; tempMean = Eigen::Rotation2Dd(0); tempMean.translation() = Eigen::Vector2d(0,0);   
      edgeFirst_to_Prior->setMeasurement(tempMean);
      edgeFirst_to_Prior->information() = Lambda;      
      // finally add the edge to the graph
      if(!graph_ptr->addEdge(edgeFirst_to_Prior))
      {
	assert(false);
      } 
      
      //U_prior = Lambda.determinant();
      
    }    

   vertextCounter_++; 
   (*totalVertextCounter)++;
   
   ///... here is where the sliding window solver is invoked..
   
   //cout<<"Total robot(s) state vertices = "<< (*totalVertextCounter) <<endl;
   
   
   if(vertextCounter_ > WINDOW_SIZE)
   {
      if(vertextCounter_ == MAX_VERTEX_COUNT-5)
      {
	cout<<"avgComputationTime = "<<avgComputationTime<<endl;
	cout<<"windowSolverInvokeCount = "<<windowSolverInvokeCount<<endl;	
	//compute average time of solver per iteration of the estimator
	avgComputationTime = avgComputationTime/windowSolverInvokeCount;
	printf("\n\nverage time taken by the solver per iteration of the estimator = %f\n\n\n",avgComputationTime);
      }     
      //Now solve the resulting graph
      timespec ts1,ts2;
      clock_gettime(CLOCK_REALTIME, &ts1);
      //cout<<"some time =  "<< ts.tv_nsec <<" (nano)seconds"<<'\n';

      solveSlidingWindowGraph(graph_ptr);
      

      clock_gettime(CLOCK_REALTIME, &ts2);
      float diff_millisec = fabs(((float)ts2.tv_nsec - (float)ts1.tv_nsec)/1000000);
      cout<<"time taken =  "<< diff_millisec <<" milli seconds"<<'\n';
      
      computationTime[vertextCounter_] = diff_millisec;
      
      if(diff_millisec<30) //removing outliers
      {
	avgComputationTime += diff_millisec;
        windowSolverInvokeCount++;
      }
      
      
      //cout<<"avgComputationTime = "<<avgComputationTime<<endl;
      //cout<<"windowSolverInvokeCount = "<<windowSolverInvokeCount<<endl;
      
      //cout<<"I am here"<<endl;
      Eigen::Isometry2d priorPose;
      
      // Prior (Mean) will be set to the vertex pose that is going to be deleted in the sliding window case 
      VertexSE2* vertex_to_become_prior = dynamic_cast<VertexSE2*>(graph_ptr->vertices()[SE2vertexID-WINDOW_SIZE]);
      priorPose = vertex_to_become_prior->estimate().toIsometry();
      
      //We now have to remove the last added prior node and a new prior node that contains the updated past. One may simply update the previous prior node but seems like it is not so straightforward to do so using g2o. Therefore I am copying the prious prior, deleting it, updating locally and then adding a new prior node which is the updated prior node
      VertexSE2* v_prior_old = dynamic_cast<VertexSE2*>(graph_ptr->vertices()[(RobotNumber)*MAX_INDIVIDUAL_STATES - 1]);
      graph_ptr->removeVertex(v_prior_old,/*bool detach=*/false);
      
      VertexSE2 * v_prior = new VertexSE2();
      v_prior->setId((RobotNumber)*MAX_INDIVIDUAL_STATES - 1); //The ID is one before the first node pose of this robot  
      v_prior->setEstimate(priorPose);
      v_prior->timestamp =  vertex_to_become_prior->timestamp;
      v_prior->agentID = RobotNumber;
      v_prior->setFixed(true);
      graph_ptr->addVertex(v_prior);
      
  
      //Now remove the oldest node (vertex_to_become_prior), which has already been used to update the new prior node to maintain the sliding window. However, grab the last hessian block of this vertex. This hessian will become the information contained by the prior node
      
      //cout<<"vertex_to_become_prior->hessian is = "<<endl<<vertex_to_become_prior->hessian(0,0)<<" "<<vertex_to_become_prior->hessian(0,1)<<"  "<<vertex_to_become_prior->hessian(0,2)<<endl;
      //cout<<vertex_to_become_prior->hessian(1,0)<<"  "<<vertex_to_become_prior->hessian(1,1)<<"  "<<vertex_to_become_prior->hessian(1,2)<<endl;
      //cout<<vertex_to_become_prior->hessian(2,0)<<"  "<<vertex_to_become_prior->hessian(2,1)<<"  "<<vertex_to_become_prior->hessian(2,2)<<endl;	
      Eigen::Matrix<double, 3, 3> Hessian;      
      Hessian<<vertex_to_become_prior->hessian(0,0), vertex_to_become_prior->hessian(0,1), vertex_to_become_prior->hessian(0,2), vertex_to_become_prior->hessian(1,0),vertex_to_become_prior->hessian(1,1), vertex_to_become_prior->hessian(1,2),vertex_to_become_prior->hessian(2,0),vertex_to_become_prior->hessian(2,1), vertex_to_become_prior->hessian(2,2);
      //Hessian<<50000, 0.0, 0.0, 0.0,50000, 0.0,0.0,0.0, 500000; 
      //New hessian is an exponentially decayed version of the old
      decayCoeff = 0.1;
      Hessian = Hessian*(exp(-decayCoeff*DECAY_LAMBDA)); //decay is exponentially with time.
      
      graph_ptr->removeVertex(vertex_to_become_prior,/*bool detach=*/false);
      
      //Now connect the fixed prior node to the first node of the rest of the seequence through a very "loose" edge! An edge that means we have poorinformation regarding the prior
	      
      EdgeSE2 * edgeFirst_to_Prior = new EdgeSE2;
      VertexSE2* firstVertex = dynamic_cast<VertexSE2*>(graph_ptr->vertices()[SE2vertexID-WINDOW_SIZE+1]); //remember it is the next node after the removed one
      
      edgeFirst_to_Prior->vertices()[0] = v_prior;
      edgeFirst_to_Prior->vertices()[1] = firstVertex;	  

      // set the observation and imformation matrix
      Eigen::Isometry2d odomFromFirstVertex = firstVertex->odometryReading; //measurement to connect the first node to prior node   
      //cout<<"odomFromFirstVertex = "<<odomFromFirstVertex.translation().x()<<" , "<<odomFromFirstVertex.translation().y()<<" agent ID = "<<firstVertex->agentID<<endl;
      edgeFirst_to_Prior->setMeasurement(odomFromFirstVertex);
      edgeFirst_to_Prior->information() = Hessian;      
      // finally add the edge to the graph
      if(!graph_ptr->addEdge(edgeFirst_to_Prior))
      {
	assert(false);
      } 
      
      //Calculating Arrival Cost at T-N
      double x_t = v_prior->estimate().translation().x();
      double y_t = v_prior->estimate().translation().y();
      double theta_t = v_prior->estimate().rotation().angle();
      
      double x_t_1 = firstVertex->estimate().translation().x();
      double y_t_1 = firstVertex->estimate().translation().y();
      double theta_t_1 = firstVertex->estimate().rotation().angle();
      
      Eigen::Vector3d delta = Eigen::Vector3d(odomVector[0], odomVector[1], odomVector[2]);
      
      Eigen::Vector3d h_t_t_1 = Eigen::Vector3d((x_t_1-x_t)*cos(theta_t) + (y_t_1-y_t)*sin(theta_t),
      -(x_t_1-x_t)*sin(theta_t) + (y_t_1-y_t)*cos(theta_t),
	normalizeAngle(theta_t_1-theta_t)
      );
      
      Eigen::Vector3d error = Eigen::Vector3d((delta(0)-h_t_t_1(0))*cos(h_t_t_1(2)) + (delta(1)-h_t_t_1(1))*sin(h_t_t_1(2)),
      -(delta(0)-h_t_t_1(0))*sin(h_t_t_1(2)) + (delta(1)-h_t_t_1(1))*cos(h_t_t_1(2)),
	normalizeAngle((delta(2)-h_t_t_1(2)))
      );
      
      //       cout<<"Hessian is = "<<endl<<
      //             Hessian(0,0)<<"  "<<Hessian(0,1)<<"  "<<Hessian(0,2)<<endl;
      //       cout<<Hessian(1,0)<<"  "<<Hessian(1,1)<<"  "<<Hessian(1,2)<<endl;
      //       cout<<Hessian(2,0)<<"  "<<Hessian(2,1)<<"  "<<Hessian(2,2)<<endl;	
      //cout<<"error.transpose()*Hessian*error = "<<error.transpose()*Hessian*error<<endl;
      //cout<<"Hessian.determinant() = "<<Hessian.determinant()<<endl;
      //double beta_t_n = U[vertextCounter_-1-WINDOW_SIZE]/();
   }
   else
     if(vertextCounter_ >1)
     {
	//simply solve it
	solveSlidingWindowGraph(graph_ptr);
	//U[vertextCounter_-1] = graph_ptr->chi2() + U_prior;
       
     }   
   
  }
  
  else
  {
    
    //Solve once more before exiting
    fclose(mhls_g2o);
    //solveSlidingWindowGraph(graph_ptr);
    (*totalVertextCounter) = 0;
    std::cout << "Saving original graph to temp_graph.g2o...\n";
    graph_ptr->save("original_graph.g2o");  
    graph_ptr->clear();
    ros::shutdown();
  }
  

}


void SelfRobot::selfLandmarkDataCallback(const read_omni_dataset::LRMLandmarksData::ConstPtr& landmarkData, int RobotNumber, g2o::SparseOptimizer* graph_ptr)
{
  if(!(*ifRobotIsStarted)[RobotNumber-1])
      return;
  
  //ROS_INFO(" got landmark from robot %d with current vertex id as %d",RobotNumber,SE2vertexID);  
  
  uint seq = landmarkData->header.seq;
  
  int curRobotPoseVertexID = SE2vertexID;

  
  for(int i=0;i<10; i++)
  {
    if(landmarkData->found[i])
    {
     
      //cout<<"landmark "<<i<<" is at X_rob = "<<landmarkData->x[i]<< " and Y_rob = "<<landmarkData->y[i]<<endl;
      Eigen::Vector2d tempLandmarkObsVec = Eigen::Vector2d(landmarkData->x[i],landmarkData->y[i]);
      // add information matrix
      Eigen::Matrix<double, 2, 2> tempInformationOnLandmark;
      
      //here we calculate the information matrix.
      
      double d = tempLandmarkObsVec.norm(),
	      phi = atan2(landmarkData->y[i],landmarkData->x[i]);
      
      double covDD = (K1*fabs(1.0-(landmarkData->AreaLandMarkActualinPixels[i]/landmarkData->AreaLandMarkExpectedinPixels[i])))*(d*d);
      double covPhiPhi = K2*(1/(d+1));
      
      double covXX = pow(cos(phi),2) * covDD 
				  + pow(sin(phi),2) * ( pow(d,2) * covPhiPhi + covDD * covPhiPhi );
      double covYY = pow(sin(phi),2) * covDD 
				  + pow(cos(phi),2) * ( pow(d,2) * covPhiPhi + covDD * covPhiPhi );
      
      //cout<<"landmark "<<i<<" has at 10/covXX = "<<1/covXX<< " and 10/covYY = "<<1/covYY<<endl;			  
				  

      if(covXX<0||covYY<0)
	ROS_WARN(" covariance negative!!! ");
//       tempInformationOnLandmark<<1/covXX,0,
//       				 0,1/covYY;
           tempInformationOnLandmark<<10,0,
                                        0,10;                                 
				 
	//cout<<" information gained by landmark number "<<i<<" = "<<endl<<
	//1/covXX<<" "<<0<<endl<<
	//0<<"  "<<1/covYY<<endl;
      
      EdgeSE2PointXY * e = new  EdgeSE2PointXY();
      // retrieve vertex pointers from graph with id's
      g2o::OptimizableGraph::Vertex * pose_a_vertex
	      = dynamic_cast<g2o::OptimizableGraph::Vertex*>
		(graph_ptr->vertices()[SE2vertexID]);//current robot pose vertex
      g2o::OptimizableGraph::Vertex * pose_b_vertex
	      = dynamic_cast<g2o::OptimizableGraph::Vertex*>
		(graph_ptr->vertices()[i]);//landmark Vertex	
      
      if(pose_a_vertex!=NULL && pose_a_vertex->dimension() == 3)		
	e->vertices()[0] = pose_a_vertex;
      else
      {
	ROS_WARN(" Robot pose does not exist ");
	break;
      }   	
	
      if(pose_b_vertex!=NULL && pose_b_vertex->dimension() == 2)
	e->vertices()[1] = pose_b_vertex;
      else
      {
	ROS_WARN(" Landmark does not exist ??? ");
	break;
      }   	
		
      
      e->setMeasurement(tempLandmarkObsVec);
      e->information() = tempInformationOnLandmark;   
      if(!graph_ptr->addEdge(e))
      {
// 	ROS_WARN(" Edge not added because it may already exist ");
      }    
    }
  }
  
  
  
 }


void SelfRobot::selfGTDataCallback(const geometry_msgs::PoseStamped::ConstPtr& gtMsgReceived,int RobotNumber, g2o::SparseOptimizer* graph_ptr)
{
  // This can be used for comparison w.r.t. ground truth data.  
  gtRobPose = *gtMsgReceived;
}
    

void SelfRobot::solveSlidingWindowGraph(g2o::SparseOptimizer* graph_ptr)
{
      /// start Optimizatin here
     
      char filename[100];
      sprintf(filename, "original_graph_only.g2o");
#ifdef  SAVE_GRAPHFILES
      graph_ptr->save(filename);
#endif
      std::string strSolver = "lm_var_cholmod";
      g2o::OptimizationAlgorithmFactory* solverFactory = g2o::OptimizationAlgorithmFactory::instance();
      g2o::OptimizationAlgorithmProperty solverProperty;
      //solverFactory->listSolvers(cerr);
      graph_ptr->setAlgorithm(solverFactory->construct(strSolver, solverProperty));    
      
      string strRobustKernel = "Cauchy";
      AbstractRobustKernelCreator* creator = RobustKernelFactory::instance()->creator(strRobustKernel);
      for (SparseOptimizer::EdgeSet::const_iterator it = graph_ptr->edges().begin(); it != graph_ptr->edges().end(); ++it) 
      {
	OptimizableGraph::Edge* e = static_cast<OptimizableGraph::Edge*>(*it);
	  e->setRobustKernel(creator->construct());
	  e->robustKernel()->setDelta(1.0);
      }    
      
      graph_ptr->setVerbose(false);
      bool flagtostop = 0;
      graph_ptr->setForceStopFlag(&flagtostop);
      int maxIterations = 100;  

      if (! graph_ptr->solver()) {
	std::cout << "Error allocating solver. Allocating \"" << strSolver << "\" failed!" << std::endl;
      }  


      graph_ptr->initializeOptimization();
      graph_ptr->computeActiveErrors();

      int result = graph_ptr->optimize(maxIterations);
      if (maxIterations > 0 && result==g2o::OptimizationAlgorithm::Fail){
	std::cout << "Optimization failed at iteration = " << result<<", result might be invalid" << std::endl;
      }
      else
      {
	double finalChi=graph_ptr->chi2();
	std::cout << "Optimization complete with final ChiSquare value = "<<finalChi<<" and iterations = "<< result<<std::endl; 
      }
      
      sprintf(filename, "solved_graph_last.g2o");
#ifdef  SAVE_GRAPHFILES
      graph_ptr->save(filename);
#endif 
      
      
      VertexSE2* mostRecetPoseVertex = dynamic_cast<VertexSE2*>(graph_ptr->vertices()[SE2vertexID]);
      
      
      // extracting most recent self pose
      //cout<<"Most recent Pose theta is = "<<mostRecetPoseVertex->estimate().rotation().angle()<<endl;
      curPose = mostRecetPoseVertex->estimate().toIsometry();      
      
      //cout<<"mostRecetPoseVertex->hessian is = "<<endl<<mostRecetPoseVertex->hessian(0,0)<<" "<<mostRecetPoseVertex->hessian(0,1)<<"  "<<mostRecetPoseVertex->hessian(0,2)<<endl;
      //cout<<mostRecetPoseVertex->hessian(1,0)<<"  "<<mostRecetPoseVertex->hessian(1,1)<<"  "<<mostRecetPoseVertex->hessian(1,2)<<endl;
      //cout<<mostRecetPoseVertex->hessian(2,0)<<"  "<<mostRecetPoseVertex->hessian(2,1)<<"  "<<mostRecetPoseVertex->hessian(2,2)<<endl;
      //cout<<"Eigen::Rotation2Dd(((VertexSE2*)mostRecetPoseVertex)->estimate().rotation().angle()).toRotationMatrix() = "<<endl<<Eigen::Rotation2Dd(((VertexSE2*)mostRecetPoseVertex)->estimate().rotation().angle()).toRotationMatrix()<<endl;   
      initPose = curPose;
      publishSelfState(graph_ptr);
      
      
      //fixing the initial nodes
      
//       graph_ptr->clear();
//       addFixedLandmarkNodes(graph_ptr);
//       SE2vertexID = 0;
//       vertextCounter_ = 0;
//       (*totalVertextCounter)= 0;   
   
      solverStep++;  
}


void SelfRobot::publishSelfState(g2o::SparseOptimizer* graph_ptr)
{
    //using the first robot for the globaltime stamp of this message
    
    msg.header.stamp = curTime; //time of the self-robot must be in the full state
    estimatedRobPose.header.stamp = curTime;
    estimatedRobPose.header.frame_id = "/world"; 
    
    for(int i=0;i<MAX_ROBOTS;i++)
    {
      if((*ifRobotIsStarted)[i])
      {
      
	int latestOptimizedRobPoseVer = (*currentPoseVertexIDs)[i];
	bool poseExists = false;
	//cout<<"latestOptimizedRobPoseVer = "<<latestOptimizedRobPoseVer<<endl;
	//This loop is to check whether the latest pose exists or not
	for(int j=0; j<WINDOW_SIZE;j++)
	{
	  if((graph_ptr->vertices()[latestOptimizedRobPoseVer]))
	  {
	    j=WINDOW_SIZE+1; //break and come out of the loop
	    poseExists = true;
	  }
	  else
	    latestOptimizedRobPoseVer--; //go one ID step below
	}
      
	if(poseExists)
	{
	  VertexSE2* mostRecetPoseVertex = dynamic_cast<VertexSE2*>(graph_ptr->vertices()[latestOptimizedRobPoseVer]);
	  mostRecetPoseVertex->isOptimizedAtLeastOnce = true;            
          
          //fill in the generic message for estimated robot pose
          estimatedRobPose.pose.pose.position.x = mostRecetPoseVertex->estimate().translation().x();
	  estimatedRobPose.pose.pose.position.y = mostRecetPoseVertex->estimate().translation().y();
	  estimatedRobPose.pose.pose.position.z = 0.81; //fixed height aboveground
	
	  estimatedRobPose.pose.pose.orientation.x = 0;
	  estimatedRobPose.pose.pose.orientation.y = 0;
	  estimatedRobPose.pose.pose.orientation.z = sin(mostRecetPoseVertex->estimate().rotation().angle()/2);
	  estimatedRobPose.pose.pose.orientation.w = cos(mostRecetPoseVertex->estimate().rotation().angle()/2);          
	  
	}
      }
    }
    
    //cout<<"Robot orientation angle = "<<angle<<" cos (angle) = "<<cos(angle)<<" and sin(angle) = "<<sin(angle)<<endl;
    selfState_publisher_generic.publish(estimatedRobPose);
}



int main (int argc, char* argv[])
{
  ros::init(argc, argv, "single_robot_localization");
  ros::NodeHandle nh("~");
  
  if (argc != 3)
  {
    ROS_WARN("WARNING: you should specify the window size\n");
  }
  else
  {
    WINDOW_SIZE = atoi(argv[1]);
    DECAY_LAMBDA = atoi(argv[2]);
  }
  
  printf(" WINDOW_SIZE set to %d and DECAY_LAMBDA set to %d \n",WINDOW_SIZE,DECAY_LAMBDA);
    //registering all the types from the libraries
    //int argc = 0; char** argv = 0;
    g2o::DlWrapper dlTypesWrapper;
    g2o::loadStandardTypes(dlTypesWrapper, argc, argv);

    //register all the solvers
    g2o::DlWrapper dlSolverWrapper;
    g2o::loadStandardSolver(dlSolverWrapper, argc, argv);    
  
  g2o::SparseOptimizer graph;

  addFixedLandmarkNodes(&graph);
  GenerateGraph node(nh, &graph);//argument corresponds to the number of robots in the team
  

  spin();
  return 0;
}




























