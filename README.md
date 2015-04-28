# mh_solver_frontend

A frontend for "moving-horizon nonlinear least squares-based multirobot cooperative perception" applied to the soccer robot dataset. The dataset can be found here: http://datasets.isr.ist.utl.pt/lrmdataset/4_Robots_DataSet/. Please choose the rosbags folder for this frontend. This front end is based on the basic template available at https://github.com/aamirahmad/read_omni_dataset to systematically read the soccer dataset. 

Important: Please note that this package requires g2o libraries. A fork https://github.com/aamirahmad/g2o of the original g2o https://github.com/RainerKuemmerle/g2o package is mandatory because the fork contains necessary additional process and measurement models for randomly moving targets.

Optimization (using the backend g2o) is performed over a dynamically moving window of fixed timesteps. This package is a frontend that creates moving windows of nodes and edges and calls g2o functions to solve them online. The package also has a full offline graph generator and solver also for the sake of completeness and benchmarking

