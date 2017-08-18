# mh_solver_frontend

A front-end for "moving-horizon nonlinear least squares-based multi-robot cooperative perception" applied to the soccer robot datasets (real or simulated). 

The Lucia branch is created specifically for the Lucia summer school 2017 lecture on cooperative perception. 

Pre-requisites for this package:

1. Install a copy of g2o from this location: <To be inserted just before the start of the school>
   Install location can be /usr/local (default). Otherwise, please set the location manually in the CMakeLists.txt of this (mh_solver_frontend) package.
2. Compile package read_omni_dataset
   2. Clone https://github.com/aamirahmad/read_omni_dataset.git
   3. Switch to branch infinite-robots
3. 
  


The dataset can be found here: http://datasets.isr.ist.utl.pt/lrmdataset/4_Robots_DataSet/. Please choose the rosbags folder for this frontend. This front end is based on the basic template available at https://github.com/aamirahmad/read_omni_dataset to systematically read the soccer dataset. 

Important: Please note that this package requires g2o libraries. A fork https://github.com/aamirahmad/g2o of the original g2o https://github.com/RainerKuemmerle/g2o package is mandatory because the fork contains necessary additional process and measurement models for randomly moving targets. Therefore, please download the g2o sources from the fork https://github.com/aamirahmad/g2o, read through the compilation instructions, compile it and install it somewhere (can be done as a normal user without being root.) Once installed, take note of the installation prefixes, meaning the path to the folders where the binaries and libraries are placed after installation. Copy these paths and open the package.xml of this package (https://github.com/aamirahmad/mh_solver_frontend/blob/master/package.xml) and replace the cflags appropriately. Only then this package would compile!

Optimization (using the backend g2o) is performed over a dynamically moving window of fixed timesteps. This package is a frontend that creates moving windows of nodes and edges and calls g2o functions to solve them online. The package also has a full offline graph generator and solver also for the sake of completeness and benchmarking

