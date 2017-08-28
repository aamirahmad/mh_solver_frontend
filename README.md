# mh_solver_frontend

A front-end for "moving-horizon nonlinear least squares-based multi-robot cooperative perception" applied to the soccer robot datasets (real or simulated). 

The Lucia branch is created specifically for the Lucia summer school 2017 lecture on cooperative perception. 

Pre-requisites for this package:

1. Install a copy of g2o from this location: https://github.molgen.mpg.de/aamir/g2o_lucia_tutorial.git
   Install location can be /usr/local (default). Otherwise, please set the location manually in the CMakeLists.txt of this (mh_solver_frontend) package.
2. Get the following package: read_omni_dataset
   1. Clone https://github.com/aamirahmad/read_omni_dataset.git
   2. Switch to branch infinite-robots.
3. Get the following package: randgen_omni_dataset
   1. Clone https://github.com/guilhermelawless/randgen_omni_dataset.git
   3. Read the instructions carefully and gnerate/play/observe data from multiple robots. This is going to be used by mh_solver_frontend.
4. There are 2 launch files in this (mh_solver_frontend) package. They correspond to problem 1 and 2 of the tutorial. Problem 3 requires the student to write his/her own launch file also.

5. Execute the launch files with the robotID and numRobots parameters of your choice after completing the missing methods in the g2o library (in the version obtained in step 1).