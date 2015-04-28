# mh_solver_frontend

A Moving-horizon nonlinear least squares-based multirobot cooperative perception. Optimization (using the backend g2o) is performed over a dynamically moving window of fixed timesteps. This package is a frontend that creates moving windows of nodes and edges and calls g2o functions to solve them online. The package also has a full offline graph generator and solver also for the sake of completeness and benchmarking
