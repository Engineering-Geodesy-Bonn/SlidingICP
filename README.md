<figure>
  <img src="./images/teaser.png" alt="Teaser Image">
  <figcaption>Consistency of the point clouds created with the two laser scanners of our field robot for a) Initial rigid mounting calibration, b) our ICP method (rigid transformation), and c) our sliding ICP method.</figcaption>
</figure>


### Overview

This repository contains a sliding ICP approach implemented for the kinematic dual laser scanning system of our ground Unmanned Ground Vehicle (UGV), as described in our previous work ([IEEE Xplore Paper](https://ieeexplore.ieee.org/abstract/document/10302421), [Arxiv Version](https://arxiv.org/pdf/2310.11516 ). The robot is designed to generate high-resolution point clouds of various crops, including beans, wheat, soybeans, sugar beets, corn, and potatoes, in agricultural fields.

## Key Features

    The U-shape design of the 2×2×2m robot causes the scanner calibration relative to the GNSS/IMU trajectory to change over time when driving in uneven field environments.
    To address this, the method performs sequential calibration updates by:
        Creating initial point clouds using rigid mounting calibration of both scanners.
        Cuts these point clouds into overlapping windows along the trajectory.
        Aligning these window point clouds by using an ICP algorithms which uses a symmetric plane-to-plane objective function and robust point matching matching.
    The calibration updates are obtained by interpolating them to laser data timestamps.
    Final, the aligned point cloud is creating by integrating the calibration updates into the direct georeferencing equations for both scanners.

## Installation (Docker)

The repository also contains Dockerfile. Please build and run the docker using the following commands:
  ```bash
  docker build -t sliding_icp_docker .
  docker run -it --name your_test_run sliding_icp_doc:latest 
  ```
## Data

The wheat, corn and sugar beet dataset is provided via sciebo share: https://uni-bonn.sciebo.de/s/qgLQ8wfS7oMWase. Download the laser profile data (.bin) for both scanners and the trajectory files (.trj) and copy them into /input folder of the repository. Note that only one dataset can be placed in the input folder. 


