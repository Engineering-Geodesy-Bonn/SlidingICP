![](./images/teaser.png)

This repository contains the code of a sliding ICP approach for kinematic dual laser scanning systems whose mounting calibration is not rigid over time. The method computes sequential updates on the calibration parameters for both scanners by first creating initial point clouds with the rigid mounting calibration and then cutting these into overlapping point cloud windows along the trajectory. These point clouds are then ICP-registered using a symmetric plane-to-plane objective function and robust matching procedure. To obtain the mounting calibration updates for both scanners, the estimated transformations of the windows are then interpolated to the timestamps of the laser data. The final, aligned point clouds are then created by integrating the calibration updates into the direct georeferencing equation for both scanners.

## Installation

## Documentation

## Key Features

The Sliding ICP method offers a range of features, including:

- __Robust plant-to-plane point matching__:

