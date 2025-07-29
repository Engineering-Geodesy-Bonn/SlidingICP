<figure>
  <img src="./images/teaser.png" alt="Teaser Image">
  <figcaption>Consistency of the point clouds created with the two laser scanners of our field robot for a) Initial rigid mounting calibration, b) our ICP method (rigid transformation), and c) our sliding ICP method.</figcaption>
</figure>

## Description

This repository contains the code of a sliding ICP approach implemented for kinematic dual laser scanning system of a ground Unmanned Ground Vehicle (UGV) designed to create high-resolution crop models in aggricultural fields. Due to the U-shape design of the 2x2x2m large vehicle the mounting calibration of the scanners w.r.t. the GNSS/IMU trajectory changes over time while driving in uneven field terrains. To compensate for these changing calibration the method computes sequential updates on the calibration parameters for both scanners by first creating initial point clouds with the rigid mounting calibration and then cutting these into overlapping point cloud windows along the trajectory. These point clouds are then ICP-registered using a symmetric plane-to-plane objective function and robust matching procedure. To obtain the mounting calibration updates for both scanners, the estimated transformations of the windows are then interpolated to the timestamps of the laser data. The final, aligned point clouds are then created by integrating the calibration updates into the direct georeferencing equation for both scanners.

## Installation

## Documentation

## Key Features

The Sliding ICP method offers a range of features, including:

- __Robust plant-to-plane point matching__:

