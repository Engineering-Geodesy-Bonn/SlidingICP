# Imports
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from colorama import init, Fore, Style

# Import from src folder
from src.calibration.calibration import calibration
from src.calibration.kinematiccalibration import kinematiccalibration
from src.dataclasses.trajectory import Trajectory
from src.dataclasses.LMIdata import LaserdataLMI
from src.directgeoreferencing.directgeoreferencing import directgeoreferencing
from src.pointcloud.pointcloud import pointcloud
from src.config.sICPconfig import sICPconfig
from src.icp.SymPlane2PlaneICP import SymPlane2PlaneICP

# Import base functions
from src.base.base import RotmatX, RotmatY, RotmatZ, Rotmat2Euler, Euler2RotMat

class SlidingICP:

    # System calibration of the left and right scanner
    calL: calibration
    calR: calibration

    # Kinematic system calibration
    kcalL: kinematiccalibration
    kcalR: kinematiccalibration

    # Trajectory of the left and right scanner
    TL: Trajectory
    TR: Trajectory

    # LMI laser profiles of the left and right scanner
    lmidataL: LaserdataLMI
    lmidataR: LaserdataLMI

    # Config file of the sequential strip alignment
    config: sICPconfig

    xlist: list # List to store estimated parametes
    idxL: list
    idxR: list

    # Input & Output pathes
    path_data: str
    path_out: str
    path_calibration: str
    configfile: str

    def __init__(self, path_data, path_out, path_calibration, configfile):
        """
        """

        self.calL = calibration()
        self.calR = calibration()
        self.kcalL = kinematiccalibration()
        self.kcalR = kinematiccalibration()
        self.TL = Trajectory()
        self.TR = Trajectory()
        self.lmidataL = LaserdataLMI()
        self.lmidataR = LaserdataLMI()
        self.config = sICPconfig()
        self.xlist = []
        self.idxL = []
        self.idxR = []
        self.path_data = path_data
        self.path_out = path_out
        self.path_calibration = path_calibration
        self.configfile = configfile

    def print_info(self):
        """
        """

        print(( " ____________________________________________________________________________\n"
                "| \n"
               f"| {Style.BRIGHT}{Fore.MAGENTA}{'Kinematic ICP strip alignment'}{Style.RESET_ALL}\n"
               f"| ___________________________________________________________________________\n"
               f"|"  ))
            
        t = time.strftime("%H:%M:%S")
        print(( f"| {Style.BRIGHT}{Fore.GREEN}{t + ' Dataset info '}{Style.RESET_ALL}" ))
        print("| - path to data:   ", self.path_data)
        print("| - path to output: ", self.path_out)
        print("| - path to calibration file: ", self.path_calibration)
        print("| - path to config file: ", self.configfile)
        print("| ")

    def loaddata(self):
        """
        """   

        # Get filenames of the trajectories
        trj_l_str = [f for f in os.listdir(self.path_data) if f.endswith('l.trj')][0]
        trj_r_str = [f for f in os.listdir(self.path_data) if f.endswith('r.trj')][0]
        
        # Read trajectories
        self.TL.read_from_file( path_to_file = self.path_data + trj_l_str, offset_xyz = self.config.txyz )
        self.TR.read_from_file( path_to_file = self.path_data + trj_r_str, offset_xyz = self.config.txyz )
        
        # Get filenames of the laserprofiles
        lmi_l_str = [f for f in os.listdir(self.path_data) if f.endswith('l.bin')][0]
        lmi_r_str = [f for f in os.listdir(self.path_data) if f.endswith('r.bin')][0]

        # Read laser data
        self.lmidataL.readbin( self.path_data+lmi_l_str )
        self.lmidataR.readbin( self.path_data+lmi_r_str )

        # __________________________________________________________
        # Intersect and interpolate data
        #
        
        self.lmidataL.intersecting( self.TL.time )
        self.lmidataR.intersecting( self.TR.time )

        self.TL = self.TL.interpolate( self.lmidataL.timestamps, kind = "cubic")
        self.TR = self.TR.interpolate( self.lmidataR.timestamps, kind = "cubic")

    def loadconfig(self):
        """
        """

        self.config.readfromjson( self.configfile )

    def writeconfig(self):
        """
        """

        self.config.writeToJson( self.path_out+"config/sICPconfig.json" )

    def loadcalibration(self):
        """
        """

        self.calL.read_calibration_from_xml( self.path_calibration + "system_config_lmi_l.xml" )
        self.calR.read_calibration_from_xml( self.path_calibration + "system_config_lmi_r.xml" )

    def create_pointcloud(self, calibration = "static"):
        """
        """

        if calibration == "static": 
            call = self.calL
            calr = self.calR
        elif calibration == "kinematic":
            call = self.kcalL
            calr = self.kcalR 

        georefL = directgeoreferencing( trajectory = self.TL,
                                        laserlines = self.lmidataL,
                                        systemcalibration = call )
        
        georefR = directgeoreferencing( trajectory = self.TR,
                                        laserlines = self.lmidataR,
                                        systemcalibration = calr )    

        pcl = georefL.run( calibration=calibration )
        pcr = georefR.run( calibration=calibration )

        return pcl, pcr

    def run(self):
        """
        """

        # Get intervals
        idxL, idxR = self.get_alignment_intervals()

        # Initial guess parameter
        x0 = np.zeros(6)

        # Loop over intervals and run ICP alignment
        for i in range(len(idxL)):

            # ___________________________________________________________________________________
            # A) Create point clouds

            # Get trajectory and laserdata of the current interval
            idxleft = np.arange(idxL[i][0],idxL[i][1])
            idxright = np.arange(idxR[i][0],idxR[i][1])

            TLi = self.TL.crop_by_index( idxleft )
            LMIl_i = self.lmidataL.crop_by_index( idxleft )

            TRi = self.TR.crop_by_index( idxright )
            LMIr_i = self.lmidataR.crop_by_index( idxright )

            # Run point cloud creation
            georefL = directgeoreferencing( TLi, LMIl_i, self.calL )
            pcl_i = georefL.run( calibration="static" )

            georefR = directgeoreferencing( TRi, LMIr_i, self.calR )
            pcr_i = georefR.run( calibration="static" )

            # ___________________________________________________________________________________
            # B) Transform point clouds into mean body frame of the current interval

            # Mean trajectory state of the interval
            idxmL = round((idxL[i][0] + idxL[i][1]) / 2)
            Tmil =  self.TL.statesall[idxmL, :]
            timei = self.TL.time[idxmL]

            # Transform point clouds
            pc_l = pcl_i.xyz 
            pc_r = pcr_i.xyz

            # Translation
            XG_l = (pc_l[:,0] - Tmil[1])
            YG_l = (pc_l[:,1] - Tmil[2]) 
            ZG_l = (pc_l[:,2] - Tmil[3]) 
                
            XG_r = (pc_r[:,0] - Tmil[1])
            YG_r = (pc_r[:,1] - Tmil[2]) 
            ZG_r = (pc_r[:,2] - Tmil[3])

            xyz_e_left = np.column_stack((XG_l, YG_l, ZG_l))
            xyz_e_right = np.column_stack((XG_r, YG_r, ZG_r))

            # Rotation
            R_B_NED_left = np.dot(np.dot(RotmatZ(Tmil[9]), RotmatY(Tmil[8])), RotmatX(Tmil[7]))
                
            # Transformation
            pc_l = (R_B_NED_left.T @ xyz_e_left.T).T
            pc_r = (R_B_NED_left.T @ xyz_e_right.T).T

            # ___________________________________________________________        
            # C) Run ICP on the point clouds

            print(( f"|______________________________________________________________________________________________________\n"
                    f"| {Style.BRIGHT}{Fore.MAGENTA}{'Running ICP alignment ('+str(i+1)+' /'+str(len(idxL))+')'}{Style.RESET_ALL} \n"
                    f"|  \n"
                    f"| - x0:  {Style.BRIGHT}{Fore.WHITE}{x0}{Style.RESET_ALL}\n"
                    f"| - number of point left:   {Style.BRIGHT}{Fore.WHITE}{len(pcl_i.xyz)}{Style.RESET_ALL}\n"
                    f"| - number of point right:  {Style.BRIGHT}{Fore.WHITE}{len(pcr_i.xyz)}{Style.RESET_ALL}"))
            
            # Set up ICP
            icp_instance = SymPlane2PlaneICP(pc_l, pc_r, x=x0)
            xi, suc = icp_instance.runICP( self.config )
            
            # Update initial guess for the next interval
            if suc == True:
                x0 = xi.copy()
            else:
                x0 = np.zeros(6)

            print(( f"| - Final transformation:  {Style.BRIGHT}{Fore.WHITE}{xi}{Style.RESET_ALL} "))
            print(( f"| - Number of point matches:  {Style.BRIGHT}{Fore.WHITE}{len(icp_instance.xyzm1)}{Style.RESET_ALL}"))
            
            # Store transformation parameters
            self.xlist.append({"left_indices": [idxL[i][0], idxL[i][1]],
                               "right_indices": [idxR[i][0], idxR[i][1]],
                               "transformation": xi.tolist(),
                               "timestamp": timei })
        
        # Write parameter correlation matrix to file
        with open(self.path_out+"parameter/Px.txt", "w") as f:   
            for transformation in self.xlist:
                left_indices = ",".join(map(str, transformation['left_indices']))
                right_indices = ",".join(map(str, transformation['right_indices']))
                transformation_values = ",".join(map(str, transformation['transformation']))
                timestamp = transformation['timestamp']
                f.write(f"{left_indices},{right_indices},{transformation_values},{timestamp}\n")

    def compute_kinematic_calibration_parameter(self):
        """
        """
        # Load icp parameter from file
        icp_param = np.loadtxt( fname=self.path_out+"parameter/Px.txt", delimiter="," )

        icp_param = icp_param[~np.isnan(icp_param).any(axis=1)]

        # Initialize kinematic calibration parameter
        self.kcalL.x = np.zeros((icp_param.shape[0], 7))
        self.kcalR.x = np.zeros((icp_param.shape[0], 7))

        # _____________________________________________________________________________
        # Get homogeneous transformation matrices for static calibration

        R_BS_L = RotmatZ( np.deg2rad(self.calL.rz) ) @ RotmatY( np.deg2rad(self.calL.ry) ) @ RotmatX( np.deg2rad(self.calL.rx) )
        H_sbl = self.create_homogeneous_matrix(R_BS_L.T, np.array((self.calL.tx, self.calL.ty, self.calL.tz))) 

        R_BS_R = RotmatZ( np.deg2rad(self.calR.rz) ) @ RotmatY( np.deg2rad(self.calR.ry) ) @ RotmatX( np.deg2rad(self.calR.rx) )
        H_sbr = self.create_homogeneous_matrix(R_BS_R.T, np.array((self.calR.tx, self.calR.ty, self.calR.tz)))

        # _____________________________________________________________________________
        # Get homogeneous transformation matrices for kinematic calibration

        for i in range(icp_param.shape[0]):

            # Get time stamp
            timei = icp_param[i,-1]

            # Translation and rotation estimated by ICP
            rx, ry, rz = icp_param[i, 4], icp_param[i, 5], icp_param[i, 6]
            R = Euler2RotMat( rx, ry, rz )
            t = np.array([icp_param[i,7], icp_param[i,8], icp_param[i,9]])

            # Transformation matrix
            H_bbl = self.create_homogeneous_matrix(R.T, -t/2)
            H_bbr = self.create_homogeneous_matrix(R,    t/2)

            # __________________________________________________________
            # Kinematic calibration scanner left
            #

            # Update transformation
            H_sb_newl = H_bbl @ H_sbl

            # Compute euler angles
            rXrYrZ_l = Rotmat2Euler(H_sb_newl[:3, :3].T)

            # store kinematic calibration
            self.kcalL.x[i,0] = timei
            self.kcalL.x[i,1:4] = rXrYrZ_l
            self.kcalL.x[i,4:7] = H_sb_newl[:3, 3]

            # __________________________________________________________
            # Kinematic calibration scanner right
            #

            # Update transformation
            H_sb_newr = H_bbr @ H_sbr

            # Compute euler angles
            rXrYrZ_r = Rotmat2Euler(H_sb_newr[:3, :3].T)

            # store kinematic calibration
            self.kcalR.x[i,0] = timei
            self.kcalR.x[i,1:4] = rXrYrZ_r
            self.kcalR.x[i,4:7] = H_sb_newr[:3, 3]

        # _____________________________________________________________________________
        # Interpolate kinematic calibration parameters
        #

        # Fill borders and interpolate using trajectory timestamps
        self.kcalL.fill_borders( self.TL.time )
        self.kcalL.interpolate( self.TL.time )

        self.kcalR.fill_borders( self.TR.time )
        self.kcalR.interpolate( self.TR.time )

        # Visualize kinematic calibration parameter
        #self.kcalL.plot( self.calL )
        #self.kcalR.plot( self.calR )

        self.kcalL.write_to_file(path_out=self.path_out, fname="l")
        self.kcalR.write_to_file(path_out=self.path_out, fname="r")

        # __________________________________________________________
        # Create point cloud with kinematic parameters

        georefL = directgeoreferencing( trajectory = self.TL,
                                        laserlines = self.lmidataL,
                                        systemcalibration = self.kcalL )
    
        georefR = directgeoreferencing( trajectory = self.TR,
                                        laserlines = self.lmidataR,
                                        systemcalibration = self.kcalR )    
 
        pcl_ = georefL.run(calibration="kinematic")
        pcr_ = georefR.run(calibration="kinematic")

        pcl_.write_to_file( path = self.path_out, filename = "PCL", offset = self.config.txyz )
        pcr_.write_to_file( path = self.path_out, filename = "PCR", offset = self.config.txyz )



    def load_transformation_parameters(self, filename, left):

        df = pd.read_csv(filename, header=None)
        df.columns = ['left_start', 'left_end', 'right_start', 'right_end', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'timestamp']
        df = df[['left_start', 'left_end', 'right_start','right_end','param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'timestamp']].to_numpy()

        """
        if left_right == "left":
            df_transformation = df_transformation[['left_start', 'left_end', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6']]
            df_transformation = df_transformation.to_numpy()
        else:
            df_transformation = df_transformation[['right_start', 'right_end', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6']].to_numpy()
        """

        # Initialize kinematic calibration
        self.kcalL.x = np.zeros((len(df), 7))
        self.kcalR.x = np.zeros((len(df), 7))

        print(df)

        return df
    
    def create_homogeneous_matrix(self, R, t):
        """
        Create a 4x4 homogeneous transformation matrix from a 3x3 rotation matrix and a 3x1 translation vector.
        
        Parameters:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
        Returns:
        H: 4x4 homogeneous transformation matrix
        """
        # Check if R is a 3x3 matrix
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix R must be 3x3.")
        
        # Check if t is a 3x1 vector
        if t.shape != (3,) and t.shape != (3, 1):
            raise ValueError("Translation vector t must be a 3x1 vector.")
        
        # Ensure t is a column vector
        t = t.reshape(3, 1)
        
        # Create the homogeneous transformation matrix
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t.flatten()
        
        return H

    def get_alignment_intervals(self):

        distance = 0
        index_end = None

        # Index offset where to start in the beginning
        idx0 = 0
        
        trajectory_length = self.TL.statesall.shape[0]
        all_indices = []
        index_start = 0

        while index_start < trajectory_length - 1:
            distance = 0
            for i in range(index_start, trajectory_length - 1):
                dist = math.sqrt((self.TL.statesall[i, 1] - self.TL.statesall[i+1, 1])**2 + (self.TL.statesall[i, 2] - self.TL.statesall[i+1, 2])**2)
                distance += dist

                if distance >= self.config.window_size:
                    index_end = i + 1
                    all_indices.append((index_start+idx0, index_end+idx0))
                    break

            # Find new index_start based on the step size
            distance = 0
            for j in range(index_start, trajectory_length - 1):
                dist = math.sqrt((self.TL.statesall[j, 1] - self.TL.statesall[j+1, 1])**2 + (self.TL.statesall[j, 2] - self.TL.statesall[j+1, 2])**2)
                distance += dist
                if distance >= self.config.step_size:
                    index_start = j + 1
                    break

            # If the step size loop completes without breaking, end the main loop
            else:
                break  

        # _____________________________________________________________
        # Find corresponding interval indices of the right trajectory
        #

        all_indices2 = []

        for i in range(len(all_indices)):
            idxA_t1, idxB_t1 = np.array(all_indices[i])

            timet1_A, timet1_B = self.TL.statesall[idxA_t1,0], self.TL.statesall[idxB_t1,0]

            idxA_t2 = np.argmin( np.abs(self.TR.statesall[:,0] - timet1_A ) )
            idxB_t2 = np.argmin( np.abs(self.TR.statesall[:,0] - timet1_B ) )

            all_indices2.append((idxA_t2, idxB_t2))

        return all_indices, all_indices2       