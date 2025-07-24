import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class kinematiccalibration:

    x: np.ndarray

    def __init__(self, x = np.ndarray, xint = np.ndarray):
        
        # kinematic calibration parameter 
        self.x = x # [t, rx, ry, rz, x, y, z]
        self.xint = xint # [t, rx, ry, rz, x, y, z]

    def fill_borders(self, timestamps):

        # Rotation
        rx_m, ry_m, rz_m = np.median(self.x[:,1]), np.median(self.x[:,2]), np.median(self.x[:,3])

        # Translation
        tx_m, ty_m, tz_m = np.median(self.x[:,4]), np.median(self.x[:,5]), np.median(self.x[:,6])
        
        self.x = np.insert(self.x, 0, np.c_[timestamps[0], rx_m, ry_m, rz_m, tx_m, ty_m, tz_m], axis=0)
        self.x = np.insert(self.x, self.x.shape[0], np.c_[timestamps[-1], rx_m, ry_m, rz_m, tx_m, ty_m, tz_m], axis=0)

    def write_to_file(self, path_out, fname):

        np.savetxt( fname=path_out+"parameter/x_"    +fname+".txt", X=self.x )
        np.savetxt( fname=path_out+"parameter/xint_" +fname+".txt", X=self.xint )

    def interpolate(self, timestamps):

        # Rotation
        f_rx = interp1d(self.x[:,0], self.x[:,1], kind="cubic", fill_value="extrapolate") 
        f_ry = interp1d(self.x[:,0], self.x[:,2], kind="cubic", fill_value="extrapolate") 
        f_rz = interp1d(self.x[:,0], self.x[:,3], kind="cubic", fill_value="extrapolate") 
        
        rx_int = f_rx( timestamps )
        ry_int = f_ry( timestamps )
        rz_int = f_rz( timestamps )

        # Translation
        f_tx = interp1d(self.x[:,0], self.x[:,4], kind="cubic", fill_value="extrapolate") 
        f_ty = interp1d(self.x[:,0], self.x[:,5], kind="cubic", fill_value="extrapolate") 
        f_tz = interp1d(self.x[:,0], self.x[:,6], kind="cubic", fill_value="extrapolate") 

        tx_int = f_tx( timestamps )
        ty_int = f_ty( timestamps )
        tz_int = f_tz( timestamps )

        self.xint = np.c_[timestamps, rx_int, ry_int, rz_int, tx_int, ty_int, tz_int ]

    def plot(self, scal):

        """
        scal: static calibration
        """
        
        # Change in the calibration with respect to static calibration
        drpy = (self.x[:, 1:4]* 180 / np.pi - np.array([scal.rx, scal.ry, scal.rz]))
        dxyz = (self.x[:, 4:7] - np.array([scal.tx, scal.ty, scal.tz])) * 1000

        drpy_int = (self.xint[:, 1:4] * 180 / np.pi - np.array([scal.rx, scal.ry, scal.rz]))
        dxyz_int = (self.xint[:, 4:7] - np.array([scal.tx, scal.ty, scal.tz])) * 1000

        t0 = self.x[0,0]
        t1 = self.x[-1,0] - t0
        
        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot positions (xyz)
        axs[0].plot(self.x[:,0]-t0, dxyz[:, 0], "xr", label='$\Delta$x')   
        axs[0].plot(self.xint[:,0]-t0, dxyz_int[:, 0], color='r', linestyle='-')

        axs[0].plot(self.x[:,0]-t0, dxyz[:, 1], "xg", label='$\Delta$y')
        axs[0].plot(self.xint[:,0]-t0, dxyz_int[:, 1], color='g', linestyle='-')

        axs[0].plot(self.x[:,0]-t0, dxyz[:, 2], "xb", label='$\Delta$z')
        axs[0].plot(self.xint[:,0]-t0, dxyz_int[:, 2], color='b', linestyle='-')

        axs[0].set_title(f'Translation', fontsize=18)
        axs[0].set_ylabel('Translation (mm)', fontsize=18)
        axs[0].legend(loc='upper left', fontsize=13)
        axs[0].tick_params(axis='both', which='major', labelsize=14)
        axs[0].set_xlim((0,t1))

        self.get_plt_style(axs[0])
            
        # Plot orientations (rpy)
        axs[1].plot(self.x[:,0]-t0, drpy[:, 0], "xr", label='$\Delta$rx')
        axs[1].plot(self.xint[:,0]-t0, drpy_int[:, 0], color='r', linestyle='-')

        axs[1].plot(self.x[:,0]-t0, drpy[:, 1], "xg", label='$\Delta$ry')
        axs[1].plot(self.xint[:,0]-t0, drpy_int[:, 1], color='g', linestyle='-')

        axs[1].plot(self.x[:,0]-t0, drpy[:, 2], "xb", label='$\Delta$rz')
        axs[1].plot(self.xint[:,0]-t0, drpy_int[:, 2], color='b', linestyle='-')

        axs[1].set_title(f'Rotation', fontsize=18)
        axs[1].set_xlabel('Time (s)', fontsize=18)
        axs[1].set_ylabel('Rotation (°)', fontsize=18)
        axs[1].legend(loc='upper left', fontsize=13)
        axs[1].tick_params(axis='both', which='major', labelsize=14)
        axs[1].set_xlim((0,t1))

        self.get_plt_style(axs[1])
        
        plt.subplots_adjust(hspace=0.5)  # Increase space between plots
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def get_plt_style(self, ax):
        ax.grid(which='both', axis='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.5)
        font = {'size': 12}
        plt.rc('font', **font)




    def read_calibration_from_file( self, pathx, path_intx ):
        
        print("--------------------------------------------------------------------------------")
        print("Reading system calibration parameters ")


        self.x = np.loadtxt( fname = pathx, delimiter = " ")
        self.xint = np.loadtxt( fname = path_intx, delimiter = " ")

        print("... done ")
        print("--------------------------------------------------------------------------------")



    