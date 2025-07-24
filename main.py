from src.core.SlidingICP import SlidingICP

from colorama import init, Fore, Style
import click


@click.command()
@click.option("--path_data", "-p", default="input/", type=str, help="Path to the data directory. Directory should contain ...")
@click.option("--path_out", "-p", default="output/", type=str, help="Path to the output data directory")
@click.option("--path_calibration", "-p", default="input/calibration/", type=str, help="Path to the static calibration of the laser scanners")
@click.option("--configfile", "-p", default="config/sICPconfig_withrejection.json", type=str, help="Config file of the sliding ICP")

def main(path_data,
         path_out,
         path_calibration,
         configfile):
    
    #####################################################################################
    # 1) SlidingICP
    #####################################################################################

    # Initialize 
    sICP = SlidingICP(path_data,
                      path_out,
                      path_calibration,
                      configfile)
    
    sICP.print_info()
    
    # Load sICP configfile
    sICP.loadconfig()

    # Load static calibration from path
    sICP.loadcalibration()

    # Load data from path
    sICP.loaddata()
    
    # Create initial point cloud and write to outpath
    pcl, pcr = sICP.create_pointcloud( calibration = "static" )
    pcl.write_to_file( path = sICP.path_out, filename = "pc_l_i", offset = sICP.config.txyz )
    pcr.write_to_file( path = sICP.path_out, filename = "pc_r_i", offset = sICP.config.txyz )
    
    # Run alignment
    sICP.run()
    
    #####################################################################################
    # 2) Compute time-dependent calibration from SlidingICP transformations
    #####################################################################################

    sICP.compute_kinematic_calibration_parameter()

    # Create point cloud with kinematic parameters and write to outpath
    pcl, pcr = sICP.create_pointcloud( calibration = "kinematic" )
    pcl.write_to_file( path = sICP.path_out, filename = "pc_l_sICP", offset = sICP.config.txyz )
    pcr.write_to_file( path = sICP.path_out, filename = "pc_r_sICP", offset = sICP.config.txyz )
    
    print("SlidingICP run successful")

if __name__ == "__main__":
    main()