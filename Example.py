import pandas as pd
from Nanoscope_converter import nanoscope_converter
from nanoscope_CurveFit import indentation_fit, area_viscoelasticity
import os
import matplotlib.pyplot as plt


"""This script shows an example of how it is possible to automatically read and process all the force curves 
contained within a specific folder"""

# finding the path of all the force curves stored within Example_RawData folder
local_dir = os.path.dirname(os.path.abspath(__file__))
curves = [local_dir + "/Example_RawData/" + curve for curve in os.listdir("Example_RawData")]

# Plotting and saving individual curves
print(curves)
for curve in curves:
    curve_data = nanoscope_converter(curve)
    # Indenting curve data
    ind_sep, ind_force = curve_data[0], curve_data[1]
    # Retracting curve data
    rt_sep, rt_force = curve_data[2], curve_data[3]
    curve_df = pd.DataFrame({"Separation_(nm)": ind_sep,
                             "Force_(pN)": ind_force,
                             "Separation_(nm)_rt": rt_sep,
                             "Force_(pN)_rt": rt_force,
                             })
#   To save the data of the current curve in a csv file
#     curve_df.to_csv(curve.split("/")[-1] + ".csv")

#   Plotting
    fig, ax = plt.subplots()
    ax.plot(ind_sep, ind_force, c="b", zorder=2)
    ax.plot(rt_sep, rt_force, c="r", zorder=1)
    ax.set_ylabel("Force (pN)")
    ax.set_xlabel("Separation (nm)")
    plt.title(curve.split("/")[-1])
    plt.show()


# Fitting and plotting individual curves
for curve in curves:
    # The plotting can be activated/deactivated within the indentation_fit function
    young_modulus = indentation_fit(curve)[2]
    print("Young Modulus (kPa): ", young_modulus)

