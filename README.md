# Nanoscope_AFM_ForceCurve_analysis


## _Introduction_ 
These scripts are made with the intent of processing force curves obtained using most of the commercially available Atomic Force Microscopes (AFMs) from Bruker (https://en.wikipedia.org/wiki/Atomic_force_microscopy). A force curve describes the process of indentation of a biological sample (chromosomes in this particular case) by a sharp tip (https://www.nature.com/articles/s42254-018-0001-7). The curves record the force related to the indentation of the tip and its retraction from the sample, from them, multiple biophysical parameters related to the mechanical and structural properties of the sample can be retrieved.

## _Contents_

Scripts for processing the force curves from Bruker AFMs. The folder "_Example_RawData_" contains examples of raw data used to test the scripts. The folder "_Example_figures_" contains the figures that are generated by the various scripts.

_**- Nanoscope_converter.py**:_ this script is a parser that reads the raw binary file containing the info on the force curve coming from the microscope and extract the information that allow plotting and processing the force curve (a representative figure of its output is present in the folder).

_**- nanoscope_CurvesContactPoint_determination.py**:_ this script contains two different functions that can be used to identify the contact point along the indenting force curve.

_The above-mentioned scripts are used within the other ones to process multiple curves and perform various analyses._

_**- nanoscope_CurveFit.py**:_ this script contains 2 different functions that allow processing and analysing the force curves. The first one is used to fit the indenting force curve with appropriate contact mechanics models; the second functions calculates the area under the indenting and retracting curve in order to extract information on the viscoelasticity of the sample (representative figures of its output is present in the folder).

## _Workflow_

_Nanoscope_converter.py_ contains a function to parse the raw AFM force curve file and export the necessary data. _nanoscope_CurvesContactPoint_determination.py_ contains two different functions to estimate the contact point along the indenting force curve. These two scripts are used by _nanoscope_CurveFit.py_, which provides two different analyses of the force curves, it can fit the indenting curve with different contact mechanics models or it can calculate the area under the indenting and retracting curves to estimate the energy dissipation. More details can be found below and also within each script.
The idea is to use the functions within these scripts to perform automated analyses of multiple force curves. The functions can be inserted into a _for loop_ in order to parse and extract the data of several curves or to automatically fit them.
