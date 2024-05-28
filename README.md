# NanoscopeAFM_ForceCurve_analysis


## _Introduction_ 
These scripts are made with the intent of processing force curves obtained using most of the commercially available Atomic Force Microscopes (AFMs) from Bruker (https://en.wikipedia.org/wiki/Atomic_force_microscopy). A force curve describes the process of indentation of a biological sample (chromosomes in this particular case) by a sharp tip (https://www.nature.com/articles/s42254-018-0001-7). The curves record the force related to the indentation of the tip and its retraction from the sample, from them, multiple biophysical parameters related to the mechanical and structural properties of the sample can be retrieved.

After building the pipeline for importing, parsing and processing the curves, a clustering analysis is performed on a batch of 100+ curves using _Dynamic Time Warping_ (DTW), which compares traces based solely on their shape; more information on DTW can be found here: https://paul-mora.com/classification/time-series/clustering/python/Dynamic-Time-Warping-Explanation-and-Testing-on-Audio-and-Tabular-Data/. 

## _Workflow_

Both _Nanoscope_converter.py_ , _nanoscope_CurvesContactPoint_determination.py_ and _nanoscope_CurveFit.py_ are used within the different clustering analyses (within _PCA_KMeans_Clustering_ and _LongitudinalClustering_DTW_ directories). For both clustering analyses, a set of 1000+ curves (stored as raw binary files coming from the instrument) is parsed, converted into arrays which are then processed in order to be used by clustering algorithms. 
For both clustering analyses, curves are corrected for background noise, aligned in order to be better compared, fitted using splines etc...
- **_PCA_KMeans_DataPrep.py_** processes and extracts different features describing the curves; these features correspond to biophysical parameters of interest, like the "contact point" (i.e., where the first value of force > 0 is recorded), the max pushing force, derivatives of force, energy of indentation etc...All the parameters are then stored in a dataframe that is used for further clustering analyses (_**PCA_analysis.py**_ and **_KMeans.py_**, within the _PCA_KMeans_Clustering_ directory).
- **_LongitudinalClustering_DataPrep.py_** processes the curves and stores their y- and x- values into two separate dataframes that are then used to perform Dynamic Time Warping analysis. The DTW analysis is performed within the _LongitudinalClustering_DTW_ directory; since DTW is quite demanding in terms of computing power, the DTW analysis has been also parallelized in _**DTW_parallelization.py**_. 


## _Contents_

Scripts for processing the force curves from Bruker AFMs. The folder "_Example_RawData_" contains examples of raw data used to test the scripts. The folder "_Example_figures_" contains the figures that are generated by the various scripts.

_**- Nanoscope_converter.py**:_ this script is a parser that reads the raw binary file containing the info on the force curve coming from the microscope and extract the information that allow plotting and processing the force curve (a representative figure of its output is present in the folder).

_**- nanoscope_CurvesContactPoint_determination.py**:_ this script contains two different functions that can be used to identify the contact point along the indenting force curve.

_The above-mentioned scripts are used within the other ones to process multiple curves and perform various analyses._

_**- nanoscope_CurveFit.py**:_ this script contains 2 different functions that allow processing and analysing the force curves. The first one is used to fit the indenting force curve with appropriate contact mechanics models; the second functions calculates the area under the indenting and retracting curve in order to extract information on the viscoelasticity of the sample (representative figures of its output is present in the folder).

_**- PCA_KMeans_DataPrep.py**:_ this script is used to analyse the force curves within the Example_RawData/Chromosomes folder, which refers to indentation experiments on multiple chromosomes. The script processes each curve (employing the previously mentioned scripts/functions) and extract useful parameters, like max force, derivatives, area under curve etc..., saving them into a .csv file for further clustering analyses.

_**- PCA_analysis.py**:_ reads the .csv file generated by Chrm_IndentationAnalysis.py and performs a PCA analysis in order to cluster curves having different shape and hence properties into distinct clusters. (a representative figure of its output is present in the folder)

_**- KMeans.py**:_ similar to PCA_analysis.py but performs KMean clustering on the data contained in the .csv file generated by Chrm_IndentationAnalysis.py (a representative figure of its output is present in the folder).

_**- KMeans_after_PCA.py**:_ in this script, the dataset containing the curve features is scaled, and PCA is performed to understand which features can describe most of the variance. Based on that, KMeans is performed for different number of clusters and the better one is selected using the elbow method. Clusters and respective curves are finally plotted together.

_**- LongitudinalClustering_DataPrep.py**:_ this script is used to analyse the force curves within the Example_RawData/Chromosomes folder, which refers to indentation experiments on multiple chromosomes. The script processes all the traces and store their x and y values into separate .csv files for the longitudinal clustering analysis.

_**- DTW_analysis.py**:_ this script performs longitudinal clustering based on DTW. The results of these analysis is the so-called distance matrix (_nxn_), whose elements contains the computed distance between all the *n* curves, the matrix is stored in a .csv file.

_**- DTW_parallelization.py**:_ DTW is computationally expensive; in this script the DTW analysis is parallelized in order to be faster and more efficient (thanks to parallelization the DTW analysis is more than 4x faster); the result (the distance matrix) is always stored in .csv file.

_**- Clustering_ResultsPlotting.py**:_ this script takes the distance matrices as input and produce a dendrogram that cluster the curves (based on the distances computed using DTW and stored in the distance matrix) into multiple hierarchical clusters. Choosing the number of clusters we want to produce, it is then possible to see the different curves classified in separate clusters only based on their shapes (refer to appropriate figures in the _Example_figures_ folder).