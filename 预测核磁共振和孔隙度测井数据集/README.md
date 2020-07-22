## Constructing a Synthetic NMR Well-log using Machine Learning


### By Abhishek Bihani

### Final Project for PGE 383 – Subsurface Machine Learning taught by Dr. Michael Pyrcz (Fall - 2019)

### Hildebrand Department of Petroleum and Geosystems Engineering

### The University of Texas at Austin

****


**Executive Summary:** 

The nuclear magnetic resonance (NMR) log is a useful tool to understand lithological information such as the variation of pore size distribution with depth, but it may not be measured in all wells. The [project](https://github.com/abhishekdbihani/synthetic_well-log_polynomial_regression/blob/master/KC151%20-%20PGE383%20v1.ipynb) attempts to predict a missing well log from other available well logs using machine learning tools, more specifically an NMR well log from the measured Gamma Ray (GR) log, Caliper log, Resistivity log, and the interpreted porosity from one well at the Keathley Canyon in the Gulf of Mexico. The constructed model is then used to predict the NMR log at Walker Ridge in Gulf of Mexico, which is another nearby site of methane hydrate accumulation. 

In Keathley Canyon Block 151 (KC-151), the analyzed well was drilled and logged during Leg I of the U.S. Department of Energy/Chevron Gas Hydrate Joint Industry Project (JIP) (Ruppel et al., 2008). At Walker Ridge 313 (WR-313), the analyzed well was drilled and logged during JIP Leg II (Collett et al., 2012). The raw well logs for KC-151 are available [here](http://mlp.ldeo.columbia.edu/data/ghp/JIP1/KC151-2/index.html?) and for WR-313 are available [here](http://mlp.ldeo.columbia.edu/data/ghp/JIP2/WR313-H/). The processed well logs used in this project for KC-151 are available [here](https://github.com/abhishekdbihani/synthetic_well-log_polynomial_regression/blob/master/KC151_logs.csv) and for WR-313 are available [here](https://github.com/abhishekdbihani/synthetic_well-log_polynomial_regression/blob/master/WR313H_logs.csv).

**Approach:**

1) For an easier characterization of the NMR data, the NMR log, i.e. relaxation time distribution was converted into Mean of T2 (MLT2) and Standard Deviation of T2 (SDT2) which are considered as the two response features to be predicted. The other well logs: GR, Caliper, Resistivity, and the interpreted porosity are the predictor features used to train the model.

2) An initial analysis is conducted on the well logs to check the univariate and bivariate distributions of the data, and the well-logs are plotted with depth. 

3) Then a linear regression is conducted for both MLT2 and SDT2 using the other predictor variables to observe the behavior with a basic model. It is seen that the linear regression could not capture the response behavior well due to noise, i.e. short-distance variations as well as non-linearities in the data relationships. 

4) This is followed by feature standardization before applying more complex models to reduce effect of outliers and predictor features having different units. Feature ranking was conducted to compare the order in which predictor variables affect the response variables.

5) Then, the logs are processed to reduce noise, and after a train-test split, polynomial regression modeling is conducted to predict the NMR log at Keathley Canyon until a good fit is obtained.

6) Finally, the trained model is used to predict the NMR log at Walker Ridge where it was not recorded.

**Pre-requisites:**

1. Python3

1. Anaconda

**Instructions:**

Run the following commands using the anaconda command line utility (after navigating to the project folder), to install the required packages, activate the environment and the notebook. 

Commands:
```bash

conda create --name swlpr
conda activate swlpr
pip install -r requirements.txt --ignore-installed --user
jupyter notebook "KC151 - PGE383 v1.ipynb"

```

*Note: The code and procedures used for this project have been adapted from the workflows followed by Dr. Pyrcz in the class (Pyrcz, 2019 a, b, c, d) and my Master's thesis supervised by Dr. Daigle (Bihani, 2016).*

<img src="https://github.com/abhishekdbihani/synthetic_well-log_polynomial_regression/blob/master/KC151-logs.png" align="middle" width="800" height="600" alt="Well-logs at KC-151" >

                                    Figure- Well logs from Keathley Canyon 151


**Assumptions:**

1) The conditions at both KC-151 and WR-313 locations are assumed to be similar enough so the same model can be applied.

2) The model is assumed to be sufficiently trained to make predictions but can be improved if more training data is available.

3) The porosity has been calculated from the bulk density log since porosity is a function of the grain density of the formation (2.65 gm/cm3 in sands, 2.70 gm/cm3 in clays; Daigle et al., 2015) and of the pore-filled fluid (assumed to be water, with a density of 1.03 gm/cm3; Daigle et al., 2015).

4) During polynomial regression, it was assumed that all the relationships between predictors and response features could be captured by basis expansion until the 3rd power.

**Citation:**
 
 If you find this repository useful, please cite as-
 
 Bihani A. Constructing a Synthetic NMR Well-log using Machine Learning. Git code (2019)  https://github.com/abhishekdbihani/synthetic_well-log_polynomial_regression.
 
**Related publications:**

Bihani A., Pore Size Distribution and Methane Equilibrium Conditions at Walker Ridge Block 313, Northern Gulf of Mexico, M.S. thesis, University of Texas, Austin, Texas, 2016. doi:10.15781/T2542J80Z

Bihani A., Daigle H., Cook A., Glosser D., Shushtarian A. (2015). OS23B-1999: Pore Size Distribution and Methane Equilibrium Conditions at Walker Ridge Block 313, Northern Gulf of Mexico. AGU Fall Meeting, 14-18 December, San Francisco, USA. 

**References:**

Collett, T. S., Lee, M. W., Zyrianova, M. V., Mrozewski, S. a., Guerin, G., Cook, A. E., and Goldberg, D. S. (2012). Gulf of Mexico Gas Hydrate Joint Industry Project Leg II logging- while-drilling data acquisition and analysis. Marine and Petroleum Geology, 34(1),41-61, doi:10.1016/j.marpetgeo.2011.08.003

Daigle, H., Cook, A., and Malinverno, A. (2015). Permeability and porosity of hydrate- bearing sediments in the northern Gulf of Mexico. Marine and Petroleum 	Geology, 68, 	551–564, doi:10.1016/j.marpetgeo.2015.10.004

Pyrcz M., (2019a) Feature Selection for Subsurface Data Analytics in Python. Retrieved December 5, 2019, from https://github.com/GeostatsGuy/PythonNumericalDemos/blob/master/SubsurfaceDataAnalytics_Feature_Ranking.ipynb

Pyrcz M., (2019b) Principal Component Analysis for Subsurface Data Analytics in Python. Retrieved December 5, 2019, from
https://github.com/GeostatsGuy/PythonNumericalDemos/blob/master/SubsurfaceDataAnalytics_PCA.ipynb

Pyrcz M., (2019c) Time Series Analysis for Subsurface Modeling in Python. Retrieved December 5, 2019, from
https://github.com/GeostatsGuy/PythonNumericalDemos/blob/master/SubsurfaceDataAnalytics_TimeSeries.ipynb

Pyrcz M., (2019d) Polygonal Regression for Subsurface Data Analytics in Python. Retrieved December 5, 2019, from
https://github.com/GeostatsGuy/PythonNumericalDemos/blob/master/SubsurfaceDataAnalytics_PolygonalRegression.ipynb

Ruppel, C., Boswell, R., and Jones, E. (2008). Scientific results from Gulf of Mexico Gas Hydrates Joint Industry Project Leg 1 drilling: Introduction and overview. Marine and Petroleum Geology, 25(9), 819–829. doi:10.1016/j.marpetgeo.2008.02.007

*****








