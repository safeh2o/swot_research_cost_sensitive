# swot_research_cost_sensitive
This repo contains code and a test dataset for the SWOT cost sensitive learning paper. We have included two codes: "ErrorWeightingInvestigation.py" and "ErrorInvestigation_ExperimentFigs.py". Make sure that they are in the same directory as some of the saving and loading assumptions assume that these files will have the same working directory.

Prior to running the code you will need to install the following Python packages:

Tensorflow 2.0 or higher
numpy
keras
matplotlib
pandas
sklearn
time
datetime
os
scipy

To run the code:

1. Paste the file path as a string in the brackets of the pd.read_csv functions at lines 253 and 257 of the "ErrorWeightingInvestigation.py" (for those new to Python, replace all of the backslashes in the file path with double backslashes. For example, instead of "C:\Users\me\my\path\Tanzania_Out.csv", use "C:\\Users\\me\\my\\path\\Tanzania_Out.csv"
2. Run the "ErrorWeightingInvestigation.py" code first - this will train and test all of the ANN ensemble forecasting systems and save the model outputs in subfolders for each alternative.
3. Run the "ErrorInvestigation_ExperimentFigs.py". This will make some figures in the subfolder for each model an will output a .csv file in the main directory entitled "Super_Array_Results.csv". This will contain all of the ensemble verification scores for all alternatives (without skill scores as these can be easily calculated in excel).

In addition to the above code, we have provided both datasets referenced in the paper. Currently it uses the Tanzania dataset, to change this, change 
