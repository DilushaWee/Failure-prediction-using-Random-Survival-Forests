1. The data for the 3 regions is contained inside the /data folder. All the output shall be written to the /Output folder. 

2. The python file "rsf_run.py" is responsible for running the RSF algorithm on the dataset and generating predictions. 

3. "rf_gb_run.py" runs the Random Forest and Gradient Boosted regression algorithms.

4. All generated files are named in the format xyear1_year2.csv format. The year1 is the year upto which the data was used for training. year2 is the year for which the prediction is done. x=gb for gradient boosting, =rf for random forest and, =rsf for random survival forest.

5. getAUCROC.m matlab function calculates the ROC curves and the AUC values for any such generated dataset.

6. train1_weibull.m and trani2_weibull.m functions create the baseline weibull predictions for the pipe failures.