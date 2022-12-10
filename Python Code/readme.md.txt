Packages needed to run Python code: 
- numpy
- requests
- pandas
- matplotlib.pyplot
- sklearn.impute
- sklearn.experimental
- sklearn.linear_model
- sklearn.model_selection
- sklearn.metrics
- sklearn.feature_selection
- sklearn.svm
- sklearn.preprocessing
- sklearn.ensemble
- dash
- plotly

Files:
1. imputation.py - this file was run to impute the missing government and satellite dataset
2. openweatherdata.py - this file was run to train and test models on the OpenWeatherData set from 2020-2022, in order to determine the best model for use in backfilling the data
3. openweatherfill.py - this file used the best models from openweatherdata.py to backfill the data for each climate/emission combination. 
4. dash_viz.py - this file pulls the aggregated climate and unemployment data and creates a dynamic visualization of the data in Dash. 