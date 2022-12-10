The folder 'R Code' contains all of the RStudio files for the Time Series analysis methods conducted for this project.

Data Check.Rmd
Required libraries: mgcv, lubridate, dplyr
These can be installed id needed using the code below:
install.packages("package_name")
This code was used as a check for each of the OpenWeather data sources. 
It checks for NA values using the is.na() function in R and checks for outliers using boxplots. 
It also plots the time series and acf plots for each variable within the data sets.
The code is meant to take one data source at a time. It does not read all of the data at once.

Imputed_Analysis_V3.Rmd (previous versions 'Imputed_Analysis_V2.Rmd' and 'Imputed_Analysis.Rmd')
Required libraries: data.table, vars, xts, stats, tseries, aod, mgcv, lubridate, dplyr, TSA, dynlm, tidyverse
This code reads all of the data containing the imputed values and true values for the OpenWeather data (from 2018-2022) and the unemployment data.
The data is split up into climate groups (mentioned in our report) adn then aggregated quarterly. 
The percent change across all climates for each quarter was calculated and plotted from 2018-2022.
The emissions trends for 2018-2022 are recorded in matrices in the file "Matrices_V2.docx" (preious version excluding imputed data trends "Matrices.docx")
The trend and seasonality were estimate for each emission for each climate.
The trend and seasonaliy were removed from each emission.
Time series objects were create for each emission and unemployment rate by climate.
Each climate's multivariate time series wwas modeled using the VAR model.
The lag value was selected using the VarSelct function.
A wald test was run for all emissions on unemployment to test for granger causality.
The time series for each variable by climate was plotted at the end of the file.
The aggregated data for each climate was exported as csv files.

Project_Code_SV_2022-10-10.Rmd (previous version 'Project_Code_SV_2022-09-25.Rmd')
Required libraries: mgcv, lubridate, dplyr, TSA, dynlm, xta, vars, stats, tseries, aod, data.table
This code is part of the EDA for this project.
It reads the OpenWeather data then extracts the latitude and longitude for each city and stores it in a new data frame that was 
used for a quick visualization in another program.
The data was combined to make the different climate groupings.
The time series for each emissino was plotted.
Initial trend and seasonality were estimated and plotted.

Unemployment_Analysis_2022-11-20.Rmd
Required libraries: data.table, vars, xts, mgcv, stats, tseries, aod, quantmod, lubridate
The OpenWeather data and unemployment data were loaded in.
The data was cleaned up and aggregated quarterly by climate group.
The Unemployment data was merged with the OpenWeather data.
Percent change in emissions and unemployment were calculated from 2020-2022.
The results were input into matrices in the MS Word document 'Matrices.docx' (Newst version is "Matrices_V2.docx")




