# Data Sources
### Irish Weather Data
The main data source used for testing the Irish Weather Data which can be downloaded [here](https://www.kaggle.com/conorrot/irish-weather-hourly-data).

I acknowledge that this may not be the most appropriate data set for a Survival Analysis, but once modified, it met the core criteria for testing the model:
1. It is freely available
2. It has a time series of X values 
3. It has an acceptable event (precipitation) for which we can attempt to predict

### Aids2
This repo also includes an example of a tuned Random Forest Survival model on the Aids2 dataset which can be downloaded [here](https://forge.scilab.org/index.php/p/rdataset/source/tree/master/csv/MASS/Aids2.csv) or loaded from the `MASS` library in R.

The original article used this dataset as a benchmark comparison for the RNN-SURV model.
