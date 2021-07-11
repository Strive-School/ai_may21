# Feature Engineering Module

Content for 2 weeks of class (10 days).

|     | Topic of the day                                       | Things to cover                         
|-----|--------------------------------------------------------|--------------------------------------------------------------|
| Mon | [1. Robust ML](./01.%20Robust%20ML)                    | Tree & Mult Models, Validation, ColumnTransformer, Pipelines |
| Tue | [2. Data Cleaning](./02.%20Data%20Cleaning)            | Missings & Outliers (Drop vars, impute vars)                 |
| Wed | [3. Numerical encodings](./03.%20Numerical%20Enc)      | MinMaxScaler, StandardScaler, BoxCox, QuantileTransformer    | 
| Thu | [4. Categorical encodings](./04.%20Categorical%20Enc)  | Ordinal, Binary, OneHot, Mean Enc., CatBoost                 |
| Fri | [5. Feature Selection & Dim Reduction](./05.%20FeatSel%20%26%20DimRed) | PCA, tSNE, UMAP, VarianceThreshold           |
| Mon | [6. FE for Geographic data](./06.%20Geoposition%20Enc) | Lat, lon. population                                         |
| Tue | [7. FE for NLP](./07.%20Text%20Enc)                    | BoW, TFIDF, N-Grams                                          |
| Wef | [8. FE for Time Series](./08.%20Date%20Enc)            | Lag features, TSfresh                                        |
| Thu | [9. FE for Several tables](./09.%20Combine%20tables)   | Manually merge & join, featuretools                          |
| Fri | 10. Kaggle challenge                                   |                                                              |


# Tabular Playground Series competitions

|                                                                         | X (data)          | Y (target)  | Metric   |
|-------------------------------------------------------------------------|-------------------|-------------|----------|
| [Jan 2021](https://www.kaggle.com/c/tabular-playground-series-jan-2021) | 14 nums           | Numeric     | RMSE     |
| [Feb 2021](https://www.kaggle.com/c/tabular-playground-series-feb-2021) | 14 nums & 10 cats | Numeric     | RMSE     |
| [Mar 2021](https://www.kaggle.com/c/tabular-playground-series-mar-2021) | 11 nums & 19 cats | Binary      | AUC      |
| [Apr 2021](https://www.kaggle.com/c/tabular-playground-series-apr-2021) | Titanic           | Binary      | Accuracy |
| [May 2021](https://www.kaggle.com/c/tabular-playground-series-may-2021) |                   | multi-class | Log loss |

# References

- https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/
- https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/