# Introduction

Cider provides tooling for poverty prediction and home location inference from individual-level mobile phone metadata 
for the purpose of algorithmic targeting of social protection. Cider supports records of CDR (call and text), recharges,
mobile data usage, and mobile money usage. Cider provides three main features: 
- **Survey outcome generation** to generate unidimensional poverty indices from a set of survey responses 
- **Featurization** to calculate over a thousand statistical features from underlying individual mobile phone metadata, built on PySpark and bandicoot.
- **Machine learning** to infer poverty from CDR features given a dataset of features where a subset of observations are 
labeled with a poverty outcomes, built on scikit-learn. 
- **Home location inference** to infer home tower or region using CDR with a set of built-in algorithms, built on PySpark 
and GeoPandas. 
<br>
Download cider from [github](https://github.com/emilylaiken/cider). 
