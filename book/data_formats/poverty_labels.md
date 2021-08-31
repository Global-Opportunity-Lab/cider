# Poverty Labels

Example file: `synthetic_data/labels.csv` <br>

| name     | label  | weight |
|----------|--------|--------|
| jb9f3j   | 100.12 | 12.80  |
| ad8b3b   | 136.91 | 21.59  |
| oewf9ras | 78.52  | 7.55   |
| bncz83   | 121.00 | 22.12  | 


## Fields <br>
***name**: _string_ <br>
Unique ID for each subscriber

***label**: _float_ or _bool_  <br>
Wealth label for the observation; can be numeric for regression, or categorical for classification

**weight**: _float_ <br>
Weight to assign the observation (i.e. from a survey sampling weight or response weight)

```{note}
Columns without a preceding asterisk '*' are optional
```
