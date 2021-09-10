# Features
Example file: `synthetic_outputs/featurizer/datasets/features.csv` <br>
| name     | feature1 | feature2 | ... | feature185 |
|----------|----------|----------|-----|------------|
| jb9f3j   | 590      | 86       | ... | 5          |
| ad8b3b   | 389      | 55       | ... | 12         |
| oewf9ras |          |          | ... | 3          |
| bncz83   | 198      | 51       | ... |            |


## Fields <br>
***name**: _string_ <br>
Unique ID for the subscriber

**features**: _float_ <br>
The file can have any number of features, named uniquely. This format is automatically produced by the featurizer class,
but custom features can be added to feature files.

```{note}
Columns without a preceding asterisk '*' are optional
```
