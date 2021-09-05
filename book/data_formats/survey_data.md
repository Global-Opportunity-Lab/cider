# Survey Data

Example file: `synthetic_data/survey.csv` <br>

| unique_id | weight | outcome | col1 | ... | col35     |
|-----------|--------|---------|------|-----|-----------|
| jb9f3j    | 12.6   | 1.5     | 5.2  | ... | category3 |
| ad8b3b    | 1.6    | 2.7     | 6.1  | ... | category6 |
| oewf9ras  | 3.2    | 4.1     | 1.5  | ... |           |
| bncz83    | 6.0    | 0.9     | 2.6  | ... | category2 |


## Fields <br>
***unique_id**: _string_ <br>
Unique ID for each survey observation

**weight**: _float_ <br>
Weight for each survey observation (if sample weights are used)

***outcome** and **col1**...**col35**: _mixed types_ <br>
Survey data. Can have any column names and be of any type (continuous, categorical, or binary)

```{note}
Columns without a preceding asterisk '*' are optional
```
