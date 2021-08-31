# Home Location Labels

Example file: `synthetic_data/home_locations.csv` <br>

| subscriber_id | antenna_id | tower_id | polygon1 | ... | polygon12 |
|---------------|------------|----------|----------|-----|-----------|
| jb9f3j        | a135       | t23      | A        | ... | J         |
| ad8b3b        | a121       | t1       | B        | ... | K         |
| oewf9ras      | a12        |          | D        | ... | N         |
| bncz83        | a52        | t98      | F        | ... | P         |


## Fields <br>
***subscriber_id**: _string_ <br>
Unique ID for each subscriber

**antenna_id**: _string_ <br>
Ground truth for home antenna (if available)

**tower_id**: _string_ <br>
Ground truth for home tower (if available)

**polygon1**...: _strings_ <br>
Ground truth for polygons, must match to shapefile names


```{note}
Columns without a preceding asterisk '*' are optional
```
