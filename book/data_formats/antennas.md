# Antennas
Example file: `synthetic_data/antennas.csv` <br>
| antenna_id | tower_id | latitude | longitude |
|------------|----------|----------|-----------|
| KB2        | KB       | 5.1632   | 11.5091   |
| KB8        | KB       | 5.1632   | 11.5091   |
| AJK1       | AJ       | 6.9012   | 12.3911   |
| OI12       | OI       |          |           |


## Fields <br>
***antenna_id**: _string_ <br>
Unique ID for the antenna, matches to antenna_id in CDR datasets


**tower_id**: _string_ <br>
Unique ID for the cell tower. Many antennas can (and typically will) have a single tower ID. Tower ID is optional, it 
need not be included if not provided by the mobile network operator (or tower IDs could be inferred using spatial 
clustering of antennas). 


**latitude**: _float_ <br>
Latitude of the antenna


**longitude**: _float_ <br>
Longitude of the antenna


```{note}
Columns without a preceding asterisk '*' are optional
```
