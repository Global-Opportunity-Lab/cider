# Mobile Data Usage
Example file: `synthetic_data/mobiledata.csv` <br>
| caller_id | volume | timestamp           |
|-----------|--------|---------------------|
| ioa032    | 50.12  | 2020-02-09 17:18:56 |
| j510da    | 59.32  | 2020-02-18 21:45:42 |
| g831d     | 12.56  | 2020-03-16 22:56:01 |
| g913kj9a  | 108.91 | 2020-03-30 08:31:33 |   


## Fields <br>
***caller_id**: _string_ <br>
Unique ID for the subscriber

**volume**: _float_ <br>
Volume of data consumed (upload or download), in any denomination

***timestamp**: _string_ <br>
String in format ‘YYYY-MM-DD hh:mm:ss’ representing the time of the recharge


```{note}
Columns without a preceding asterisk '*' are optional
```
