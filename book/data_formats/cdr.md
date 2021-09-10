# CDR (call and text)

Example file: `synthetic_data/cdr.csv` <br>

| txn_type | caller_id | recipient_id | timestamp           | duration | caller_antenna | recipient_antenna | international |
|----------|-----------|--------------|---------------------|----------|----------------|-------------------|---------------|
| call     | jb9f3j    | ad8b3b       | 2020-01-01 00:00:42 | 253.0    | AJK3           | KB3               | domestic      |
| text     | ad8b3b    | bncz83       | 2020-02-12 10:12:08 |          | KB9            | KB8               | domestic      |
| text     | oewf9ras  | bncz83       | 2020-03-30 14:50:01 | 15.0     | KB8            | OI14              | domestic      |
| call     | bncz83    | jb9f3j       | 2020-06-15 22:38:31 |          | OI12           | AJK1              | international |


## Fields <br>
***txn_type**: _string_ <br>
Chosen from {call, text}

***caller_id**: _string_ <br>
Unique ID for the caller or sender

***recipient_id**: _string_ <br>
Unique ID for the recipient 

***timestamp**: _string_ <br>
String in format ‘YYYY-MM-DD hh:mm:ss’ representing the start time of a call or the time an text is sent

**duration**: _float_ <br>
Call duration in seconds. NaN or empty string for SMS messages or calls missing duration

```{note}
Columns without a preceding asterisk '*' are optional
```
