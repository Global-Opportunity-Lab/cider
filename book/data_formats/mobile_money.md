# Mobile Money
Example file: `synthetic_data/mobilemoney.csv` <br>
| txn_type | caller_id | recipient_id | timestamp           | amount | sender_balance_before | sender_balance_after | recipient_balance_before | recipient_balance_after |
|----------|-----------|--------------|---------------------|--------|-----------------------|----------------------|--------------------------|-------------------------|
| cashin   | jb9f3j    |              | 2020-01-01 00:00:42 | 234.0  | 500.0                 | 734.0                |                          |                         |
| cashout  | ad8b3b    |              | 2020-02-12 10:12:08 | 195.0  | 200.5                 | 5.5                  |                          |                         |
| p2p      | oewf9ras  | bncz83       | 2020-03-30 14:50:01 | 2.4    | 100.0                 | 97.6                 | 78.1                     | 80.5                    |
| billpay  | bncz83    | jb9f3j       | 2020-06-15 22:38:31 | 76.1   | 150.0                 | 73.9                 | 12.9                     | 89.0                    |
| other    | bncz83    | ad8b3b       | 2020-08-16 23:10:10 | 502.0  | 872.0                 | 370.0                | 0.00                     | 502.0                   |


## Fields <br>
***txn_type**: _string_
Any string describing the type of the transaction; the dataset can define any number of categories for transactions, 
but keep in mind that features will be calculated separately for each transaction type so compute time scales with the 
number of categories. Common categories include {cashin, cashout, billpay, p2p, other}.

***caller_id**: _string_ <br>
Unique ID for the caller or sender

***recipient_id**: _string_ <br>
Unique ID for the recipient 

***timestamp**: _string_ <br>
String in format ‘YYYY-MM-DD hh:mm:ss’ representing the start time of a call or the time an text is sent

***amount**: _float_ <br>
Amount of money transferred (in any currency)

**sender_balance_before**: _float_ <br>
Balance of sender before transaction (in any currency). Only provided by some mobile network operators.

**sender_balance_after**: _float_ <br>
Balance of sender after transaction (in any currency).  Only provided by some mobile network operators.

**recipient_balance_after**: _float_ <br>
Balance of the recipient after transaction, where applicable (in any currency).  Only provided by some mobile network 
operators.

**recipient_balance_before**: _float_ <br>
Balance of the recipient after transaction, where applicable (in any currency).  Only provided by some mobile network 
operators.

```{note}
Columns without a preceding asterisk '*' are optional
```
