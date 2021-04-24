# Short-Term-Load-Forecasting-Pecan-Street



## Dataset Overview

A total of 346 residents were recorded in the [Pecanstreet Dataset](https://www.dropbox.com/sh/m20yh5v2yb72o5h/AAACsgKYdIehhV3zhiQOBb4Ka?dl=0). Nearly half of them contained missing data.

| # Data points | Timespan                        | Resident counts |
| ------------- | :------------------------------ | --------------- |
| 172980        | 01-01 00:00:00 ~ 04-30 02:59:00 | 26 (7.51%)      |
| 173100        | 01-01 00:00:00 ~ 04-30 04:59:00 | 26 (7.51%)      |
| 172920        | 01-01 00:00:00 ~ 04-30 01:59:00 | 35 (10.11%)     |
| 173040        | 01-01 00:00:00 ~ 04-30 03:59:00 | 25 (7.23%)      |
| 133417        | 01-01 00:00:00 ~ 04-02 15:36:00 | 49 (14.16)      |
| 90800         | 01-01 00:00:00 ~ 03-04 01:19:00 | 81 (23.41%)     |

***Overview of major data clusters (resident count >= 5)***



## Prediction Methods

### Naive Method

For stable period data, we directly compute the period factors and the "base" load, then multiply each factor with the base load to get a naive prediction. 

This can be also used for 