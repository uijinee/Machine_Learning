# Machine_Learning

```python
from tools.preprocess import LoadDataset

train_path = "../data/train.csv"
test_path = "../data/valid.csv"
valid_path = "../data/valid.csv"

# do with train, test, valid
data = LoadDataset(train_path,
                   test_path, 
                   valid_path)

data.nan_processing({"HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam": "FILL MOST FREQ",
                     "OTHERS": "DROP COLUMN"})
data.sampling("OFF")
data.scaling({"Production Qty Collect Result_Fill2":"STANDARD",
              "OTHERS": "MINMAX"})
data.encoding({"Equipment_Dam": "ONEHOT",
               "OTHERS": "LABEL"})
data.feature_engineering("PCA", {"n_components": 5})
```

### Result

> Missing Value Processing...
> 
> (before processing)The number of nan
> - train: 59028
> - valid: 13149
> 
>   1. Processing <HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam> Columns with 'FILL MOST FREQ'
>   2. Processing <OTHERS> Columns with 'DROP COLUMN'
> 
> Finish!
> (after processing)The number of nan
> - train: 0
> - valid: 0
> 
> ====================
> Sampling with OFF...
> 
> (before sampling)Value count
> target
> 0         31156
> 1          2000
> Name: count, dtype: int64 
> 
> Finish!
> (after sampling)Value count
> target
> 0         31156
> 1          2000
> Name: count, dtype: int64
> 
> ====================
> Scaling our dataset...
> 
>   1. Scaling <Production Qty Collect Result_Fill2> Columns with 'STANDARD'
>   2. Scaling <OTHERS> Columns with 'MINMAX'
> 
> Finish!
> 
> ====================
> Encoding our dataset...
> 
>   1. Encoding <Equipment_Dam> Columns with 'ONEHOT'
>   2. Encoding <OTHERS> Columns with 'LABEL'
> 
> Finish!
> 
> ====================
> Feature Engineering with PCA...
> 
>   Execute PCA with 5 components...
> 
>   Explained Variance Ratio: 0.9999848915330183
>       ※ details = [9.99826078e-01 9.28367807e-05 3.45329186e-05 1.84270374e-05
>  1.30168761e-05]
> 
> Finish!
> 
> ====================

---

```python
# 2. do with train, test
_data = LoadDataset(train_path, 
                    test_path)

_data.nan_processing({"ALL": "DROP COLUMN"})
_data.feature_engineering("PCA", {"n_components": 5})
_data.sampling("UNDER")
_data.scaling({"ALL": "STANDARD"})
_data.encoding({"ALL": "TARGET"})
```

> Missing Value Processing...
> 
> (before processing)The number of nan
> - train: 59028
>   1. Processing <ALL> Columns with 'DROP COLUMN'
> 
> Finish!
> (after processing)The number of nan
> - train: 0
> 
> ====================
> Feature Engineering with PCA...
> 
>   Execute PCA with 5 components...
>   Warning: We only transform data with numeric value, now we have 12 categorical column!
> 
>   Explained Variance Ratio: 0.9966137041315634
>       ※ details = [0.77551949 0.15837124 0.04262567 0.01529109 0.00480621]
> 
> Finish!
> 
> ====================
> Sampling with UNDER...
> 
> (before sampling)Value count
> target
> 0         31156
> 1          2000
> Name: count, dtype: int64 
> 
> Finish!
> (after sampling)Value count
> target
> 0         2000
> 1         2000
> Name: count, dtype: int64
> 
> ====================
> Scaling our dataset...
> 
>   1. Scaling <ALL> Columns with 'STANDARD'
> 
> Finish!
> 
> ====================
> Encoding our dataset...
> 
>   1. Encoding <ALL> Columns with 'TARGET'
> 
> Finish!
> 
> ====================

---
**And now we can use transformed feature**

data.train.x
data.valid.x
data.test.x