# Solution of the MSPD by FU-Berlin
by Nils Seitz, Luis Schulze, Alexander Schwarz

If you need help, send a mail to: [nils.seitz@googlemail.com](mailto:nils.seitz@googlemail.com)


## Setup
1. Get testcases from original contest repo (and put inside contest/testcases)
2. Install necessary packages (`conda create --name <env> --file requirements.txt`, `pip install -r requirements.txt`)
3. Run inference (see Use)

## Packages used
In case `requirements.txt` doesn't work, a quick list of necessary packages:
- spektral (GNN)
- numpy (arrays)
- pandas (dataframes)
- tensorflow (ML/NN)
- sklearn (specifically: clustering)

You should install them in this order, as e.g. spektral depends on TF and numpy.

## Use
1. `cd contest`
2. `python -m inference`

## Explanation of Code
... coming soon

### Weight Issue
If results are of you might need to retrain weights shortly (local).  
To do this, a `train.py` is provided in `contest/utils`, which is run with `cd contest/utils && python -m train`
It generates weights to `contest/utils/weights/best_weights.h5`, which you can then copy to `contest/weights` (and rename to `model_{obj_num}.h5`) such that they get used in the `inference.py`.

At the top of `train.py`, there are adjustable settings:
```python
DATA_PATH = './testcases'
SAVE_PATH = './weights'
params = {
    'classes' : 1,
    'col_name' : 'rating',
    'obj_num' : 3,
    # start: 0,
    'norm' : True,
    'TIMEIT': 1,
}
```

Here, you can also modify data and save paths. 
Note that you can adjust the `'obj_num'`, depending on which `model_{obj_num}.h5` you want to (re-)train.
Note also, that the column name is `'rating'`, which is a custom row we calculated from the given dataframes. 
This column norms all values (each source combination per net) for a given objective, e.g. 3W+S, between 0 and 1, such that 1 is the best source combination and 0 the worst.

In case you are interested in that data, you can send a mail - it's too large for GitHub.

You can basically adjust anything in the constructor of `MSPDMixedNet` and this function:
```python
def calc_y(self, src_idx_enum):

    src_row = self.graph_df[self.graph_df['sourceIdx'] == src_idx_enum]
    s = (src_row[f'skew{self.objnum}'] - self.s_min) / self.s_max
    w = (src_row[f'wireLength{self.objnum}'] - self.w_min) / self.w_max
    #   return (1-w)

    x = np.array(src_row[[self.objective]], dtype=np.float32).reshape(1, )
    return x / 100
    # MIN_VAL = 50
    # x = (x-MIN_VAL)/ (100-MIN_VAL)
    #
    # return np.where(x > 0 , x , 0)
```

The logic should match with the col_name you chose for training in `params`.
Additionally, if choosing something not normed, you would need to adjust the specific part in `inference.py` as well, since e.g. its written to use normed values, etc.