# Solution of the MSPD by FU-Berlin
by Nils Seitz, Luis Schulze, Alexander Schwarz

If you need help, send a mail to: [nils.seitz@googlemail.com](mailto:nils.seitz@googlemail.com)


## Setup
1. Get testcases from original contest repo (and put inside contest/testcases)
2. Install necessary packages (`conda create --name <env> --file requirements.txt`)
3. Run inference (see next)

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
To do this, a `train.py` is provided in `contest/utils`.
It generates weights to `contest/utils/weights`, which you can then copy to `contest/weights` such that they get used in the `inference.py`.
