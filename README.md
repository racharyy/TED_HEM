# VaryFairyTED : A Fair in Rating Predictor for Public Speeches by Awareness of  Verbal and Gesture Quality
Codes and appendix for the paper *VaryFairyTED : A Fair in Rating Predictor for Public Speeches by Awareness of  Verbal and Gesture Quality*

## Appendix

Additional figures and tables are added in the appendix.pdf file 

## Create Environment

```
$ cd TED_HEM
$ conda env create --file ted_hem.yml
```

## Generate Figures and Tables for the paper

Run the notebook './Code/plotter.py'

- Section 5 and 6 of the notebook generates the figures (Figure 3 - 10).
- Section 7 of the notebook generates the table.


## Running the Prediction Model

```
$ cd Code
$ python -m Prediction_model.runner_div --conf confs/experiment_div.yaml --eps "${eps}" --lam "${lam}" --div True
```

- conf is a dictionary containing all the hyperparameters of the model. 
- eps and the lam are the hyperparameters of the model.
- div tells whether to use the HEM loss or not.

More help is below:

```
parser = argparse.ArgumentParser('Train and Evaluate Neural Networks')
  parser.add_argument('--conf', dest='config_filepath', 
    help='Full path to the configuration file')

  parser.add_argument('--bin', dest='num_bin', type=int, default=None,
    help='number of bins for diversity')
  parser.add_argument('--eps', dest='eps',  type=float, default=None,
    help='epsilon')
  parser.add_argument('--lam', dest='lam',  type=float, default=None,
    help='lambda for diversity loss')
  parser.add_argument('--div', dest='div',  type=bool, default=False,
    help='Whether to use HEM during training')
  parser.add_argument('--split', dest='split',  type=bool, default=None,
    help='Whether to split the data before training')
```
