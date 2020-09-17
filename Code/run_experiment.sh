for eps in 0.017; do
    for lam in 5; do
         python -m Prediction_model.runner_div --conf confs/experiment_div.yaml --eps "${eps}" --lam "${lam}"
    done
done