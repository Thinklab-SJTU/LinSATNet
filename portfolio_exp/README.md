# Portfolio Allocation Experiment in LinSAT Paper

To run the code, first install all the necessary packages with

```pip install linsatnet cvxpy cvxpylayers numpy==1.19.5 scipy==1.5.4 pandas==1.1.5 torch==1.7.1```

## Run the experiment

You can reproduce the main result of our portfolio optimization experiment with LinSATNet by running

```python main.py --use_linsatnet```

You can run the experiment with original StemGNN with

```python main.py```
