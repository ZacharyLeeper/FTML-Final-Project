# FTML-Final-Project

This repo served as our project for Fair and Transparent Machine Learning during the Fall 2022 semester.
In this project, we trained a neural network under various fairness constraints and ran an HCI study
to evaluate how human stakeholders view fairness metrics used in practice.

To use this repo, some setup is required:
1. Clone the repo: `git clone https://github.com/ZacharyLeeper/FTML-Final-Project`
2. In the top-level directory, create the two directories `models` and `results`.
   These are used for saving models and storing graphs of results, respectively.
3. Install the python packages required to run the scripts `pip install -r requirements.txt`.
   You may want to create a virtual environment for this first.

To train the neural network, run the script `train_and_eval_models.py` from the `src` directory.

To interactively use the models like we did in the experiments, run the script `run_experiment.py`
script from the `src` directory. You will need to specify the model to load with the `--model <path>` flag.
