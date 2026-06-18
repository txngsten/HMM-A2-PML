# Sequential Modeling and Scalable Inference System (COMP318) - Oliver Wuttke (WUTT0019)
A full sequential modeling and inference system powered by Hidden Markov Models.
The project needs to be setup and ran correctly in order for it to run as intended so please follow these setup instructions.

These steps all assume you are currently in the `Source/` directory where this README.md file is contained.
## Setup Python (3.14.5)
Run this to install and activate the correct python version virtual environment (venv):
```bash
python3.14 -m venv .venv
source .venv/bin/activate
```
Then we need to install our [requirements](requirements.txt) like so:
```bash
pip install -r requirements.txt
```

## Running the Project
The order of execution is important for this project to be run correctly so please do these steps in order.
First the [train_test_val.py](train_test_val.py) file needs to be run first to ensure that the model params are saved, so [probabilistic_model.py](probabilistic_model.py) and [visualise.py](visualise.py) can run correctly.
```bash
python train_test_val.py
```
Once this has finished executing a *hmm_params.npz* file should be saved into the current directory.
Then to execute the inference engine run the following:
```bash
python probabilistic_model.py
```

### Optional
I've also included a file [visualise.py](visualise.py) that was used to generate some of the figures in the report.
To run this just ensure [train_test_val.py](train_test_val.py) and the *hmm_params.npz* are saved:
```bash
python visualise.py
```