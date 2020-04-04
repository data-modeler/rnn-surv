# rnn-surv
==============================

For cases where one wishes to predict the amount of time until an event occurs, but the some of the observations have yet to see the event occur (that is, it is right-censored), survival analysis is family of methods used to perform the analysis and begin making predictions. Most survival models only regard observations at a particular point in time, or perhaps averaged over time, however, sometimes there is a need to develop a model that regards the time-series of dependent variables.

This repository represents an implementation of a survival analysis via recurrent neural network, and is based upon the [RNN-SURV model described here](http://medianetlab.ee.ucla.edu/papers/RNN_SURV.pdf)

------------
## Setup
------------
### AWS
If you haven't already, set up [AWS CLI credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html) by editing the file ~/.aws/credentials (Linux & Mac) or %USERPROFILE%\.aws\credentials (Windows)

```bash
[default]
aws_access_key_id=AKIAIOSFODNN7EXAMPLE
aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

[user1]
aws_access_key_id=AKIAI44QH8DHBEXAMPLE
aws_secret_access_key=je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY
```
Then edit the `Makefile` sections related to the `BUCKET` and `PROFILE` in order to enable sync to S3.

### Environment Setup
Simply run
```bash
make create_environment
make requirements
```

### Data Sources
See the `References` directory for sources of raw data.  Once downloaded and placed into `/data/raw` run 
```bash
make data
```

### More Commands
```bash
make help
```

------------
## Training
------------
### From Docker
The Dockerfile included here will launch a tensorflow gpu container for training the model. Please note that nVidia drivers and a compatible GPU must be installed, otherwise it will train on CPU.
Build the container with:
```
sudo docker build . -t rnnsurv
```

To train the model, run:
```
sudo docker run --gpus all -u 1000:1000 -v $(pwd):/mnt -it rnnsurv:latest python mnt/src/models/train_model.py
```

Or for an interactive session, run:
```
sudo docker run --gpus all -u 1000:1000 -v $(pwd):/mnt -it rnnsurv:latest bash
```



------------
## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
