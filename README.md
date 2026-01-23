# MindRec: Mind-inspired Progressive Generative Recommendation

This repository hosts the official PyTorch-based implementation of the method presented in the paper "MindRec: Mind-inspired Progressive Generative Recommendation".

## Installation

To install the project, follow these steps:

1. Clone this git repository and change the directory to this repository.

2. Create a conda environment and activate.

```conda create --name MindRec python=3.10 -y```

```conda activate MindRec```

3. Install the required modules from pip.

```pip install -r requirements.txt```

## :bar_chart: Dataset

Our complete dataset is hosted on Hugging Face :hugging_face: and will be made publicly available.

Alternatively, you can run the code in process.ipynb to reproduce it from scratch.

## MindRec

To reproduce the results of MindRec, follow these steps:

```cd /MindRec```

```bash run_train.sh```

```bash run_test_ddp.sh```
