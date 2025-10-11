# MindRec: Mind-inspired Coarse-to-fine Decoding for Generative Recommendation

This repository hosts the official PyTorch-based implementation of the method presented in the paper "MindRec: Mind-inspired Coarse-to-fine Decoding for Generative Recommendation".

## Installation

To install the project, follow these steps:

1. Clone this git repository and change directory to this repository.

2. Create a conda environment and activate.

```conda create --name MindRec python=3.10 -y```

```conda activate MindRec```

3. Install the required modules from pip.

```pip install -r requirements.txt```

## Data processing

Our complete dataset is hosted on Hugging Face and will be released publicly soon.

Alternatively, you can run the code in process.ipynb to reproduce it from scratch.

## MindRec

To reproduce the results in of MindRec, follow these steps:

```cd /MindRec```

```bash run_train.sh```

```bash run_test_ddp.sh```

## Baseline

To reproduce the results of baselines, follow these steps:

```cd /baseline```

1. Tiger

```cd /Tiger```

```bash run_train.sh```

```bash run_test.sh```

2. IDGenRec

```cd /IDGenRec/command```

```bash train_standard.sh```

3. LETTER-Qwen & LETTER-Llama

```cd /LETTER-LC-Rec```

```bash run_train.sh```

```bash run_test_ddp.sh```

4. LETTER-LLaDA

```cd /LETTER-LLaDA```

```bash run_train.sh```

```bash run_test_ddp.sh```
