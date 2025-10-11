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

1. SASRec

```bash run_SASRec.sh```

2. Tiger

```bash run_sft.sh```

```bash evaluate_sft.sh```

3. IDGenRec

```bash run_sft-gfn_logp_div_s.sh```

4. LETTER-Qwen & LETTER-Llama

```bash run_train.sh```

```bash run_test_ddp.sh```

5. LETTER-LLaDA

```bash run_train.sh```

```bash run_test_ddp.sh```
