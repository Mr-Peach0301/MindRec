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
Due to file size limitations of GitHub, the files of training set are not uploaded to the repository, other files are available. The following steps describe our data processing procedure, using the Video Games dataset as an example.

1. Download the dataset

```wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Video_Games.json.gz```

```wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz```

2. Unzip

```gunzip Video_Games.json.gz```

```gunzip meta_Video_Games.json.gz```

```cd /with_history```

3. Process for BIGRec and IFairLRS

```python ./code/process.py --category "Video_Games"```

4. Process for SASRec

```bash to_SASRec.sh```

5. Process for Flower

run the code in process.ipynb

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

```bash item_side_reweight.sh```

5. LETTER-LLaDA

```bash item_side_reweight.sh```
