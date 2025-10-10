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

To reproduce the results in RQ1, follow these steps:

1. Data processing

run the code in /without_history/gfn/process.ipynb 

2. Baseline

```cd /without_history/base_line```

```bash ./shell/train_sft_100.sh```

```bash ./shell/train_sft_1500.sh```

```bash ./shell/ppo.sh```

```bash ./shell/dpo.sh```

3. Flower

```cd /without_history/gfn```

```CUDA_VISIBLE_DEVICES=X python train.py task=movie_all_param_1.5B_100 device=gpu > movie_1.5B_0.00001_0.05.out &```

```CUDA_VISIBLE_DEVICES=X python train.py task=movie_all_param_3B_1500 device=gpu > movie_3B_0.00001_0.4.out &```

## Baseline

To reproduce the results in RQ2, follow these steps:

```cd /with_history```

1. Train SASRec

```bash run_SASRec.sh```

2. Train and evaluate BIGRec

```bash run_sft.sh```

```bash evaluate_sft.sh```

3. Train Flower

```bash run_sft-gfn_logp_div_s.sh```

4. Train IFairLRS

```bash item_side_reweight.sh```
