# Automatic speech recognition project

## Report

Experimental details and heuristics can be found in 
[wandb report](https://wandb.ai/practice-cifar/asr_project/reports/ASR-Report--Vmlldzo1Nzc5NTQy).

## Installation guide

To get started run the following command.

```shell
pip install -r ./requirements.txt
```

*Remark:* Since `kenlm` library cannot be installed by `pip` on Windows. Windows users can either remove this
requirement, or try [official guide](https://kheafield.com/code/kenlm/). 

Before starting training check correctness on test cases
```shell
python -m unittest discover hw_asr/tests
```

## Preparing datasets (optional)

Datasets are loaded automatically, however, downloading archives from web might be slow.
To tackle with problem pre downloading applied. For example, to install `train-clean-100`
part of librispeech dataset with multithreading
```shell
sudo apt install axel
axel -n <num_threads> https://www.openslr.org/resources/12/train-clean-100.tar.gz
mkdir -p data/datasets/librispeech
mv train-clean-100.tar.gz data/datasets/librispeech
```

## Model evaluation

To download checkpoint of pretrained model, execute the following python code from repository root
```python3
import gdown
gdown.download("https://drive.google.com/uc?id=1au11O0p-orSg8h31gEQP5lRQ1Vk0GrSI", "default_test_model/checkpoint.pth")
```

It loads model parameters to directory `default_test_model`, where evaluation config is located.

The next step is running `test.py` to evaluate model on given datasets. Datasets
can be changed in "data" section of `config.json` and with command line argument `-t`
```shell
python test.py \
   -c default_test_model/config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json
```
To run it with the default args
```shell
python3 test.py
```

Results are logged into stdout and saved in two files

- `output.json` (or another name provided in `-o`) contains model predictions with different heuristics
- `output.csv`  (the same file with extension `.csv`) contains metrics for given predictions

## Model training and fine-tuning

To start training execute
```shell
python train.py \
   -c path/to/config.json \
   -r path/to/checkpoint.pth
```

If `-r` (`--resume`) parameter is specified, training from random initialization will be performed.
For example, recommended training pipeline from scratch can be applied by
```shell
python train.py -c final_model/pretrain.json
```

with optional fine-tuning on train-other `librispeech` dataset
```shell
python train.py \
   -c final_model/finetune.json \
   -r saved/models/final_pretrain/<run_id>/model_best.pth
```

Run id is an unique identifier of execution - datetime by default.

## Additional features

#### Inference stage

Model evaluation supports `beam search` and `argmax` algorithms to 
find the most possible output sequences.
In addition, inference stage equipped by pretrained language model, which significantly boosts Word Error Rate metrics

                    arg max   | beam_search  | model_search
    test-clean  	0.162679  |    0.160069  |    0.107235
    test-other  	0.361537  |    0.358296  |    0.263807

This model works with output logits and evaluates beam search with 
modified scores, including prior knowledge of text structure.

#### Another tokenizer

By default, char tokenization is applied for texts. However, 
BPE tokenizer is also implemented and can be used by adding to config file
```json
  "text_encoder": {
    "type": "CTCBPETextEncoder",
    "args": {"vocab_size": <your_vocab_size>}
  }
```


## Credits

This repository is based on an
[asr-template](https://github.com/WrathOfGrapes/asr_project_template) repository.
