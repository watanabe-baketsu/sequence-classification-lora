# Sequence(text) Classification with Transformers
## Introduction
This repository contains the code for phishing html classification PoC.
The model can be fine-tuned with LoRA.
Besides, the transformer model can be used for text embedding, and the embedding can be used as the features for the custom classification head.

## LoRA Fine-tuning
### Fine-tune the model
```bash
python3 lora-finetune.py --model_name=<Your-Model-Name> --dataset_path=<Your-Dataset-Path> --target_modules=<Your target_module>
```
> Note: You should edit the `--target_modules` according to this mapping.
> [https://github.com/huggingface/peft/blob/632997d1fb776c3cf05d8c2537ac9a98a7ce9435/src/peft/utils/other.py#L202](https://github.com/huggingface/peft/blob/632997d1fb776c3cf05d8c2537ac9a98a7ce9435/src/peft/utils/other.py#L202)

### Inference
```bash
python3 lora-inference.py --model_name=<Your-Model-Name> --dataset_path=<Your-Dataset-Path>
```

## Transformer Embedding and Classification with simple head
