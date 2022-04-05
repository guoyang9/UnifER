# A Unified End-to-End Retriever-Reader Framework for Knowledge-based VQA

There are two config files in [cfgs](./cfgs/)
## Prerequisites
    * python==3.7
    * pytorch==1.10.0
## Dataset
First of all, make all the data in the right position according to the config file settings. 

* Please download the OK-VQA dataset from the link of the original paper.
* The image features can be found at the [LXMERT](https://github.com/airsplay/lxmert).


## Pre-processing

1. Process answers:
    ```
    python tools/answer_parse_okvqa.py
    ```

2. Extract knowledge base with Roberta:

    ```
    python tools/kb_parse.py
    ```
3. Convert image features to h5 (optional):
    ```
    python tools/detection_features_converter.py 
    ```
## Model Training
```
python main.py --name unifer --gpu 0
```

## Model Evaluation 
```
python main.py --name unifer --test-only
```
## Citation
```
@inproceedings{transearch,
  author    = {Yangyang Guo and
               Liqiang Nie and
               Yongkang Wong and
               Yibing Liu and
               Zhiyong Cheng and
               Mohan S. Kankanhalli},
  title     = {A Unified End-to-End Retriever-Reader Framework for Knowledge-based VQA},
  booktitle = {ACM Multimedia Conference},
  publisher = {ACM},
  year      = {2022}
}
```
