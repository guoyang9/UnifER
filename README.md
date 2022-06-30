# A Unified End-to-End Retriever-Reader Framework for Knowledge-based VQA

Official implementation for the MM'22 paper.

![model structure](/imgs/structure.png)

There are two config files in [cfgs](./cfgs/), individually for the OK-VQA and FVQA datasets. Note that we mainly test our method on the OK-VQA dataset.
## Prerequisites
* python==3.7
* pytorch==1.10.0
## Dataset
First of all, make sure all the data in the right position according to the config file settings. 

* Please download the OK-VQA dataset from the link of the original paper.
* The image features can be found at the [LXMERT](https://github.com/airsplay/lxmert) (If you need only the ViLT model, then skip these features and only download the mscoco images.).


### Pre-processing:
The last step is optional for LXMERT and VisualBert only.
1. Process answers:
    ```python
    python tools/answer_parse_okvqa.py 
    ```

2. Extract knowledge base with Roberta:
    ```python
    python tools/kb_parse.py
    ```
3. Convert image features to h5 (optional):
    ```python
    python tools/detection_features_converter.py 
    ```
### Model Training:
```python
python main.py --name unifer --gpu 0
```

### Model Evaluation: 
```python
python main.py --name unifer --test-only
``` 

### Citation:
If you found this repo helpful, please consider cite the following paper :+1: :
```ruby
@inproceedings{unifer,
  author    = {Yangyang Guo and Liqiang Nie and Yongkang Wong and Yibing Liu and Zhiyong Cheng and Mohan S. Kankanhalli},
  title     = {A Unified End-to-End Retriever-Reader Framework for Knowledge-based VQA},
  booktitle = {ACM Multimedia Conference},
  publisher = {ACM},
  year      = {2022}}
```
