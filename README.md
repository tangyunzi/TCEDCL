# Trigger-free Cybersecurity Event Detection Based on Contrastive Learning
_Last update: 5/25/2023_  


### Dependencies
- Python 3.8
- Numpy 1.16.2
- transformers 4.29.1 
- pytorch 2.0.1 
- simcse 0.4 

### File description
1.`Unsuper_train.txt` is used to train an unsupervised contrastive learning models to generate the model file used in our project. The model is named `my-unsup-simcse-bert-base-uncased`.
The unsupervised contrastive learning model is an unsupervised contrastive learning method used in the SimCSE model.  The link to SimCSE is https://github.com/princeton-nlp/SimCSE.

2.`finetune.py`  file is used for fine-tuning the model.

3.`simed.py`  file is used for detecting events.

4.`annotation.jsonl` is the annotated data file that needs to be divided into train, dev, and test in a 3:1:1 ratio.

## References
1. Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Online and Punta Cana, Dominican Republic. Association for Computational Linguistics., pp. 6894–6910, 2021