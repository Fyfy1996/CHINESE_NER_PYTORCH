# CHINESE_NER_PYTORCH
Pytorch Version for NER, Name Entity Recognization

Include LSTM (Random initialized Embeddings)+CRF or BERT Embeddings+CRF

## Requirements
   pytorch_pretrained_bert 0.6.2, can be changed to transformers(by huggging face), needs to modify the model.py
   
   pytorch, test on 1.8.0, not tested on lower verisons
   
      
## How to use

Edit your own Dataset class (torch.utils.data.Dataset)
See dataset.py for help.

      python run_lstmcrf.py
      python run_bertcrf.py
      
Params can be changed in the two py files by changing the argument class

one_sent_entities.py is a sample that can input a sent in the ternimal and then print the entities inside the sentence
