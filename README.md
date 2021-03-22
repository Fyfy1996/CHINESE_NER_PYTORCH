# CHINESE_NER_PYTORCH
Pytorch Version for NER, Name Entity Recognization

Include LSTM (Random initialized Embeddings)+CRF or BERT Embeddings+CRF

## Requirements
   pytorch_pretrained_bert 0.6.2, can be changed to transformers(by huggging face), needs to modify the import in model.py (Not tested yet)
   
   pytorch, test on 1.8.0, haven't tested on lower verisons
      
## How to use

Should Edit your own Dataset class (torch.utils.data.Dataset) to load your own data First

Bert version of NER is using Bert-base-chinese, have been downloaded in pretrained_models fold

See dataset.py for help.

Then adjust the params. Params can be changed in the two py files by changing the argument class 

      python run_lstmcrf.py
      python run_bertcrf.py
      

one_sent_entities.py is a sample that can input a sent in the ternimal and then print the entities inside the sentence. Can modify the modelpath to support your own model
