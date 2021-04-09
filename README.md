# CHINESE_NER_PYTORCH
Pytorch Version for NER, Name Entity Recognization

Include LSTM (Random initialized Embeddings)+CRF

## Requirements
   
   pytorch, test on 1.8.0, haven't tested on lower verisons
      
## How to use

Should Edit your own Dataset class (torch.utils.data.Dataset) to load your own data First



See dataset.py for help.

Then adjust the params. Params can be changed in the two py files by changing the argument class 

      python run_lstmcrf.py
      

one_sent_entities.py is a sample that can input a sent in the ternimal and then print the entities inside the sentence. Can modify the modelpath to support your own model
