# CHINESE_NER_PYTORCH
Pytorch Version for NER, Name Entity Recognization<br>
Tools including:<br> 
	lstm + crf<br>
	bert + softmax<br>
	bert + crf<br>
	
## Requirements
   
		pytorch 1.8.0
		transformers, 4.15.0

## How to use
Make sure your data is label in BIO methods <br>
Then modify the data path in python scripts. See dataset.py for help.<br>


Then adjust the params. Params can be checked in python scripts.
Run this code to start the training:

		python run_lstmcrf.py
      
## Results
These results are tested on the provided dataset
||Precision|Recall|F1|
|------|:-:|:-:|:-:|
|LSTM+CRF|-|-|-|
|BERT+SOFTMAX|91.65|92.13|91.18|
|BERT+CRF|91.60|91.95|91.77|
