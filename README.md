# CHINESE_NER_PYTORCH
Pytorch Version for NER, Name Entity Recognization<br>
torch版本的命名实体识别<br>
Tools including:<br> 
有如下的模型（脚本）<br>
	lstm + crf<br>
	bert + softmax<br>
	bert + crf<br>
	
## Requirements
   主要使用的包，其他包（如numpy等）就不赘述<br>
		pytorch 1.8.0
		transformers, 4.15.0

## How to use
1. 使用BIO标注 <br>
2. 替换dataset文件夹的训练和测试txt <br>
3. 修改 `run_MODEL_NAME.py` 中的参数 <br>
4. 使用如下命令运行模型

		python run_lstmcrf.py
      
## Results
Dataset文件夹中有一个样例数据，在此数据上，结果如下
||Precision|Recall|F1|
|------|:-:|:-:|:-:|
|LSTM+CRF|-|-|-|
|BERT+SOFTMAX|91.65|92.13|91.18|
|BERT+CRF|91.60|91.95|91.77|

## 其他
Feel Free to use it！
如有问题和bug等，欢迎交流 yonttavan@126.com
