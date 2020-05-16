## Image Prediction With Knowledge Graph Prediction and Integration


#### Set Parameters
```
python parameters.py
```

##### Notes
* For using image data, comment out the part marked with (*) in batch_util.py
* Make sure all lines are uncommented in run.sh, running run.sh will run all the following steps sequentially
* Remove [:5] from read_data_train.py or read_data_val_or_test.py for complete training
----------------------------------------------- <br>
----------------------------------------------- <br>

#### Process data into binary format
```
python read_data_train.py
```
In : dataset/v1/train 

Out : train_targets_pos <br>
	  train_targets_negs <br>
	  train_contexts_text <br>
	  train_contexts_image <br>
	  train_binarized_data.pkl <br>
	  vocab_word_to_id.pkl <br>
	  vocab_id_to_word.pkl <br>

Uses : read_data_util.py <br>
	   parameters.py <br>
----------------------------------------------- <br>
----------------------------------------------- <br>

#### Get word embeddings for training vocabulary 
```
python embeddings.py
```
In : vocab_id_to_word.pkl <br>
	 GoogleNews-vectors-negative300.bin

Out : vocab_embeddings.pkl

Uses : parameters.py <br>
----------------------------------------------- <br>
----------------------------------------------- <br>

#### Binarize validation data 
```
python read_data_val_or_test.py val
```
In : dataset/v1/val <br>
	 vocab_word_to_id.pkl
	 
Out : val_targets_pos <br>
	  val__targets_negs <br>
	  val_contexts_text <br>
	  val_contexts_image <br>
	  val_binarized_data.pkl

Uses : read_data_util.py <br>
	   parameters.py <br>
----------------------------------------------- <br>
----------------------------------------------- <br>

#### Train
```
python train.py
```
In : train_binarized_data.pkl <br>
	 val_binarized_data.pkl <br>
	 vocab_word_to_id.pkl <br>

Out : model

Uses : HREI.py and batch_util.py <br>
	   parameters.py <br>
----------------------------------------------- <br>
----------------------------------------------- <br>

#### Binarize test data 
```
python read_data_val_or_test.py test
```
In : dataset/v1/test <br>
	 vocab_word_to_id.pkl
	 
Out : test_targets_pos <br>
	  test_targets_negs <br>
	  test_contexts_text <br>
	  test_contexts_image <br>
	  test_binarized_data.pkl <br>

Uses : read_data_util.py <br>
	   parameters.py	   <br>
----------------------------------------------- <br>
----------------------------------------------- <br>

#### Test
```
python test.py
```
In : test_binarized_data.pkl <br>
	 vocab_id_to_word.pkl <br>
	 model <br>
	 test_targets

Out : test_outputs

Uses : HREI.py and batch_util.py <br>
	   parameters.py <br>
----------------------------------------------- <br>
----------------------------------------------- <br>

* For using image data, comment out the part marked with (*) in batch_util.py