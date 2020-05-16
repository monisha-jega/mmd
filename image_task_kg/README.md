## Image Prediction With Knowledge Graph Integration


### Set Parameters
```
python parameters.py
```

##### Notes
* For using image data, comment out the part marked with (*) in batch_util.py
* Make sure all lines are uncommented in run.sh, running run.sh will run all the following steps sequentially
* Remove [:5] from read_data_train.py or read_data_val_or_test.py for complete training <br>



### Process data into binary format
```
python read_data_train.py
```
In : <br>
dataset/v1/train 

Out : <br>
train_targets_pos <br>
	  train_targets_negs <br>
	  train_contexts_text <br>
	  train_contexts_image <br>
	  train_binarized_data.pkl <br>
	  vocab_word_to_id.pkl <br>
	  vocab_id_to_word.pkl <br>

Uses : <br> read_data_util.py <br>
	   parameters.py <br>
----------------------------------------------- <br>
----------------------------------------------- <br>

### Get word embeddings for training vocabulary 
```
python embeddings.py
```
In : <br> vocab_id_to_word.pkl <br>
	 GoogleNews-vectors-negative300.bin

Out : <br> vocab_embeddings.pkl

Uses : <br> parameters.py <br>

### Binarize validation data 
```
python read_data_val_or_test.py val
```
In : <br> dataset/v1/val <br>
	 vocab_word_to_id.pkl
	 
Out : <br> val_targets_pos <br>
	  val__targets_negs <br>
	  val_contexts_text <br>
	  val_contexts_image <br>
	  val_binarized_data.pkl

Uses : <br> read_data_util.py <br>
	   parameters.py <br>


### Train
```
python train.py
```
In : <br> train_binarized_data.pkl <br>
	 val_binarized_data.pkl <br>
	 vocab_word_to_id.pkl <br>

Out : <br> model

Uses : <br> HREI.py and batch_util.py <br>


### Binarize test data 
```
python read_data_val_or_test.py test
```
In : <br> dataset/v1/test <br>
	 vocab_word_to_id.pkl
	 
Out : <br> test_targets_pos <br>
	  test_targets_negs <br>
	  test_contexts_text <br>
	  test_contexts_image <br>
	  test_binarized_data.pkl <br>

Uses : <br> read_data_util.py <br>
	   parameters.py	   <br>


### Test
```
python test.py
```
In : <br> test_binarized_data.pkl <br>
	 vocab_id_to_word.pkl <br>
	 model <br>
	 test_targets

Out : <br> test_outputs

Uses : <br> HREI.py and batch_util.py <br>
	   parameters.py <br>
	   

* For using image data, comment out the part marked with (*) in batch_util.py