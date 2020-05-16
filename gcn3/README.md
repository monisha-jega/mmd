## Graph Convolutional Network - Connecting Words and Sentences

### Set Parameters
```
python parameters.py
```
-----------------------------------------------


### Process training data
```
python read_data_train.py
```
In : <br>  train_dir <br>
Out : <br> data_dump_dir/train_targets_pos <br>
	  data_dump_dir/train_targets_negs <br>
	  data_dump_dir/train_contexts_text <br>
	  data_dump_dir/train_contexts_image <br>
	  data_dump_dir/train_binarized_data.pkl <br>
	  data_dump_dir/vocab_word_to_id.pkl <br>
	  data_dump_dir/vocab_id_to_word.pkl <br>
-----------------------------------------------


### Get pretrained word embeddings for training vocabulary 
```
python embeddings.py
```
In : <br>data_dump_dir/vocab_id_to_word.pkl <br>
	 embed_file <br>
Out : <br>data_dump_dir/vocab_embeddings.pkl <br>
-----------------------------------------------


### Process validation data
```
python read_data_val_or_test.py val
```
In : <br>val_dir <br>
	 data_dump_dir/vocab_word_to_id.pkl	 <br> 
Out :<br>data_dump_dir/val_targets_pos <br>
	  data_dump_dir/val__targets_negs <br>
	  data_dump_dir/val_contexts_text <br>
	  data_dump_dir/val_contexts_image <br>
	  data_dump_dir/val_binarized_data.pkl <br>
-----------------------------------------------


### Train
```
python train.py
```
In : <br>data_dump_dir/train_binarized_data.pkl <br>
	 data_dump_dir/val_binarized_data.pkl <br>
	 data_dump_dir/vocab_word_to_id.pkl <br>
Out : <br>model_dump_dir/* <br>
-----------------------------------------------


### Process test data 
```
python read_data_val_or_test.py test
```
In : <br>test_dir <br>
	 data_dump_dir/vocab_word_to_id.pkl	 <br> 
Out : <br>data_dump_dir/test_targets_pos <br>
	  data_dump_dir/test_targets_negs <br>
	  data_dump_dir/test_contexts_text <br>
	  data_dump_dir/test_contexts_image <br>
	  data_dump_dir/test_binarized_data.pkl <br>
-----------------------------------------------


### Test
```
python test.py
```
In : <br> data_dump_dir/test_binarized_data.pkl <br>
	 data_dump_dir/vocab_id_to_word.pkl <br>
	 model_dump_dir <br>
	 data_dump_dir/test_targets/* <br>
Out : <br> data_dump_dir/test_outputs <br>
-----------------------------------------------
