#Process
python read_data_train.py
python embeddings.py
python read_data_val_or_test.py val

#Train
python train.py

#Test
python read_data_val_or_test.py test
python test.py