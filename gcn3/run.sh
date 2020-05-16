#Process
python -u read_data_train.py
python -u embeddings.py
python -u read_data_val_or_test.py val

#Train
python -u train.py

#Test
#python -u read_data_val_or_test.py test
#python -u test.py