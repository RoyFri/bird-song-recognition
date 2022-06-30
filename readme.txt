code file guide:
---------------

1.	main.py:
		the 'main' function:
			- creating data files of processed data
			  (using 'data_processing' function)
			- extracting the data files
			- praparing the dataset
			  (using 'prepare_data' function)
			- training, testing and evaluating
			  (using 'train' function)

2.	data_processing_Bulbul5_lite.py:
		'data_processing' function:
			- extracting labels
			  (using 'read_label_file' function)
			- division into frams, creating Mel-spectrograms
			- determining labels
			  (using 'get_bin_label' function)
			- noise cleaning
			  (using 'blobRemove' and 'medclip')

3.	parsing_functions.py:
		'read_label_file' function

4.	audio_functions.py:
		'get_bin_label' function

5.	training_functions.py:
		'prepare_data' function:
			- converting to numpy array
			- ajusting dimentions
			- train/test split
		'train' function:
			- ploting random examples from the data set
			- building and and training a CNN model
			- ploting loss and accuracy progress graphs

6.	Utils.py (code file by AM and YL):
		'blobRemove' function
		'medclip' function



Thanks,

Roy.
