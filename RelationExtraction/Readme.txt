Python Packages used:
---------------------
		NLTK
		Spacy
		networkx
		sklearn
		matplotlib


Dataset - RelationExtraction/dataset/
Report  - RelationExtraction/Report

How to run the model
--------------------
	1) First run Model.py under RelationExtraction Folder - This will create train_df.csv and test_df.csv under RelationExtraction which will have all the features extracted from train and test data respectively.
	2) Run model_run.ipynb to train the model and test on new input sentence 
		-> For new input sentence, update the entity anotated sentence in RelationExtraction/dataset/predict_input.txt file.
                -> After you train the model using model_run.ipynb, run the last cells after "####Run below for predict new input sentenc ###", to predict on new sentence using trained classifier.

