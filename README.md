# NLP : Classifying Semantic Relation between Entities using LinearSVM
It presents a computational linguistics approach for extraction of entity relations. First, data preprocessing and cleaning is performed on our dataset. After preprocessing, relevant features are extracted for each of the entities. These features include word tokens, word prefixes, shortest paths between root and entities, hypernyms, part of speech tags, and more relevant information that helps the machine better understand the sentence. A feature vector extraction method is used to create our input to our machine learning model. We apply a Linear Support Vector Machine (SVM) to build our model. Finally, we test this model using unseen entities and evaluate our model based on metrics such as f-score, precision, recall, and accuracy. We found our model predicted relations with an accuracy of 75.5% and F1 score of 77%.

 
Features:
 
 We trained our model with 21 features that belong to categories such as lexical features, wordnet features like hypernyms, and information about the sentence’s dependency parse tree. The following explains each feature.


## Lexical Features

1. Entities: We extracted the annotated nominals from the given sentences. If the entity is composed of more than one word, we parse the noun phrase and take the head word. (eg., San Jose University -> University).
2. Entity Tag: Using NLTK POS tagger, we attached Part of Speech Tag for each entity.
3. Words between nominals: Using the words between the nominals is a highly relevant feature regarding a relation in a sentence.
4. Prefix of Length 5: Using the prefixes of length 5 for the words between the nominals provides a kind of stemming (produced -> produ, caused -> cause). This stemming effect
reduces the complexity of our features and generalizes the words, producing a better generalization of other potential similar sentences.
5. Distance between nominals: The number of words between the two entities mentioned. This feature will mainly help the model predict the classes such as Product-Producer and Entity-Origin because these relations usually would not have intervening tokens. For example, organ builder, Chestnut flour.
6. POS Tag Between: We extract a coarse-grained part of the speech sequence for the words between the nominals. This feature is produced by extracting the first letter of each token’s POS tag. This feature is included mainly to find the class such as Member-Collection which usually invoke prepositional phrases such as : of, in the and its corresponding POS TAG Between I and ‘I_D’ respectively.
7. Words before E1, After E2, Before E1, and Single word after E2: Extracting all the tokens before the e1 entity, after the e2 entity, and all the tokens before e1 and a single token after e2.


## WordNet Features
1. E1 and E2 Hypernyms: Using the wordnet package, we extract hypernyms for both e1 and e2.
2. Lowest Common Hypernym: We utilize Wordnet's lowest_common_hypernyms method to extract the lowest common hypernym between the two named entities. It will perform a search of hypernyms until there is a potential match between the two.
3. Wu-Palmer Similarity: It calculates relatedness by considering the depths of the two synsets in the WordNet taxonomies, along with the depth of the LCS (Least Common Subsumer). The score can be 0 < score <= 1. This feature mainly helps in finding how similar are the given two nominals.


## Dependency Parsing
Motivated from the paper (A Shortest Path Dependency Kernel for Relation Extraction), If e1 and e2 are two entities mentioned in the same sentence such that they are observed to be in a relationship R, the contribution of the sentence dependency graph to establishing the relationship R(e1, e2) is almost exclusively concentrated in the shortest path between e1 and e2 in the undirected version of the dependency graph.

1) Dependency Path length1: Using Spacy and networkx, produced the dependency parse tree and find the shortest path dependency between two entities. This feature mainly helps in finding the parent node which connects the two entities.
2) Dependency Path length2: Encode the complete path between e1 and e2 including dependency features.
3) Connecting Path: Finding the tokens between the two entities from the path extracted by shortest path dependency.
4) SDP Root Node Lemma: This feature will extract the lemmatized head word of the phrase/token connecting two entities in the two entities.
5) Shortest Path Length: This feature will reveal that the shortest path length given the dependency parse.
Figure 5: Shortest Path between fire and fuel and length is 4
6) Root Word Location: This feature will give the location (BEFORE, BETWEEN, AFTER) of the root word between two entities, found from dependency parse



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


