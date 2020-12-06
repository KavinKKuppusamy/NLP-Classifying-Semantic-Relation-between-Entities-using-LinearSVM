import pandas as pd
from CorpusReader import CorpusReader

import os

file_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(file_dir, 'train_df.csv')
test_file_path = os.path.join(file_dir, 'test_df.csv')

train_cr = CorpusReader('./dataset/semeval_train.txt')
train_features = train_cr.feature_extract()

test_cr = CorpusReader('./dataset/semeval_test.txt')
test_features = test_cr.feature_extract()

train_df = pd.DataFrame([t.__dict__ for t in train_features])
train_df.to_csv(train_file_path, index=False, header=True)

test_df = pd.DataFrame([t.__dict__ for t in test_features])
test_df.to_csv(test_file_path, index=False, header=True)

