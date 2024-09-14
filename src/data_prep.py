# Loading data
import collections
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def z_normed(features):
	scaler = StandardScaler()
	scaler.fit(features['train'])
	return scaler.transform(features['train']), scaler.transform(features['dev']), scaler.transform(features['test'])

def get_features():
	features = collections.defaultdict(list)
	for split in ['train', 'dev', 'test']:
		with open(f'../features/SBERT_LIWC_STATS_{split}.pkl', "rb") as fIn:
			stored_data = pickle.load(fIn)
			features[split] = stored_data['SBERT']
	return z_normed(features)


def convert_data(X, y, gender_arr):
	features = []
	gender = []
	concept_labels = []
	final_labels = []
	# Convert DataFrames to PyTorch Tensors
	for i in [0, 1, 2]:
		features.append( torch.tensor(X[i].values, dtype=torch.float32))
		gender.append(torch.tensor(gender_arr[i].values, dtype=torch.float32))
	
	for i in [0, 1]:
		concept_labels.append(torch.tensor(y[i].values, dtype=torch.float32))
		final_labels.append(torch.tensor(np.sum(y[i].values, axis=1, keepdims=True),
	                            dtype=torch.float32))
	
	# # Convert Test Set Data to PyTorch Tensors
	final_labels.append(torch.tensor(y[2].values, dtype=torch.float32).unsqueeze(1))
	
	return features, gender, concept_labels, final_labels
	

def prepare_data(labels):
	X = []
	y = []
	features = get_features()
	for data in features:
		X.append(pd.DataFrame(data))
	
	test_data = pd.read_csv('../data/test_split.csv', sep=',')
	symptoms_data = pd.read_csv('../data/Detailed_PHQ8_Labels_mapped.csv', sep=';')
	metadata = pd.read_csv("../data/metadata.csv")
	symptoms = pd.merge(symptoms_data, metadata, on='Participant_ID')
	from sklearn.preprocessing import LabelEncoder
	label_encoder = LabelEncoder()
	symptoms['Gender'] = label_encoder.fit_transform(symptoms['Gender'])
	gender_arr = []
	gender_arr.append(pd.DataFrame(symptoms[56:]['Gender']))
	gender_arr.append(pd.DataFrame(symptoms[:56]['Gender']))
	gender_arr.append(pd.DataFrame(label_encoder.transform(test_data['Gender'])))
	
	all = symptoms_data[labels]
	y.append(all[56:])
	y.append(all[:56])
	y.append(test_data.loc[:, 'PHQ_Score'])
	return X, y, gender_arr

