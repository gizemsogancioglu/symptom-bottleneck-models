import pickle
import numpy as np
import pandas as pd

import torch.nn.functional as F
import torch.optim as optim

import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler

class CCCLoss(nn.Module):
	def __init__(self):
		super(CCCLoss, self).__init__()
	
	def forward(self, y_pred, y_true):
		y_true_mean = torch.mean(y_true)
		y_pred_mean = torch.mean(y_pred)
		covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
		y_true_var = torch.mean((y_true - y_true_mean) ** 2)
		y_pred_var = torch.mean((y_pred - y_pred_mean) ** 2)
		
		ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
		return 1 - ccc  # We return 1 - ccc because we want to maximize CCC, but PyTorch optimizers minimize the loss.

def z_normed(train, dev, test):
	scaler = StandardScaler()
	scaler.fit(train)
	return scaler.transform(train), scaler.transform(dev), scaler.transform(test)

# Loading data
with open('../features/SBERT_LIWC_STATS_train.pkl', "rb") as fIn:
	stored_data = pickle.load(fIn)
	SBERT_train = stored_data['SBERT']
	LIWC_train = stored_data['LIWC']
	STATS_train = stored_data['STATS']
	

with open('../features/SBERT_LIWC_STATS_dev.pkl', "rb") as fIn:
	stored_data = pickle.load(fIn)
	SBERT_dev = stored_data['SBERT']
	LIWC_dev = stored_data['LIWC']
	STATS_dev = stored_data['STATS']

with open('../features/SBERT_LIWC_STATS_test.pkl', "rb") as fIn:
	stored_data = pickle.load(fIn)
	SBERT_test = stored_data['SBERT']
	LIWC_test = stored_data['LIWC']
	STATS_test = stored_data['STATS']

STATS_train, STATS_dev, STATS_test = z_normed(STATS_train, STATS_dev, STATS_test)
LIWC_train, LIWC_dev, LIWC_test = z_normed(LIWC_train, LIWC_dev, LIWC_test)
SBERT_train, SBERT_dev, SBERT_test = z_normed(SBERT_train, SBERT_dev, SBERT_test)

def calculate_ccc(y_pred, y_true):
	y_true_mean = torch.mean(y_true)
	y_pred_mean = torch.mean(y_pred)
	covariance = torch.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
	y_true_var = torch.mean((y_true - y_true_mean) ** 2)
	y_pred_var = torch.mean((y_pred - y_pred_mean) ** 2)
	
	ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
	return ccc.item()

def calculate_rmse(y_pred, y_true):
	# Assuming y_pred and y_true are torch tensors
	y_pred = y_pred.view(-1)
	y_true = y_true.view(-1)
	
	mse = F.mse_loss(y_pred, y_true)
	rmse = torch.sqrt(mse)
	return rmse
def prepare_data(labels):
	X = []
	y = []
	X_train = SBERT_train
	X_dev = SBERT_dev
	X_test = SBERT_test
	
	for data in [X_train, X_dev, X_test]:
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


class StandardModel(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(StandardModel, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(input_dim, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 8),
			nn.ReLU(),
			nn.Linear(8, output_dim),
			
		)
	
	def forward(self, x):
		return self.mlp(x)
class JointBottleneckModel(nn.Module):
	def __init__(self, input_dim, concept_dim, len_stats, output_dim):
		super(JointBottleneckModel, self).__init__()
		# Concept layer: Linear regression layer for predicting concepts
		self.concept_layer = nn.Sequential(
			nn.Linear(input_dim, 100),  # Increased the number of neurons
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, concept_dim),
		
		)
		#Task layer: Linear regression layer for final prediction based on concepts
		self.task_layer = nn.Sequential(
			nn.Linear(concept_dim, output_dim),  # Increased the number of neurons
		)

	def forward(self, x, stats):
		concept_preds = self.concept_layer(x)  # Predict concepts as continuous values
		# Concatenate concept predictions and stats
		#concatenated_input = torch.cat((concept_preds, stats), dim=1)  # Concatenate along the feature dimension
		final_pred = self.task_layer(concept_preds)  # Predict final output based on concepts
		return final_pred, concept_preds

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def experiment(X, y, lam, type='joint'):
	ccc_criterion = CCCLoss()  # CCC for optimization
	input_dim = X[0].shape[1]  # Input feature dimension from the dataframe
	concept_dim = y[0].shape[1]  # Number of concepts from the dataframe
	output_dim = 1  # Final label is a single continuous value (sum of symptoms)
	# # Define loss and optimizer
	len_stats = 1
	# Instantiate the model
	if type == 'standard':
		model = StandardModel(input_dim=input_dim, output_dim=output_dim)
	else:
		model = JointBottleneckModel(input_dim, concept_dim, len_stats, output_dim)
	
	best_hyperparameters = {}
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
	
	# Convert DataFrames to PyTorch Tensors
	data = torch.tensor(X[0].values, dtype=torch.float32)
	stats = torch.tensor(gender_arr[0].values, dtype=torch.float32)
	
	concept_labels = torch.tensor(y[0].values, dtype=torch.float32)  # Regression labels for concepts
	final_labels = torch.tensor(np.sum(y[0].values, axis=1, keepdims=True),
	                            dtype=torch.float32)  # Regression label for final output
	
	# Convert Development Set Data to PyTorch Tensors
	data_dev = torch.tensor(X[1].values, dtype=torch.float32)
	stats_dev = torch.tensor(gender_arr[1].values, dtype=torch.float32)
	concept_labels_dev = torch.tensor(y[1].values, dtype=torch.float32)
	final_labels_dev = torch.tensor(np.sum(y[1].values, axis=1, keepdims=True), dtype=torch.float32)
	
	# # Convert Test Set Data to PyTorch Tensors
	data_test = torch.tensor(X[2].values, dtype=torch.float32)
	final_labels_test = torch.tensor(y[2].values, dtype=torch.float32).unsqueeze(1)
	stats_test = torch.tensor(gender_arr[2].values, dtype=torch.float32)
	
	# Get the final_labels_dev where stats_dev score is 0
	final_labels_dev_female = final_labels_dev[stats_dev[:, 0] == 0]
	final_labels_test_female = final_labels_test[stats_test[:, 0] == 0]
	final_labels_dev_male = final_labels_dev[stats_dev[:, 0] == 1]
	final_labels_test_male = final_labels_test[stats_test[:, 0] == 1]
	
	num_epochs = 200
	best_ccc = 0
	patience_counter = 0
	
	for epoch in range(num_epochs):
		model.train()
		optimizer.zero_grad()
		
		if type == 'standard':
			final_pred = model(data)
			loss = ccc_criterion(final_pred, final_labels)
		else:
			final_pred, concept_preds = model(data, stats)
			concept_loss = ccc_criterion(concept_preds, concept_labels)
			concept_sum = torch.sum(concept_preds, dim=1).unsqueeze(1)
			# Compute loss for final prediction using CCC
			final_loss = ccc_criterion(final_pred, final_labels)
			# Combine loss
			loss = ((1 - lam) * concept_loss) + ((lam) * final_loss)
		
		loss.backward()
		optimizer.step()
		
		model.eval()
		with torch.no_grad():
			if type == 'standard':
				final_pred_dev = model(data_dev)
				final_pred_test = model(data_test)
			else:
				final_pred_dev, concept_preds_dev = model(data_dev, stats_dev)
				final_pred_test, concept_preds_test = model(data_test, stats_test)
				if type == 'sequential':
					final_pred_dev = torch.sum(concept_preds_dev, dim=1).unsqueeze(1)
					final_pred_test = torch.sum(concept_preds_test, dim=1).unsqueeze(1)
			
			final_pred_dev_female = final_pred_dev[stats_dev[:, 0] == 0]
			final_pred_dev_male = final_pred_dev[stats_dev[:, 0] == 1]

			final_pred_test_female = final_pred_test[stats_test[:, 0] == 0]
			final_pred_test_male = final_pred_test[stats_test[:, 0] == 1]
			final_ccc = calculate_ccc(final_pred_dev, final_labels_dev)
			final_rmse = calculate_rmse(final_pred_dev, final_labels_dev)
			
			final_ccc_test = calculate_ccc(final_pred_test, final_labels_test)
			final_rmse_test = calculate_rmse(final_pred_test, final_labels_test)
			
			# print(f"Epoch {epoch + 1}, CCC on development set: {calculate_ccc(final_pred_dev, final_labels_dev)}")
			# print(f"Epoch {epoch + 1}, CCC on test set: {calculate_ccc(final_pred_test, final_labels_test)}")
			scheduler.step(-final_ccc)
			
			if final_ccc > best_ccc:
				best_ccc = final_ccc
				patience_counter = 0  # Reset the patience counter
				female_dev= calculate_ccc(final_pred_dev_female, final_labels_dev_female)
				male_dev = calculate_ccc(final_pred_dev_male, final_labels_dev_male)
				female_test = calculate_ccc(final_pred_test_female, final_labels_test_female)
				male_test = calculate_ccc(final_pred_test_male, final_labels_test_male)
				fairness_dev= female_dev / male_dev
				ccc_scores = []
				if (type == 'joint') or (type == 'sequential'):
					for i in range(concept_preds_dev.shape[1]):  # Iterate over each concept
						ccc = calculate_ccc(concept_preds_dev[:, i], concept_labels_dev[:, i])
						ccc_scores.append(ccc)
						
					for i, ccc in enumerate(ccc_scores):
						best_hyperparameters[f'concept_ccc_symptom{i}'] = ccc
				
				fairness_test = female_test / male_test
				hyperparameters = {
					'learning_rate': optimizer.param_groups[0]['lr'],
					'weight_decay': 1e-4,
					'lambda': lam,
					'dev_ccc': final_ccc,
					'test_ccc': final_ccc_test,
					'dev_rmse': final_rmse.item(),
					'test_rmse': final_rmse_test.item(),
					'epoch': epoch,
					'fairness-dev': fairness_dev,
					'fairness-test': fairness_test,
					'female-dev': female_dev,
					'male-dev': male_dev,
					'female-test': female_test,
					'male-test': male_test,
				}
				best_hyperparameters.update(hyperparameters)
				
			else:
				patience_counter += 1  # Increment the patience counter
		#if patience_counter >= 10:
			#print("Early stopping due to no improvement in CCC score.")
		#	break
	#print(f"Best Hyperparameters: {best_hyperparameters}")
	return best_hyperparameters


if __name__ == "__main__":
	symptoms = ['PHQ_8NoInterest', 'PHQ_8Depressed', 'PHQ_8Sleep', 'PHQ_8Tired', 'PHQ_8Appetite',
	            'PHQ_8Failure', 'PHQ_8Concentrating', 'PHQ_8Moving']
	depression_label = ['PHQ_8Total']
	X, y, gender_arr = prepare_data(labels=symptoms)
	method_type = 'standard'
	#0.6, 0.7, 0.8, 0.9,
	# sequential: lambda = 0
	for lam in [0.7]:
		results = []
		for seed in [0, 1, 2, 3, 4]:
			set_random_seed(seed)
			results.append(experiment(X, y, lam, type='standard'))
		
		df = pd.DataFrame(results)
		if method_type == 'standard':
			average_result = df[
			['dev_ccc', 'test_ccc', 'dev_rmse', 'test_rmse', 'fairness-dev', 'fairness-test', 'female-dev', 'male-dev',
			 'female-test', 'male-test']].mean()  # Averaging only 'key1' and 'key2'
		else:
			average_result = df[['dev_ccc', 'test_ccc', 'dev_rmse', 'test_rmse', 'fairness-dev', 'fairness-test', 'female-dev', 'male-dev',
		                     'female-test', 'male-test', 'concept_ccc_symptom0', 'concept_ccc_symptom1',
		                     'concept_ccc_symptom2', 'concept_ccc_symptom3', 'concept_ccc_symptom4',
		                     'concept_ccc_symptom5', 'concept_ccc_symptom6', 'concept_ccc_symptom7']].mean()  # Averaging only 'key1' and 'key2'
		
		#std_result = df[['dev_ccc', 'test_ccc', 'dev_rmse', 'test_rmse']].std()  # Averaging only 'key1' and 'key2'
		#print(f"lamda value {lam}: {average_result}, {std_result}")
		print(f"lamda value {lam}: {average_result}")
