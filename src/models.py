import collections
import copy

import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch
from measures import CCCLoss, calculate_ccc, calculate_rmse, measure_fairness
from data_prep import prepare_data, convert_data

# Define the concept prediction model for sequential pipeline.
class ConceptPredictionModel(nn.Module):
	def __init__(self, input_dim, concept_dim):
		super(ConceptPredictionModel, self).__init__()
		# MLP to predict concepts from input features
		self.concept_layer = nn.Sequential(
			nn.Linear(input_dim, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, concept_dim),
		)
	
	def forward(self, x):
		# Predict concepts from input features
		concept_preds = self.concept_layer(x)
		return concept_preds
	
	def get_loss(self, y_pred, y):
		loss = ccc_criterion(y_pred, y)
		return loss


# Define the final prediction model for sequential pipeline.
class FinalPredictionModel(nn.Module):
	def __init__(self, concept_dim, output_dim):
		super(FinalPredictionModel, self).__init__()
		# MLP to predict the final output from concepts
		self.output_layer = nn.Sequential(
			nn.Linear(concept_dim, 100),
			nn.Linear(100, output_dim),
		)
	
	def forward(self, concepts):
		# Predict final output from predicted concepts
		final_output = self.output_layer(concepts)
		return final_output
	
	# initializing weights close to 1/n, where n is the number of concepts
	def initialize_weights(model):
		with torch.no_grad():
			for param in model.parameters():
				nn.init.constant_(param, 1.0 / 8)  # Initialize weights close to equal contribution
	
	def get_loss(self, y_pred, y):
		loss = ccc_criterion(y_pred, y)
		return loss


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
			nn.Linear(8, 100),
			nn.ReLU(),
			nn.Linear(100, output_dim),
		)
	
	def forward(self, x):
		return self.mlp(x)
	
	def get_loss(self, y_pred, y):
		loss = ccc_criterion(y_pred, y)
		return loss


class JointBottleneckModel(nn.Module):
	def __init__(self, input_dim, concept_dim, len_stats, output_dim):
		super(JointBottleneckModel, self).__init__()
		self.concept_layer = nn.Sequential(
			nn.Linear(input_dim, 100),  # Increased the number of neurons
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, 100),
			nn.ReLU(),
			nn.Linear(100, concept_dim),
		)
		self.task_layer = nn.Sequential(
			nn.Linear(concept_dim, 100),
			nn.ReLU(),
			nn.Linear(100, output_dim),
		)
	
	def forward(self, x, stats):
		concept_preds = self.concept_layer(x)  # Predict concepts as continuous values
		final_pred = self.task_layer(concept_preds)  # Predict final output based on concepts
		return final_pred, concept_preds
	
	def initialize_weights(model):
		with torch.no_grad():
			for name, param in model.named_parameters():
				if 'task_layer' in name:
					nn.init.constant_(param, 1.0 / model.task_layer[0].in_features)  # Equal contribution
	
	def get_loss(self, final_pred, final_label, concept_pred, concept_label, weight):
		concept_loss = ccc_criterion(concept_pred, concept_label)
		# Compute loss for final prediction using CCC
		final_loss = ccc_criterion(final_pred, final_label)
		loss = ((1 - weight) * concept_loss) + ((weight) * final_loss)
		return loss

def compute_concept_ccc(concept_pred, concept_label):
	concept_cccs = []
	for i in range(8):
		concept_ccc = ccc_criterion(concept_pred[:, i], concept_label[:, i])
		concept_cccs.append(concept_ccc)
	concept_cccs_tensor = torch.stack(concept_cccs)
	average_ccc = torch.mean(concept_cccs_tensor)
	return average_ccc

def set_random_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True)


def add_concept_scores(best_hyperparameters, concept_preds_dev):
	ccc_scores = []
	for i in range(concept_preds_dev.shape[1]):  # Iterate over each concept
		ccc = calculate_ccc(concept_preds_dev[:, i], concept_labels[1][:, i])
		ccc_scores.append(ccc)
	for i, ccc in enumerate(ccc_scores):
		best_hyperparameters[f'concept_ccc_symptom{i}'] = ccc
	return best_hyperparameters


def experiment(features, gender, concept_labels, final_labels, lam, type='joint'):
	# Instantiate the model
	if type == 'standard':
		model = StandardModel(input_dim=input_dim, output_dim=output_dim)
	elif type == 'sequential-1':
		model = ConceptPredictionModel(input_dim, concept_dim)
	elif type == 'sequential-2':
		model = FinalPredictionModel(concept_dim, output_dim)
		model.initialize_weights()
	else:
		model = JointBottleneckModel(input_dim, concept_dim, len_stats, output_dim)
		model.initialize_weights()
	
	best_hyperparameters = {}
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
	
	num_epochs = 100
	best_ccc = 0
	patience_counter = 0
	best_loss = 1
	
	best_model_state = 0
	for epoch in range(num_epochs):
		model.train()
		optimizer.zero_grad()
		
		if type == 'standard':
			final_pred = model(features[0])
			loss = model.get_loss(final_pred, final_labels[0])
		elif type == 'sequential-1':
			concept_predictions = model(features[0])
			loss = model.get_loss(concept_predictions, concept_labels[0])
		elif type == 'sequential-2':
			# features are concept predictions.
			final_predictions = model(features[0])
			loss = model.get_loss(final_predictions, final_labels[0])
		else:
			final_predictions, concept_predictions = model(features[0], gender)
			loss = model.get_loss(final_predictions, final_labels[0], concept_predictions, concept_labels[0], lam)
		
		loss.backward()
		optimizer.step()
		model.eval()
		
		with torch.no_grad():
			if type == 'standard' or (type == 'sequential-1') or (type == 'sequential-2'):
				final_pred_dev = model(features[1])
				final_pred_test = model(features[2])
				if type == 'sequential-2':
					concept_preds_dev = features[1]
			
			else:
				final_pred_dev, concept_preds_dev = model(features[1], gender[1])
				final_pred_test, concept_preds_test = model(features[2], gender[2])
				if type == 'symptom':
					final_pred_dev = torch.sum(concept_preds_dev, dim=1).unsqueeze(1)
					final_pred_test = torch.sum(concept_preds_test, dim=1).unsqueeze(1)
			
			final_score = calculate_ccc(final_pred_dev, final_labels[1])
		
			#if type == 'joint':
			#	final_score = 1 - (model.get_loss(final_pred_dev, final_labels[1], concept_preds_dev, concept_labels[1], lam)).item()
			scheduler.step(1-final_score)
			
			if final_score > best_ccc:
				best_ccc = final_score
				patience_counter = 0  # Reset the patience counter
				best_hyperparameters.update(measure_fairness(final_labels, [final_pred_dev, final_pred_test], gender))
				if (type == 'joint') or (type == 'symptom') or (type == 'sequential-2'):
					best_hyperparameters = add_concept_scores(best_hyperparameters, concept_preds_dev)
				
				hyperparameters = {
					'learning_rate': optimizer.param_groups[0]['lr'],
					'lambda': lam,
					'dev_ccc': calculate_ccc(final_pred_dev, final_labels[1]),
					'test_ccc': calculate_ccc(final_pred_test, final_labels[2]) if len(final_labels) == 3 else None,
					'dev_rmse': calculate_rmse(final_pred_dev, final_labels[1]).item(),
					'test_rmse': calculate_rmse(final_pred_test, final_labels[2]).item() if len(
						final_labels) == 3 else None,
					'epoch': epoch,
					'loss': 1 - final_score
				}
				best_hyperparameters.update(hyperparameters)
				best_model_state = copy.deepcopy(model.state_dict())
			else:
				patience_counter += 1  # Increment the patience counter
		if patience_counter >= 10:
			print("Early stopping due to no improvement in CCC score.")
			break
	
	return best_hyperparameters, best_model_state


symptoms = ['PHQ_8NoInterest', 'PHQ_8Depressed', 'PHQ_8Sleep', 'PHQ_8Tired', 'PHQ_8Appetite',
            'PHQ_8Failure', 'PHQ_8Concentrating', 'PHQ_8Moving']
scores = ['loss', 'dev_ccc', 'test_ccc', 'dev_rmse', 'test_rmse', 'fairness-dev',
          'fairness-test', 'female-dev', 'male-dev', 'female-test', 'male-test']
symptoms_scores = ['concept_ccc_symptom0', 'concept_ccc_symptom1',
                   'concept_ccc_symptom2', 'concept_ccc_symptom3', 'concept_ccc_symptom4',
                   'concept_ccc_symptom5', 'concept_ccc_symptom6',
                   'concept_ccc_symptom7']
depression_label = ['PHQ_8Total']


def get_preds_freezed_model(model_state):
	concept_model = ConceptPredictionModel(input_dim, concept_dim)
	# Load the best concept model state and freeze its weights
	concept_model.load_state_dict(model_state)
	concept_model.eval()  # Set concept model to evaluation mode
	for param in concept_model.parameters():
		param.requires_grad = False  # Freeze the concept model weights
	
	# Generate frozen concept predictions after loading the best state
	with torch.no_grad():
		concept_train_preds = concept_model(features[0])
		concept_dev_preds = concept_model(features[1])
		concept_test_preds = concept_model(features[2])
	
	print("dev concept ccc: ", calculate_ccc(concept_dev_preds, concept_labels[1]))
	return [concept_train_preds, concept_dev_preds, concept_test_preds]


def seq_experiment():
	X, y, gender_arr = prepare_data(labels=symptoms)
	features, gender, concept_labels, final_labels = convert_data(X, y, gender_arr)
	
	results = []
	for seed in [0, 1, 2, 3, 4]:
		set_random_seed(seed)
		method_type = 'sequential-1'
		hyperparams, model_state = experiment(features, gender, concept_labels, concept_labels, 0, type=method_type)
		
		method_type = 'sequential-2'
		concept_preds = get_preds_freezed_model(model_state)
		hyperparams, model_state = experiment(concept_preds, gender, concept_labels, final_labels, 0, type=method_type)
		
		results.append(hyperparams)
	df = pd.DataFrame(results)
	
	average_result = df[scores + symptoms_scores].mean()  # Averaging only 'key1' and 'key2'
	
	print(average_result)


if __name__ == "__main__":
	X, y, gender_arr = prepare_data(labels=symptoms)
	features, gender, concept_labels, final_labels = convert_data(X, y, gender_arr)
	
	ccc_criterion = CCCLoss()  # CCC for optimization
	input_dim = X[0].shape[1]  # Input feature dimension from the dataframe
	concept_dim = y[0].shape[1]  # Number of concepts from the dataframe
	output_dim = 1  # Final label is a single continuous value (sum of symptoms)
	len_stats = 1
	method_type = 'joint'
	
	if method_type == 'sequential':
		seq_experiment()
	elif (method_type == 'standard') or (method_type == 'symptom') or (method_type == 'joint'):
		res = collections.defaultdict(list)
		for lam in [0, 0.5, 0.6, 0.7, 0.8, 0.9]:
			results = []
			for seed in [0, 1, 2, 3, 4]:
				set_random_seed(seed)
				hyperparams, model_state = experiment(features, gender, concept_labels, final_labels, lam,
				                                      type=method_type)
				results.append(hyperparams)
			
			df = pd.DataFrame(results)
			if method_type == 'standard':
				average_result = df[scores].mean()  # Averaging only 'key1' and 'key2'
			else:
				average_result = df[scores + symptoms_scores].mean()  # Averaging only 'key1' and 'key2'
			res[str(lam)] = average_result
			if not method_type == 'joint':
				break
		print(res)
	else:
		print("ERROR: method type is unknown...")
