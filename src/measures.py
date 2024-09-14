import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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

def measure_fairness(final_labels, final_preds, gender_arr):
	# Get the final_labels_dev where stats_dev score is 0
	final_labels_dev_female = final_labels[1][gender_arr[1][:, 0] == 0]
	final_labels_test_female = final_labels[2][gender_arr[2][:, 0] == 0]
	final_labels_dev_male = final_labels[1][gender_arr[1][:, 0] == 1]
	final_labels_test_male = final_labels[2][gender_arr[2][:, 0] == 1]
	
	final_pred_dev_female = final_preds[0][gender_arr[1][:, 0] == 0]
	final_pred_dev_male = final_preds[0][gender_arr[1][:, 0] == 1]
	
	final_pred_test_female = final_preds[1][gender_arr[2][:, 0] == 0]
	final_pred_test_male = final_preds[1][gender_arr[2][:, 0] == 1]
	
	female_dev = calculate_ccc(final_pred_dev_female, final_labels_dev_female)
	male_dev = calculate_ccc(final_pred_dev_male, final_labels_dev_male)
	female_test = calculate_ccc(final_pred_test_female, final_labels_test_female)
	male_test = calculate_ccc(final_pred_test_male, final_labels_test_male)
	fairness_dev = female_dev / male_dev
	fairness_test = female_test / male_test
	
	hyperparameters = {
		'fairness-dev': fairness_dev,
		'fairness-test': fairness_test,
		'female-dev': female_dev,
		'male-dev': male_dev,
		'female-test': female_test,
		'male-test': male_test,
	}
	
	return hyperparameters