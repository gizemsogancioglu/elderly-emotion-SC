import pandas as pd
import numpy as np

def calc_min_from_list(score_list):

	df = pd.DataFrame(eval(score_list), columns=['word', 'pos', 'score'])
	return df['score'].min()


def calc_max_from_list(score_list):

	df = pd.DataFrame(eval(score_list), columns=['word', 'pos', 'score'])
	return df['score'].max()


def calc_mean_from_list(score_list):

	df = pd.DataFrame(eval(score_list), columns=['word', 'pos', 'score'])
	return df['score'].mean()


def calc_sum_from_list(score_list):

	df = pd.DataFrame(eval(score_list), columns=['word', 'pos', 'score'])
	return df['score'].sum()


def calc_num_from_list(score_list):

	return len(eval(score_list))

def calc_num_pos_from_list(score_list):

	df = pd.DataFrame(eval(score_list), columns=['word', 'pos', 'score'])
	df = df[df['score'] > 0]

	return len(df)

def calc_num_neg_from_list(score_list):

	df = pd.DataFrame(eval(score_list), columns=['word', 'pos', 'score'])
	df = df[df['score'] < 0]

	return len(df)


def calc_range_from_list(score_list):

	df = pd.DataFrame(eval(score_list), columns=['word', 'pos', 'score'])
	max_val = df['score'].max()
	min_val = df['score'].min()
	return max_val - min_val


def calc_sum_from_dict(score_dict):

	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass

	total_sum = 0
	if score_dict:
		for k in score_dict.keys():
			total_sum += score_dict[k]

		return total_sum 
	else:
		return np.nan


def calc_min_from_dict(score_dict):
	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass
	if score_dict:
		return np.min(list(score_dict.values()))
	else: 
		return np.nan


def calc_max_from_dict(score_dict):
	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass
	if score_dict:
		return np.max(list(score_dict.values()))
	else: 
		return np.nan


def calc_range_from_dict(score_dict):
	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass
	
	if score_dict:
		values = list(score_dict.values())
		return np.max(values) - np.min(values)
	else: 
		return np.nan


def calc_mean_from_dict(score_dict):
	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass
	if score_dict:
		return np.mean(list(score_dict.values()))
	else: 
		return np.nan


def calc_num_neg_from_dict(score_dict):
	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass
	count = 0 
	if score_dict:
		for val in score_dict.values():
			if val < 0:
				count += 1
		return count
	else: 
		return np.nan


def calc_num_pos_from_dict(score_dict):
	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass
	count = 0 
	if score_dict:
		for val in score_dict.values():
			if val > 0:
				count += 1
		return count
	else: 
		return np.nan
