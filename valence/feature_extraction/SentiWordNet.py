import pandas as pd
import numpy as np
import re
import os
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
#import stats
import valence.feature_extraction.stats as stats

lemmatizer = WordNetLemmatizer()

def load(path):
	path = os.path.join(path, 'SentiWordNet.csv')
	senti_dict = pd.read_csv(path, dtype={'ID': int})
	return senti_dict

def tokenize_text(txt, negation=False):

	# find all the combinations of letters (1+)
	tokens = nltk.word_tokenize(txt)

	if negation:

		i = 0
		token_list = []

		while i < len(tokens):	
			if tokens[i] == 'not' and i != len(tokens)-1:
				token_list.append(tokens[i] + ' ' + tokens[i+1])
				i += 1
			else:
				token_list.append(tokens[i])
			i += 1
		return token_list

	else:
		return tokens


def map_pos(pos):

	if pos == 'j':
		return 'a'
	else:
		return pos


def look_up(token, pos, ref_dict, score_type, flip):

	# Find the right token
	score_list = ref_dict[ref_dict['Word'] == token][score_type].to_list()

	# Find the right Part of Speech
	# score_list = score_list[score_list['POS']==pos]

	if not all(v == 0 for v in score_list):
		score = np.round(np.mean(score_list), 5)
		if flip:
			score = -score
			token = flip + ' ' + token
		return (token, pos, score)

	else:
		return None

def calc_score(tokens, score_type, ref_dict, lemmatizer, pbar):
	'''
	Find the scores associated with each token return a dictionary with words as keys and scores as values  

	Accepts:
		tokens - a list of words
		ref_dict - sentiment dictionary 

	Returns:
		score_dict - a dictionary of scores 
	'''
	scores = []
	for token in tokens:
			
		# find an average score for all different uses of the same word
		pos_tag = token[1][0].lower()
		pos_tag = map_pos(pos_tag)
		token = token[0]

		flip = False
		if len(token.split(' ')) == 2 and token.startswith('no'):
			flip = token.split(' ')[0]
			token = token.split(' ')[1]
			

		new_score = look_up(token, pos_tag, ref_dict, score_type, flip=flip)
		if new_score:
			scores.append(new_score)

		# Repeat for the lemmata
		else:
			try:
				token = lemmatizer.lemmatize(token, pos=pos_tag)
				new_score = look_up(token, pos_tag, ref_dict, score_type, flip=flip)
				if new_score:
					scores.append(new_score)
			except:
				pass

	pbar.update(1)


		# if len(t_score) == 0:

		# 	inf_df = ref_dict[ref_dict['Word'] == t_lemma]
		# 	if len(inf_df) >= 1:
				
		# 		score_list = list(inf_df['PosScore'].values)
		# 		if not all(v == 0 for v in score_list):
		# 			if flip:
		# 				score_list = [-x for x in score_list]
		# 				token = neg_part + ' ' + token

		# 			score_dict[token] = score_list

		# else:
		# 	score_list = list(ref_dict[ref_dict['Word'] == token]['PosScore'].values)

		# 	if not all(v == 0 for v in score_list):
		# 		if flip:
		# 			score_list = [-x for x in score_list]
		# 			token = neg_part + ' ' + token

		# 		score_dict[token] = score_list

	return scores

def calc_pos_score_old(tokens, ref_dict):
	'''
	Find the scores associated with each token return a dictionary with words as keys and scores as values  

	Accepts:
		tokens - a list of words
		ref_dict - sentiment dictionary 

	Returns:
		score_dict - a dictionary of scores 
	'''
	score_dict = {}
	for token in tokens:

		if token in score_dict.keys():
			break

		flip = False

		# Process tokens with negation
		if len(token.split(' ')) == 2:
			neg_part = token.split(' ')[0]
			token = token.split(' ')[1]
			flip = True

		try:
			t_lemma = lemmatizer.lemmatize(token)
		except:
			t_lemma = None
			
		# find an average score for all different uses of the same word
		t_score = ref_dict[ref_dict['Word'] == token]['PosScore']
		
		if len(t_score) == 0:

			inf_df = ref_dict[ref_dict['Word'] == t_lemma]
			if len(inf_df) >= 1:
				
				score_list = list(inf_df['PosScore'].values)
				if not all(v == 0 for v in score_list):
					if flip:
						score_list = [-x for x in score_list]
						token = neg_part + ' ' + token

					score_dict[token] = score_list

		else:
			score_list = list(ref_dict[ref_dict['Word'] == token]['PosScore'].values)

			if not all(v == 0 for v in score_list):
				if flip:
					score_list = [-x for x in score_list]
					token = neg_part + ' ' + token

				score_dict[token] = score_list

	return score_dict


def calc_neg_score(tokens, ref_dict):
	'''
	Find the scores associated with each token return a dictionary with words as keys and scores as values 

	Accepts:
		tokens - a list of words
		ref_dict - sentiment dictionary 

	Returns:
		score_dict - a dictionary of scores 
	'''

	score_dict = {}
	for token in tokens:

		if token in score_dict.keys():
			break

		flip = False

		# Process tokens with negation
		if len(token.split(' ')) == 2:
			neg_part = token.split(' ')[0]
			token = token.split(' ')[1]
			flip = True
		
		try:
			t_lemma = lemmatizer.lemmatize(token)
		except:
			t_lemma = None
	
		# find an average score for all different uses of the same word
		t_score = ref_dict[ref_dict['Word'] == token]['NegScore']
		
		if len(t_score) == 0:

			inf_df = ref_dict[ref_dict['Word'] == t_lemma]
			if len(inf_df) >= 1:
				
				score_list = list(inf_df['NegScore'].values)
				if not all(v == 0 for v in score_list):
					if flip:
						score_list = [-x for x in score_list]
						token = neg_part + ' ' + token

					score_list = [-x for x in score_list]
					score_dict[token] = score_list

		else:
			score_list = list(ref_dict[ref_dict['Word'] == token]['NegScore'].values)

			if not all(v == 0 for v in score_list):
				if flip:
					score_list = [-x for x in score_list]
					token = neg_part + ' ' + token

				score_list = [-x for x in score_list]
				score_dict[token] = score_list

	return score_dict


def calc_score_old(text, function, ref_dict, negation):


	text = text.lower()
	tokens = tokenize_text(text, negation)
	pos_tokens = nltk.pos_tag(tokens)
	score = function(tokens, ref_dict)
	return score


def merge_dicts(dict1, dict2):
	'''
	Arguments:
		dict1 - series with dict of positive values
		dict2 - series with dict of negative values
	'''

	for key in dict2.keys():
		if key not in dict1.keys():
			dict1[key] = []

		for val in dict2[key]:
			dict1[key].append(val)

	return dict1


def merge_scores(pos, neg):
	'''
	Arguments:
		pos - series with dict of positive values
		neg - series with dict of negative values
	'''

	total_scores = []
	assert len(pos) == len(neg)
	for i in range(len(pos)):
		total_scores.append(merge_dicts(pos.iloc[i], neg.iloc[i]))

	return total_scores


def find_sum(score_dict):
	'''
	Arguments:
		score_dict - dictionary with original scores for every word
	Returns:
		dictionary with a single sum score for every word
	'''
	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass
	new_dict = {}
	for key in score_dict.keys():
		new_dict[key] = np.sum(list(score_dict[key]))

	return new_dict


def find_mean(score_dict):
	'''
	Arguments:
		score_dict - dictionary with original scores for every word
	Returns:
		dictionary with a single mean score for every word
	'''
	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass
	new_dict = {}
	for key in score_dict.keys():
		new_dict[key] = np.mean(list(score_dict[key]))

	return new_dict


def find_min(score_dict):
	'''
	Arguments:
		score_dict - dictionary with original scores for every word
	Returns:
		dictionary with a single mean score for every word
	'''
	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass
	new_dict = {}
	for key in score_dict.keys():
		new_dict[key] = np.min(list(score_dict[key]))
	return new_dict


def find_max(score_dict):
	'''
	Arguments:
		score_dict - dictionary with original scores for every word
	Returns:
		dictionary with a single mean score for every word
	'''
	try:
		score_dict = json.loads(score_dict.replace('\'', '\"'))
	except:
		pass
	new_dict = {}
	for key in score_dict.keys():
		new_dict[key] = np.max(list(score_dict[key]))
	return new_dict


def get_tokens(txt_list, negation):
	'''
	Creates tokens of type (word, pos)

	Arguments:
		txt_list - list of strings

	Returns:
		list of tuples   
	'''
	final_token_list = []
	for txt in nltk.sent_tokenize(txt_list):
		print(txt)
		txt = txt.lower()
		txt = txt.replace(r'&quot;', r'"')
		tokens = nltk.word_tokenize(txt)
		tokens = nltk.pos_tag(tokens)
		final_token_list.append(tokens)

		# ignore for now
		if negation:

			i = 0
			token_list = []

			while i < len(tokens):	

				if tokens[i][0] in ['n\'t', 'no', 'not'] and i != len(tokens)-1:
					if tokens[i][0] == 'n\'t':
						n = 'not'
					else:
						n = tokens[i][0]

					new_token = (n + ' ' + tokens[i+1][0], tokens[i+1][1])
					token_list.append(new_token)
					i += 1
				else:
					token_list.append(tokens[i])
				i += 1
			return token_list

	return final_token_list


def get_scores(data, data_column,  s_dict, negation=False):
	print(data.columns)
	tokens = data[data_column].apply(get_tokens, args=(negation,))
	print(tokens)
	lemmatizer = WordNetLemmatizer()

	
	print('Calculating positive scores')
	pbar = tqdm(total=len(data))
	pos = tokens.apply(calc_score, args=('PosScore', s_dict, lemmatizer, pbar))
	pbar.close()

	print('Calculating negative scores')
	pbar = tqdm(total=len(data))
	neg = tokens.apply(calc_score, args=('NegScore', s_dict, lemmatizer, pbar))
	pbar.close()

	data['scores_pos_SentiWordNet'] = pos
	data['scores_neg_SentiWordNet'] = neg
	return data


def get_scores_old(data, s_dict, negation=False):
	'''
	Add a new column to dataframe that holds a dictionary with words as keys and 
	sentiment scores from the dictionary as values

	Accepts:
		data - a series with text
		s_dict - a dataframe with sentiment dictionary words and scores
	Returns:
		a series holding sentiment scores as values of a dictionary 
	'''
	try:
		lemma_check = lemmatizer.lemmatize('check')
	except:
		pass

	print('    Calculating positive scores')
	pos = data['machine_translation'].apply(calc_score, args=(calc_pos_score_old, s_dict, negation))
	print(pos)
	
	print('    Calculating negative scores')
	neg = data['machine_translation'].apply(calc_score, args=(calc_neg_score, s_dict, negation))	
	print(neg)

	print('    Merging')
	both = merge_scores(pos, neg)
	data['scores_SentiWordNet'] = both

	print('    Calculating statistics')
	data['scores_SentiWordNet_min'] = data['scores_SentiWordNet'].apply(find_min)
	data['scores_SentiWordNet_max'] = data['scores_SentiWordNet'].apply(find_max)
	data['scores_SentiWordNet_mean'] = data['scores_SentiWordNet'].apply(find_mean)
	data['scores_SentiWordNet_sum'] = data['scores_SentiWordNet'].apply(find_sum)

	return data


def add_stats_from_list(data):

	for s in ['pos', 'neg']:

		scores = data['scores_{}_SentiWordNet'.format(s)]

		data['min_{}_SentiWordNet'.format(s)] = scores.apply(stats.calc_min_from_list)
		data['max_{}_SentiWordNet'.format(s)] = scores.apply(stats.calc_max_from_list)
		data['range_{}_SentiWordNet'.format(s)] = scores.apply(stats.calc_range_from_list)
		data['sum_{}_SentiWordNet'.format(s)] = scores.apply(stats.calc_sum_from_list)
		data['mean_{}_SentiWordNet'.format(s)] = scores.apply(stats.calc_mean_from_list)
		data['num_{}_SentiWordNet'.format(s)] = scores.apply(stats.calc_num_from_list)

	return data


