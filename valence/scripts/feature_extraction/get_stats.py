import pandas as pd
import os

if __name__ == '__main__':

	ROOT = r'/Users/Gizem/PycharmProjects/compare-esc'
	score_path = os.path.join(ROOT, 'dictionaries')

	SentiWordNet_path = os.path.join(score_path, 'personality_traits_dict_scorestraining.csv')
	SentiWordNet_scores = pd.read_csv(SentiWordNet_path)

	# Add statistics 
	print('\nAdding statistics')
	SentiWordNet_stats = SentiWordNet.add_stats_from_list(SentiWordNet_scores)
	drop_col = ['scores_pos_SentiWordNet', 'scores_neg_SentiWordNet']
	SentiWordNet_stats = SentiWordNet_stats.drop(drop_col, axis=1)
	SentiWordNet_stats.to_csv(os.path.join(score_path, 'personality_traits_dict_stats_training.csv'), index=None)


