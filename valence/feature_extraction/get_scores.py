import pandas as pd
import os
import valence.feature_extraction.SentiWordNet as sentiwordnet

def write_senti_scores(data, split, text_column):
	# Get dictionary scores
	print('SentiWordNet')
	SentiWordNet_scores = sentiwordnet.get_scores(data, text_column, SentiWordNet_dict, negation=True)

	# Remove unnecessary columns
	drop_col = ['id', 'text']

	# PolArt_scores = PolArt_scores.drop(drop_col, axis=1)
	SentiWordNet_scores = SentiWordNet_scores.drop(drop_col, axis=1)
	# print('SentiWordNet\n', SentiWordNet_scores.head())

	# Save the scores
	SentiWordNet_scores.to_csv(os.path.join(ROOT, 'dictionaries', 'personality_traits_dict_scores'+split+".csv"), index=None)


if __name__ == '__main__':
	ROOT = r'/Users/Gizem/PycharmProjects/compare-esc'
	#data_path = os.path.join(ROOT+"/raw_data", 'T03_transcripts_translated.csv')
	#data = pd.read_csv(data_path, delimiter=',')

	transcription_test = pd.DataFrame.from_dict(
		pd.read_pickle(ROOT+"/raw_data/transcription_test.pkl"),
		orient='index', columns=['text'])
	transcription_training = pd.DataFrame.from_dict(
		pd.read_pickle(ROOT+"/raw_data/transcription_training.pkl"),
		orient='index', columns=['text'])
	transcription_validation = pd.DataFrame.from_dict(pd.read_pickle(
		ROOT+"/raw_data/transcription_validation.pkl"),
													  orient='index', columns=['text'])
	transcription_test['id'] = transcription_test.index
	transcription_training['id'] = transcription_training.index
	transcription_validation['id'] = transcription_validation.index

	print('SentiWordNet')
	SentiWordNet_path = os.path.join(ROOT, 'dictionaries')
	SentiWordNet_dict = sentiwordnet.load(SentiWordNet_path)

	write_senti_scores(transcription_training, "training", "text")
	write_senti_scores(transcription_validation, "validation", "text")
	write_senti_scores(transcription_test, "test", "text")
