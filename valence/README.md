# elderly-emotion-SC

## Description
This project contains the scripts for our entry in the Elderly Emotion Sub-Challenge which is part of the INTERSPEECH 2020 Computational Paralinguistics Challenge (ComParE)

    .
    ├── scripts                    # source 
    │   ├── feature_extraction     # scripts that perform feature extraction
    │   ├── valence_classifier.py  # Main module
    │   └── ...                
    └── data
      │   ├── raw_data             # should contain challenge dataset
      │   ├── features             # linguistic features
      │   ├── models               # trained linguistic models
      │   └── predictions          # contains dev/test preds as an output of valence_classifier
    └── download_model_resources.sh     # (Optional) script to download trained models and locate them into data/models folder
    └── download_fasttext_model.sh     # (Optional) script to download fine-tuned fasttext model and locate it into data/models folder
      
        
> If features (fasttext*.csv, polarity*.csv, dict*.csv, TFIDF*.csv) and models (bows*.pkl, dict*.pkl, ft_polarity*.pkl) folders contain required data, then 
feature extraction and model training stages will be skipped. 
If empty, valence_classifier script will extract all by using data/raw_data/data.csv)    

Trained models and fine-tuned fasttext models can be downloaded by running the scripts as mentioned above. In case of any technical difficulty, resources are located in `https://bitbucket.org/gizemsogancioglu/model-resources/src/master/` and can be downloaded from the repo and located to data/models folder manually.  
## License
[MIT](https://choosealicense.com/licenses/mit/)

## Installation
Recommended (Optional) :
```bash
$ conda create --name YOUR_ENV_NAME python=3.8
$ conda activate YOUR_ENV_NAME
``` 

Project can be installed simply with the following command. 
```bash
python setup.py install
```

## ISSUES
### Done:
1. Module is working for the batch prediction on development and test set when required feature files are in the source/features folder.
    Output - UAR score for FOLD 4 and dev/test predictions.  
2. Module is working when translated data is placed under the source/data folder. Feature can be automatically extracted.
3. Project can be packaged and installed successfully simply with `python setup.py install` command. 

### TODO:
4. Module is working when raw data is placed under source/data folder. Translation can be done automatically with the help of Google translate API.   
5. There is a script that can predict the valence score of a single story. 
6. There are necessary tests (analysis - pylint) in the valence/tests folder. 
