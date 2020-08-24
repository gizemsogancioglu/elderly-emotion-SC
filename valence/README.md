# elderly-emotion-SC
## Description
This project contains the scripts for our entry in the Elderly Emotion Sub-Challenge which is part of the INTERSPEECH 2020 Computational Paralinguistics Challenge (ComParE)

## Installation
Recommended (Optional) :
`conda create --name YOUR_ENV_NAME python=3.8` 
`conda activate YOUR_ENV_NAME` 

Project can be installed simply with the following command. 
`python setup.py install`

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
