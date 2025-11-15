Deep Learning DANN implementation built on TensorFlow 2 for Change Detection with Multi Target and Multi Source support.

Startup classes:

Classes starting with "Main_Script_Executor" contain the hyperparameters and configuration required for training.

Examples:
Main_Script_Executor_Tr_MA_Eval_MA - Trains on MA dataset and evaluates on the same source dataset.

Main_Script_Executor_Tr_PA_Domain_Adaptation_Multi - Trains on PA dataset and evaluates on MA and RO target datasets.


Startup parameters:
- train: whether to do the training or not (boolean - default:True)
- test: whether to test or not (boolean - default:True)
- metrics: whether to compute metrics or not (boolean - default:True)

python Main_Script_Executor_Tr_RO_Eval_RO.py --train True --test False --metrics False 2>&1 | tee Output_Tr_RO_Eval_RO.txt


The datasets used in this work are available through the following links. 

Images of Amazon biome: https://drive.google.com/drive/folders/1V4UdYors3m3eXaAHXgzPc99esjQOc3mq?usp=sharing; 
Images of Cerrado biome: https://drive.google.com/drive/folders/14Jsw0LRcwifwBSPgFm1bZeDBQvewI8NC?usp=sharing; 
References of Amazon domains: https://drive.google.com/drive/folders/15i04inGjme56t05gk98lXErSRgRnU30x?usp=sharing; 
References of Cerrado domain: https://drive.google.com/drive/folders/1n9QZA_0V0Xh8SrW2rsFMvpjonLNQPJ96?usp=sharing.
