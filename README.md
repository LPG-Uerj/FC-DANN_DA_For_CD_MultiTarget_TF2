Deep Learning DANN implementation built on TensorFlow 2 + Keras API for Change Detection with Multi Target support.

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
