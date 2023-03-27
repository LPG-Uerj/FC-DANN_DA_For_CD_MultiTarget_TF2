import os

home = os.getenv("HOME")
Dataset_MAIN_PATH = str(home)+"/workspace/dataset/"
METHOD = "DeepLab"

Train_MAIN_COMMAND = "Main_Train_FC114.py"
Test_MAIN_COMMAND = "Main_Test_FC114.py"
Metrics_05_MAIN_COMMAND = "Main_Compute_Metrics_05.py"
Metrics_th_MAIN_COMMAND = "Main_Compute_Average_Metrics_MT.py"

ROLE_SOURCE = "S"
ROLE_TARGET = "T"

PHASE_TRAIN = "train"
PHASE_TEST = "test"
PHASE_METRICS = "compute_metrics"

LR = str(0.00001)
#LR = str(0.0001)

#TRAINING_BATCH_SIZE = 32
TRAINING_BATCH_SIZE = "64"
TESTING_BATCH_SIZE = "500"

Full_Path_Train_MAIN_COMMAND = "$HOME/workspace/FC-DANN_DA_For_CD_MultiTarget_TF2/"+Train_MAIN_COMMAND
Full_Path_Test_MAIN_COMMAND = "$HOME/workspace/FC-DANN_DA_For_CD_MultiTarget_TF2/"+Test_MAIN_COMMAND
Full_Path_Metrics_05_MAIN_COMMAND = "$HOME/workspace/FC-DANN_DA_For_CD_MultiTarget_TF2/"+Metrics_05_MAIN_COMMAND
Full_Path_Metrics_th_MAIN_COMMAND = "$HOME/workspace/FC-DANN_DA_For_CD_MultiTarget_TF2/"+Metrics_th_MAIN_COMMAND

IMAGES_SECTION = "Organized/Images/"
REFERENCE_SECTION = "Organized/References/"
T1_YEAR = "2016"
T2_YEAR = "2017"
DATA_TYPE = ".npy"

BUFFER_DIMENSION_IN = 0
BUFFER_DIMENSION_OUT = 2

TRAINING_TYPE_CLASSIFICATION = "classification"
TRAINING_TYPE_DOMAIN_ADAPTATION = "domain_adaptation"

DA_CHART_TITLE = 'DA multi-target experiments comparison'

LOWER_BOUND_LABEL = 'Train(X)|Test(Y)]\n(Source only training)'
UPPER_BOUND_SOURCE_ONLY_LABEL = 'Train(X)|Test(X)\n(Source only training)'
UPPER_BOUND_DA_LABEL = 'Train X,Y1,Y2 |Test((Y)Y1,Y2 → X)\n(DA training on multi-target)'
SINGLE_TARGET_LABEL = 'Train X |Test(Y → X)\n(DA single-target)'
MULTI_TARGET_LABEL = 'Train X |Test((Y)Y1,Y2 → X)\n(DA training on source)'

EXPERIMENTS_LABELS = [    
    '3 neurons fc discr. unblcd mixed datasets',
    #'3 neurons fc discr.',
    '2 neurons fc discr.',    
    '3 neurons conv discr.',    
    '3 neurons fc discr. 1 run warmup',
    '3 neurons fc discr. 5 runs warmup',
    '3 neurons fc discr. 1 run warmup 64 batch size'
]

EXPERIMENTS_LABELS_LB = [    
    '3 neurons fc discr.\nunblcd mixed datasets',
    #'3 neurons fc discr.',
    '2 neurons fc discr.',    
    '3 neurons conv discr.',    
    '3 neurons fc discr.\n1 run warmup',
    '3 neurons fc discr.\n5 runs warmup',
    '3 neurons fc discr.\n1 run warmup\n64 batch size'
]