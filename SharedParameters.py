import os

home = os.getenv("HOME")
Dataset_MAIN_PATH = str(home)+"/workspace/dataset/"
METHOD = "DeepLab"

Train_MAIN_COMMAND = "Main_Train_FC114.py"
Test_MAIN_COMMAND = "Main_Test_FC114.py"
Metrics_05_MAIN_COMMAND = "Main_Compute_Metrics_05.py"
Metrics_th_MAIN_COMMAND = "Main_Compute_Average_Metrics_MT.py"

AVG_MAIN_PATH = "./results/results_avg/"

RESULTS_MAIN_PATH = "./results/results/"

CHECKPOINTS_MAIN_PATH = "./results/checkpoints/"

ROLE_SOURCE = "S"
ROLE_TARGET = "T"

PHASE_TRAIN = "train"
PHASE_TEST = "test"
PHASE_METRICS = "compute_metrics"

LR = str(0.0001)


#2.5
GAMMA = str(2.5)

PATIENCE = str(10)

TRAINING_BATCH_SIZE = "32"
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

SKIP_CONNECTIONS = str(True)

TRAINING_TYPE_CLASSIFICATION = "classification"
TRAINING_TYPE_DOMAIN_ADAPTATION = "domain_adaptation"

DA_MULTI_SOURCE_TITLE = 'DA Multi-Source '
DA_MULTI_TARGET_TITLE = 'DA Multi-Target '
DA_CHART_TITLE = 'DA Multi-Target experiments comparison'

LOWER_BOUND_LABEL = 'Source only training\\\(lowerbound)'
UPPER_BOUND_SOURCE_ONLY_LABEL = 'Training on target\\\(upperbound)'
SINGLE_TARGET_LABEL = 'DA single-target'
MULTI_TARGET_LABEL = 'DA multi-target'
MULTI_SOURCE_LABEL = 'DA multi-source'

FORMAT_CHART_TITLE = 'Training on {} | Validating on {}'
FORMAT_LOWER_BOUND_LABEL = 'Train({})|Test({})]\nSource only training'
FORMAT_UPPER_BOUND_SOURCE_ONLY_LABEL = 'Train({})|Test({})\nTraining on target'
FORMAT_SINGLE_TARGET_LABEL = 'Train {} |Test({} → {})\nDA Single-Target'
FORMAT_MULTI_TARGET_LABEL = 'Train {} |Test({},{} → {})\nDA Multi-Target'
FORMAT_MULTI_SOURCE_LABEL = 'Train {},{} |Test({} → {},{})\nDA Multi-Source'

formatted_chart_title = lambda x, y: FORMAT_CHART_TITLE.format(x, y)
formatted_lower_bound_label = lambda x, y: FORMAT_LOWER_BOUND_LABEL.format(x, y)
formatted_upper_bound_source_only_label = lambda y: FORMAT_UPPER_BOUND_SOURCE_ONLY_LABEL.format(y,y)
formatted_single_target_label = lambda x, y: FORMAT_SINGLE_TARGET_LABEL.format(x, y, x)
formatted_multi_target_label = lambda x, y1, y2: FORMAT_MULTI_TARGET_LABEL.format(x, y1, y2, x)
formatted_multi_source_label = lambda x, y, z: FORMAT_MULTI_SOURCE_LABEL.format(x, y, z, x, y)

EXPERIMENTS_LABELS = [    
    '3 neurons',
    '2 neurons'
]

EXPERIMENTS_LABELS_LB = [    
    '3 neurons',
    '2 neurons'
]

MULTI_SOURCE_EXPERIMENTS_LABELS = [    
    '2 neurons discr.'
]


INCLUDE_DA_RESULTS = True

DISCRIMINATE_DOMAIN_TARGETS = str(True)