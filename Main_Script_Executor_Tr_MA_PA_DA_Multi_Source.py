import os
import warnings
import argparse
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
import SharedParameters

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='')

parser.add_argument('--train', dest='train', type=eval, choices=[True, False], default=True, help='whether to run train script or not')
parser.add_argument('--test', dest='test', type=eval, choices=[True, False], default=True, help='whether to run test script or not')
parser.add_argument('--metrics', dest='metrics', type=eval, choices=[True, False], default=True, help='whether to run metrics script or not')
parser.add_argument('--discriminate_domain_targets', dest='discriminate_domain_targets', type=eval, choices=[True, False], default=False, help='Applies for Multi-target training. Decides whether each target dataset will be assigned a different domain label or every target dataset will get the same label.')

parser.add_argument('--running_in', dest='running_in', type=str, default='Datarmor_Interactive', help='Decide wether the script will be running')
#parser.add_argument('--phase', dest = 'phase', type = str, default = 'train', help = 'Decide wether the phase: Train|Test will be running')

args = parser.parse_args()

if args.running_in == 'Datarmor_Interactive':
    Train_MAIN_COMMAND = SharedParameters.Train_MAIN_COMMAND
    Test_MAIN_COMMAND = SharedParameters.Test_MAIN_COMMAND
    Metrics_05_MAIN_COMMAND = SharedParameters.Metrics_05_MAIN_COMMAND
    Metrics_th_MAIN_COMMAND = SharedParameters.Metrics_th_MAIN_COMMAND
if args.running_in == 'Datarmor_PBS':
    Train_MAIN_COMMAND = SharedParameters.Full_Path_Train_MAIN_COMMAND
    Test_MAIN_COMMAND = SharedParameters.Full_Path_Test_MAIN_COMMAND
    Metrics_05_MAIN_COMMAND = SharedParameters.Full_Path_Metrics_05_MAIN_COMMAND
    Metrics_th_MAIN_COMMAND = SharedParameters.Full_Path_Metrics_th_MAIN_COMMAND

warnings.filterwarnings("ignore")
Schedule = []

Checkpoint_Results_MAIN_PATH = "./results/"

training_type = SharedParameters.TRAINING_TYPE_DOMAIN_ADAPTATION

#Deforastation / No Deforastation
num_classes = "2"


source_dataset = CERRADO_MA.DATASET + "_" + AMAZON_PA.DATASET
target_dataset = AMAZON_RO.DATASET 
source_to_target = source_dataset + "_to_" + target_dataset

checkpoint_dir = "checkpoint_tr_"+source_to_target+"_"
results_dir = "results_tr_"+source_to_target+"_"
runs = "5"

domain_regressor_type = "FC"
warmup = "1"

DR_LOCALIZATION = ['55']
METHODS  = [SharedParameters.METHOD]
DA_TYPES = ['DR']
TARGET_DATASETS = [AMAZON_RO.DATASET]

for dr_localization in DR_LOCALIZATION:
    for method in METHODS:
        for da in DA_TYPES:
            
            
            checkpoint_dir_param = checkpoint_dir + training_type + "_" + da + "_" + domain_regressor_type + "_multi_source_discriminate_target_" + args.discriminate_domain_targets

            if args.train:
                
                Schedule.append("python " + Train_MAIN_COMMAND + " "
                                "--classifier_type " + method + " "
                                "--domain_regressor_type " + domain_regressor_type + " "
                                "--DR_Localization " + dr_localization + " "
                                "--skip_connections " + SharedParameters.SKIP_CONNECTIONS + " "
                                "--epochs 100 "
                                "--batch_size " + SharedParameters.TRAINING_BATCH_SIZE + " "
                                "--lr " + SharedParameters.LR + " "
                                "--gamma " + SharedParameters.GAMMA + " "
                                "--beta1 0.9 "
                                "--data_augmentation True "                                                              
                                "--fixed_tiles True "
                                "--defined_before False "
                                "--image_channels 7 "
                                "--patches_dimension 64 "
                                "--overlap_s 0.9 "
                                "--overlap_t 0.9 "
                                "--compute_ndvi False "
                                "--balanced_tr False "
                                "--buffer True "                                
                                "--porcent_of_last_reference_in_actual_reference 100 "
                                "--porcent_of_positive_pixels_in_actual_reference_s 2 "
                                "--porcent_of_positive_pixels_in_actual_reference_t 2 "
                                "--num_classes " + num_classes + " "
                                "--discriminate_domain_targets " + str(args.discriminate_domain_targets) + " "
                                "--phase train "
                                "--training_type " + training_type + " "
                                "--da_type " + da + " "
                                "--runs " + runs + " "
                                "--warmup " + warmup + " "
                                "--patience " + SharedParameters.PATIENCE + " "
                                "--checkpoint_dir " + checkpoint_dir_param + " "
                                "--source_dataset " + source_dataset + " "
                                "--target_dataset " + target_dataset + " "
                                "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")

            for target_ds in TARGET_DATASETS:
                
                results_dir_param = results_dir + training_type + "_" + da + "_" + domain_regressor_type + "_multi_source_discriminate_target_" + args.discriminate_domain_targets

                if args.test:                    
                    Schedule.append("python " + Test_MAIN_COMMAND + " "
                                "--classifier_type " + method + " "
                                "--domain_regressor_type " + domain_regressor_type + " "
                                "--DR_Localization " + dr_localization + " "
                                "--skip_connections " + SharedParameters.SKIP_CONNECTIONS + " "
                                "--batch_size " + SharedParameters.TESTING_BATCH_SIZE + " "                                
                                "--overlap 0.75 "
                                "--image_channels 7 "
                                "--beta1 0.9 "
                                "--patches_dimension 64 "
                                "--compute_ndvi False "
                                "--num_classes " + num_classes + " " 
                                "--discriminate_domain_targets " + str(args.discriminate_domain_targets) + " "                              
                                "--phase test "
                                "--training_type " + training_type + " "
                                "--da_type " + da + " "
                                "--checkpoint_dir " + checkpoint_dir_param + " "
                                "--results_dir " + results_dir_param + " "
                                "--dataset " + target_ds + " "                                                           
                                "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")                

                if args.metrics:
                    Schedule.append("python " + Metrics_05_MAIN_COMMAND + " "
                                "--classifier_type " + method + " "
                                "--domain_regressor_type " + domain_regressor_type + " "
                                "--skip_connections " + SharedParameters.SKIP_CONNECTIONS + " "                                
                                "--patches_dimension 64 "
                                "--fixed_tiles True "
                                "--overlap 0.75 "
                                "--image_channels 7 "
                                "--buffer True "                                
                                "--eliminate_regions True "                                
                                "--compute_ndvi False "
                                "--phase compute_metrics "
                                "--training_type " + training_type + " "
                                "--save_result_text True "
                                "--checkpoint_dir " + checkpoint_dir_param + " "
                                "--results_dir " + results_dir_param + " "
                                "--dataset " + target_ds + " "                                
                                "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")

                    Schedule.append("python " + Metrics_th_MAIN_COMMAND + " "
                                "--classifier_type " + method + " "
                                "--domain_regressor_type " + domain_regressor_type + " "
                                "--skip_connections " + SharedParameters.SKIP_CONNECTIONS + " "                                
                                "--patches_dimension 64 "
                                "--fixed_tiles True "
                                "--overlap 0.75 "
                                "--image_channels 7 "
                                "--buffer True "                                
                                "--eliminate_regions True "                                
                                "--Npoints 100 "
                                "--compute_ndvi False "
                                "--phase compute_metrics "
                                "--training_type " + training_type + " "
                                "--save_result_text False "
                                "--checkpoint_dir " + checkpoint_dir_param + " "
                                "--results_dir " + results_dir_param + " "
                                "--dataset " + target_ds + " "                                
                                "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")

for i in range(len(Schedule)):
    if os.system(Schedule[i]) != 0:
        exit()