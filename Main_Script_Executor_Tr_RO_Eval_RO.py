import os
import warnings
import argparse
from Amazonia_Legal_RO import AMAZON_RO
import SharedParameters

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='')

parser.add_argument('--train', dest='train', type=eval, choices=[True, False], default=True, help='whether to run train script or not')
parser.add_argument('--test', dest='test', type=eval, choices=[True, False], default=True, help='whether to run test script or not')
parser.add_argument('--metrics', dest='metrics', type=eval, choices=[True, False], default=True, help='whether to run metrics script or not')

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

source_dataset = AMAZON_RO.DATASET
training_type = SharedParameters.TRAINING_TYPE_CLASSIFICATION
checkpoint_dir = "checkpoint_tr_"+source_dataset+"_"
results_dir = "results_tr_"+source_dataset+"_"
runs = "5"

#Deforastation / No Deforastation
num_classes = "2"

#Source, Target Ma, Target RO
num_targets = "2" 

#TARGET: RO
target_dataset = AMAZON_RO.DATASET
source_to_target = AMAZON_RO.DATASET + "_to_" + target_dataset

DR_LOCALIZATION = ['55']
METHODS  = [SharedParameters.METHOD]
DA_TYPES = ['None']
DATASETS = [AMAZON_RO.DATASET]

for dr_localization in DR_LOCALIZATION:
    for method in METHODS:
        for da in DA_TYPES:
            if args.train:
                
                Schedule.append("python " + Train_MAIN_COMMAND + " "
                                "--classifier_type " + method + " "
                                "--domain_regressor_type FC "
                                "--DR_Localization " + dr_localization + " "
                                "--skip_connections False "
                                "--epochs 100 "
                                "--batch_size 32 "
                                "--lr 0.0001 "
                                "--beta1 0.9 "
                                "--data_augmentation True "                                                              
                                "--fixed_tiles True "
                                "--defined_before False "
                                "--image_channels 7 "
                                "--patches_dimension 64 "
                                "--overlap_s 0.9 "
                                "--overlap_t 0.9 "
                                "--compute_ndvi False "
                                "--balanced_tr True "
                                "--buffer True "
                                "--source_buffer_dimension_out 2 "
                                "--source_buffer_dimension_in 0 "
                                "--target_buffer_dimension_out 2 "
                                "--target_buffer_dimension_in 0 "
                                "--porcent_of_last_reference_in_actual_reference 100 "
                                "--porcent_of_positive_pixels_in_actual_reference_s 2 "
                                "--porcent_of_positive_pixels_in_actual_reference_t 2 "
                                "--num_classes " + num_classes + " "
                                "--num_targets " + num_targets + " "
                                "--phase train "
                                "--training_type " + training_type + " "
                                "--da_type " + da + " "
                                "--runs " + runs + " "
                                "--patience 10 "
                                "--checkpoint_dir " + checkpoint_dir + method + "_" + da + "_" + target_dataset + " "
                                "--source_dataset " + source_dataset + " "
                                "--target_dataset " + target_dataset + " "
                                "--images_section " + SharedParameters.IMAGES_SECTION + " "
                                "--reference_section " + SharedParameters.REFERENCE_SECTION + " "
                                "--data_type " + SharedParameters.DATA_TYPE + " "
                                "--source_data_t1_year " + SharedParameters.T1_YEAR + " "
                                "--source_data_t2_year " + SharedParameters.T2_YEAR + " "
                                "--target_data_t1_year " + SharedParameters.T1_YEAR + " "
                                "--target_data_t2_year " + SharedParameters.T2_YEAR + " "
                                "--dataset_main_path " + SharedParameters.Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")

            for target_ds in DATASETS:

                if args.test:                    
                    Schedule.append("python " + Test_MAIN_COMMAND + " "
                                "--classifier_type " + method + " "
                                "--domain_regressor_type FC "
                                "--DR_Localization " + dr_localization + " "
                                "--skip_connections False "
                                "--batch_size 500 "                                
                                "--overlap 0.75 "
                                "--image_channels 7 "
                                "--beta1 0.9 "
                                "--patches_dimension 64 "
                                "--compute_ndvi False "
                                "--num_classes " + num_classes + " "
                                "--num_targets " + num_targets + " "
                                "--phase test "
                                "--training_type " + training_type + " "
                                "--da_type " + da + " "
                                "--checkpoint_dir " + checkpoint_dir + method + "_" + da + "_" + target_dataset + " "
                                "--results_dir " + results_dir + method + "_" + da + "_" + target_dataset + "_multi_" + target_ds + " "
                                "--dataset " + target_ds + " "
                                "--images_section " + SharedParameters.IMAGES_SECTION + " "
                                "--reference_section " + SharedParameters.REFERENCE_SECTION + " "                                
                                "--dataset_main_path " + SharedParameters.Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")                

                if args.metrics:
                    Schedule.append("python " + Metrics_05_MAIN_COMMAND + " "
                                "--classifier_type " + method + " "
                                "--domain_regressor_type FC "
                                "--skip_connections False "                                
                                "--patches_dimension 64 "
                                "--fixed_tiles True "
                                "--overlap 0.75 "
                                "--image_channels 7 "
                                "--buffer True "
                                "--buffer_dimension_out 2 "
                                "--buffer_dimension_in 0 "
                                "--eliminate_regions True "                                
                                "--compute_ndvi False "
                                "--phase compute_metrics "
                                "--training_type " + training_type + " "
                                "--save_result_text True "
                                "--checkpoint_dir " + checkpoint_dir + method + "_" + da + "_" + target_dataset + " "
                                "--results_dir " + results_dir + method + "_" + da + "_" + target_dataset + "_multi_" + target_ds + " "
                                "--dataset " + target_ds + " " 
                                "--images_section " + SharedParameters.IMAGES_SECTION + " "
                                "--reference_section " + SharedParameters.REFERENCE_SECTION + " "
                                "--data_type " + SharedParameters.DATA_TYPE + " "
                                "--data_t1_year " + SharedParameters.T1_YEAR + " "
                                "--data_t2_year " + SharedParameters.T2_YEAR + " "                                
                                "--dataset_main_path " + SharedParameters.Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")

                    Schedule.append("python " + Metrics_th_MAIN_COMMAND + " "
                                "--classifier_type " + method + " "
                                "--domain_regressor_type FC "
                                "--skip_connections False "                                
                                "--patches_dimension 64 "
                                "--fixed_tiles True "
                                "--overlap 0.75 "
                                "--image_channels 7 "
                                "--buffer True "
                                "--buffer_dimension_out 2 "
                                "--buffer_dimension_in 0 "
                                "--eliminate_regions True "                                
                                "--Npoints 100 "
                                "--compute_ndvi False "
                                "--phase compute_metrics "
                                "--training_type " + training_type + " "
                                "--save_result_text False "
                                "--checkpoint_dir " + checkpoint_dir + method + "_" + da + "_" + target_dataset + " "
                                "--results_dir " + results_dir + method + "_" + da + "_" + target_dataset + "_multi_" + target_ds + " "
                                "--dataset " + target_ds + " "
                                "--images_section " + SharedParameters.IMAGES_SECTION + " "
                                "--reference_section " + SharedParameters.REFERENCE_SECTION + " "
                                "--data_type " + SharedParameters.DATA_TYPE + " "
                                "--data_t1_year " + SharedParameters.T1_YEAR + " "
                                "--data_t2_year " + SharedParameters.T2_YEAR + " "
                                "--dataset_main_path " + SharedParameters.Dataset_MAIN_PATH + " "
                                "--checkpoint_results_main_path " + Checkpoint_Results_MAIN_PATH + " ")


for i in range(len(Schedule)):
    os.system(Schedule[i])