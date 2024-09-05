import Charts
import argparse
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
import SharedParameters

parser = argparse.ArgumentParser()

parser.add_argument('--mapchart', dest='mapchart', type=eval, choices=[True, False], default=True)
parser.add_argument('--f1chart', dest='f1chart', type=eval, choices=[True, False], default=True)

args = parser.parse_args()
num_samples = 100

baseline_paths = [
    'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_RO/',
    'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_RO/',
    'results_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO_wrmp1_gamma_2.5_skipconn_True/',
    'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_skipconn_True/',
    'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_skipconn_True/',
    
    'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_RO_skipconn_True/',
    'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_RO_skipconn_True/',    

    'results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/',
    'results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_classification_None_FC_multi_source_discriminate_target_False/',
]

baseline_checkpoints = [
    'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
    'checkpoint_tr_Amazon_PA_classification_Amazon_PA/',
    'checkpoint_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO_wrmp1_gamma_2.5_skipconn_True/',
    'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/',
    'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/',
    
    'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/',
    'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/',
    
    'checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/',
    'checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_classification_None_FC_multi_source_discriminate_target_False/',
]

baseline_labels = [
    SharedParameters.formatted_upper_bound_source_only_label("RO"),
    SharedParameters.formatted_lower_bound_label("PA","RO"),
    SharedParameters.formatted_single_target_label("PA","RO"),
    SharedParameters.formatted_multi_target_label("PA","RO","MA") + "\n" + SharedParameters.EXPERIMENTS_LABELS[0],
    SharedParameters.formatted_multi_target_label("PA","RO","MA") + "\n" + SharedParameters.EXPERIMENTS_LABELS[1],
    
    SharedParameters.formatted_multi_target_label("PA","RO","MA") + "\n" + SharedParameters.EXPERIMENTS_LABELS[2],
    SharedParameters.formatted_multi_target_label("PA","RO","MA") + "\n" + SharedParameters.EXPERIMENTS_LABELS[3],
    
    SharedParameters.formatted_multi_source_label("MA","PA","RO"),
    SharedParameters.formatted_multi_source_no_da_label("MA","PA","RO")
]

args.checkpoint_results_main_path = "./results/"

target = AMAZON_RO.DATASET

titles = SharedParameters.formatted_chart_title("PA","RO")+'\n'
map_file = f'{SharedParameters.DA_MULTI_TARGET_FILE_TITLE}_Ts_PA_Eval_RO'
metrics_file = f'Metrics_{SharedParameters.DA_MULTI_TARGET_FILE_TITLE}_Ts_PA_Eval_RO'

Charts.create_all_charts(args, baseline_paths,baseline_labels,baseline_checkpoints,titles, map_file,metrics_file,num_samples,target)