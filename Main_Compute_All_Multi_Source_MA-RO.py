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
    'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_RO/',
    'results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_classification_None_FC_multi_source_discriminate_target_False/',
    'results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_True/',
    'results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/',
    
]

baseline_checkpoints = [
    'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
    'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',
    'checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_classification_None_FC_multi_source_discriminate_target_False/',
    'checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_True/',
    'checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/',
    
]

baseline_labels = [
    SharedParameters.formatted_upper_bound_source_only_label('RO'),
    SharedParameters.formatted_lower_bound_label('MA','RO'),
    SharedParameters.formatted_multi_source_no_da_label('MA','PA','RO'),
    SharedParameters.formatted_multi_source_label('MA','PA','RO') + "\n" + SharedParameters.EXPERIMENTS_LABELS[0],
    SharedParameters.formatted_multi_source_label('MA','PA','RO') + "\n" + SharedParameters.EXPERIMENTS_LABELS[1],
]

args.checkpoint_results_main_path = "./results/"

path_to_export_chart = SharedParameters.RESULTS_MAIN_PATH

source = CERRADO_MA.DATASET
target = AMAZON_RO.DATASET

titles = SharedParameters.formatted_chart_title('MA','RO')+'\n'
map_file = f'{SharedParameters.DA_MULTI_SOURCE_FILE_TITLE}_Ts_MA_Eval_RO'
metrics_file = f'Metrics_{SharedParameters.DA_MULTI_SOURCE_FILE_TITLE}_Ts_MA_Eval_RO'

Charts.create_all_charts(args, baseline_paths,baseline_labels,baseline_checkpoints,titles, map_file,metrics_file,num_samples,target)