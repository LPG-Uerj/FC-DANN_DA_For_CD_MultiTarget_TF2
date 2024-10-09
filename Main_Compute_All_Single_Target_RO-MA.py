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
    'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/',
    'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Cerrado_MA/',
    'results_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_wrmp1_gamma_2.5_skipconn_True/',
]

baseline_checkpoints = [
    'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',
    'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
    'checkpoint_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_gamma_2.5_skipconn_True/',
]

baseline_labels = [
    SharedParameters.formatted_upper_bound_source_only_label("MA"),
    SharedParameters.formatted_lower_bound_label("RO","MA"),
    SharedParameters.formatted_single_target_label("RO","MA"),
]

args.checkpoint_results_main_path = "./results/"

target = CERRADO_MA.DATASET

titles = SharedParameters.formatted_chart_title("RO","MA")+'\n'
map_file = f'{SharedParameters.DA_MULTI_TARGET_FILE_TITLE}_Ts_RO_Eval_MA'
metrics_file = f'Metrics_{SharedParameters.DA_MULTI_TARGET_FILE_TITLE}_Ts_RO_Eval_MA'

Charts.create_all_charts(args, baseline_paths,baseline_labels,baseline_checkpoints,titles, map_file,metrics_file,num_samples,target)