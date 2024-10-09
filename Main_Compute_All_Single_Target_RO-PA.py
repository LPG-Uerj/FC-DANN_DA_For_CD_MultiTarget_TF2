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
result_path = []

baseline_paths = [
    'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/',
    'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_PA/',
    'results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_wrmp1_gamma_2.5_skipconn_True/',
]

baseline_checkpoints = [
    'checkpoint_tr_Amazon_PA_classification_Amazon_PA/',
    'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_gamma_2.5_skipconn_True/',
]

baseline_labels = [
    SharedParameters.formatted_upper_bound_source_only_label("PA"),
    SharedParameters.formatted_lower_bound_label("RO","PA"),
    SharedParameters.formatted_single_target_label("RO","PA"),
]

args.checkpoint_results_main_path = "./results/"

target = AMAZON_PA.DATASET

titles = SharedParameters.formatted_chart_title("RO","PA")+'\n'
map_file = f'{SharedParameters.DA_MULTI_TARGET_FILE_TITLE}_Ts_RO_Eval_PA'
metrics_file = f'Metrics_{SharedParameters.DA_MULTI_TARGET_FILE_TITLE}_Ts_RO_Eval_PA'

Charts.create_all_charts(args, baseline_paths,baseline_labels,baseline_checkpoints,titles, map_file,metrics_file,num_samples,target)