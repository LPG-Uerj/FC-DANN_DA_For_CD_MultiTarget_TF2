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
    'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Cerrado_MA/',
    'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Cerrado_MA/',
    
    'results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_classification_None_FC_multi_source_discriminate_target_False/',
    'results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Cerrado_MA/',
    #'results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/',
    
]

baseline_checkpoints = [
    'checkpoint_tr_Amazon_PA_classification_Amazon_PA/',
    'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
    
    'checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_classification_None_FC_multi_source_discriminate_target_False/',
    'checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/',
    #'checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/',
    
]

baseline_labels = [
    SharedParameters.formatted_upper_bound_source_only_label("MA"),
    SharedParameters.formatted_lower_bound_label("PA","MA"),
    
    SharedParameters.formatted_multi_source_no_da_label("PA","RO","MA"),
    SharedParameters.formatted_multi_source_label("PA","RO","MA") + " " + SharedParameters.EXPERIMENTS_LABELS[0],
    #SharedParameters.formatted_multi_source_label("PA","RO","MA") + " " + SharedParameters.EXPERIMENTS_LABELS[1],
]

args.checkpoint_results_main_path = "./results/"
path_to_export_chart = SharedParameters.RESULTS_MAIN_PATH

target = CERRADO_MA.DATASET

#titles = SharedParameters.formatted_chart_title("PA","MA")+'\n'
#map_file = f'{SharedParameters.DA_MULTI_SOURCE_FILE_TITLE}_Ts_PA_Eval_MA'
#metrics_file = f'Metrics_{SharedParameters.DA_MULTI_SOURCE_FILE_TITLE}_Ts_PA_Eval_MA'

#Charts.create_all_charts(args, baseline_paths,baseline_labels,baseline_checkpoints,titles, map_file,metrics_file,num_samples,target)

f1_array, f1_std, f1_mean = Charts.get_stats(baseline_paths, SharedParameters.RESULTS_MAIN_PATH)

print('f1_mean')
print(f1_mean)

print('f1_std')
print(f1_std)

Charts.t_test(f'{baseline_labels[2]} vs Lower baseline 0',f1_mean[0],f1_std[0],f1_mean[2],f1_std[2], n=5)
Charts.t_test(f'{baseline_labels[2]} vs Lower baseline 1',f1_mean[1],f1_std[1],f1_mean[2],f1_std[2], n=5)

Charts.t_test(f'{baseline_labels[3]} vs Lower baseline 0',f1_mean[0],f1_std[0],f1_mean[3],f1_std[3], n=5)
Charts.t_test(f'{baseline_labels[3]} vs Lower baseline 1',f1_mean[1],f1_std[1],f1_mean[3],f1_std[3], n=5)