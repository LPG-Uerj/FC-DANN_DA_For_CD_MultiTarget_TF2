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

main_path = "./results/results_avg/"
#MA RO -> PA
baseline_paths = [
    'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/',
    'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_PA/',
    'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_PA/',
    'results_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_wrmp1_gamma_2.5_skipconn_True/',
    'results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_wrmp1_gamma_2.5_skipconn_True/'
]

baseline_checkpoints = [
    'checkpoint_tr_Amazon_PA_classification_Amazon_PA/',
    'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',
    'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
    'checkpoint_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_gamma_2.5_skipconn_True/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_gamma_2.5_skipconn_True/'
]

baseline_labels = [
    SharedParameters.formatted_upper_bound_source_only_label('PA'),
    SharedParameters.formatted_lower_bound_label('MA','PA'),
    SharedParameters.formatted_lower_bound_label('RO','PA'),
    SharedParameters.formatted_single_target_label('MA','PA'),
    SharedParameters.formatted_single_target_label('RO','PA')
]

result_path = [
    'results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/'
]

checkpoint_list = [
    'checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/'
]

args.checkpoint_results_main_path = "./results/"
path_to_export_chart = "./results/results/"


target = AMAZON_PA.DATASET

cont = 1

titles = 'X=MA,RO Y=PA\n'
map_file = 'Multi_Source_Ts_MA_RO_Eval_PA_'
metrics_file = 'Metrics_Multi_Source_Ts_MA_RO_Eval_PA_'
for i in range(0, len(result_path)):
    result_path_ = baseline_paths + [result_path[i]]
    labels_ = baseline_labels + [SharedParameters.formatted_multi_source_label('MA','RO','PA')]
    checkpoint_list_ = baseline_checkpoints + [checkpoint_list[i]]

    title = titles + SharedParameters.DA_MULTI_SOURCE_TITLE + SharedParameters.MULTI_SOURCE_EXPERIMENTS_LABELS[i]
    if args.mapchart:
        file_title = map_file+str(cont)
        map_list = Charts.create_map_chart(result_path_,labels_,main_path,path_to_export_chart,file_title,title,num_samples,(8,8))
    if args.f1chart:
        file_title = metrics_file+str(cont)
        Charts.create_chart(args,labels_,target,result_path_,checkpoint_list_,map_list,path_to_export_chart,file_title,title)
    cont += 1