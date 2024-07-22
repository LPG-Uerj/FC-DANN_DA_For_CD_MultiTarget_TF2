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
    'results_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO_wrmp1_gamma_2.5_skipconn_True/'
]

baseline_checkpoints = [
    'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
    'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',
    'checkpoint_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO_gamma_2.5_skipconn_True/'
]

baseline_labels = [
    SharedParameters.formatted_upper_bound_source_only_label('RO'),
    SharedParameters.formatted_lower_bound_label('MA','RO'),
    SharedParameters.formatted_single_target_label('MA','RO')
]

result_path = [    
    'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_skipconn_True/',
    'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_skipconn_True/'
]

checkpoint_list = [    
    'checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/',
    'checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/'
]

args.checkpoint_results_main_path = "./results/"

path_to_export_chart = SharedParameters.RESULTS_MAIN_PATH

source = CERRADO_MA.DATASET
target = AMAZON_RO.DATASET

cont = 1


titles = 'X=MA, Y=RO(Y1=RO,Y2=PA)\n'
map_file = 'Multi_Target_Ts_MA_Eval_RO_'
metrics_file = 'Metrics_Multi_Target_Ts_MA_Eval_RO_'
for i in range(0, len(result_path)):
    result_path_ = baseline_paths + [result_path[i]]
    labels_ = baseline_labels + [SharedParameters.formatted_multi_target_label('MA','RO','PA')]
    checkpoint_list_ = baseline_checkpoints + [checkpoint_list[i]]

    title = titles + SharedParameters.DA_MULTI_TARGET_TITLE + SharedParameters.EXPERIMENTS_LABELS[i]
    if args.mapchart:   
        file_title = map_file+str(cont)        
        map_list = Charts.create_map_chart(result_path_,labels_,SharedParameters.AVG_MAIN_PATH,SharedParameters.RESULTS_MAIN_PATH,file_title,title,num_samples,(7,7))
        Charts.create_map_f1_boxplot(result_path_,labels_,SharedParameters.RESULTS_MAIN_PATH, SharedParameters.RESULTS_MAIN_PATH, file_title,title)
    if args.f1chart:
        file_title = metrics_file+str(cont)        
        Charts.create_chart(args,labels_,target,result_path_,checkpoint_list_,map_list,SharedParameters.RESULTS_MAIN_PATH,file_title,title)
    cont += 1