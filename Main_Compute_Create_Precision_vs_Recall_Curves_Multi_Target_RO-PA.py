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

main_path = "./results/results_avg/"

lower_bound_path = 'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_PA/'
lower_bound_checkpoint = 'checkpoint_tr_Amazon_RO_classification_Amazon_RO/'

upper_bound_source_only_path = 'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/'
upper_bound_source_only_checkpoint = 'checkpoint_tr_Amazon_PA_classification_Amazon_PA/'

upper_bound_da_path = 'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_PA/'
upper_bound_da_checkpoint = 'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_PA_Cerrado_MA/'

single_target_path = 'results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/'
single_target_checkpoint = 'checkpoint_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/'

result_path = [    
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Amazon_PA/',
    #'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA/',        
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_PA/',    
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_CONV_multi_balanced_domain_labels_True_Amazon_PA/',
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_balanced_domain_labels_True_Amazon_PA/',
    'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_balanced_domain_labels_True_wrmp5_Amazon_PA/',
    #'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_balanced_domain_labels_True_wrmp1_64_Amazon_PA/'
]

titles = 'X=RO, Y=PA(Y1=PA,Y2=MA)\n'

checkpoint_list = [    
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Amazon_PA_Cerrado_MA/',
    #'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA_Cerrado_MA/',    
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_PA_Cerrado_MA/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_CONV_multi_balanced_domain_labels_True_Amazon_PA_Cerrado_MA/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_balanced_domain_labels_True_Amazon_PA_Cerrado_MA/',
    'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_balanced_domain_labels_True_wrmp5_Amazon_PA_Cerrado_MA/',
    #'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_balanced_domain_labels_True_wrmp1_64_Amazon_PA_Cerrado_MA/'
] 

args.checkpoint_results_main_path = "./results/"
path_to_export_chart = "./results/results/"

map_file = 'Multi_Target_Ts_RO_Eval_PA_'
metrics_file = 'Metrics_Multi_Target_Ts_RO_Eval_PA_'

target = AMAZON_PA.DATASET

cont = 1
for i in range(0, len(result_path)):
    result_path_ = [upper_bound_source_only_path,lower_bound_path,single_target_path,result_path[i]]
    labels_ = [SharedParameters.UPPER_BOUND_SOURCE_ONLY_LABEL,SharedParameters.LOWER_BOUND_LABEL,SharedParameters.SINGLE_TARGET_LABEL,SharedParameters.MULTI_TARGET_LABEL]
    checkpoint_list_ = [upper_bound_source_only_checkpoint,lower_bound_checkpoint,single_target_checkpoint,checkpoint_list[i]]

    title = titles + "DA multi-target " + SharedParameters.EXPERIMENTS_LABELS[i]
    if args.mapchart:   
        file_title = map_file+str(cont)        
        map_list = Charts.create_map_chart(result_path_,labels_,main_path,path_to_export_chart,file_title,title,num_samples,(7,7))
    if args.f1chart:
        file_title = metrics_file+str(cont)        
        Charts.create_chart(args,labels_,target,result_path_,checkpoint_list_,map_list,path_to_export_chart,file_title,title)
    cont += 1

title = titles + SharedParameters.DA_CHART_TITLE
if args.mapchart:   
    file_title = map_file+str(cont)    
    map_list = Charts.create_map_chart(result_path,SharedParameters.EXPERIMENTS_LABELS_LB,main_path,path_to_export_chart,file_title,title,num_samples,(7,7))
if args.f1chart:    
    file_title = metrics_file+str(cont)  
    Charts.create_chart(args,SharedParameters.EXPERIMENTS_LABELS_LB,target,result_path,checkpoint_list,map_list,path_to_export_chart,file_title,title)