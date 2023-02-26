import argparse
import Charts
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA

parser = argparse.ArgumentParser(description='')

parser.add_argument('--debugMode',dest='debugMode', type=eval, choices=[True, False], default=False)

args = parser.parse_args()

def main():
    
    experiments = [
        'Tr:PA-Ts:PA',                        
        'Tr:RO->PA-Ts:PA',    
        'Tr:RO->PA,RO-\nTs:PA(Upper)',       
        'Tr:RO->PA,RO(unblcd)-Ts:PA',    
        'Tr:RO->PA,RO-\nTs:PA',        
        'Tr:RO-Ts:PA',
        'Tr:RO->PA,RO-Ts:PA\n(2 neurons discriminator)',
        'Tr:RO->PA,RO-Ts:PA\n(3 neurons conv discriminator)'
        ]

    args.checkpoint_results_main_path = "./results/"

    #Accuracy, F1-Score, Recall, Precision
    
    
    path_to_export_chart = "./results/results/"
    #path_to_export_chart = "../"

   
    result_list = [
        'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/',        
        'results_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
        'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_PA/',
        'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_Amazon_PA/',
        'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA/',
        'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_PA/',
        'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_PA/',
        'results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_CONV_multi_balanced_domain_labels_True_Amazon_PA/'
        ]
    checkpoint_list = [
        'checkpoint_tr_Amazon_PA_classification_Amazon_PA/',        
        'checkpoint_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
        'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_PA_Cerrado_MA/',
        'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_Amazon_PA_Cerrado_MA/',
        'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_PA_Cerrado_MA/',
        'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
        'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_PA_Cerrado_MA/',
        'checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_CONV_multi_balanced_domain_labels_True_Amazon_PA_Cerrado_MA/'
        ] 
    target_list = [
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET,
        AMAZON_PA.DATASET]

    cont = 1
    for i in range(0, len(result_list), 4):
        result_list_ = result_list[i : i + 4]
        checkpoint_list_ = checkpoint_list[i : i + 4]
        target_list_ = target_list[i : i + 4]
        experiments_ = experiments[i : i + 4]
        
        title = "Metrics_Tr_RO_Ts_PA"+str(cont)

        Charts.create_chart(args,experiments_,target_list_,result_list_,checkpoint_list_,path_to_export_chart,title)
        cont += 1

if __name__=='__main__':
    main()