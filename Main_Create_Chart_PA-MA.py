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
        'Tr:MA-Ts:MA',                        
        'Tr:PA->MA-Ts:MA',        
        'Tr:PA->RO,MA-\nTs:MA(Upper)',
        'Tr:PA->RO,MA(unblcd)-Ts:MA',
        'Tr:PA->RO,MA-\nTs:MA',
        'Tr:PA-Ts:MA',
        'Tr:PA->RO,MA-Ts:MA\n(2 neurons discriminator)',
        ]

    args.checkpoint_results_main_path = "./results/"

    #Accuracy, F1-Score, Recall, Precision

    
    path_to_export_chart = "./results/results/"
    #path_to_export_chart = "../"

   
    result_list = [
        'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/',        
        'results_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/',
        'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Cerrado_MA/',
        'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_Cerrado_MA/',
        'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_Cerrado_MA/',
        'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Cerrado_MA/',
        'results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Cerrado_MA/'
        ]
    checkpoint_list = [
        'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',        
        'checkpoint_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/',
        'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DRCL_multi_balanced_Amazon_RO_Cerrado_MA/',
        'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_Amazon_RO_Cerrado_MA/',
        'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_Amazon_RO_Cerrado_MA/',
        'checkpoint_tr_Amazon_PA_classification_Amazon_PA/',
        'checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_multi_balanced_domain_labels_False_Amazon_RO_Cerrado_MA/'
        ]
    target_list = [
        CERRADO_MA.DATASET,
        CERRADO_MA.DATASET,        
        CERRADO_MA.DATASET, 
        CERRADO_MA.DATASET, 
        CERRADO_MA.DATASET,       
        CERRADO_MA.DATASET,
        CERRADO_MA.DATASET]

    for i in range(0, len(result_list), 4):
        result_list_ = result_list[i : i + 4]
        checkpoint_list_ = checkpoint_list[i : i + 4]
        target_list_ = target_list[i : i + 4]
        experiments_ = experiments[i : i + 4]
        
        title = "Metrics_Tr_PA_Ts_MA"+str(i)

        Charts.create_chart(args,experiments_,target_list_,result_list_,checkpoint_list_,path_to_export_chart,title)

if __name__=='__main__':
    main()