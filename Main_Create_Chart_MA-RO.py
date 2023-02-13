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
        'Tr:RO-Ts:RO',                        
        'Tr:MA->RO-Ts:RO',        
        'Tr:MA->RO,PA-Ts:RO',  
        'Tr:MA->RO,PA(blcd)-Ts:RO',        
        'Tr:MA-Ts:RO'
        ]

    args.checkpoint_results_main_path = "./results/"

    #Accuracy, F1-Score, Recall, Precision
    title = "Metrics_Tr_MA_Ts_RO"
    
    path_to_export_chart = "./results/results/"
    #path_to_export_chart = "../"

   
    result_list = [
        'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_RO/',        
        'results_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/',
        'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_Amazon_RO/',
        'results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_balanced_Amazon_RO/',
        'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_RO/'      
        ]
    checkpoint_list = [
        'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',        
        'checkpoint_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/',
        'checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_Amazon_RO_Amazon_PA/',
        'checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_multi_balanced_Amazon_RO_Amazon_PA/',
        'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/'
        ]
    target_list = [
        AMAZON_RO.DATASET,
        AMAZON_RO.DATASET,        
        AMAZON_RO.DATASET,
        AMAZON_RO.DATASET,        
        AMAZON_RO.DATASET]

    Charts.create_chart(args,experiments,target_list,result_list,checkpoint_list,path_to_export_chart,title)

if __name__=='__main__':
    main()