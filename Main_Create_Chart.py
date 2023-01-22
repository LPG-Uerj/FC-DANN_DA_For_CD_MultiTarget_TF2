import argparse
import Charts
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA

parser = argparse.ArgumentParser(description='')

parser.add_argument('--source_dataset', dest='source_dataset', type=str, default=None,help='The name of the original source dataset used')
parser.add_argument('--target_dataset', dest='target_dataset', type=str, default=None,help='The name of the target dataset whose metrics will be computed')
args = parser.parse_args()

def main():

    if args.source_dataset == 'Amazon_RO':   
        experiments = [
            #No DA:
            'Tr: RO, Ts: RO',
            'Tr: RO, Ts: PA',
            'Tr: RO, Ts: MA',
            #Single-target DA:
            'Tr: RO->PA, Ts: RO',
            'Tr: RO->PA, Ts: PA'
            'Tr: RO->MA, Ts: RO',
            'Tr: RO->MA, Ts: MA',
            #Multi-target DA:
            'Tr: RO->PA,MA, Ts: PA',
            'Tr: RO->PA,MA, Ts: MA',
            'Tr: RO->PA,MA, Ts: RO'
                    ]
    elif args.source_dataset == 'Amazon_PA':
        experiments = [
            'Tr: PA, Ts: PA',
            'Tr: PA, Ts: RO',
            'Tr: PA, Ts: MA',
            #Single-target DA:
            'Tr: PA->RO, Ts: PA',
            'Tr: PA->RO, Ts: RO',
            'Tr: PA->MA, Ts: PA',
            'Tr: PA->MA, Ts: MA',
            #Multi-target DA:
            'Tr: PA->RO,MA, Ts: RO',
            'Tr: PA->RO,MA, Ts: MA',
            'Tr: PA->RO,MA, Ts: PA'
                    ]
    elif args.source_dataset == 'Cerrado_MA':
        experiments = [
            'Tr: MA, Ts: MA',
            'Tr: MA, Ts: PA',
            'Tr: MA, Ts: RO',
            #Single-target DA:
            'Tr: MA->RO, Ts: MA',
            'Tr: MA->RO, Ts: RO',
            'Tr: MA->PA, Ts: MA',
            'Tr: MA->PA, Ts: PA',
            #Multi-target DA:
            'Tr: MA->RO,PA, Ts: RO',
            'Tr: MA->RO,PA, Ts: PA',
            'Tr: MA->RO,PA, Ts: MA'
                        ]
    else:
        raise Exception("Invalid argument source_dataset: " + args.source_dataset) 

    args.checkpoint_results_main_path = "./results/"

    #Accuracy, F1-Score, Recall, Precision
    title = "Metrics_Source_"+args.source_dataset
    
    path_to_export_chart = "./results/results/"

    if args.source_dataset == 'Amazon_RO':  
        result_list = [
            'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_RO/',
            'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_PA/',
            'results_tr_Amazon_RO_classification_S_Amazon_RO_T_Cerrado_MA/',

            'results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_RO/',
            'results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
            'results_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Amazon_RO/',
            'results_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/'            
            ]
        checkpoint_list = [
            'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
            'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
            'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',

            'checkpoint_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
            'checkpoint_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
            'checkpoint_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/',
            'checkpoint_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/',
            ] 
        target_list = [
            AMAZON_RO.DATASET,
            AMAZON_PA.DATASET,
            CERRADO_MA.DATASET,
            AMAZON_RO.DATASET,
            AMAZON_PA.DATASET,
            AMAZON_RO.DATASET,
            CERRADO_MA.DATASET
        ] 

    elif args.source_dataset == 'Amazon_PA':  
        result_list = [
            'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/',
            'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_RO/',
            'results_tr_Amazon_PA_classification_S_Amazon_PA_T_Cerrado_MA/',

            'results_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_PA/',
            'results_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/',
            'results_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Amazon_PA/',
            'results_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/'            
            ]
        checkpoint_list = [
            'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
            'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',
            'checkpoint_tr_Amazon_RO_classification_Amazon_RO/',

            'checkpoint_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/',
            'checkpoint_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/',
            'checkpoint_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/',
            'checkpoint_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA/',
            ] 
        target_list = [
            AMAZON_PA.DATASET,
            AMAZON_RO.DATASET,
            CERRADO_MA.DATASET,
            AMAZON_PA.DATASET,
            AMAZON_RO.DATASET,
            AMAZON_PA.DATASET,
            CERRADO_MA.DATASET
        ]

    elif args.source_dataset == 'Cerrado_MA':  
        result_list = [
            'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/',
            'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_PA/',
            'results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_RO/',            
            
            'results_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Cerrado_MA/',
            'results_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/',
            'results_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Cerrado_MA/',
            'results_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
            ]
        checkpoint_list = [
            'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',
            'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',
            'checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/',

            'checkpoint_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/',
            'checkpoint_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO/',
            'checkpoint_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
            'checkpoint_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA/',
            ] 
        target_list = [
            CERRADO_MA.DATASET,
            AMAZON_PA.DATASET,
            AMAZON_RO.DATASET,
            CERRADO_MA.DATASET,
            AMAZON_RO.DATASET,
            CERRADO_MA.DATASET,
            AMAZON_PA.DATASET
        ]  
    else:
        raise Exception("Invalid argument source_dataset: " + args.source_dataset)  

    Charts.create_chart(args,experiments,target_list,result_list,checkpoint_list,path_to_export_chart,title)

if __name__=='__main__':
    main()