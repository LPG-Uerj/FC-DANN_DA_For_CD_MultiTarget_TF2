import Charts
import argparse
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
import SharedParameters


parser = argparse.ArgumentParser()

parser.add_argument(
    "--recalculate", dest="recalculate", type=eval, choices=[True, False], default=True
)

parser.add_argument(
    "--method", dest="method", type=str, default="multi-source"
)

args = parser.parse_args()


num_samples = 100
result_path = []


#Multisource cenario example: source=['PA-MA','PA-RO','MA-RO'] target= ['RO','MA','PA']
'''
sources = [
    {
        "name": "MA-PA",
        "targets": [
            {
                "name": "RO",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL_NO_DA}",
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_classification_None_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_classification_None_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_no_da_label('MA','PA','RO'),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','PA','RO')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[1],
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','PA','RO')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[1],
                    }
                ],
            }
        ],
    },
    
    {
        "name": "PA-RO",
        "targets": [
            {
                "name": "MA",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL_NO_DA}",
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_classification_None_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_classification_None_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_no_da_label('PA','RO','MA'),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('PA','RO','MA')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[1],
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('PA','RO','MA')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[1],
                    }
                ],
            }
        ],
    },
    
    {
        "name": "MA-RO",
        "targets": [
            {
                "name": "PA",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL_NO_DA}",
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_classification_None_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_classification_None_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_no_da_label('MA','RO','PA'),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','RO','PA')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[1],
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','RO','PA')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[1],
                    }
                ],
            }
        ],
    }
]
'''

sources = [
    {
        "name": "MA-PA",
        "targets": [
            {
                "name": "MA",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Cerrado_MA/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','PA','MA')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    }
                ],
            },
            {
                "name": "PA",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Amazon_PA/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','PA','PA')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    }
                ],
            },
            {
                "name": "RO",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Amazon_RO/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','PA','RO')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    }
                ],
            }
        ],
    },
    
    {
        "name": "PA-RO",
        "targets": [
            {
                "name": "MA",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Cerrado_MA/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('PA','RO','MA')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    }
                ],
            },
            {
                "name": "PA",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Amazon_PA/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('PA','RO','PA')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    }
                ],
            },
            {
                "name": "RO",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Amazon_RO/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('PA','RO','RO')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    }
                ],
            }
        ],
    },
    
    {
        "name": "MA-RO",
        "targets": [
            {
                "name": "MA",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Cerrado_MA/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','RO','MA')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    }
                ],
            },
            {
                "name": "PA",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Amazon_PA/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','RO','PA')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    }
                ],
            },
            {
                "name": "RO",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}" + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True_Amazon_RO/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_True/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','RO','RO')
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    }
                ],
            }
        ],
    }
]

'''
# Print the structure to verify
for source in sources:
    print(f"Source: {source['name']}")
    for target in source["targets"]:
        print(f"  Target: {target['name']}")
        for method in target["methods"]:
            print(f"    Method: {method['name']}")
            print(f"      Results: {method['results']}")
            print(f"      Checkpoints: {method['checkpoints']}")
            print(f"      Labels: {method['labels']}")
'''

args.recalculate = True
args.method = "multi-source"

Charts.generate_tables(
    args,
    sources
)
#Charts.generate_tables_uncertainty(args,sources)