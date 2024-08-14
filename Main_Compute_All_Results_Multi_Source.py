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

args = parser.parse_args()


num_samples = 100
result_path = []


#Multisource cenario example: source=['PA-MA','PA-RO','MA-RO'] target= ['RO','MA','PA']
sources = [
    {
        "name": "MA-PA",
        "targets": [
            {
                "name": "RO",
                "methods": [
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}",
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','PA','RO'),
                    },
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
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}",
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('PA','RO','MA'),
                    },
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
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}",
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','RO','PA'),
                    },
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

args.recalculate = False
args.method = "multi-source"
Charts.generate_tables(
    args,
    sources
)