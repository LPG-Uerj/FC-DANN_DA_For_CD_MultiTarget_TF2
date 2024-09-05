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
    "--method", dest="method", type=str, default="multi-target"
)

args = parser.parse_args()


num_samples = 100
result_path = []

sources = [
    {
        "name": "MA",
        "targets": [
            {
                "name": "PA",
                "methods": [
                    {
                        "name": SharedParameters.UPPER_BOUND_SOURCE_ONLY_LABEL,
                        "results": "results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_classification_Amazon_PA/",
                        "labels": SharedParameters.formatted_upper_bound_source_only_label(
                            "PA"
                        ),
                    },
                    {
                        "name": SharedParameters.LOWER_BOUND_LABEL,
                        "results": "results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_PA/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/",
                        "labels": SharedParameters.formatted_lower_bound_label(
                            "MA", "PA"
                        ),
                    },
                    {
                        "name": SharedParameters.SINGLE_TARGET_LABEL,
                        "results": "results_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_wrmp1_gamma_2.5_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_gamma_2.5_skipconn_True/",
                        "labels": SharedParameters.formatted_single_target_label(
                            "MA", "PA"
                        ),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[0]}",
                        "results": "results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_PA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "MA", "RO", "PA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[1]}",
                        "results": "results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_PA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "MA", "RO", "PA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[1],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[2]}",
                        "results": "results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_PA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "MA", "RO", "PA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[2],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[3]}",
                        "results": "results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_PA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "MA", "RO", "PA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[3],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL_NO_DA}",
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_classification_None_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_classification_None_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_no_da_label('MA','RO','PA'),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}",
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','RO','PA'),
                    }
                ],
            },
            {
                "name": "RO",
                "methods": [
                    {
                        "name": SharedParameters.UPPER_BOUND_SOURCE_ONLY_LABEL,
                        "results": "results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_RO/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_classification_Amazon_RO/",
                        "labels": SharedParameters.formatted_upper_bound_source_only_label(
                            "RO"
                        ),
                    },
                    {
                        "name": SharedParameters.LOWER_BOUND_LABEL,
                        "results": "results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Amazon_RO/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/",
                        "labels": SharedParameters.formatted_lower_bound_label(
                            "MA", "RO"
                        ),
                    },
                    {
                        "name": SharedParameters.SINGLE_TARGET_LABEL,
                        "results": "results_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO_wrmp1_gamma_2.5_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO_gamma_2.5_skipconn_True/",
                        "labels": SharedParameters.formatted_single_target_label(
                            "MA", "RO"
                        ),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[0]}",
                        "results": "results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "MA", "RO", "PA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[1]}",
                        "results": "results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "MA", "RO", "PA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[1],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[2]}",
                        "results": "results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_RO_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "MA", "RO", "PA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[2],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[3]}",
                        "results": "results_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_RO_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_to_Amazon_RO_Amazon_PA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_RO_Amazon_PA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "MA", "RO", "PA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[3],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL_NO_DA}",
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_classification_None_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_classification_None_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_no_da_label('MA','PA','RO'),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}",
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','PA','RO'),
                    }
                ],
            },
        ],
    },
    {
        "name": "PA",
        "targets": [
            {
                "name": "MA",
                "methods": [
                    {
                        "name": SharedParameters.UPPER_BOUND_SOURCE_ONLY_LABEL,
                        "results": "results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/",
                        "labels": SharedParameters.formatted_upper_bound_source_only_label(
                            "MA"
                        ),
                    },
                    {
                        "name": SharedParameters.LOWER_BOUND_LABEL,
                        "results": "results_tr_Amazon_PA_classification_S_Amazon_PA_T_Cerrado_MA/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_classification_Amazon_PA/",
                        "labels": SharedParameters.formatted_lower_bound_label(
                            "PA", "MA"
                        ),
                    },
                    {
                        "name": SharedParameters.SINGLE_TARGET_LABEL,
                        "results": "results_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_wrmp1_gamma_2.5_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_wrmp1_gamma_2.5_skipconn_True/",
                        "labels": SharedParameters.formatted_single_target_label(
                            "PA", "MA"
                        ),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[0]}",
                        "results": "results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Cerrado_MA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "PA", "RO", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[1]}",
                        "results": "results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Cerrado_MA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "PA", "RO", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[1],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[2]}",
                        "results": "results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Cerrado_MA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "PA", "RO", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[2],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[3]}",
                        "results": "results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Cerrado_MA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "PA", "RO", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[3],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL_NO_DA}",
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_classification_None_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_classification_None_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_no_da_label('PA','RO','MA'),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}",
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('PA','RO','MA'),
                    }
                ],
            },
            {
                "name": "RO",
                "methods": [
                    {
                        "name": SharedParameters.UPPER_BOUND_SOURCE_ONLY_LABEL,
                        "results": "results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_RO/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_classification_Amazon_RO/",
                        "labels": SharedParameters.formatted_upper_bound_source_only_label(
                            "RO"
                        ),
                    },
                    {
                        "name": SharedParameters.LOWER_BOUND_LABEL,
                        "results": "results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_RO/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_classification_Amazon_PA/",
                        "labels": SharedParameters.formatted_lower_bound_label(
                            "PA", "RO"
                        ),
                    },
                    {
                        "name": SharedParameters.SINGLE_TARGET_LABEL,
                        "results": "results_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO_wrmp1_gamma_2.5_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_single_Amazon_RO_wrmp1_gamma_2.5_skipconn_True/",
                        "labels": SharedParameters.formatted_single_target_label(
                            "PA", "RO"
                        ),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[0]}",
                        "results": "results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "PA", "RO", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[1]}",
                        "results": "results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "PA", "RO", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[1],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[2]}",
                        "results": "results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_RO_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "PA", "RO", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[2],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[3]}",
                        "results": "results_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_RO_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_to_Amazon_RO_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_RO_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "PA", "RO", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[3],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL_NO_DA}",
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_classification_None_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_classification_None_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_no_da_label('MA','PA','RO'),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}",
                        "results": "results_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_PA_to_Amazon_RO_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','PA','RO'),
                    }
                ],
            },
        ],
    },
    {
        "name": "RO",
        "targets": [
            {
                "name": "MA",
                "methods": [
                    {
                        "name": SharedParameters.UPPER_BOUND_SOURCE_ONLY_LABEL,
                        "results": "results_tr_Cerrado_MA_classification_S_Cerrado_MA_T_Cerrado_MA/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_classification_Cerrado_MA/",
                        "labels": SharedParameters.formatted_upper_bound_source_only_label(
                            "MA"
                        ),
                    },
                    {
                        "name": SharedParameters.LOWER_BOUND_LABEL,
                        "results": "results_tr_Amazon_RO_classification_S_Amazon_RO_T_Cerrado_MA/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_classification_Amazon_RO/",
                        "labels": SharedParameters.formatted_lower_bound_label(
                            "RO", "MA"
                        ),
                    },
                    {
                        "name": SharedParameters.SINGLE_TARGET_LABEL,
                        "results": "results_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_wrmp1_gamma_2.5_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_single_Cerrado_MA_gamma_2.5_skipconn_True/",
                        "labels": SharedParameters.formatted_single_target_label(
                            "RO", "MA"
                        ),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[0]}",
                        "results": "results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Cerrado_MA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_PA_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "RO", "PA", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[0],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[1]}",
                        "results": "results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Cerrado_MA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_PA_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "RO", "PA", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[1],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[2]}",
                        "results": "results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Cerrado_MA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_PA_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "RO", "PA", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[2],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[3]}",
                        "results": "results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Cerrado_MA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_PA_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label(
                            "RO", "PA", "MA"
                        )
                        + " "
                        + SharedParameters.EXPERIMENTS_LABELS[3],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL_NO_DA}",
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_classification_None_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_classification_None_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_no_da_label('PA','RO','MA'),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}",
                        "results": "results_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_Amazon_RO_to_Cerrado_MA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('PA','RO','MA'),
                    }
                ],
            },
            {
                "name": "PA",
                "methods": [
                    {
                        "name": SharedParameters.UPPER_BOUND_SOURCE_ONLY_LABEL,
                        "results": "results_tr_Amazon_PA_classification_S_Amazon_PA_T_Amazon_PA/",
                        "checkpoints": "checkpoint_tr_Amazon_PA_classification_Amazon_PA/",
                        "labels": SharedParameters.formatted_upper_bound_source_only_label("PA"),
                    },
                    {
                        "name": SharedParameters.LOWER_BOUND_LABEL,
                        "results": "results_tr_Amazon_RO_classification_S_Amazon_RO_T_Amazon_PA/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_classification_Amazon_RO/",
                        "labels": SharedParameters.formatted_lower_bound_label("RO","PA"),
                    },
                    {
                        "name": SharedParameters.SINGLE_TARGET_LABEL,
                        "results": "results_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_wrmp1_gamma_2.5_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_single_Amazon_PA_gamma_2.5_skipconn_True/",
                        "labels": SharedParameters.formatted_single_target_label("RO","PA"),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[0]}",
                        "results": "results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_PA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_True_wrmp_1_Amazon_PA_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label("RO","PA","MA") + " " + SharedParameters.EXPERIMENTS_LABELS[0],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[1]}",
                        "results": "results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_PA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_FC_multi_discriminate_target_False_wrmp_1_Amazon_PA_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label("RO","PA","MA") + " " + SharedParameters.EXPERIMENTS_LABELS[1],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[2]}",
                        "results": "results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_PA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_True_wrmp_1_Amazon_PA_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label("RO","PA","MA") + " " + SharedParameters.EXPERIMENTS_LABELS[2],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_TARGET_LABEL}\\\{SharedParameters.EXPERIMENTS_LABELS[3]}",
                        "results": "results_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_PA_skipconn_True/",
                        "checkpoints": "checkpoint_tr_Amazon_RO_to_Amazon_PA_Cerrado_MA_domain_adaptation_DR_Dense_multi_discriminate_target_False_wrmp_1_Amazon_PA_Cerrado_MA_skipconn_True/",
                        "labels": SharedParameters.formatted_multi_target_label("RO","PA","MA") + " " + SharedParameters.EXPERIMENTS_LABELS[3],
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL_NO_DA}",
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_classification_None_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_classification_None_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_no_da_label('MA','RO','PA'),
                    },
                    {
                        "name": f"{SharedParameters.MULTI_SOURCE_LABEL}",
                        "results": "results_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "checkpoints": "checkpoint_tr_Cerrado_MA_Amazon_RO_to_Amazon_PA_domain_adaptation_DR_FC_multi_source_discriminate_target_False/",
                        "labels": SharedParameters.formatted_multi_source_label('MA','RO','PA'),
                    }
                ],
            },
        ],
    },
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
args.method = "multi-target"
Charts.generate_tables(
    args,
    sources
)

Charts.generate_tables_uncertainty(args,sources)