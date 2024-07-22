import os
import warnings

warnings.filterwarnings("ignore")
Schedule = []

REFERENCES = [
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_MA-PA.py --f1chart False',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_MA-RO.py --f1chart False',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_MA-MA.py --f1chart False',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_PA-MA.py --f1chart False',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_PA-RO.py --f1chart False',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_PA-PA.py --f1chart False',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_RO-MA.py --f1chart False',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_RO-PA.py --f1chart False',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_RO-RO.py --f1chart False'
]

REFERENCES = [
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_MA-PA.py',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_MA-RO.py',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_PA-MA.py',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_PA-RO.py',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_RO-MA.py',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_RO-PA.py'
]

REFERENCES = [
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Source_MA_PA.py ',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Source_MA_RO.py ',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Source_PA_RO.py '
]

REFERENCES = [
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_MA-PA.py',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_MA-RO.py',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_PA-MA.py',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_PA-RO.py',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_RO-MA.py',
    'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_RO-PA.py'
]

for reference in REFERENCES:
    Schedule.append("python " + reference + " 2>&1 | tee mAP.txt ")

for i in range(len(Schedule)):
    try:
        os.system(Schedule[i])
    except Exception as e:
        print(e)
        continue