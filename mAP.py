import os
import warnings

warnings.filterwarnings("ignore")
Schedule = []

REFERENCES = [
'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_MA-PA.py',
'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_MA-RO.py',
'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_PA-MA.py',
'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_PA-RO.py',
'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_RO-MA.py',
'Main_Compute_Create_Precision_vs_Recall_Curves_Multi_Target_RO-PA.py'
]

for reference in REFERENCES:
    Schedule.append("python " + reference + " ")

for i in range(len(Schedule)):
    try:
        os.system(Schedule[i])
    except Exception as e:
        print(e)
        continue