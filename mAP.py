import os
import warnings

warnings.filterwarnings("ignore")
Schedule = []

REFERENCES = [
'Main_Compute_Create_Precision_vs_Recall_Curves_MA.py',
'Main_Compute_Create_Precision_vs_Recall_Curves_PA.py',
'Main_Compute_Create_Precision_vs_Recall_Curves_RO.py',
]

for reference in REFERENCES:
    Schedule.append("python " + reference + " ")

for i in range(len(Schedule)):
    try:
        os.system(Schedule[i])
    except Exception as e:
        print(e)
        continue