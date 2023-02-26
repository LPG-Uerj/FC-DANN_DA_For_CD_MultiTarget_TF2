import os
import warnings

warnings.filterwarnings("ignore")
Schedule = []

REFERENCES = [
'Main_Script_Executor_Tr_MA_DA_Multi_Balanced.py --train False',
'Main_Script_Executor_Tr_PA_DA_Multi_Balanced.py --train False',
'Main_Script_Executor_Tr_RO_DA_Multi_Balanced.py --train False'
]

for reference in REFERENCES:
    Schedule.append("python " + reference + " ")

for i in range(len(Schedule)):
    try:
        os.system(Schedule[i])
    except Exception as e:
        print(e)
        continue