import os
import warnings

warnings.filterwarnings("ignore")
Schedule = []


REFERENCES = [
    'Main_Script_Executor_Tr_RO_Eval_RO_Uncertainty.py',
]

for reference in REFERENCES:
    Schedule.append("python " + reference + " 2>&1 | tee Main_Script_Executor_Uncertainty.txt ")

for i in range(len(Schedule)):
    try:
        os.system(Schedule[i])
    except Exception as e:
        print(e)
        continue