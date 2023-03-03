import os
import warnings

warnings.filterwarnings("ignore")
Schedule = []

REFERENCES = [
    'Main_Script_Executor_Tr_PA_Eval_PA-RO-MA.py 2>&1 | tee Main_Script_Executor_Tr_PA_Eval_PA-RO-MA.txt',
    'Main_Script_Executor_Tr_MA_Eval_MA-PA-RO.py 2>&1 | tee Main_Script_Executor_Tr_MA_Eval_MA-PA-RO.txt',
    'Main_Script_Executor_Tr_RO_Eval_RO-PA-MA.py 2>&1 | tee Main_Script_Executor_Tr_RO_Eval_RO-PA-MA.txt'
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_DA_Single.py 2>&1 | tee Main_Script_Executor_Tr_MA_DA_Single.txt',
    'Main_Script_Executor_Tr_PA_DA_Single.py 2>&1 | tee Main_Script_Executor_Tr_PA_DA_Single.txt',
    'Main_Script_Executor_Tr_RO_DA_Single.py 2>&1 | tee Main_Script_Executor_Tr_RO_DA_Single.txt'
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_DA_Multi_Balanced.py 2>&1 | tee Main_Script_Executor_Tr_MA_DA_Multi_Balanced.txt',
    'Main_Script_Executor_Tr_PA_DA_Multi_Balanced.py 2>&1 | tee Main_Script_Executor_Tr_PA_DA_Multi_Balanced.txt',
    'Main_Script_Executor_Tr_RO_DA_Multi_Balanced.py 2>&1 | tee Main_Script_Executor_Tr_RO_DA_Multi_Balanced.txt'
]

for reference in REFERENCES:
    Schedule.append("python " + reference + " ")

for i in range(len(Schedule)):
    try:
        os.system(Schedule[i])
    except Exception as e:
        print(e)
        continue