import os
import warnings

warnings.filterwarnings("ignore")
Schedule = []


REFERENCES = [
    'Main_Script_Executor_Tr_MA_DA_Single.py 2>&1 | tee Main_Script_Executor_Tr_MA_DA_Single.txt',
    'Main_Script_Executor_Tr_PA_DA_Single.py 2>&1 | tee Main_Script_Executor_Tr_PA_DA_Single.txt',
    'Main_Script_Executor_Tr_RO_DA_Single.py 2>&1 | tee Main_Script_Executor_Tr_RO_DA_Single.txt'
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_DA_Single.py --train False 2>&1 | tee Main_Script_Executor_Tr_MA_DA_Single_test.txt',
    'Main_Script_Executor_Tr_PA_DA_Single.py --train False 2>&1 | tee Main_Script_Executor_Tr_PA_DA_Single_test.txt',
    'Main_Script_Executor_Tr_RO_DA_Single.py --train False 2>&1 | tee Main_Script_Executor_Tr_RO_DA_Single_test.txt'
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_DA_Multi_Balanced.py 2>&1 | tee Main_Script_Executor_Tr_MA_DA_Multi_Balanced.txt',
    'Main_Script_Executor_Tr_PA_DA_Multi_Balanced.py 2>&1 | tee Main_Script_Executor_Tr_PA_DA_Multi_Balanced.txt',
    'Main_Script_Executor_Tr_RO_DA_Multi_Balanced.py 2>&1 | tee Main_Script_Executor_Tr_RO_DA_Multi_Balanced.txt'
]

REFERENCES = [
    'Main_Script_Executor_Tr_PA_Eval_PA-RO-MA.py 2>&1 | tee Main_Script_Executor_Tr_PA_Eval_PA-RO-MA.txt',
    'Main_Script_Executor_Tr_MA_Eval_MA-PA-RO.py 2>&1 | tee Main_Script_Executor_Tr_MA_Eval_MA-PA-RO.txt',
    'Main_Script_Executor_Tr_RO_Eval_RO-PA-MA.py 2>&1 | tee Main_Script_Executor_Tr_RO_Eval_RO-PA-MA.txt'
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_DA_Multi_Balanced.py 2>&1 | tee Main_Script_Executor_Tr_MA_DA_Multi_Balanced.txt',
    'Main_Script_Executor_Tr_PA_DA_Multi_Balanced.py 2>&1 | tee Main_Script_Executor_Tr_PA_DA_Multi_Balanced.txt',
    'Main_Script_Executor_Tr_RO_DA_Multi_Balanced.py 2>&1 | tee Main_Script_Executor_Tr_RO_DA_Multi_Balanced.txt',
    'Main_Script_Executor_Tr_MA_DA_Single.py 2>&1 | tee Main_Script_Executor_Tr_MA_DA_Single.txt',
    'Main_Script_Executor_Tr_PA_DA_Single.py  2>&1 | tee Main_Script_Executor_Tr_PA_DA_Single.txt',
    'Main_Script_Executor_Tr_RO_DA_Single.py 2>&1 | tee Main_Script_Executor_Tr_RO_DA_Single.txt'
]


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
    'Main_Script_Executor_Tr_MA_DA_Multi_Target.py 2>&1 | tee Main_Script_Executor_Tr_MA_DA_Multi_Target.txt',
    'Main_Script_Executor_Tr_PA_DA_Multi_Target.py 2>&1 | tee Main_Script_Executor_Tr_PA_DA_Multi_Target.txt',
    'Main_Script_Executor_Tr_RO_DA_Multi_Target.py 2>&1 | tee Main_Script_Executor_Tr_RO_DA_Multi_Target.txt'
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_PA_DA_Multi_Source.py 2>&1 | tee Main_Script_Executor_Tr_MA_PA_DA_Multi_Source.txt',
    'Main_Script_Executor_Tr_MA_RO_DA_Multi_Source.py 2>&1 | tee Main_Script_Executor_Tr_MA_RO_DA_Multi_Source.txt',
    'Main_Script_Executor_Tr_PA_RO_DA_Multi_Source.py 2>&1 | tee Main_Script_Executor_Tr_PA_RO_DA_Multi_Source.txt'
]


REFERENCES = [
    'Main_Script_Executor_Tr_MA_DA_Multi_Target.py --train False --test False ',
    'Main_Script_Executor_Tr_PA_DA_Multi_Target.py --train False --test False ',
    'Main_Script_Executor_Tr_RO_DA_Multi_Target.py --train False --test False ',
    'Main_Script_Executor_Tr_MA_DA_Multi_Target.py --train False --test False --discriminate_domain_targets True ',
    'Main_Script_Executor_Tr_PA_DA_Multi_Target.py --train False --test False --discriminate_domain_targets True ',
    'Main_Script_Executor_Tr_RO_DA_Multi_Target.py --train False --test False --discriminate_domain_targets True ',
    'Main_Script_Executor_Tr_MA_PA_DA_Multi_Source.py --train False --test False ',
    'Main_Script_Executor_Tr_MA_RO_DA_Multi_Source.py --train False --test False ',
    'Main_Script_Executor_Tr_PA_RO_DA_Multi_Source.py --train False --test False ',
    'Main_Script_Executor_Tr_MA_DA_Single.py --train False --test False ',
    'Main_Script_Executor_Tr_PA_DA_Single.py --train False --test False ',
    'Main_Script_Executor_Tr_RO_DA_Single.py --train False --test False '
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_Eval_MA-PA-RO.py --train False ',
    'Main_Script_Executor_Tr_PA_Eval_PA-RO-MA.py --train False ',
    'Main_Script_Executor_Tr_RO_Eval_RO-PA-MA.py --train False '
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_DA_Multi_Target.py',
    'Main_Script_Executor_Tr_PA_DA_Multi_Target.py',
    'Main_Script_Executor_Tr_RO_DA_Multi_Target.py'
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_DA_Multi_Target.py --discriminate_domain_targets True ',
    'Main_Script_Executor_Tr_PA_DA_Multi_Target.py --discriminate_domain_targets True ',
    'Main_Script_Executor_Tr_RO_DA_Multi_Target.py --discriminate_domain_targets True '
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_Eval_MA-PA-RO.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_PA_Eval_PA-RO-MA.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_RO_Eval_RO-PA-MA.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_MA_DA_Single.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_PA_DA_Single.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_RO_DA_Single.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_MA_DA_Multi_Target.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_PA_DA_Multi_Target.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_RO_DA_Multi_Target.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_MA_DA_Multi_Target.py --train False --test False --metrics False --discriminate_domain_targets True ',
    'Main_Script_Executor_Tr_PA_DA_Multi_Target.py --train False --test False --metrics False --discriminate_domain_targets True ',
    'Main_Script_Executor_Tr_RO_DA_Multi_Target.py --train False --test False --metrics False --discriminate_domain_targets True ',
    'Main_Script_Executor_Tr_MA_PA_DA_Multi_Source.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_MA_RO_DA_Multi_Source.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_PA_RO_DA_Multi_Source.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_MA_PA_DA_Multi_Source_No_DA.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_MA_RO_DA_Multi_Source_No_DA.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_PA_RO_DA_Multi_Source_No_DA.py --train False --test False --metrics False ',
]

REFERENCES = [
    'Main_Script_Executor_Tr_MA_DA_Multi_Target.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_PA_DA_Multi_Target.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_RO_DA_Multi_Target.py --train False --test False --metrics False ',
    'Main_Script_Executor_Tr_MA_DA_Multi_Target.py --train False --test False --metrics False --discriminate_domain_targets True ',
    'Main_Script_Executor_Tr_PA_DA_Multi_Target.py --train False --test False --metrics False --discriminate_domain_targets True ',
    'Main_Script_Executor_Tr_RO_DA_Multi_Target.py --train False --test False --metrics False --discriminate_domain_targets True ',
]

REFERENCES = [
    'Main_Compute_All_Single_Target_MA-PA.py ',
    'Main_Compute_All_Single_Target_MA-RO.py ',
    'Main_Compute_All_Single_Target_PA-MA.py ',
    'Main_Compute_All_Single_Target_PA-RO.py ',
    'Main_Compute_All_Single_Target_RO-MA.py ',
    'Main_Compute_All_Single_Target_RO-PA.py ',
]

for reference in REFERENCES:
    Schedule.append("python " + reference + " 2>&1 | tee Main_Script_Executor_All_Multi.txt ")

for i in range(len(Schedule)):
    try:
        os.system(Schedule[i])
    except Exception as e:
        print(e)
        continue