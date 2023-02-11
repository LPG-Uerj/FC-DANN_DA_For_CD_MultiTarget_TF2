import os
import warnings

warnings.filterwarnings("ignore")
Schedule = []

REFERENCES = [
'Main_Create_Chart_MA-PA.py',
'Main_Create_Chart_MA-RO.py',
'Main_Create_Chart_PA-MA.py',
'Main_Create_Chart_PA-RO.py',
'Main_Create_Chart_RO-MA.py',
'Main_Create_Chart_RO-PA.py'
]

for reference in REFERENCES:
    Schedule.append("python " + reference + " ")

for i in range(len(Schedule)):
    try:
        os.system(Schedule[i])
    except Exception as e:
        print(e)
        continue