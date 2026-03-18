import subprocess
import os

# model types: cot, coe, std, ph, optrust , dataset: LPWP, IndustryOR, MAMOEASY, MAMOCOMPLEX, NLP4LP
# model_types = ['cot', 'ph', 'opt', 'std', 'coe']
# model_types = ['cot', 'ph', 'opt', 'std']
model_types = ['coe']
# datasets = ['LPWP', 'IndustryOR', 'MAMOEASY', 'MAMOCOMPLEX', 'NLP4LP']
datasets = ['IndustryOR', 'MAMOEASY', 'MAMOCOMPLEX', 'NLP4LP', 'LPWP']



# already_run_log = os.listdir('app_logs/')
# already_run = []

# for model_type in model_types:
#     for dataset in datasets:
#         log_name = f'app_{model_type}_{dataset}.log'
#         if log_name in already_run_log:
#             already_run.append((model_type, dataset))

# print(f"Already run combinations: {already_run}")

# assert False, "Stop here to avoid rerunning completed experiments."



for model_type in model_types:
    for dataset in datasets:
        # if (model_type, dataset) in already_run:
        #     print(f"Skipping already run combination: {model_type}, {dataset}")
        #     continue
        app_log_name = f'app_coe/app_{model_type}_{dataset}.log'
        cmd = ['python', '-u', 'demo.py', '--algorithm', model_type, '--dataset', dataset]
        print(f"Running command: {' '.join(cmd)}")
        with open(app_log_name, 'w') as app_log_file:
            process = subprocess.Popen(cmd, stdout=app_log_file, stderr=subprocess.STDOUT)
            process.wait()
        print(f"Finished running command: {' '.join(cmd)}. Output logged to {app_log_name}")