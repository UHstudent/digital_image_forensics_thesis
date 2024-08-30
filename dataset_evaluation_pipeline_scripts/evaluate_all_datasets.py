import json
import os
import time
import subprocess

#add guarding code block to ensure this is only excecuted when script is called directly, and not when imported by others processes
if __name__ == '__main__':

    scripts = [
        "evaluate_columbia.py",
        "evaluate_coverage.py",
        "evaluate_IMD2020.py",
        "evaluate_cocoglide.py",
        "evaluate_IFS.py",
        "evaluate_in_the_wild.py",
    ]

    #create error log json file
    with open('error_log_main.json', 'w') as f:
        json.dump('hopefully this remains empty', f)
        f.write('\n')

    for dataset in scripts:
        try:
            print("Starting: " + dataset)
            start_time = time.time()
            result = subprocess.run(['python', dataset], check=True, capture_output=True, text=True)
            end_time = time.time()
            info_log = {
                "dataset" : dataset,
                "Time elapsed" : str(end_time - start_time)
            }
            with open('error_log_main.json', 'a') as f:
                json.dump(info_log, f)
                f.write('\n')
            
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            info_log = {
                "dataset" : dataset,
                "error": str(e),
                "Time elapsed" : str(end_time - start_time)
            }
            with open('error_log_main.json', 'a') as f:
                json.dump(info_log, f)
                f.write('\n')

    
