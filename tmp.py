import os
import json

paths = [
    'dataset/NLP4LP', 'dataset/NLP4LP2']


for path in paths:
    problems = os.listdir(path)
    for problem in problems:
        problem_path = os.path.join(path, problem)
        if 'infeasible' in problem or 'unsolved' in problem:
            # delete the directory
            os.system(f'rm -rf {problem_path}')
            print(f'Deleted {problem_path}')
            continue
        sol = os.path.join(problem_path, 'solution.json')

        if not os.path.exists(sol):
            print(f'solution.json not found in {problem_path}, skipping...')
            os.system(f'rm -rf {problem_path}')
            continue
        
        ob = os.path.join(problem_path, 'sample.json')
        with open(sol, 'r', encoding='utf8') as f:
            sol_data = json.load(f)
        ans = sol_data['objective']
        sample = [
    {
        "input": {},
        "output": [ans]
    }

]
        with open(ob, 'w', encoding='utf8') as f:
            json.dump(sample, f, indent=4)
        print(f'Processed {problem_path}')

