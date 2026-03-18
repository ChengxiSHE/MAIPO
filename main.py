import os
from utils import extract_code_from_string, get_response
import argparse
from natsort import natsorted
import time
from logger import get_logger
from tqdm import tqdm
from base_line import Standard, Process_Hint, Chain_of_Thought
from utils import read_problem, read_test_samples, test_generated_code, Result
from concurrent.futures import ThreadPoolExecutor
import asyncio


logger = get_logger()

async def main(save_path, dataset='LPWP', problem_extent=':', model_type='std'):

    base_dir = 'dataset'
    dataset_dir = os.path.join(base_dir, dataset)
    # 读取所有dataset目录下的子目录
    problems = [os.listdir(dataset_dir)][0]
    # 排序
    problems = natsorted(problems)
    # 根据problem_extent筛选问题
    if problem_extent != ':':
        extent_parts = problem_extent.split('-')
        start_idx = int(extent_parts[0]) if extent_parts[0] else 0
        end_idx = int(extent_parts[1]) if len(extent_parts) > 1 and extent_parts[1] else len(problems)
        problems = problems[start_idx:end_idx]

    # 逐个测试问题
    correct_count = 0
    grammer_err = 0
    wrong_err = 0

    pbar = tqdm(total=len(problems), desc="Testing Problems")

    for problem in problems:
        problem_data = read_problem(dataset, problem)
        if model_type == 'std':
            model = Standard()
        elif model_type == 'ph':
            model = Process_Hint()
        elif model_type == 'cot':
            model = Chain_of_Thought()
        else:
            raise ValueError("Invalid model type. Choose from 'std', 'ph', 'cot'.")

        response = await model.run(problem_description=problem_data['description'], code_example=problem_data['code_example'])
        generated_code = extract_code_from_string(response)

        # 保存回答和code
        with open(os.path.join(save_path, f"{problem}_response.txt"), 'w', encoding='utf8') as f:
            f.write(response)

        code_path = os.path.join(save_path, f"{problem}_code.py")

        with open(code_path, 'w', encoding='utf8') as f:
            f.write(generated_code)

        test_samples = read_test_samples(dataset, problem)

        res = test_generated_code(problem, code_path, test_samples)

        if res == Result.ACCEPT:
            correct_count += 1
        elif res == Result.COMPILE_ERROR:
            grammer_err += 1
        elif res == Result.WRONG_ANSWER:
            wrong_err += 1

        pbar.update(1)

    pbar.close()

    return correct_count, grammer_err, wrong_err, len(problems)

        
def batch_start():
    parser = argparse.ArgumentParser(description="Batch Test Optimization Problem Solvers")
    parser.add_argument('--dataset', type=str, default='LPWP', help='Dataset name')
    parser.add_argument('--model_type', type=str, default='std', help="Model type: 'std' for Standard, 'ph' for Process Hint, 'cot' for Chain of Thought")

    args = parser.parse_args()

    log_dir_name = f'run_{args.model_type}_{args.dataset}_{str(round(time.time()))}'
    save_path = os.path.join('outcomes', log_dir_name)
    logger.info(f"Results will be saved to: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 读取问题数量
    base_dir = 'dataset'
    dataset_dir = os.path.join(base_dir, args.dataset)
    # 读取所有dataset目录下的子目录
    problems_len = len([os.listdir(dataset_dir)][0])
    # 划分为10个等份
    portion = problems_len // 10
    parts = []
    for i in range(10):
        start_idx = i * portion
        end_idx = (i + 1) * portion if i < 9 else problems_len
        parts.append(f"{start_idx}-{end_idx}")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, part in enumerate(parts):
            futures.append(executor.submit(main, save_path, args.dataset, part, args.model_type))
        
        for future in futures:
            future.result()  # 等待所有线程完成











if __name__ == '__main__':
    batch_start()