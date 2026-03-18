import os
from utils import extract_code_from_string, get_response, SpacedRateLimiter
from prompts import STD_PROMPTS, PH_PROMPTS, COT_PROMPTS
import argparse
from natsort import natsorted
import time
from logger import get_logger
from tqdm.asyncio import tqdm
from base_line import Standard, Process_Hint, Chain_of_Thought
from utils import read_problem, read_test_samples, test_generated_code, Result, test_generated_code_derict
from concurrent.futures import ThreadPoolExecutor
import asyncio
from asyncio import Semaphore
from datetime import datetime
os.environ["GRB_LOG_TO_CONSOLE"] = "0"  # ← 禁止控制台输出
from optrust import Optrust
from coe import COE


global_semaphore = asyncio.Semaphore(100)  # 限制最大并发数为 10
rate_limiter = SpacedRateLimiter(interval=1.0)

logger = get_logger()

async def main(save_path, dataset='LPWP', problem='', model_type='std', prompts=None):

    if model_type == 'std':
        if not prompts:
            prompts = STD_PROMPTS
        model = Standard(prompt_data=prompts)
    elif model_type == 'ph':
        if not prompts:
            prompts = PH_PROMPTS
        model = Process_Hint(prompt_data=prompts)
    elif model_type == 'cot':
        if not prompts:
            prompts = COT_PROMPTS
        model = Chain_of_Thought(prompt_data=prompts)
    elif model_type == 'opt':
        model = Optrust()
        if prompts:
            model.update_prompts(prompts)
    else:
        raise ValueError("Invalid model type. Choose from 'std', 'ph', 'cot'.")

    # print('-'*20,'\n')
    # print(f'Solving Problem: {problem}')

    base_dir = 'dataset'
    problem_dir = os.path.join(base_dir, problem)
    # # 读取所有dataset目录下的子目录
    # problems = [os.listdir(dataset_dir)][0]

    problem_data = read_problem(dataset, problem)

    async with global_semaphore:
        time1 = time.time()
        await rate_limiter.wait_turn()
        logger.info(f"[{problem}] Sending request to LLM...")
        response = await model.run(problem_description=problem_data['description'], code_example=problem_data['code_example'])
        time2 = time.time()
        logger.info(f"[{problem}] Received response from LLM in {time2 - time1:.2f} seconds.")

    time2 = time.time()
    generated_code = extract_code_from_string(response)
    time3 = time.time()
    logger.info(f"[{problem}] Extracted code from response in {time3 - time2:.2f} seconds.")

    # 保存回答和code
    with open(os.path.join(save_path, f"{problem}_response.txt"), 'w', encoding='utf8') as f:
        f.write(response)

    code_path = os.path.join(save_path, f"{problem}_code.py")

    with open(code_path, 'w', encoding='utf8') as f:
        f.write(generated_code)

    time4 = time.time()
    logger.info(f"[{problem}] Saved generated code to {code_path} in {time4 - time3:.2f} seconds.")

    test_samples = read_test_samples(dataset, problem)



    if model_type in  ['opt', 'std'] or problem_data['code_example'] == '':
        test_code = test_generated_code_derict
    else:
        test_code = test_generated_code
    ground_truth = test_samples['output'][0]
    time5 = time.time()
    logger.info(f"[{problem}] Read test samples in {time5 - time4:.2f} seconds.")

    res = await test_code(problem=problem, test_code_path=code_path, test_samples=test_samples, ground_truth=ground_truth)
    time6 = time.time()
    logger.info(f"[{problem}] Tested generated code in {time6 - time5:.2f} seconds.")

    # res=(res,err_info)
    return res

async def main_cot(save_path, dataset='LPWP', problem='', prompts=None):

    model = COE()

    base_dir = 'dataset'
    problem_dir = os.path.join(base_dir, problem)

    if prompts:
        model.update_prompts(prompts)

    problem_data = read_problem(dataset, problem)

    async with global_semaphore:
        time1 = time.time()
        await rate_limiter.wait_turn()
        logger.info(f"[{problem}] Sending request to LLM...")
        response = await model.run(problem=problem_data)
        time2 = time.time()
        logger.info(f"[{problem}] Received response from LLM in {time2 - time1:.2f} seconds.")

    time2 = time.time()
    generated_code = extract_code_from_string(response)
    time3 = time.time()
    logger.info(f"[{problem}] Extracted code from response in {time3 - time2:.2f} seconds.")

    # 保存回答和code
    with open(os.path.join(save_path, f"{problem}_response.txt"), 'w', encoding='utf8') as f:
        f.write(response)

    code_path = os.path.join(save_path, f"{problem}_code.py")

    with open(code_path, 'w', encoding='utf8') as f:
        f.write(generated_code)

    time4 = time.time()
    logger.info(f"[{problem}] Saved generated code to {code_path} in {time4 - time3:.2f} seconds.")

    # test_code = test_generated_code_derict
    if problem_data['code_example'] == '':
        test_code = test_generated_code_derict
    else:
        test_code = test_generated_code

    test_samples = read_test_samples(dataset, problem)
    ground_truth = test_samples['output'][0]
    time5 = time.time()
    logger.info(f"[{problem}] Read test samples in {time5 - time4:.2f} seconds.")

    res = await test_code(problem=problem, test_code_path=code_path, test_samples=test_samples, ground_truth=ground_truth)
    time6 = time.time()
    logger.info(f"[{problem}] Tested generated code in {time6 - time5:.2f} seconds.")

    # res=(res,err_info)
    return res




        
async def batch_start(prompts=None, dataset='LPWP', model_type='ph', log_dir='run_test'):
    args = argparse.Namespace()
    args.dataset = dataset
    args.model_type = model_type

    log_dir_name = f'run_{args.model_type}_{args.dataset}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    save_path = os.path.join('outcomes', log_dir, log_dir_name)
    logger.info(f"Results will be saved to: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 读取问题数量
    base_dir = 'dataset'
    dataset_dir = os.path.join(base_dir, args.dataset)
    # 读取所有dataset目录下的子目录

    problems = natsorted(os.listdir(dataset_dir))

    # 剔除隐藏文件等非目录项
    problems = [p for p in problems if os.path.isdir(os.path.join(dataset_dir, p))]

    problems_len = len(problems)

    if model_type == 'coe':
        tasks = [main_cot(save_path, dataset=args.dataset, problem=problem, prompts=prompts) for problem in problems]
    else:
        tasks =[main(save_path, dataset=args.dataset, problem=problem, model_type=args.model_type, prompts=prompts) for problem in problems]


    # results = await asyncio.gather(*tasks)
        # 使用 tqdm.as_completed 显示进度

    results = []

    for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Testing Problems"):

        result = await coro

        results.append(result)


    # # 存储result
    # with open(os.path.join(save_path, "results.txt"), 'w', encoding='utf8') as f:
    #     for problem, res in zip(problems, results):
    #         if res is None:
    #             f.write(f"{problem}: No Result\n\n")
    #             continue
    #         f.write(f"{problem}: {res[0].name}\n")
    #         if res[1] is not None:
    #             f.write(f"Error Info: {res[1]}\n")
    #         f.write("\n")



    # results 存储的是 class Result(Enum):

    # ACCEPT = 0
    # WRONG_ANSWER = 1
    # RUNTIME_ERROR = 2
    # COMPILE_ERROR = 3
    total_correct = sum(1 for res in results if res[0] == Result.ACCEPT)
    total_grammer_err = sum(1 for res in results if res[0] == Result.COMPILE_ERROR)
    total_wrong_err = sum(1 for res in results if res[0] == Result.WRONG_ANSWER)
    total_problems = len(problems)

    errors = '\n'.join(res[1] for res in results if res[1] is not None)


    logger.info(f"Final Results: {total_correct}/{total_problems} correct, {total_grammer_err} grammar errors, {total_wrong_err} wrong answers.")
    logger.info(f"Errors Details:\n{errors}")
    print(f"Final Results: {total_correct}/{total_problems} correct, {total_grammer_err} grammar errors, {total_wrong_err} wrong answers.")

    return total_correct/total_problems, errors


if __name__ == '__main__':
    asyncio.run(batch_start(dataset='LPWP', model_type='cot'))