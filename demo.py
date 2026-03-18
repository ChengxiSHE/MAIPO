from llmc_experts import Student, Professor, ErrorRecorder, Distributor, Conductor, Evolver
from utils import get_response as get_model_response
from prompts import STD_PROMPTS, PH_PROMPTS, COT_PROMPTS, COE_PROMPTS
import argparse
from logger import get_logger
from main_as import batch_start
from utils import apply_optimized_json_to_dict, build_masked_json_from_dict, powerset
import asyncio
from prompts import student_knowledge
from datetime import datetime
from optrust import Optrust


log = get_logger()


async def get_suggestion():
    parser = argparse.ArgumentParser(description="Optimize LLM Prompts using Multi-Agent Collaboration")
    parser.add_argument('--algorithm', type=str, default='cot', help="Optimization algorithm to use: 'coe' for Coalition of Experts")
    parser.add_argument('--dataset', type=str, default='Test', help='Dataset name')
    parser.add_argument('--num_students', type=int, default=3, help='Number of student agents')
    parser.add_argument('--max_iterations', type=int, default=3, help='Maximum number of iterations')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Convergence threshold')
    args = parser.parse_args()

    args.algorithm = args.algorithm.lower()
    
    if args.algorithm == 'std':
        initial_prompt = STD_PROMPTS
    elif args.algorithm == 'ph':
        initial_prompt = PH_PROMPTS
    elif args.algorithm == 'cot':
        initial_prompt = COT_PROMPTS
    elif args.algorithm == 'coe':
        initial_prompt = COE_PROMPTS
    elif args.algorithm == 'opt':
        model = Optrust()
        initial_prompt = model.export()
    else:
        raise ValueError("Invalid algorithm type. Choose from 'std', 'ph', 'cot', 'coe'.")

    print(initial_prompt)

    initial_prompt, mappings, template_data = build_masked_json_from_dict(initial_prompt, prefix="KEY")

    initial_prompt = str(initial_prompt)

    log.info(f"Initial Prompt:\n{initial_prompt}")

    # 创建学生实例
    students = [Student(knowledge=student_knowledge[j]) for j in range(args.num_students)]
    log.info(f"Created {args.num_students} student agents.")

    # 创建其他角色实例
    distributor = Distributor()
    professor = Professor()
    conductor = Conductor()
    error_recorder = ErrorRecorder()
    evolver = Evolver()


    R_prev = 0.0
    cur_errors = ''


    R_prev, cur_errors = await batch_start(prompts=None, dataset=args.dataset, model_type=args.algorithm, log_dir=f'run_{args.algorithm}_{args.dataset}_test')


    cur_errors = await error_recorder.summary_error(cur_errors)



    # 生成任务集
    G = distributor.generate_sub_goals()

    n = len(G)

    # with open('res.txt', 'w', encoding='utf8') as f:
    #     # 写入所有信息
    #     f.write(f'Current task: dataset={args.dataset}, algorithm={args.algorithm}, num_students={args.num_students}, max_iterations={args.max_iterations}, epsilon={args.epsilon}\n')


    # 开始优化
    for t in range(args.max_iterations):
        # print(args.max_iterations)
        # assert False
        # 存储suggestions的结构
        suggestionss = []
        for j in range(args.num_students):
            suggestion = ''
            for gi in G:
                # 修改
                gs = str(G)
                suggestion = await students[j].generate_suggestion(suggestion, gi, initial_prompt, cur_errors)
                log.info(f"Student {j} processed suggestion of step {gi}.")
                
                while True:
                    revision, approved = await professor.evaluate(suggestion)
                    log.info(f"Professor evaluated suggestion of step {gi} by Student {j}. Approved: {approved}")
                    if approved:
                        log.info(f"Suggestion for step {gi} by Student {j} approved.")
                        break
                    suggestion = await students[j].update_suggestion(suggestion, revision)
            suggestionss.append(suggestion)
                
        # 将每个学生任务1到n的suggestion合并
        coalition_suggestions = powerset(suggestionss)
        log.info(f"epoch {t}, Generated {len(coalition_suggestions)} coalition suggestions.")

        num_coalitions = len(coalition_suggestions)

        Rq = []

        shapley_data = {}


        for idx, coalition in coalition_suggestions:
            if len(coalition) == 0:
                continue
            improved_prompt = await conductor.generate_prompt(
                base_prompt=initial_prompt,
                coalition_suggestions=list(coalition),
                errors=cur_errors
            )

            # 评估新prompt
            the_prompt = apply_optimized_json_to_dict(improved_prompt, mappings, template_data)
            log.info(f'the_prompt:{the_prompt}')
            R, errors = await batch_start(prompts=the_prompt, dataset=args.dataset, model_type=args.algorithm, log_dir=f'run_{args.algorithm}_{args.dataset}_test')
            shapley_data[idx] = R
            # with open('res.txt', 'a', encoding='utf8') as f:
            #     # 写入所有信息
            #     f.write(f'Iteration {t}, Coalition {coalition}, R: {R}\n')


            R = 0.0  # TODO: 实现评估函数
            Rq.append((improved_prompt, R, errors))
            # log.info(f"Iteration {t}, Coalition size {len(coalition)}, R: {R}")
            # store Rq
            log.info(f'Rq: {Rq}')

        with open('res.txt', 'a', encoding='utf8') as f:
            f.write('#'*20)
            # store all information, dataset, algorithm, num_students, max_iterations, epsilon, time
            f.write(f'Iteration {t}, Dataset: {args.dataset}, Algorithm: {args.algorithm}, Num Students: {args.num_students}, Max Iterations: {args.max_iterations}, Epsilon: {args.epsilon}\n')
            # 写入所有信息
            f.write(f'Shapley Data at iteration {t}: {shapley_data}\n')
            # end
            f.write('\n')

        # 寻找Rq中评分最高的prompt
        improved_prompt, R_current, the_errors = max(Rq, key=lambda x: x[1])

        the_errors = await error_recorder.summary_error(the_errors)

        # update knowledges
        current_knowledges = [students[j].knowledge for j in range(args.num_students)]
        
        current_knowledges = str(current_knowledges)

        knowledges = await evolver.evolve_prompt(current_knowledges, R_current, the_errors)

        for j in range(args.num_students):
            students[j].knowledge = knowledges[j]

        
        # 检查收敛
        if abs(R_current - R_prev) < args.epsilon:
            print(f"Converged at iteration {t}")
            break
            
        # 更新当前最优prompt和评分
        if R_current > R_prev:
            initial_prompt = improved_prompt
            R_prev = R_current
            cur_errors = the_errors
            log.info(f"New Best Prompt at iteration {t} with R: {R_current}\n{initial_prompt}")



    # 存储带时间信息的最优prompt
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log.info(f"Final Optimized Prompt:\n{initial_prompt}")



    log.info('begin test')
    test_dataset = args.dataset + '2'
    # 评估新prompt
    the_prompt = apply_optimized_json_to_dict(initial_prompt, mappings, template_data)
    R2, _ = await batch_start(prompts=the_prompt, dataset=test_dataset, model_type=args.algorithm)
    R1, _ = await batch_start(dataset=test_dataset, model_type=args.algorithm)

    with open('res.txt', 'a', encoding='utf8') as f:
        f.write(f"Test Results on {test_dataset}: Baseline R: {R1}, Optimized R: {R2}")
        f.write('#'*20)


    log.info(f"Test Results on {test_dataset}: Baseline R: {R1}, Optimized R: {R2}")

if __name__ == '__main__':
    asyncio.run(get_suggestion())