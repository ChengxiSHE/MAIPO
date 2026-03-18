from coe import COE

Prompt_std = """You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential.
In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.
You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem. Now the origin problem is as follow: {problem_description} Give your Python code directly."""
Sys_std = ''



STD_PROMPTS = {
    'prompt': Prompt_std,
    'system': Sys_std}




Prompt_ph = """You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.
You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:
{problem_description}
Let's analyse the problem step by step, and then give your Python code.
Here is a starter code:
{code_example}"""
Sys_ph = ''

PH_PROMPTS = {
    'prompt': Prompt_ph,
    'system': Sys_ph
}


Prompt_cot = """You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential.
In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.
You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.Now the origin problem is as follow:{problem_description}Let's analyse the problem step by step, and then give your Python code.Here is a starter code:{code_example}"""
Sys_cot = ''

COT_PROMPTS = {
    'prompt': Prompt_cot,
    'system': Sys_cot}
coe = COE()


COE_PROMPTS = coe.export()

student_knowledge = [
        "Describe a real-world operational decision-making scenario involving limited resources, competing objectives, or logistical constraints (e.g., workforce scheduling, supply chain distribution, or inventory management). Based on this scenario, formulate a mathematical optimization model using the framework of Operations Research. Specify the decision variables, objective function (minimize cost, maximize efficiency, etc.), and all relevant constraints (capacity, demand, time, budget, etc.). Ensure the model is linear (or specify if it should be integer, nonlinear, etc.) and explain any assumptions made.",
        "Construct an Operations Research model using mixed-integer linear programming (MILP) to solve a combinatorial optimization problem. Begin by selecting an appropriate problem class (e.g., facility location, vehicle routing, or project selection). Define the sets, parameters, decision variables, objective function, and constraints in standard mathematical notation. Justify why MILP is suitable for this problem and discuss potential solution approaches (e.g., branch-and-bound, cutting planes). Include considerations for scalability and computational tractability.",
        "You are a data scientist working with a hospital administration team aiming to optimize operating room (OR) scheduling to reduce patient waiting times and maximize surgical throughput. Build a deterministic Operations Research model that allocates surgeries to time slots over a weekly planning horizon, considering surgeon availability, room capacity, surgery durations, and priority levels. Formulate the model as a linear or integer program, clearly stating all components. Additionally, outline how this model could be integrated into a decision support system and what real-time data would be required for daily re-optimization."
    ]


if __name__ == '__main__':
    from utils import apply_optimized_json_to_dict, build_masked_json_from_dict

    print("Building masked JSON from COE_PROMPTS...")
    print(COE_PROMPTS)

    masked_prompts_json, mappings, template_data = build_masked_json_from_dict(COE_PROMPTS, prefix="STD")
    print("Masked Prompts JSON:")
    print(masked_prompts_json)
    print("\nMappings:")
    print(mappings)
    print("\nTemplate Data:")
    print(template_data)
    # 假设 optimized_json_str 是从某个地方获得的优化后的 JSON 字符串
    optimized_json_str = masked_prompts_json  # 这里仅作示例，实际使用时应替换为真实的优化结果
    final_prompts = apply_optimized_json_to_dict(optimized_json_str, mappings, template_data)
    print("\nFinal Prompts after applying optimized JSON:")
    print(final_prompts)