from base_line import Base
from utils import extract_code_from_string, test_generated_code, test_generated_code_derict

import asyncio
from utils import items_to_dict, update_items_from_uuid_dict


class Decomposition(Base):

    SYSTEM = """
    You are an expert in mathematical optimization. Your task is to identify and prepare natural language descriptions of components of an optimization problem.
    """

    TASK = """
    Upon receiving a problem description, you should:
    
    1. Carefully analyze and comprehend the problem.
    2. Summarize the decision variables related to the problem. Indicate whether each of the decision variables is required to be integer, real or binary based on the context of the problem.
    3. Summarize and define the objective of the problem. Indicate any parameters or numerical values needed to define the objective.
    4. Identify and list all constraints, including any implicit ones like non negativity. List and summarize the constraints using natural language. Indicate any parameters or numerical values needed to define each of the constraints
    5. Verify if any numerical values or parameters defined in the problem description are missing from the objective or constraints you identified, and update the list of components you prepared, if necessary.

    Note that:
    If adding any mathematical expressions, try to mathematically represent constraints and objectives as close to their natural language description as possible; you do not need to simplify any constraints or objectives.
    The final list of components should be enclosed between the "'''" lines.
    
    Here is a description of the problem we need you to find the components for:

    {description}
    ----

    Now, follow the steps outlined above. Explain your reasoning and remember to enclose the final list of components between the "'''" lines.
    """

    def __init__(self, prompt=TASK, system=SYSTEM):
        super().__init__()
        self.template = prompt
        self.sys = system
        self.id = 'decomposition'


class Decomposition_debug(Base):
    SYSTEM = """
    You are an expert in mathematical optimization. Your task is to review previously identified components of an optimization problem.
    """

    TASK = """
    Upon receiving the description of an optimization problem and a list of previously identified components of the optimization problem, you should:
    
    1. Carefully analyze and comprehend the problem.
    2. Verify if the decision variables, objectives and constraints listed in the previously prepared list of components have been identified correctly.
    3. Verify if any decision variables, objectives or constraints in the description of the optimization problem are missing from the previously prepared list of components and update the list, if necessary.
    4. Verify if any numerical values or parameters defined in the problem description are missing from the components you identified, and update the list of components you prepared to include those, if necessary.
    5. Prepare a final, revised list with the components of the optimization problem (including objectives, constraints and decision variables) in natural language. Make sure to avoid repeating components.
     
    Note that:
    - You should include any implicit constraints such as non-negativity
    - If adding any mathematical expressions, try to mathematically represent constraints and objectives as close to their natural language description as
    possible; you do not need to simplify any constraints or objectives.
    You should indicate whether each of the decision variables is required to be integer, real or binary based on the context of the problem.
    - The final list of components should be enclosed between the "'''" lines.

    Here is a description of the problem we need you to find the components for:
    
    {description}
    ----
    
    And here is the list of previously identified components:
    ----
    {previous_components}
    ----
    
    Now, follow the steps outlined above. Explain your reasoning and remember to enclose the final list of components between the "'''" lines.    
    """


    def __init__(self, prompt=TASK, system=SYSTEM):
        super().__init__()
        self.template = prompt
        self.sys = system
        self.id = 'decomposition_debug'




class Optimization_modeling(Base):

    SYSTEM = """
    You are an expert in mathematical optimization, and your task is to model an optimization problem.
    """

    TASK = """
    Upon receiving the description of an optimization problem, you should:
    
    1. Carefully analyze and comprehend the problem.
    2. Carefully review the decision variables previously identified. Define symbols representing the decision variables and indicate whether each of the decision variables is required to be integer, real or binary based on the context of the problem.
    3. Indicate whether any decision variables are required to be non-negative based on the context of the problem.
    4. Carefully review the previously identified objectives, and prepare a mathematical formulation representing the objective. If the optimization problem has multiple objectives, convert a multi-objective optimization problem into a single-objective optimization problem using linear scalarization with the weights of the objectives.
    5. Carefully review the previously identified constraints, and prepare a mathematical formulation representing each of the constraints.
    6. Prepare a mathematical formulation of the problem using LaTeX.
    7. Verify if any numerical values or parameters defined in the problem description are missing from the formulation, and update the mathematical formulation to include them, if necessary.
    
    Note that:
    - Try to mathematically represent constraints and objectives as close to their natural language description as possible.
    - You do not need to simplify any constraints or objectives.
    - Your formulation should be in LaTeX mathematical format.
    - The final mathematical formulation should be enclosed between the "'''" lines.
    
    Here is a description of the problem we need you to model:
    ----
    {description}
    ----
    
    The following components have been previously identified:
    ----
    {components}
    ----
    
    Now, solve the problem step by step. Explain your reasoning and remember to enclose the final list of components between the "'''" lines.
    """ 

    def __init__(self, prompt=TASK, system=SYSTEM):
        super().__init__()
        self.template = prompt
        self.sys = system
        self.id = 'optimization_modeling'


class Optimization_debug(Base):

    SYSTEM = """
    You're an expert in mathematical optimization. You need to revise the mathematical formulation of an optimization problem prepared by a student.
    """

    TASK = """
    Here is a description of the problem we need you to model:
    
    {description}
    ----
    
    The following components have been previously identified:
    ----
    {components}
    ----
    
    And here is the mathematical formulation we need you to verify:
    ----
    {previous_formulation}
    ----
    
    Solve the problem step by step. Explain your reasoning and remember to enclose the final list of components between the "'''" lines.
    """


    def __init__(self, prompt=TASK, system=SYSTEM):
        super().__init__()
        self.template = prompt
        self.sys = system
        self.id = 'optimization_debug'


class Programming(Base):

    SYSTEM = """
    You are an expert in mathematical optimization, and your objective is to create a Python script to solve an optimization problem using gorubi.
    """

    TASK = """
    When solving an optimization problem, you should follow a structured approach:
    
    1. Carefully analyze and comprehend the problem description.
    2. Carefully analyze and comprehend the provided decomposition of the problem into a detailed list of decision variables, objective(s) and constraints.
    3. Carefully analyze and comprehend the previously prepared mathematical formulation of the optimization problem.
    4. Prepare a well-documented Python script to solve the optimization problem using gorubi. Anchor your implementation on the context, the detailed decomposition, and the formulation of the optimization problem. Pay special attention to the domain of each decision variable, implicit constraints such as non-negativity, and that all relevant parameters are included in the script you generate.
    Note that
    - You should clearly explain your reasoning and the steps you take to solve the problem.
    - You should enclose the final code between "```python" lines, as in the provided examples.
    - You should print the optimal value of the optimization problem using 'Optimal value: ', as in the provided examples.
    Let's think step by step and clearly describe our reasoning.
    
    
    Here is the problem description:
    ----
    {description}
    ----
    
    The following components have been extracted from the problem description:
    ----
    {components}
    ----
    
    And the following mathematical formulation to represent the optimization problem has been prepared:
    ----
    {formulation}
    ----
    
    Now, follow the steps outlined above. Explain your reasoning and remember to enclose the generated code between "'''" lines.
    """



    def __init__(self, prompt=TASK, system=SYSTEM):
        super().__init__()
        self.template = prompt
        self.sys = system
        self.id = 'programming'



class Optrust():

    def __init__(self):

        self.decomposition = Decomposition()
        self.des_debug = Decomposition_debug()
        self.formulation = Optimization_modeling()
        self.formulation_debug = Optimization_debug()
        self.programming = Programming()


        self.agents = [self.decomposition, self.des_debug, self.formulation, self.formulation_debug, self.programming]

    def export(self):
        return items_to_dict(self.agents)

    def update_prompts(self, optimized_dict):
        update_items_from_uuid_dict(self.agents, optimized_dict)


    async def run(self, problem_description, **kwargs):
        pre_component = await self.decomposition.run(description=problem_description)
        verified_component = await self.des_debug.run(description=problem_description, previous_components=pre_component)
        formu = await self.formulation.run(description=problem_description, components=verified_component)
        verified_formu = await self.formulation_debug.run(description=problem_description, components=verified_component, previous_formulation=formu)
        code = await self.programming.run(description=problem_description, components=verified_component, formulation=verified_formu)
        return code



async def main():


    llm = Optrust()


    fig = llm.export()


    llm.update_prompts(fig)
    




    assert False

    file_path = 'dataset/LPWP/prob_0/description.txt'

    with open(file_path, 'r') as f:
        description = f.read()


    code = await llm.run(description=description)


    code = extract_code_from_string(code)

    code_path = 'test/test.py'

    with open(code_path, 'w') as f:
        f.write(code)


    samples = 'dataset/LPWP/prob_0/samples.json'

    res = await test_generated_code_derict('prob_0', code_path, 1300)

    print(res)



if __name__ == '__main__':

    asyncio.run(main())

