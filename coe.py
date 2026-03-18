from config import BASE_URL, MODEL_NAME, API_KEY
from utils import get_response
import numpy as np
import json
from utils import test_generated_code, test_generated_code_derict, extract_code_from_string
import random


class BaseExpert:
    
    def __init__(self):
        self.get_response = get_response

    
    def get_forward_parameters(self, problem, comment_pool):
        parameters = {
            'problem_description': problem['description'],
            'comments_text': comment_pool.get_comments(self.id)
        }

        return parameters

    def get_backward_parameters(self, problem, feedback_pool):
        parameters = {
            'problem_description': problem['description'],
            'previous_answer': self.previous_answer,
            'feedback': feedback_pool.get_comments(self.id)
        }
        return parameters



    async def forward(self, problem, comment_pool):
        # comment = comment_pool.get_comments(self.id)
        parameters = self.get_forward_parameters(problem, comment_pool)
        prompt = self.forward_template.format(**parameters)
        response = await self.get_response(prompt=prompt, system_message=self.sys)
        self.previous_answer = response
        return response

    async def backward(self, problem, feedback_pool):
        parameters = self.get_backward_parameters(problem, feedback_pool)
        prompt = self.backward_template.format(**parameters)
        response = await self.get_response(prompt=prompt, system_message=self.sys)
        return response





class CodeReviewer(BaseExpert):

    ROLE_DESCRIPTION = 'You are a code reviewer that conducts thorough reviews of the implemented code to identify any errors, inefficiencies, or areas for improvement.'
    FORWARD_TASK = '''As a Code Reviewer, your responsibility is to conduct thorough reviews of implemented code related to optimization problems. 
You will identify possible errors, inefficiencies, or areas for improvement in the code, ensuring that it adheres to best practices and delivers optimal results. Now, here is the problem: 
{problem_description}. 

You are supposed to refer to the comments given by your colleagues from other aspects: {comments_text}'''

    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The answer you give previously is as follow:
{previous_answer}
    
The feedback is as follow:
{feedback}

The output format is a JSON structure followed by refined code:
{{
    'is_caused_by_you': false,
    'reason': 'leave empty string if the problem is not caused by you',
    'refined_result': 'Your refined answer...'
}}
'''


    def __init__(self):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.backward_template = self.BACKWARD_TASK
        self.id = 'CodeReviewer'


class LPFileGenerator(BaseExpert):

    ROLE_DESCRIPTION = 'You are an LP file generator that expertises in generating LP (Linear Programming) files that can be used by optimization solvers.'
    FORWARD_TASK = '''As an LP file generation expert, your role is to generate LP (Linear Programming) files based on the formulated optimization problem. 

LP files are commonly used by optimization solvers to find the optimal solution. 
Here is the important part source from LP file format document: {knowledge}. 

Your expertise in generating these files will help ensure compatibility and efficiency. 
Please review the problem description and the extracted information and provide the generated LP file: 
{problem_description}.

The comments given by your colleagues are as follows: 
{comments_text}, please refer to them carefully.'''

    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The feedback is as follow:
{feedback}

The modeling you give previously is as follow:
{previous_answer}

The output format is a JSON structure followed by refined code:
{{
    "is_caused_by_you": false,
    "reason": "leave empty string if the problem is not caused by you",
    "refined_result": "Your refined result"
}}
'''
    def __init__(self):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.backward_template = self.BACKWARD_TASK
        self.id = 'LPFileGenerator'


    def get_forward_parameters(self, problem, comment_pool):
        parameters = super().get_forward_parameters(problem, comment_pool)
        knowledge = None
        parameters['knowledge'] = knowledge
        return parameters


class ModelingExpert(BaseExpert):

    ROLE_DESCRIPTION = 'You are a modeling expert specialized in the field of Operations Research and Optimization. Your expertise lies in Mixed-Integer Programming (MIP) models, and you possess an in-depth understanding of various modeling techniques within the realm of operations research. At present, you are given an Operations Research problem, alongside additional insights provided by other experts. The goal is to holistically incorporate these inputs and devise a comprehensive model that addresses the given production challenge.'

    FORWARD_TASK = '''Now the origin problem is as follow:
{problem_description}
And the comments from other experts are as follow:
{comments_text}

Give your MIP model of this problem. Additionally, please note that your model needs to be a solvable linear programming model or a mixed-integer programming model. This also means that the expressions of the constraint conditions can only be equal to, greater than or equal to, or less than or equal to (> or < are not allowed to appear and should be replaced to be \geq or \leq).

Your output format should be a JSON like this:
{{
    "VARIABLES": "A mathematical description about variables",
    "CONSTRAINS": "A mathematical description about constrains",
    "OBJECTIVE": "A mathematical description about objective"
}}
'''

    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The feedback is as follow:
{feedback}

The modeling you give previously is as follow:
{previous_answer}

The output format is a JSON structure followed by refined code:
{{
    "is_caused_by_you": false,
    "reason": "leave empty string if the problem is not caused by you",
    "refined_result": "Your refined result"
}}
'''

    def __init__(self):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.backward_template = self.BACKWARD_TASK
        self.id = 'ModelingExpert'



class ModelingKnowledgeSupplementExpert(BaseExpert):

    ROLE_DESCRIPTION = 'You are an experts that offers supplementary knowledge related to modeling techniques and best practices.'
    FORWARD_TASK = '''You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:
{problem_description}
Let's analyse the problem step by step, and then give your Python code.
Here is a starter code:
{code_example}
And the comments from other experts are as follow:
{comments_text}

Give your Python code directly.'''
    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The code you give previously is as follow:
{previous_answer}
    
The feedback is as follow:
{feedback}

The output format is a JSON structure followed by refined code:
{{
    'is_caused_by_you': false,
    'reason': 'leave empty string if the problem is not caused by you',
    'refined_result': 'Your refined code...'
}}
'''

    def __init__(self,):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.backward_template = self.BACKWARD_TASK
        self.id = 'ModelingKnowledgeSupplementExpert'

    def get_forward_parameters(self, problem, comment_pool):
        parameters = super().get_forward_parameters(problem, comment_pool)
        parameters['code_example'] = problem.get('code_example', '')
        return parameters


class ParameterExtractor(BaseExpert):

    ROLE_DESCRIPTION = 'You are an expert that identifies and extracts relevant variables from the problem statement.'
    FORWARD_TASK = '''As a parameter extraction expert, your role is to identify and extract the relevant variables, constrans, objective from the problem statement. 
Your expertise in the problem domain will help in accurately identifying and describing these variables. 
Please review the problem description and provide the extracted variables along with their definitions: 
{problem_description}

And the comments from other experts are as follow:
{comments_text}

Please note that the information you extract is for the purpose of modeling, which means your variables, constraints, and objectives need to meet the requirements of a solvable LP or MIP model. Within the constraints, the comparison operators must be equal to, greater than or equal to, or less than or equal to (> or < are not allowed to appear and should be replaced to be \geq or \leq).
'''
    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The code you give previously is as follow:
{previous_answer}
    
The feedback is as follow:
{feedback}

The output format is a JSON structure followed by refined code:
{{
    'is_caused_by_you': false,
    'reason': 'leave empty string if the problem is not caused by you',
    'refined_result': 'Your refined code...'
}}
'''

    def __init__(self,):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.backward_template = self.BACKWARD_TASK
        self.id = 'ParameterExtractor'

    def get_forward_parameters(self, problem, comment_pool):
        parameters = super().get_forward_parameters(problem, comment_pool)
        knowledge = None
        parameters['knowledge'] = knowledge
        return parameters

class ProgrammingExampleProvider(BaseExpert):

    ROLE_DESCRIPTION = 'You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.'
    
    FORWARD_TASK = '''You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:
{problem_description}
Let's analyse the problem step by step, and then give your Python code.
Here is a starter code:
{code_example}
And the comments from other experts are as follow:
{comments_text}

Give your Python code directly.'''
    
    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The feedback is as follow:
{feedback}

The modeling you give previously is as follow:
{previous_answer}

The output format is a JSON structure followed by refined code:
{{
    "is_caused_by_you": false,
    "reason": "leave empty string if the problem is not caused by you",
    "refined_result": "Your refined result"
}}
'''

    def __init__(self,):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.backward_template = self.BACKWARD_TASK
        self.id = 'ProgrammingExampleProvider'

    def get_forward_parameters(self, problem, comment_pool):
        parameters = super().get_forward_parameters(problem, comment_pool)
        parameters['code_example'] = problem.get('code_example', '')
        return parameters



class ProgrammingExpert(BaseExpert):

    ROLE_DESCRIPTION = 'You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.'
    FORWARD_TASK = '''You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:
{problem_description}
Let's analyse the problem step by step, and then give your Python code.
Here is a starter code:
{code_example}
And the comments from other experts are as follow:
{comments_text}

Give your Python code directly. You should follow the format of given code example strictly. No code is required outside the function except for the import package (No test code). In your code, the model must be a solvable LP or MIP model.'''
    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The code you give previously is as follow:
{previous_answer}
    
The feedback is as follow:
{feedback}

The output format is a JSON structure followed by refined code:
{{
    'is_caused_by_you': false,
    'reason': 'leave empty string if the problem is not caused by you',
    'refined_result': 'Your refined code...'
}}
'''

    def __init__(self,):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.backward_template = self.BACKWARD_TASK
        self.id = 'ProgrammingExpert'

    def get_forward_parameters(self, problem, comment_pool):
        parameters = super().get_forward_parameters(problem, comment_pool)
        parameters['code_example'] = problem.get('code_example', '')
        return parameters



class TerminologyInterpreter(BaseExpert):

    ROLE_DESCRIPTION = 'You are a terminology interpreter who provides additional domain-specific knowledge to enhance problem understanding and formulation.'
    FORWARD_TASK = '''As a domain knowledge terminology interpreter, your role is to provide additional information and insights related to the problem domain. 
Here are some relevant background knowledge about this problem: {knowledge}. 

You can contribute by sharing your expertise, explaining relevant concepts, and offering suggestions to improve the problem understanding and formulation. 
Please provide your input based on the given problem description: 
{problem_description}

Your output format should be a JSON like this (choose at most 3 hardest terminology):
[
  {{
    "terminology": "...",
    "interpretation": "..."
  }}
]
'''

    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The feedback is as follow:
{feedback}

The answer you give previously is as follow:
{previous_answer}

The output format is a JSON structure followed by refined code:
{{
    'is_caused_by_you': false,
    'reason': 'leave empty string if the problem is not caused by you',
    'refined_result': 'Your refined result'
}}
'''

    def __init__(self,):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.backward_template = self.BACKWARD_TASK
        self.id = 'TerminologyInterpreter'

    def get_forward_parameters(self, problem, comment_pool):
        parameters = super().get_forward_parameters(problem, comment_pool)
        knowledge = None
        parameters['knowledge'] = knowledge
        return parameters

class Comment:
    def __init__(self, expert, text):
        self.expert = expert
        self.text = text

class CommentPool:
    def __init__(self, all_experts, visible_matrix):
        self.comment = []
        self.all_experts = all_experts
        self.visible_matrix = visible_matrix
        self.expert_name_to_id = {expert.id: expert for expert in all_experts}

    def add_comment(self, comment: Comment):
        self.comment.append(comment)

    def pop_comment(self):
        return self.comment.pop()

    # def get_comments(self, expert_name):
    #     id_ = self.expert_name_to_id[expert_name]
    #     visible_experts = self.visible_matrix[id_]
    #     comment_list = []
    #     for comment in self.comment:
    #         target_id = self.expert_name_to_id[comment.expert.id]
    #         if visible_experts[target_id]:
    #             comment_list.append(comment.text)
    #     return comment_list

    def get_comments(self, expert_name):
        return [comment.text for comment in self.comment]


    def get_current_comment_text(self):
        comments_text = ''
        if len(self.comment) == 0:
            comments_text = 'There is no comment available, please ignore this section.\n'
        else:
            for comment in self.comment:
                comments_text += comment.expert.id + ': ```' + comment.text + '```\n'

        return comments_text

    def __len__(self):
        return len(self.comment)




##########################################################


class Conductor(BaseExpert):
    ROLE_DESCRIPTION='''you will take on the role of the conductor for a multi-expert system.'''
    FORWARD_TASK = '''Now, you are presented with an operational optimization-related problem: 
{problem_description}

In this multi-expert system, there are many experts, each of whom is responsible for solving part of the problem.
Your task is to CHOOSE THE NEXT EXPERT TO CONSULT.

The names of the experts and their capabilities are listed below:
{experts_info} 

Experts that have already been commented include: 
{commented_experts}

Please select an expert to consult from the remaining expert names {remaining_experts}.

Please note that the maximum number of asked experts is {max_collaborate_nums}, and you can ask {remaining_collaborate_nums} more times.

You should output the name of expert directly. The next expert is:'''

    def __init__(self,):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.id = 'Conductor'



    async def forward(self, problem, comment_pool, max_collaborate_nums):
        all_experts = comment_pool.all_experts
        all_experts_name = [e.id for e in all_experts]
        commented_experts_name = [c.expert.id for c in comment_pool.comment]

        experts_info = '\n'.join([str(e) for e in all_experts])
        commented_experts = str(commented_experts_name)     
        remaining_experts = str(list(set(all_experts_name) - set(commented_experts_name)))


        parameters = {
            'problem_description': problem['description'],
            'experts_info': experts_info,
            'commented_experts': commented_experts,
            'remaining_experts': remaining_experts,
            'max_collaborate_nums': max_collaborate_nums,
            'remaining_collaborate_nums': max_collaborate_nums - len(commented_experts_name)
        }

        prompt = self.forward_template.format(**parameters)

        answer = await self.get_response(prompt=prompt, system_message=self.sys)

        print(f'Conductor selected expert: {answer}')
        
    

        expert_name_to_obj = { e.id: e for e in all_experts }
        for name, expert in expert_name_to_obj.items():
            if name.lower() in answer.lower():
                return expert

        print('Can not find expert, random choice!')
        return random.choice(list(expert_name_to_obj.values()))

    

class Reducer(BaseExpert):
    
    ROLE_DESCRIPTION = 'You are an expert that responsible for summarize the comment of all other experts then conclude the final answer'
    FORWARD_TASK = '''Now, you are an expert of Operations Research.
You are supposed to give the final code of an problem.
Text description of the problem: {problem_description}
Your colleagues are all experts in various related fields. They have given their own insights. I hope you will carefully refer to these comments when giving the final code:
{comment_text}

No code is required outside the function except for the import package (No test code).
Your final code is as following:
'''

    def __init__(self,):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.id = 'Reducer'

    def get_forward_parameters(self, problem, workspace):
        parameters = {
            'problem_description': problem['description'],
        }
        comment_text = workspace.get_current_comment_text()
        parameters['comment_text'] = comment_text
        return parameters


class Evaluator(BaseExpert):

    ROLE_DESCRIPTION = '''You are an evaluator.'''
    FORWARD_TASK = '''You will be responsible for generating test samples for verifying the correctness of a program.

You will be given an operations research optimization problem and its function signature, and you are responsible for generating an input example for testing the function.
The test data you generate must be reasonable, solvable, and realistic.
Output JSON directly without any other information!

Input:
problem: A candy store mixes regular candy and sour candy to prepare two products, regular mix and sour surprise mix. Each kilogram of the regular mix contains 0.8 kg of regular candy and 0.2 kg of sour candy. The profit per kilogram of the regular mix is $3. Each kilogram of the sour surprise mix contains 0.1 kg of regular candy and 0.9 kg of sour candy. The profit per kilogram of the sour surprise mix is $5. The candy store has 80 kg of regular candy and 60 kg of sour candy available. How many kilograms of each type of candy mix should be created to maximize profits?
code:
def prob_29(regular_mix, sour_surprise_mix, constraint1, constraint2):
    """
    Args:
        regular_mix: a float, the amount of regular mix candy created
        sour_surprise_mix: a float, the amount of sour surprise mix candy created
        constraint1: an integer, the limit of available regular candy
        constraint2: an integer, the limit of available sour candy
    Returns:
        obj: a float, the maximum profit achieved
    """
    obj = 1e9
    # To be implemented
    return obj

Output:
{{
    "input": {{
        "regular_mix": 94.2,
        "sour_surprise_mix": 45.7,
        "constraint1": 80,
        "constraint2": 60
    }}
}}

Input:
problem: {problem_description}
code:
{code_example}

Output:
'''

    def __init__(self,):
        super().__init__()
        self.sys = self.ROLE_DESCRIPTION
        self.forward_template = self.FORWARD_TASK
        self.id = 'Evaluator'

    def get_forward_parameters(self, problem, comment_pool):
        parameters = super().get_forward_parameters(problem, comment_pool)
        parameters['code_example'] = problem.get('code_example', '')
        return parameters


    async def evaluate(self, problem, test_code_path, test_samples=None, ground_truth=None):
        feed_back = ''
        if test_samples is None and ground_truth is None:
            raise ValueError("Either test_samples or ground_truth must be provided.")
        if test_samples is None:
            _, feed = await test_generated_code_derict(problem, test_code_path, ground_truth=ground_truth)
        else:
            _, feed = await test_generated_code(problem, test_code_path, test_samples=test_samples)

        return feed_back













class COE:

    def __init__(self, max_trials=3, max_collaborate_nums=4, enable_evaluator=False):
        self.code_reviewer = CodeReviewer()
        self.lp_file_generator = LPFileGenerator()
        self.modeling_expert = ModelingExpert()
        self.modeling_knowledge_supplement_expert = ModelingKnowledgeSupplementExpert()
        self.parameter_extractor = ParameterExtractor()
        self.programming_example_provider = ProgrammingExampleProvider()
        self.programming_expert = ProgrammingExpert()
        self.terminology_interpreter = TerminologyInterpreter()

        self.agents = [
            self.code_reviewer,
            self.lp_file_generator,
            self.modeling_expert,
            self.modeling_knowledge_supplement_expert,
            self.parameter_extractor,
            self.programming_example_provider,
            self.programming_expert,
            self.terminology_interpreter]

        self.num_experts = len(self.agents)

        self.comment_pool = CommentPool(
            all_experts=self.agents,
            visible_matrix=np.ones((self.num_experts, self.num_experts), dtype=bool)
        )

        self.conductor = Conductor()
        self.evaluator = Evaluator()
        self.reducer = Reducer()
        self.max_trials = max_trials
        self.max_collaborate_nums = max_collaborate_nums
        self.enable_evaluator = enable_evaluator



    async def run(self, problem):
        # print('Starting COE framework...')
        expert_stack = []
        for _ in range(self.max_trials):
            for _ in range(self.max_collaborate_nums):
                next_expert = await self.conductor.forward(
                    problem=problem,
                    comment_pool=self.comment_pool,
                    max_collaborate_nums=self.max_collaborate_nums
                )
                # print(f'Next expert selected: {next_expert.id}')
                comment_text = await next_expert.forward(problem=problem, comment_pool=self.comment_pool)
                # print(f'Comment from {next_expert.id} collected.')
                self.comment_pool.add_comment(Comment(expert=next_expert, text=comment_text))
                expert_stack.append(next_expert.id)
            answer = await self.reducer.forward(
                problem=problem,
                comment_pool=self.comment_pool
            )

            return answer

            # code = extract_code_from_string(answer)

            # with open('final_code.py', 'w', encoding='utf8') as f:
            #     f.write(code)

            # if self.enable_evaluator:
            #     test_samples = None
            #     ground_truth = 100
            #     feedback = await self.evaluator.evaluate(
            #         problem=problem,
            #         test_code_path='final_code.py',
            #         test_samples=test_samples,
            #         ground_truth=ground_truth
            #     )
            #     if feedback == '':
            #         print('Final code passed all tests!')
            #         return answer
            #     else:
            #         print('Final code failed tests, continuing optimization...')
            #         # You can implement further optimization based on feedback here

            #     assert False, 'Debug: Evaluator is not fully implemented yet.'


    def export(self):
        return {
            item.id: {
                "sys": item.sys,
                "forward_template": item.forward_template,
                "backward_template": item.backward_template
            }
            for item in self.agents
        }


    def update_prompts(self, optimized_dict: dict):
        uid_index = {item.id: item for item in self.agents}
        for uid, data in optimized_dict.items():
            if uid not in uid_index:
                raise KeyError(f"Unknown UUID returned by LLM: {uid}")

            item = uid_index[uid]

            if isinstance(data, dict):
                if "forward_template" in data:
                    item.forward_template = data["forward_template"]
                if "backward_template" in data:
                    item.backward_template = data["backward_template"]
                if "sys" in data:
                    item.sys = data["sys"]

        missing = set(uid_index.keys()) - set(optimized_dict.keys())
        if missing:
            raise KeyError(f"UUIDs missing in LLM response: {missing}")





if __name__ == '__main__':
    # code_reviewer = CodeReviewer()
    # async def main():
    #     result = await code_reviewer.forward(problem_description="Solve the following linear programming problem: Maximize z = 3x + 4y subject to the constraints: 2x + y <= 20, 4x + 5y <= 40, x >= 0, y >= 0.", comments_text="The model formulation looks good, but consider adding more constraints related to resource limitations.")
    #     print(result)

    # import asyncio
    # asyncio.run(main())

    coe_frame = COE(enable_evaluator=True)

    problem_description = '''You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:
Minimize the cost of production while meeting the demand for products A and B. Each unit of product A requires 2 hours of labor and 3 units of raw material, while each unit of product B requires 4 hours of labor and 2 units of raw material. The profit per unit of product A is $5, and for product B, it is $8. The total available'''

    problem = {
        'description': problem_description,
        'code_example': '',
        'knowledge': ''
    }

    import asyncio
    asyncio.run(coe_frame.run(problem=problem))