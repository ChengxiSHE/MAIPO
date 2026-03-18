from utils import get_response as get_model_response


class Distributor:

    def generate_sub_goals(self, dataset=None):
        """

        You are a goal decomposition assistant. Your response must follow this exact template:

        Total steps: [number]
        Step 1: [description]
        Step 2: [description]
        ...
        Step N: [description]

        For example:
        Total steps: 3
        Step 1: Analyze the task requirements.
        Step 2: Identify the resources needed.
        Step 3: Execute the plan.

        Please follow this format strictly.

        :param dataset: 训练数据 D
        :param K: 学生的知识或个性集 K = {k1, ..., km}
        :return: 子目标列表 G = [g1, ..., gn]
        """

        steps = [ \
"Interpret the prompt's purpose, scope, and expected outputs, and identify missing elements or unclear specifications.", \
"Rewrite and reorganize the prompt to improve rigor, reduce ambiguity, and enhance operational usability for modeling tasks.", \
"Review the optimized prompt for correctness, logical coherence, and execution feasibility, ensuring alignment with OR modeling standards."
]

        return steps


class Student:

    SYSTEM = '''
You are an expert in prompt engineering for automatic modeling and problem solving.

You are given:
- Current Suggestion: {suggestion}
- Current Prompt: {current_prompt}
- Current step: {g_i}
- Errors observed for the current prompt: {Errors}

Your goal:
Based on the current step Current step and the corresponding Errors observed for the current prompt, provide concrete, step-specific suggestions to improve Current Prompt so that the same errors are less likely to occur in the next iteration.

Instructions:
- Focus ONLY on how to modify or extend the current prompt in the NEXT revision.
- Use Current step to keep your suggestions specific to the current step (do not discuss other steps).
- Use Errors observed for the current prompt to:
  - Identify which parts or omissions in Current Prompt likely caused the issues.
  - Suggest how to clarify, constrain, or reorganize the prompt to avoid these errors.

Constraints:
- Do NOT rewrite, replace, or directly edit the full prompt text.
- Do NOT output a new or fully rewritten prompt.
- Do NOT change, rename, or introduce any placeholder variables (anything inside curly braces in Current Prompt must remain exactly as is).

Output format:
- Output ONLY a numbered list of concrete, actionable improvement suggestions (1., 2., …).
- List up to 2 suggestions.
- Each suggestion should:
  - Refer explicitly to a specific part / behavior of Current Prompt, and
  - Explain how adjusting it at this step will help fix or reduce the given Errors observed for the current prompt.
'''


    OUTPUT = ''


    def __init__(self, knowledge):
        self.knowledge = knowledge
        # prompt = ChatPromptTemplate([
        #     ("system",self.knowledge), 
        #     ("ai", self.OUTPUT),
        #     ("human", self.SYSTEM),
        # ])
        # api_key = "sk-13426092a6f04f28bf95f2c4da7317e4"
        # llm = ChatOpenAI(model_name="deepseek-chat", temperature=0, openai_api_key=api_key, base_url="https://api.deepseek.com")
        # self.chain = prompt | llm | StrOutputParser()

    async def generate_suggestion(self, suggestion:str, g_i, current_prompt, Errors=None):
        task = self.SYSTEM.format(suggestion=suggestion, g_i=g_i, current_prompt=current_prompt, Errors=Errors)
        return await get_model_response(task, system_message=self.knowledge)


    async def update_suggestion(self, suggestion, revision):
        # 拼prompt
        prompt = f'''Original Suggestion:
{suggestion}

Revision:
{revision}

Task:
Update the original suggestion according to the revision.

Rules:
- Output only the updated suggestion.
- Do NOT include explanations or justification.
- Do NOT repeat the revision text verbatim; integrate it into the suggestion.
'''
        # 调用模型获取更新后的建议
        updated_suggestion = await get_model_response(prompt)

        return updated_suggestion


class Professor:

    TASK = '''
You are an evaluator. Your job is to review the given suggestion, revise it if needed,
and decide whether it should be approved based on the following criteria:

- Feasibility: the suggestion can be realistically implemented.
- Accuracy: the suggestion is factually and technically correct for the task.
- Rationality: the suggestion is logically consistent and makes sense in context.

Suggestion to evaluate:
{suggestion}

Your required output format is STRICTLY the following two lines:

suggestion: <revised suggestion>;
approved: <True or False>

Approval rule:
- If the revised suggestion satisfies feasibility, accuracy, and rationality, set approved to True.
- If any of these criteria is clearly not satisfied, set approved to False.

Rules you MUST follow:
1. Your output MUST contain EXACTLY two lines.
2. Do NOT output explanations, analysis, or reasoning.
3. Do NOT add any extra text before or after the two required lines.
4. "suggestion:" MUST appear exactly as written, as the first line.
5. after the colon of "suggestion:" and the revised suggestion, it must has a ";".
6. "approved:" MUST appear exactly as written, as the second line.
7. approved MUST be either "True" or "False" (capital T/F).
8. If no revision is needed, copy the original suggestion into the revised suggestion field.

Your output should be the following format:
suggestion: <revised suggestion>;
approved: <True or False>

'''

    def __init__(self):
        # prompt = ChatPromptTemplate([
        #     ("system", "You are a strict professor who evaluates student suggestions."),
        #     ("ai", self.OUTPUT),
        #     ("human", self.TASK),
        # ])
        # api_key = "sk-13426092a6f04f28bf95f2c4da7317e4"
        # llm = ChatOpenAI(model_name="deepseek-chat", temperature=0, openai_api_key=api_key, base_url="https://api.deepseek.com")
        # self.chain = prompt | llm | StrOutputParser()
        pass

    async def evaluate(self, suggestion):
        
        # 调用模型进行评估
        # evaluation = self.chain.invoke({
        #     "suggestion": suggestion
        # })

        return '', True


        evaluate_prompt = self.TASK.format(suggestion=suggestion)
        evaluation = await get_model_response(evaluate_prompt,)


        # print(evaluation)


        # 解析模型输出，假设格式为 "Revision: ...; Approved: True/False"
        parts = evaluation.split(';', 1)
        revision_part = parts[0].replace("suggestion:", "").strip()
        approved_part = parts[1].replace("approved:", "").strip()
        # approved_part = evaluation[:-4]
        # # print(approved_part.lower())
        if 'approved: True' in evaluation:
            approved_part = 'True'
        elif 'approved: False' in evaluation:
            approved_part = 'False'
        approved = approved_part.lower() == 'true'
        return revision_part, approved

class Professor_V2(Professor):

    async def evaluate(self, suggestion):
        
        # 调用模型进行评估
        # evaluation = self.chain.invoke({
        #     "suggestion": suggestion
        # })

        return '', True



class Evolver:

    TASK = """
You are an expert in prompt engineering for automatic modeling and problem solving.

You are given the following information:

1. The agent's current prompt, provided as a LIST of prompt components:
{prompt}

2. The accuracy rate of the agent's last response:
{r}

3. The error information from the last attempt:
{error}

Your task is to refine the agent's prompt to improve correctness, robustness, and problem-solving effectiveness,
based on the observed accuracy and error information.

IMPORTANT CONSTRAINTS:
- The prompt is a LIST. You MUST preserve the list structure.
- Do NOT remove list elements unless they are redundant or directly responsible for the observed errors.
- Do NOT merge all elements into a single paragraph unless the original list has only one element.
- Do NOT change the underlying task, goal, or intent of any list element.
- Do NOT modify or remove any placeholder variables enclosed in curly braces.

Optimization guidelines:
- Improve clarity and precision within each list element.
- Strengthen constraints or instructions related to the observed errors.
- Reduce ambiguity that may cause incorrect modeling, reasoning, or implementation.
- Maintain logical consistency and ordering among list elements.

OUTPUT REQUIREMENTS:
- Output ONLY the refined prompt as a LIST.
- Do NOT include explanations, comments, headings, or additional text.
- The output list must be directly usable as a replacement for the original prompt list.
"""



    async def evolve_prompt(self, prompt, r, errpr):
        task = self.TASK.format(prompt=prompt, r=r, error=errpr)
        return await get_model_response(task)



class ErrorRecorder:

    @staticmethod
    async def summary_error(err_info):
        return await get_model_response(f"Please summarize the errors occurred in the following log for problem {err_info}", system_message="You are an expert log analyzer.")



class Conductor:

    async def generate_prompt(self,
                        base_prompt,
                        coalition_suggestions='',
                        errors=''):
        prompt = prompt = f"""
You are provided with the following information:

1. Base Prompt:
{base_prompt}

2. Suggestions:
{coalition_suggestions}

3. Errors observed in previous attempts:
{errors}

Your task is to refine the Base Prompt to improve its clarity, coherence, and effectiveness,
while strictly preserving its original intent and functionality.

STRICT RULES (must be followed):
- You MUST NOT change, rename, remove, reorder, or alter in any way any placeholder variables enclosed in curly braces (e.g., {{variable_name}}) that appear in the Base Prompt.
- You MUST NOT add new placeholders, remove existing placeholders, or modify the content inside any curly braces.
- You MUST NOT introduce new information, assumptions, examples, or external context.
- You MUST NOT change the meaning, scope, or objective of the original Base Prompt.
- You MAY only improve wording, sentence structure, and clarity.

OUTPUT REQUIREMENTS:
- Output ONLY the refined version of the Base Prompt.
- Do NOT include explanations, commentary, headings, markdown, or any additional text.
- Do NOT restate the rules or mention the input sections.

Failure to follow these rules will be considered an incorrect response.
"""

        improved_prompt = await get_model_response(prompt)
        return improved_prompt


class Conductor_V2:

    async def generate_prompt(self,
                        base_prompt,
                        coalition_suggestions='',
                        errors=''):
        prompt = prompt = f"""
You are provided with the following information:

1. Base Prompt:
{base_prompt}

2. Suggestions:
{coalition_suggestions}

3. Errors observed in previous attempts:
{errors}
"""

        improved_prompt = await get_model_response(prompt)
        return improved_prompt

