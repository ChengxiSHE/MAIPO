from config import BASE_URL, MODEL_NAME, API_KEY
from utils import get_response
from prompts import STD_PROMPTS, PH_PROMPTS, COT_PROMPTS




class Base:

    def __init__(self):
        self.get_response = get_response

    async def run(self, **kwargs):
        prompt = self.template.format(**kwargs)
        response = await self.get_response(prompt=prompt, system_message=self.sys)
        return response



class Standard(Base):
    def __init__(self, prompt_data=STD_PROMPTS):
        super().__init__()
        self.template = prompt_data['prompt']
        self.sys = prompt_data['system']


class Process_Hint(Base):
    def __init__(self, prompt_data=PH_PROMPTS):
        super().__init__()
        self.template = prompt_data['prompt']
        self.sys = prompt_data['system']

class Chain_of_Thought(Base):
    def __init__(self, prompt_data=COT_PROMPTS):
        super().__init__()
        self.template = prompt_data['prompt']
        self.sys = prompt_data['system']




if __name__ == '__main__':
    base_line = Standard()
    result = base_line.run(problem_description="Solve the following linear programming problem: Maximize z = 3x + 4y subject to the constraints: 2x + y <= 20, 4x + 5y <= 40, x >= 0, y >= 0.")
    print(result)