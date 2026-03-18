import re
import json
import os
import requests
# from config import BASE_URL, API_KEY, MODEL_NAME
from openai import AsyncOpenAI as OpenAI
from openai import BadRequestError, RateLimitError, AuthenticationError
from enum import Enum
from logger import get_logger
import importlib
import sys
import ast
from typing import Dict, Tuple
import itertools
import time
import asyncio
import subprocess


BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
API_KEY = os.environ.get('API_KEY')


PLACEHOLDER_PATTERN = re.compile(r"(\{\{.*?\}\}|\{[a-zA-Z0-9_]+\})")

PLACEHOLDER_PATTERN = re.compile(r"(\{\{.*?\}\}|\{[a-zA-Z0-9_]+\})")

TOKENS = [0, 0, 0]



def mask_placeholders(text: str, prefix: str):
    """
    把 text 中的 {xxx} / {{xxx}} 替换成 __PH_{prefix}_0__ 之类安全 token，
    返回: (new_text, mapping)
      - new_text: 替换后的文本
      - mapping: {token: 原始占位符} 映射，用于之后恢复
    """
    mapping = {}

    def repl(match):
        key = f"__PH_{prefix}_{len(mapping)}__"
        mapping[key] = match.group(0)
        return key

    new_text = PLACEHOLDER_PATTERN.sub(repl, text)
    return new_text, mapping


def unmask_placeholders(text: str, mapping: dict):
    """根据 mapping 把 token 恢复成原始占位符。"""
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def build_masked_json_from_dict(data: dict, prefix: str = "PROMPTS"):
    """
    通用版本：对 data 这个 dict 的所有 **字符串 value** 做占位符 mask，
    返回:
      - masked_json: 可以直接给 LLM 的 JSON 字符串
      - mappings: {field_name: {token: 原始占位符}} 的字典
      - template_data: 原始字典的一个拷贝，用于之后对齐 key 和做 fallback
    """
    if isinstance(data, str):
        try:
            data = ast.literal_eval(data)
        except Exception as e:
            raise ValueError("Input data string is not a valid dictionary.") from e

    masked = {}
    mappings = {}
    template_data = dict(data)  # 备份一份原始的，用于 fallback

    for field_name, raw_value in data.items():
        if isinstance(raw_value, str):
            field_prefix = f"{prefix}_{field_name}"
            new_text, mapping = mask_placeholders(raw_value, field_prefix)
            masked[field_name] = new_text
            mappings[field_name] = mapping
        else:
            # 非字符串，不做占位符处理，原样保留
            masked[field_name] = raw_value
            mappings[field_name] = {}

    masked_json = json.dumps(masked, indent=2, ensure_ascii=False)
    return masked_json, mappings, template_data

def apply_optimized_json_to_dict(optimized_json_str: str,
                                 mappings: dict,
                                 template_data: dict):
    """
    通用版本：
      - optimized_json_str: LLM 返回的 JSON 字符串
      - mappings: build_masked_json_from_dict 返回的占位符映射
      - template_data: 原始 dict，用于保证 key 集合不变 & 兜底

    返回：
      - final_data: 新的 dict，结构/字段名与 template_data 完全一致，
                    字符串中的占位符已恢复为原样。
    """
    # 处理 LLM 返回 ```json ... ``` 的情况
    fenced = re.search(r"```json\s*(\{.*\})\s*```", optimized_json_str, re.DOTALL)
    if fenced:
        optimized_json_str = fenced.group(1)

    try:
        optimized_masked = json.loads(optimized_json_str)
    except json.JSONDecodeError:
        # LLM 返回的不是合法 JSON，直接退回原始字典
        return dict(template_data)

    final_data = {}
    for field_name, original_value in template_data.items():
        mapping = mappings.get(field_name, {})

        # 只按原始 key 取值，如果 LLM 没给这个 key，就用原始值兜底
        masked_value = optimized_masked.get(field_name, original_value)

        if isinstance(masked_value, str):
            restored = unmask_placeholders(masked_value, mapping)
        else:
            # 如果模型乱改成非字符串，就沿用原始 value
            restored = original_value

        final_data[field_name] = restored

    return final_data




logger = get_logger()

# 存储结果的枚举类
class Result(Enum):

    ACCEPT = 0
    WRONG_ANSWER = 1
    RUNTIME_ERROR = 2
    COMPILE_ERROR = 3


def extract_code_from_string(input_string):
    # Match code within ```python ... ``` or ``` ... ``` blocks
    pattern = r'```(?:python)?\s*(.*?)\s*```'
    
    # Find all matches in the input string
    code_blocks = re.findall(pattern, input_string, re.DOTALL)

    if len(code_blocks) == 0:
        # print(f'Parse code error! {input_string}')
        return input_string
    elif len(code_blocks) == 1:
        return code_blocks[0]

    code_blocks = [code for code in code_blocks if 'pip' not in code]
    return '\n'.join(code_blocks)


def read_problem(dataset, problem_name):
    base_dir = 'dataset'
    with open(os.path.join(base_dir, dataset, problem_name, 'description.txt'), 'r', encoding='utf8') as f:
        description = f.read()


    code_example_path = os.path.join(base_dir, dataset, problem_name, 'code_example.py')

    if not os.path.exists(code_example_path):
        code_example = ''
    else:
        with open(os.path.join(base_dir, dataset, problem_name, 'code_example.py'), 'r', encoding='utf8') as f:
            code_example = f.read()

    return {
        'description': description,
        'code_example': code_example
    }


def read_test_samples(dataset, problem):
    with open(os.path.join('dataset', dataset, problem, 'sample.json'), 'r', encoding='utf8') as f:
        test_samples = json.load(f)
    if isinstance(test_samples, list):
        return test_samples[0]
    return test_samples


async def get_response(prompt='What opportunities and challenges will the Chinese large model industry face in 2025?', system_message='', base_url=BASE_URL, model_name=MODEL_NAME, api_key=API_KEY, iters=0):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url)

    if iters > 3:
        logger.error(f"Exceeded maximum retry attempts for prompt: {prompt}")
        return ''


    # time.sleep(1)

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0.2
        )

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        usage = (prompt_tokens, completion_tokens, total_tokens)

        res = response.choices[0].message.content

    except AuthenticationError as e:
        # print(type(e).__name__)
        logger.error(f"Authentication error: {e}")
        assert False, f"Authentication error: {e}"
    except BadRequestError as e:
        logger.error(f"Bad request error: {e}")
        res = ''
        usage = (0, 0, 0)
    except RateLimitError as e:
        logger.error(f"Rate limit error: {e}")
        await asyncio.sleep(5)
        res = await get_response(prompt, system_message, base_url, model_name, api_key, iters=iters+1)
        return res

    # request time out error
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timeout error: {e}")
        await asyncio.sleep(5)
        res = await get_response(prompt, system_message, base_url, model_name, api_key, iters=iters+1)
        return res
    except asyncio.TimeoutError as e:
        logger.error(f"Asyncio timeout error: {e}")
        await asyncio.sleep(5)
        res = await get_response(prompt, system_message, base_url, model_name, api_key, iters=iters+1)
        return res

    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        await asyncio.sleep(5)
        res = await get_response(prompt, system_message, base_url, model_name, api_key, iters=iters+1)
        return res

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        assert False, f"Unexpected error: {e}"

    global TOKENS
    TOKENS[0] += usage[0]
    TOKENS[1] += usage[1]
    TOKENS[2] += usage[2]
    logger.info(f"Total tokens used so far: {TOKENS}")


    # print(res)


    return res




async def test_generated_code(problem, test_code_path, test_samples, **kwargs):
    # codepath = 'outcomes/run_2024-06-10_15-30-00/problem_xxx/generated_code.py'
    # 转化为可导入的模块路径
    code = test_code_path.replace('/', '.').rstrip('.py')
    # 获取test_code_path的文件夹
    code_dir = os.path.dirname(test_code_path)

    module_name = os.path.splitext(os.path.basename(test_code_path))[0]


    try:
        # 用 importlib 从指定文件路径加载模块
        spec = importlib.util.spec_from_file_location(module_name, test_code_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create spec for {test_code_path}")

        module = importlib.util.module_from_spec(spec)
        # 注册到 sys.modules，方便后续 reload 等操作
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except SystemExit as e:
        # 这里就专门处理 sys.exit() 的情况
        logger.error(f"Generated code for problem {problem} called sys.exit({e.code}) during import")
        return Result.RUNTIME_ERROR, f"Generated code called sys.exit({e.code}) during import"
    except Exception as e:
        logger.error(f"Failed to import generated code for problem {problem}: {e}")
        return Result.COMPILE_ERROR, f"Failed to import generated code for problem {problem}: {e}"

    try:
        func = getattr(module, problem)
    except AttributeError as e:
        logger.error(f"Function {problem} not found in generated code: {e}")
        return Result.COMPILE_ERROR, f"Function {problem} not found in generated code: {e}"

    try:
        output = func(*test_samples['input'])
        ground_truth = test_samples['output'][0]
    except Exception as e:
        logger.error(f"Runtime error when executing function {problem}: {e}")
        return Result.RUNTIME_ERROR, f"Runtime error when executing function {problem}: {e}"

    with open(os.path.join(code_dir, f'{problem}_res.txt'), 'w', encoding='utf8') as f:
        f.write(f'Output:\n{output}\n')
        f.write(f'Ground Truth:\n{ground_truth}\n')

    res = await judge_result(output, ground_truth) if output is not None else None

    if res is not None and res == "ACCEPT":
        print(f"Generated code for problem {problem} passed the test.")
        return Result.ACCEPT, None
    elif res is not None and res == "REJECT":
        print(f"Generated code for problem {problem} failed the test.")
        return Result.WRONG_ANSWER, 'Wrong answer'
    else:
        return Result.COMPILE_ERROR, 'Judgment error'
    


async def test_generated_code_derict(problem, test_code_path, ground_truth=None, **kwargs):
    '''
    直接运行生成的代码文件，适用于简单的脚本形式的代码
    problem: 问题名称，对应存储的结果名
    test_code_path: 生成代码的文件路径
    返回: (Result, error_message)
    '''
    if not os.path.exists(test_code_path):
        return Result.COMPILE_ERROR, f"code_path not found: {test_code_path}"
    # codepath = 'outcomes/run_2024-06-10_15-30-00/problem_xxx/generated_code.py'
    # 获得test_code_path的文件夹
    code_dir = os.path.dirname(test_code_path)
    # 获取环境
    env = os.environ.copy()
    # 安全执行
    try:
        # 子线程运行
        proc = subprocess.run([sys.executable, test_code_path], capture_output=True, text=True, env=env, timeout=60, stdin=subprocess.DEVNULL)

    except subprocess.TimeoutExpired as e:
        logger.error(f"Generated code for problem {problem} timed out during execution")
        return Result.RUNTIME_ERROR, f"Generated code timed out during execution"
    except Exception as e:
        logger.error(f"Failed to execute generated code for problem {problem}: {e}")
        return Result.COMPILE_ERROR, f"Failed to execute generated code for problem {problem}:"

    stdout = proc.stdout or ''
    stderr = proc.stderr or ''

    with open(os.path.join(code_dir, f'{problem}_res.txt'), 'w', encoding='utf8') as f:
        f.write(f'STDOUT:\n{stdout}\n')
        f.write(f'STDERR:\n{stderr}\n')
        f.write(f'Ground Truth:\n{ground_truth}\n')

    if ground_truth is None:
        return Result.ACCEPT, None

    # 判断结果是否正确
    res = await judge_result(stdout, ground_truth) if ground_truth is not None else None


    if res is not None and res == "ACCEPT":
        print(f"Generated code for problem {problem} passed the test.")
        return Result.ACCEPT, None
    elif res is not None and res == "REJECT":
        print(f"Generated code for problem {problem} failed the test.")
        return Result.WRONG_ANSWER, 'Wrong answer'


    if proc.returncode != 0:
        logger.error(f"Generated code for problem {problem} exited with code {proc.returncode}")
        return Result.RUNTIME_ERROR, f"Generated code exited with code {proc.returncode}"
    
    return Result.COMPILE_ERROR, 'Judgment error'




    #     exec_globals = {}
    #     with open(test_code_path, 'r', encoding='utf8') as f:
    #         code = f.read()
    #     exec(code, exec_globals)
    # except SystemExit as e:
    #     logger.error(f"Generated code for problem {problem} called sys.exit({e.code}) during execution")
    #     return Result.RUNTIME_ERROR, f"Generated code called sys.exit({e.code}) during execution"
    # except Exception as e:
    #     logger.error(f"Failed to execute generated code for problem {problem}: {e}")
    #     return Result.COMPILE_ERROR, f"Failed to execute generated code for problem {problem}: {e}"

    # # 获得输出结果
    # output = exec_globals.get('output', None)
    # ground_truth = exec_globals.get('ground_truth', None)
    # with open(os.path.join(code_dir, f'{problem}_res.txt'), 'w', encoding='utf8') as f:
    #     f.write(f'Output:\n{output}\n')
    #     f.write(f'Ground Truth:\n{ground_truth}\n')
    # if output == ground_truth:
    #     return Result.ACCEPT, None
    # else:
    #     return Result.WRONG_ANSWER, f'Wrong answer, Output: {output}, Ground Truth: {ground_truth}'


# 判断结果是否正确的agent
async def judge_result(result, ground_truth):
    prompt = f'''

You are an expert judge. Your task is to determine whether the provided result is equivalent to the ground truth, allowing for minor formatting differences and small numerical errors.


- Ignore surrounding brackets (e.g., [ ], ( )), quotes, extra whitespace, or trailing punctuation in either the result or ground truth.

- If both values are numeric (integers or decimals), consider them matching if their relative error is within 5%. 

  That is: |result - ground_truth| / max(|ground_truth|, 1e-8) ≤ 0.05.

- For non-numeric answers, they should convey the same meaning or value (e.g., "yes" vs "Yes", "42" vs "forty-two" would not match unless explicitly equivalent).


The result is:

{result}


The ground truth is:

{ground_truth}


Please respond with only one word: "ACCEPT" if they match under these rules, otherwise "REJECT".

'''
    judgment = await get_response(prompt, system_message="You are a judge.")
    # print(judgment)
    return judgment.strip().upper()

def compute_shapley_values(values: Dict[Tuple[int, ...], float],
                           m: int) -> Dict[int, float]:
    """
    根据各 coalition 的价值 v(S) 计算每个玩家的 Shapley value φ_q(v)。
    values 的 key 是排序好的 tuple, 如 (0,1,3)。
    """
    # 这里只给出结构，具体实现可以之后再补
    shapley: Dict[int, float] = {j: 0.0 for j in range(m)}
    # TODO: 填写 Shapley value 的计算逻辑
    return shapley

def powerset(iterable):
    res = []
    for r in range(len(iterable)+1):
        for idx in itertools.combinations(range(len(iterable)), r):
            combine = " and ".join(iterable[i] for i in idx)
            res.append((idx, combine))

    return res



class SpacedRateLimiter:
    """Ensure at least `interval` seconds between *starts* of requests (global)."""
    def __init__(self, interval: float):
        self.interval = interval
        self._lock = asyncio.Lock()
        self._next_time = 0.0  # monotonic time when next request may start

    async def wait_turn(self):
        async with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                await asyncio.sleep(self._next_time - now)
            # reserve the next slot
            self._next_time = time.monotonic() + self.interval


def items_to_dict(items):
    return {
        item.id: {
            "template": item.template,
            "sys": item.sys
        }
        for item in items
    }


def update_items_from_uuid_dict(
    items,
    optimized_dict: dict):
    """
    Update PromptItem objects using UUID-keyed optimized dict from LLM.

    strict=True  -> missing or extra UUIDs will raise error
    strict=False -> silently skip mismatches
    """
    def build_uid_index(items) -> dict:
        return {item.id: item for item in items}
    uid_index = build_uid_index(items)

    # 1. 更新已有项
    for uid, data in optimized_dict.items():
        if uid not in uid_index:
            raise KeyError(f"Unknown UUID returned by LLM: {uid}")

        item = uid_index[uid]

        if isinstance(data, dict):
            if "template" in data:
                item.template = data["template"]
            if "sys" in data:
                item.sys = data["sys"]

    # 2. 可选：检查是否有 item 未被返回
    missing = set(uid_index.keys()) - set(optimized_dict.keys())
    if missing:
        raise KeyError(f"UUIDs missing in LLM response: {missing}")




if __name__ == '__main__':
    import asyncio

    async def main():
        for i in range(100):
            prompt = f"请计算{i}+{i**2}的结果，并用Python代码返回答案。"
            res = await get_response(prompt)
            print(res)

    asyncio.run(main())

    # # # test_generated_code('prob_0', 'outcomes/run_ph_LPWP_1765359243/prob_0_code.py', {'input': [10, 50, 5, 20, 100], 'output': [250]})

    # code_path = 'outcomes/run_std_ComplexOR_2025-12-20_17-05-22/aircraft_assignment_code.py'
    # probelem = 'aircraft_assignment_code'
    # asyncio.run(test_generated_code_derict(probelem, code_path, ground_truth=(30200)))
