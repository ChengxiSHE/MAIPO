# MAIPO: Multi-Agent Interactive Prompt Optimization for OR/Optimization

**MAIPO** is a multi-agent, iterative prompt-optimization framework for operations research and optimization tasks. It orchestrates multiple roles (Doctor, Professor, Evolver, Conductor, Student, etc.) to mimic a research group discussion loop that improves prompts, generates code, tests solutions, and feeds back errors.

## Technical Outline
- **Multi-role loop**: 3 Doctors propose stepwise improvements; Professor checks feasibility; Evolver mutates role prompts; Conductor deduplicates and merges for the next round.
- **Forward + feedback**: Prompt optimization is framed as policy-driven, interpretable reasoning with iterative error feedback.

## Key Files
- [demo.py](demo.py): Multi-agent optimization entry; orchestrates Student/Professor/Conductor/Evolver and evaluation.
- [main_as.py](main_as.py): Async batch evaluation/generation; supports std/ph/cot/opt/coe modes.
- [main.py](main.py): Threaded batch baseline.
- [base_line.py](base_line.py): Baseline prompt wrappers (Standard/Process_Hint/Chain_of_Thought).
- [prompts.py](prompts.py): Initial prompt templates (STD/PH/COT/COE) and student_knowledge.
- [llmc_experts.py](llmc_experts.py): Role implementations (Distributor/Student/Professor/Conductor/Evolver, etc.).
- [coe.py](coe.py): Coalition-of-Experts roles and forward/backward templates.
- [optrust.py](optrust.py): Optrust decomposition and modeling workflow (decompose, model, code gen).
- [utils.py](utils.py): LLM calls, placeholder masking/restoring, code testing, rate limiting.
- [config.py](config.py): LLM access config (replace API_KEY).
- Datasets: dataset/<DATASET>/<PROBLEM>/ with description.txt, optional code_example.py, sample.json.
- Outputs: outcomes/ (code and responses), logs/ (run logs), res.txt (iteration stats).

## Requirements
- Python 3.10+
- Core libs: openai, requests, tqdm, natsort, numpy (for COE), asyncio (builtin).
- Install example:
  ```bash
  pip install openai requests tqdm natsort numpy
  ```

## API Setup
Edit [config.py](config.py) to set BASE_URL, MODEL_NAME, API_KEY for your provider. Prefer reading from environment variables to avoid leaking keys.

## Dataset Format
Each problem folder contains:
- description.txt: natural-language task.
- sample.json: test I/O (if list, first element is used).
- Optional code_example.py: starter/template code.

## Quick Start
1) **Baseline solve/eval (async)**
   ```bash
   python main_as.py --dataset LPWP --model_type std   # or ph / cot / coe / opt
   ```
   Outputs go to outcomes/run_<algo>_<dataset>_*, logs in logs/.

2) **Multi-agent prompt optimization**
   ```bash
   python demo.py --algorithm cot --dataset Test --num_students 3 --max_iterations 3 --epsilon 0.01
   ```
   - Initial prompts: std/ph/cot/coe/opt.
   - Each round calls [main_as.py](main_as.py) to evaluate, logs errors, Evolver updates role knowledge, Conductor merges suggestions.

3) **COE standalone**
   ```bash
   python main_as.py --dataset LPWP --model_type coe
   ```

## Outputs
- Generated code and LLM responses: outcomes/<run_dir>/.
- Evaluation stats: console + logs/run_*.log; error summaries in res.txt (Shapley data, convergence notes, etc.).

## Repro Tips
- Start with the small dataset Test to validate pipeline and API config.
- To customize prompts, edit [prompts.py](prompts.py) or pass custom JSON in demo.py (use build_masked_json_from_dict/apply_optimized_json_to_dict to preserve placeholders).
- To add problems, drop them into dataset/<YourSet>/<Problem>/ following the format above.

## Acknowledgments
Inspired by lab-style multi-round collaboration, combining Doctor/Professor/Evolver/Conductor roles to turn prompt optimization into a structured POMDP-style evolution, with coalition/diversity ideas to keep suggestions varied and interpretable.
