---
title: >-
  [Paper Note] RExBench: Can coding agents autonomously implement AI research extensions?
description: >-
  [ACL2026][Code Intelligence][Coding Agent] RExBench places coding agents within real AI paper codebases, requiring them to implement expert-designed research extensions and scoring them via controlled execution results.…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "Coding Agent"
  - "AI Research Agent"
  - "Research Extension"
  - "Automatic Evaluation"
  - "Code Generation Benchmark"
date: 2026-05-08
content_hash: 3ea16b456a774bd3
---

# RExBench: Can coding agents autonomously implement AI research extensions?

**Conference**: ACL2026  
**arXiv**: [2506.22598](https://arxiv.org/abs/2506.22598)  
**Code**: https://rexbench.com/  
**Area**: Code Intelligence  
**Keywords**: Coding Agent, AI Research Agent, Research Extension, Automatic Evaluation, Code Generation Benchmark

## TL;DR
RExBench places coding agents within real AI paper codebases, requiring them to implement expert-designed research extensions and scoring them via controlled execution results. It finds that even the strongest current agents achieve only about a one-third success rate, indicating a significant gap in autonomous research extension capabilities.

## Background & Motivation
**Background**: LLM agents are already capable of solving certain software engineering tasks, such as fixing GitHub issues, refactoring code, running experimental scripts, and processing data analysis workflows. Concurrently, AI for Science and automated research systems are attempting to have agents perform experimental design, code reproduction, and paper analysis.

**Limitations of Prior Work**: Existing benchmarks either lean towards general software engineering or evaluate paper reproduction, Kaggle-style modeling, or open-ended research Q&A. They rarely measure a capability closer to real research iteration: implementing a previously unpublished research extension based on existing papers and codebases and producing experimental results consistent with expert implementations.

**Key Challenge**: Research extension tasks must be realistic, open, and scientifically meaningful; however, automatic evaluation requires tasks to be executable, results to be verifiable, and environments to be controlled. Over-simplification loses the "research flavor," while excessive openness prevents stable scoring.

**Goal**: The authors aim to construct a benchmark that balances realism and automatic evaluation to assess whether coding agents can understand paper contexts, read original code, locate modification points, implement new experiments, and reproduce the numerical results of an expert gold solution in an isolated environment.

**Key Insight**: RExBench selects codebases from 12 NLP/ML papers, has domain experts design research extension tasks, and stores the solutions and success criteria in a private evaluation infrastructure. Agents only receive the paper, the codebase, and high-level extension instructions, eventually submitting a patch that the system executes to compare results.

**Core Idea**: Using "implementing unpublished research extensions" instead of "reproducing existing paper results" as a test of coding agents' research capabilities simultaneously mitigates data contamination and aligns more closely with real research workflows.

## Method
The task format of RExBench is: given one or more related papers, the original codebase, and expert-written extension instructions, the agent must edit the codebase to implement extension experiments. The system applies the agent-generated patch to the original repository, runs it in a fixed VM and container environment, and judges success based on output files or numerical results.

### Overall Architecture
The benchmark contains 12 research extension tasks covering four types of changes: models, algorithms, data, and evaluation methods. For example, the WinoDict task requires replacing synthetic target words with real English words from different frequency groups to check if existing word meanings interfere with in-context word acquisition; the Othello task requires changing the board state representation of a probe; the Tree of Thoughts task requires analyzing the algorithm's failure modes on specific models.

For each task, a domain expert first verifies that the original code can be reproduced, then implements a gold extension and records numerical results. After an agent submits a patch, the evaluation infrastructure executes it in the same hardware, random seed, and dependency environment. Final success depends on whether the agent's experimental output falls within the exact value or a narrow range of the gold results.

### Key Designs
1.  **Research Extension as the Core Task Unit**:
    - **Function**: Evaluates the agent's ability to "validate new hypotheses built upon existing research."
    - **Mechanism**: Each task is neither a minor bug fix nor a verbatim copy of paper reproduction; instead, the agent modifies the model, data, algorithm, or evaluation process based on expert instructions to obtain new experimental results.
    - **Design Motivation**: Real research often begins with "what if we replace X with Y." RExBench turns this extension into an executable task, which tests code understanding and experimental implementation in research scenarios better than simple coding problems.

2.  **Private Gold Solution and Controlled Execution Evaluation**:
    - **Function**: Reduces the risk of data contamination and improves the credibility of numerical evaluation.
    - **Mechanism**: Gold edits and evaluation scripts are not public; agents only submit patches. The system applies patches, runs experiments, and collects results and logs in task-specific Apptainer containers. For tasks with randomness, the gold solution uses 5 seeds to estimate the mean and a $\pm 2$ standard deviation range.
    - **Design Motivation**: If the solution were public, agents might succeed through memorization or training data leakage. Private evaluation makes success more representative of true autonomous implementation.

3.  **Multi-layer Metrics for Diagnosing Agent Failure**:
    - **Function**: Distinguishes between whether the code runs, whether the correct files were modified, and whether the final scientific result is correct.
    - **Mechanism**: The primary metric is the final success rate; auxiliary metrics include execution success rate and file recall. The former checks if experimental output matches the gold result, while file recall measures the overlap between files edited by the agent and those edited by the expert.
    - **Design Motivation**: Failures in research code tasks are often not simple compilation errors. An agent might find the right file but implement faulty logic, or the code might run but produce deviant results. Multi-layered metrics help analyze capability bottlenecks.

### Loss & Training
This paper does not train models but constructs a benchmark and evaluates agents. The experiments test 12 agent combinations using aider and OpenHands frameworks, with backbones including Claude 4/3.7 Sonnet, GPT-5, o1, o4-mini, and DeepSeek-R1. Each agent runs 5 times per task to estimate random fluctuation. The authors also tested two levels of human-written hints: level one helps locate information, and level two provides specific implementation steps.

## Key Experimental Results

### Main Results
The main results indicate that current agents are still significantly deficient in research extensions. The best combination, OpenHands + Claude 4 Sonnet, has an average final success rate of approximately 33%, with an execution success rate of 68%.

| Agent Setting | Final Success Rate | Execution Success Rate | Observations |
| :--- | :--- | :--- | :--- |
| OpenHands + Claude 4 Sonnet | ~33% | 68% | Strongest combo, yet fails most extensions |
| OpenHands + GPT-5 | Lower than Claude 4 | Strong and non-zero | Often syntactically correct but results deviate |
| Claude 3.7 / 4 Series | Significantly better | High | Can locate core files and generate runnable code |
| o1 / DeepSeek-R1 | Close to or 0% | DeepSeek-R1 failed | Reasoning models might overthink in agent loops |
| aider + o4-mini / R1 | Very low | Often empty patches | Non-iterative frameworks struggle with complex tasks |

These results show that while strong backbones bring agents closer to correct implementations, a large gap remains between "runnable" and "scientifically correct."

### Ablation Study
The authors provided different levels of hints to observe if agents could leverage human prompts.

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| No hints | Best final success ~33% | Realistic autonomous research extension setting |
| Info localization hints | Improvement for some strong agents | Strong models implement better when files are pointed out |
| Step-level hints | OpenHands + Claude 4/GPT-5 reach 43% | Specific steps help strong agents, but still <50% |
| Weak agents + hints | Limited improvement | Requires basic code understanding to use hints |
| Individual tasks | Hints may decrease performance | If hints specify a path the model is not good at |

### Key Findings
- Most agents find the core editing areas (high file recall) but have low final success, indicating the bottleneck is not file localization but understanding experimental logic and implementation details.
- Explicit errors include Python ValueErrors, empty patches, SyntaxErrors, and execution timeouts. Claude and GPT-5 series rarely produce SyntaxErrors.
- Implicit errors are more dangerous: the code runs, but results do not match the gold solution. Analysis of top-2 agents shows the ratio of logical errors to numerical/parameter errors in implicit failures is approximately 2:1.
- Stronger models produce more implicit errors. This implies future agents may "crash" less obviously but more frequently provide plausible-looking but scientifically incorrect implementations.
- The number of lines changed in the gold solution has a significant negative impact on final success ($\beta=-0.038, p<0.01$), indicating implementation workload is a primary source of difficulty.

## Highlights & Insights
- RExBench advances agent evaluation from "writing code that passes unit tests" to "executing verifiable research extensions." This is critical for research agents, as real risks stem from incorrect experimental conclusions rather than syntax errors.
- The design of private gold solutions is important. Compared to tasks like PaperBench, research extensions naturally reduce the possibility of training data leakage and better test online reasoning and code comprehension.
- The paper notes that failures of stronger models are harder to debug, serving as a realistic engineering warning. Runnable but incorrect agent patches could mislead researchers by introducing faulty experiments into papers.
- Hint experiments provide insights into human-agent collaboration: agents do not simply succeed with any hint. There is an interaction between hint granularity, pathing, and model capability, requiring more robust human-agent protocols in the future.

## Limitations & Future Work
- For automatic evaluation, tasks must have clear numerical targets, making them more idealized than real open research. In reality, extension ideas often require multiple rounds of trial and error and redefinition.
- Currently, there are only 12 tasks. While each task is information-dense, the statistical power to distinguish between similar models is limited. Expanding the number of tasks and domain coverage is crucial.
- The benchmark focuses on NLP/ML code. Experimental environments, data scales, simulators, and metrics in other scientific fields may introduce additional challenges.
- Process-level metrics are insufficient. The authors suggest adding landmark evaluations to alleviate the difficulty of post-hoc analysis of implicit errors.
- Executing machine-written code automatically poses security risks; the use of internet-free container environments is a necessary safeguard. Future deployments require stricter sandboxing, permission controls, and auditing.

## Related Work & Insights
- **vs SWE-bench**: SWE-bench tests GitHub issue fixing; RExBench tests research hypothesis extensions. The latter emphasizes experimental logic, paper understanding, and numerical results.
- **vs PaperBench / Paper2Code**: These focus on reproduction or paper-to-code; RExBench focuses on unpublished extensions, better mitigating data contamination.
- **vs MLE-bench / MLAgentBench**: These are more like ML pipelines or Kaggle problems; RExBench inputs include paper context and original code, making tasks closer to research iteration.
- **Insight**: Future research agent evaluation should measure "runnable," "interpretable," "correct results," and "reviewable" separately, rather than relying solely on a final leaderboard score for reliability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Evaluating agents via research extensions rather than reproduction is a realistic and contamination-resistant approach.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Tasks are deep and analysis is detailed, but the number of tasks is relatively small and concentrated in NLP/ML.
- Writing Quality: ⭐⭐⭐⭐☆ The benchmark construction and error analysis are clear.
- Value: ⭐⭐⭐⭐⭐ High reference value for research agents, secure execution, automated experimental evaluation, and human-agent collaboration design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents](codedistiller_automatically_generating_code_libraries_for_scientific_coding_agen.md)
- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](../../NeurIPS2025/code_intelligence/mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)
- [\[ICML 2026\] MARS: Modular Agent with Reflective Search for Automated AI Research](../../ICML2026/code_intelligence/mars_modular_agent_with_reflective_search_for_automated_ai_research.md)
- [\[ICLR 2026\] InnoGym: Benchmarking the Innovation Potential of AI Agents](../../ICLR2026/code_intelligence/innogym_benchmarking_the_innovation_potential_of_ai_agents.md)
- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)

</div>

<!-- RELATED:END -->
