---
title: >-
  [Paper Note] RExBench: Can coding agents autonomously implement AI research extensions?
description: >-
  [ACL 2026][Code Intelligence][Coding Agent] RExBench places coding agents into real AI paper repositories to implement expert-designed research extensions. Performance is scored via controlled execution results, revealing that even the strongest current agents achieve only about a one-third success rate, indicating a significant gap in autonomous research capabi
tags:
  - ACL 2026
  - Code Intelligence
  - Coding Agent
  - AI Research Agent
date: 2026-05-08
content_hash: f7f5a6c58c976460
---
# RExBench: Can coding agents autonomously implement AI research extensions?

**Conference**: ACL2026  
**arXiv**: [2506.22598](https://arxiv.org/abs/2506.22598)  
**Code**: https://rexbench.com/  
**Area**: Code Intelligence  
**Keywords**: Coding Agent, AI Research Agent, Research Extension, Automated Evaluation, Code Generation Benchmark

## TL;DR
RExBench places coding agents into real AI paper repositories to implement expert-designed research extensions. Performance is scored via controlled execution results, revealing that even the strongest current agents achieve only about a one-third success rate, indicating a significant gap in autonomous research capabilities.

## Background & Motivation
**Background**: LLM agents have demonstrated proficiency in certain software engineering tasks, such as fixing GitHub issues, refactoring code, executing experimental scripts, and managing data analysis pipelines. Concurrently, AI for Science and automated research systems are attempting to enable agents to perform experimental design, paper reproduction, and analysis.

**Limitations of Prior Work**: Existing benchmarks typically focus on general software engineering, paper reproduction, Kaggle-style modeling, or open-ended research Q&A. They rarely measure a capability closer to authentic scientific iteration: implementing an unpublished research extension on an existing codebase and producing experimental results consistent with an expert implementation.

**Key Challenge**: Research extension tasks must be authentic, open-ended, and scientifically meaningful; however, automated evaluation requires tasks to be executable, results to be verifiable, and environments to be controlled. Excessive simplification sacrifices the research nature of the task, while excessive openness prevents stable scoring.

**Goal**: The authors aim to construct a benchmark that balances authenticity with automated evaluation. It assesses whether coding agents can understand paper contexts, read original codebases, locate modification points, implement new experiments, and reproduce the numerical results of expert gold solutions in isolated environments.

**Key Insight**: RExBench selects codebases from 12 NLP/ML papers. Domain experts design research extension tasks and store the solutions and success criteria in a private evaluation infrastructure. Agents are provided only with the paper, the codebase, and high-level instructions; they submit patches which are then executed and compared against expert results.

**Core Idea**: By replacing "reproduction of existing results" with "implementation of unpublished research extensions" as the test for coding agent research capability, the benchmark simultaneously mitigates data contamination and aligns with actual research workflows.

## Method

### Overall Architecture
The RExBench task format is as follows: given one or more related papers, the original codebase, and expert-written extension instructions, the agent must edit the codebase to implement a research extension experiment not performed in the original paper. The system applies the agent’s patch to the repository and executes it within a fixed VM or container, judging success based on output files or numerical results. The benchmark contains 12 research extension tasks covering four categories: model, algorithm, data, and evaluation methodology. For example, the WinoDict task requires replacing synthetic target words with real English words from different frequency groups to check if existing semantic knowledge interferes with in-context word acquisition; the Othello task involves modifying the board state representation of a probe; and the Tree of Thoughts task involves analyzing the failure modes of the algorithm on specific models. Each task is first verified by experts to ensure the original code is reproducible, after which they manually implement a gold extension and record the numerical results. After an agent submits a patch, the evaluation infrastructure executes it using the same hardware, random seeds, and dependencies. Success is determined by whether the agent's experimental results fall exactly on or within a narrow range of the gold values.

### Key Designs

**1. Research Extension as the Core Task Unit: Verifying New Hypotheses Instead of Reproducing Existing Papers**

Real research often begins with "what happens if I change X to Y?" Existing benchmarks are either biased toward general software engineering or focus on reproduction and Kaggle-style modeling, rarely testing activities tied to authentic research iteration. In RExBench, tasks are not about fixing minor bugs or verbatim reproduction; instead, agents must modify models, data, algorithms, or evaluation pipelines to obtain new experimental results. Making "extensions" executable forces agents to demonstrate code understanding and experimental implementation skills required in scientific scenarios more effectively than simple coding problems.

**2. Private Gold Solution and Controlled Execution: Keeping Solutions Private to Reduce Contamination and Ensure Credibility**

If gold solutions are public, agents might succeed through memorization or training data leakage. RExBench hides gold edits and evaluation scripts within a private infrastructure. Agents submit patches, and the system applies them, runs experiments, and collects logs within task-specific Apptainer containers. For stochastic tasks, gold solutions utilize five seeds to estimate a mean and a $\pm 2$ standard deviation interval as the success criterion, ensuring "success" reflects genuine autonomous implementation rather than memorizing answers.

**3. Multi-layer Diagnostic Metrics: Separating Execution, Localization, and Scientific Correctness**

Failures in research code tasks are often more complex than simple compilation errors—an agent might find the right file but implement incorrect logic, or produce runnable code that yields incorrect experimental results. RExBench establishes a hierarchy of metrics: the primary metric is the final success rate (whether experimental output matches the gold standard), supported by execution success rate (whether the code ran to completion) and file recall (the overlap between agent-edited files and expert-edited files). This tripartite breakdown reveals whether an agent failed at localization, execution, or experimental logic.

### Loss & Training
This paper does not train models; instead, it constructs a benchmark and evaluates existing agents. The experiments test 12 agent combinations using two frameworks, aider and OpenHands, with backbones including Claude 4/3.7 Sonnet, GPT-5, o1, o4-mini, and DeepSeek-R1. Each task is run five times per agent to estimate variance. The authors also test two levels of human-written hints: Level 1 assists with information localization, and Level 2 provides specific implementation steps.

## Key Experimental Results

### Main Results
The main experiments show that current agents are significantly deficient in research extensions. The best combination, OpenHands + Claude 4 Sonnet, achieved an average final success rate of approximately 33%, with an execution success rate of 68%.

| Agent Setup | Final Success Rate | Execution Success Rate | Key Observations |
| :--- | :--- | :--- | :--- |
| OpenHands + Claude 4 Sonnet | ~33% | 68% | Strongest combination; still fails most extensions. |
| OpenHands + GPT-5 | Lower than Claude 4 | Strong and non-zero | Often syntactically correct but results deviate. |
| Claude 3.7 / Claude 4 Series | Significantly better than weak models | Higher | Capable of locating core files and generating runnable code. |
| o1 / DeepSeek-R1 | Near or equal to 0 | DeepSeek-R1 failed completely | Reasoning models may "overthink" in agent loops. |
| aider + o4-mini / DeepSeek-R1 | Very Low | Often empty patches | Non-iterative frameworks are unsuitable for complex tasks. |

These results indicate that while powerful backbones bring agents closer to correct implementations, a large gap remains between "runnable" and "scientifically correct."

### Ablation Study
The authors provided different levels of hints to observe if agents could utilize human intervention.

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| No hints | Best final success ~33% | Realistic autonomous research extension setting. |
| Localization hints | Improvement for some strong agents | Strong models implement better when files/info are pointed out. |
| Step-level hints | OpenHands + Claude 4 / GPT-5 reach 43% | Specific steps help strong agents, but still below 50%. |
| Weak agent + hints | Limited improvement | Requires baseline code understanding to utilize hints. |
| Individual tasks | Hints may decrease performance | If hints specify an implementation path a model dislikes, it may fail. |

### Key Findings
- Most agents can identify core editing regions (high file recall), but final success is low, suggesting the bottleneck is correctly understanding experimental logic rather than file localization.
- Explicit errors primarily include Python ValueErrors, empty patches, SyntaxErrors, and execution timeouts. Claude and GPT-5 series rarely produce SyntaxErrors.
- Implicit errors are more dangerous: the code runs, but the results do not match the gold standard. Analysis of top-2 agents shows the ratio of logic errors to numerical/parameter errors is approximately 2:1.
- Stronger models produce more implicit errors. This implies future agents may "crash" less frequently but more often provide implementations that are plausible yet scientifically incorrect.
- The number of lines changed in the gold solution has a significant negative impact on final success ($\beta=-0.038, p<0.01$), indicating that implementation workload is a primary source of difficulty.

## Highlights & Insights
- RExBench advances agent evaluation from "writing code that passes unit tests" to "executing verifiable research extensions." This is critical for scientific agents, where the real risk is incorrect experimental conclusions rather than syntax errors.
- The design of private gold solutions is vital. Compared to paper reproduction tasks like PaperBench, research extensions naturally reduce the likelihood of training data leakage and better test online reasoning and code comprehension.
- The paper notes that failures in stronger models are harder to debug, which is a practical engineering warning. Runnable but incorrect agent patches could mislead researchers and introduce erroneous experiments into papers.
- Hint experiments provide insights for human-agent collaboration: success is not guaranteed simply by providing hints. There is an interaction between hint granularity, implementation paths, and model capability, necessitating the design of more robust human-agent protocols.

## Limitations & Future Work
- For automated evaluation, tasks must have clear numerical targets, which is more idealized than open-ended real-world research involving trial and error and re-defining experiments.
- Currently limited to 12 tasks; while information-dense, the statistical power to distinguish between similar models is limited. Community expansion of task counts and domain coverage is important.
- The benchmark focuses on NLP/ML. Other scientific fields may present additional challenges regarding experimental environments, data scale, simulators, and evaluation metrics.
- Process-level metrics are insufficient. The authors suggest adding landmark evaluations or intermediate checks to mitigate the difficulty of post-hoc analysis for implicit errors.
- Executing machine-written code carries security risks; the use of internet-disabled containers is a necessary safeguard. Real-world deployment requires stricter sandboxing, permission controls, and auditing.

## Related Work & Insights
- **vs. SWE-bench**: SWE-bench measures GitHub issue fixes; RExBench measures research hypothesis extensions. The latter emphasizes experimental logic and numerical results.
- **vs. PaperBench / Paper2Code**: These focus on reproduction or paper-to-code translation; RExBench focuses on unpublished extensions, better mitigating data contamination.
- **vs. MLE-bench / MLAgentBench**: These are more akin to ML pipelines or Kaggle problems; RExBench includes paper context and original code, making tasks closer to research iterations.
- **Insight**: Future evaluation for research agents should separate "runnable," "interpretable," "correct results," and "reviewable" metrics rather than relying solely on a final leaderboard score.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Evaluating agents via research extensions instead of reproduction is highly authentic and contamination-resistant.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Deep tasks and detailed analysis, though the total number of tasks is relatively low and concentrated in NLP/ML.
- Writing Quality: ⭐⭐⭐⭐☆ Clear construction and error analysis; some agent results rely on charts, necessitating the appendix for full numerical data.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for research agents, secure execution, automated experimental evaluation, and human-agent collaboration design.

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
