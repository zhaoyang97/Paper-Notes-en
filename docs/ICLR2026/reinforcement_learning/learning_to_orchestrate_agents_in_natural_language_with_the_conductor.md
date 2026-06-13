---
title: >-
  [Paper Note] Learning to Orchestrate Agents in Natural Language with the Conductor
description: >-
  [ICLR 2026][Reinforcement Learning][multi-agent coordination] A 7B Qwen2.5 model is trained via GRPO as a "Conductor" that outputs complete agent workflows in natural language—comprising subtask instructions…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "multi-agent coordination"
  - "reinforcement-learning"
  - "workflow orchestration"
  - "test-time scaling"
  - "collective intelligence"
date: 2026-05-08
content_hash: 84c5f4469d4b67f3
---

# Learning to Orchestrate Agents in Natural Language with the Conductor

**Conference**: ICLR 2026
**arXiv**: [2512.04388](https://arxiv.org/abs/2512.04388)  
**Code**: Available (submitted with paper)  
**Area**: Reinforcement Learning
**Keywords**: multi-agent coordination, reinforcement-learning, workflow orchestration, test-time scaling, collective intelligence

## TL;DR
A 7B Qwen2.5 model is trained via GRPO as a "Conductor" that outputs complete agent workflows in natural language—comprising subtask instructions, worker assignments, and communication topology access lists—to coordinate frontier models such as GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro. Trained on only 960 questions × 200 iterations, the Conductor achieves an average accuracy of 77.27% across 7 reasoning benchmarks, surpassing all single-model baselines (GPT-5: 74.78%) and multi-agent baselines.

## Background & Motivation

**Background**: Different LLMs have complementary strengths across domains (e.g., GPT-5 excels at coding, Gemini at scientific reasoning), and commercial AI products rely on manually designed agent workflows to exploit the advantages of model ensembles.

**Limitations of Prior Work**:
- Handcrafted agent scaffolding demands extensive prompt engineering and lacks adaptability.
- Methods such as MoA and RouterDC perform only model routing or use fixed topologies, with expressiveness constrained by predefined option sets.
- Self-reflection strategies exhibit diminishing returns beyond five rounds, limiting intra-model improvement.
- No end-to-end approach exists for learning coordination strategies—one that lets RL automatically discover "who does what and how they collaborate."

**Key Challenge**: Flexible agent coordination strategies are needed to fully exploit heterogeneous model ensembles, yet manual design is costly and non-generalizable, while routing classifiers are restricted to predefined topologies.

**Goal**: Enable a small model to automatically learn, via RL, to design optimal multi-model coordination workflows for arbitrary problems.

**Key Insight**: Natural language is adopted as the workflow specification language—the Conductor directly outputs three Python lists (subtask descriptions, model IDs, and access lists), placing any coordination strategy expressible in natural language within the search space.

**Core Idea**: Model the task of "designing an agent workflow" as a sequence generation problem amenable to end-to-end RL optimization.

## Method

### Overall Architecture

Given a question $q_i$, the Conductor reasons within `<think>` tags and then outputs three Python lists: `subtasks` (natural-language subtask instructions), `model_ids` (worker assignments), and `access_lists` (specifying which prior outputs each worker can observe). The workflow is executed sequentially, and the output of the final step serves as the answer.

### Key Designs

1. **Natural Language Workflow Specification**:
    - **Function**: Each step output by the Conductor includes a natural-language subtask instruction, a worker ID, and an access list that together define the complete coordination topology.
    - **Mechanism**: The workflow is represented as $\{(\text{subtask}_i, \text{agent}_i, \text{access}_i)\}_{i=1}^L$, supporting topologies ranging from simple best-of-N and chain structures to parallelizable tree structures (e.g., `access=[[],[],["all"]]`). Worker context is organized via a dialogue template that encodes preceding task–response pairs.
    - **Design Motivation**: Natural language as an interface is far more expressive than a classifier—the Conductor can perform prompt engineering (writing focused instructions), task decomposition (multi-step planning), verification (delegating to another model for checking), and role assignment (e.g., "you are the planner" / "you write the code").

2. **End-to-End RL Training with GRPO**:
    - **Function**: Train the Conductor to learn coordination strategies using only outcome correctness as the reward signal.
    - **Mechanism**: The GRPO objective is $J(\theta) = \mathbb{E}[\frac{1}{G}\sum_{i=1}^{G}(\min(r_i A_i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon)A_i))]$. The reward is sparse: format error = 0, correct answer = 1, incorrect answer = 0.5. The advantage is $A_i = (r_i - \text{mean})/\text{std}$. No KL penalty is applied ($\beta=0$).
    - **Design Motivation**: Assigning a reward of 0.5 (rather than −1) to format-valid but incorrect outputs encourages exploration of diverse coordination strategies rather than collapsing to safe but uninformative outputs. Training converges with only 960 questions and 200 iterations, as frontier workers provide a strong execution foundation.

3. **Recursive Topologies & Adaptive Worker Pools**:
    - **Function**: Extend the Conductor's capabilities by (a) enabling recursive coordination with itself as a worker, and (b) supporting adaptation to arbitrary model subsets.
    - **Mechanism**: Recursion is achieved by allowing the Conductor to specify its own ID in an access list; recursive calls receive the parent output and preceding worker responses as context, with a manually imposed maximum recursion depth. Adaptive pools are obtained by fine-tuning a pretrained Conductor, sampling a random subset of $k$ workers at each step.
    - **Design Motivation**: Recursive topologies introduce a new axis for test-time scaling—the Conductor can observe initial strategy outcomes and adaptively revise (e.g., switching from GPT-5 to Claude/Gemini in recursive rounds upon detecting poor performance on BigCodeBench). Adaptive pools enable the same Conductor to operate in purely open-source or purely closed-source settings.

### Loss & Training

**Training data**: 960 questions from four domains (MATH: 300, MMLU, RLPR, LiveCodeBench V1). **Hyperparameters**: batch size = 256 (4 questions × 64 rollouts), lr = 1e-6, cosine scheduling, AdamW ($\beta_1=0.9$, $\beta_2=0.999$), max completion = 1024 tokens, 200 GRPO iterations. **Worker settings**: max 4096 output tokens, temperature 0.2, minimum inference budget. **Hardware**: 2× H100 80 GB.

## Key Experimental Results

### Main Results — Comparison with Unconstrained Best Single Models

| Model | MATH500 | LiveCodeBench | AIME25 | GPQA-D | Average |
|-------|---------|--------------|--------|--------|---------|
| GPT-5 | 99.0 | 82.90 | 90.8 | 82.3 | 74.78 |
| Gemini 2.5 Pro | 96.0 | 67.24 | 78.3 | 84.8 | 70.97 |
| Claude Sonnet 4 | 96.0 | 46.54 | 74.3 | 77.7 | 65.69 |
| R1-Distill-32B | 82.5 | 26.86 | 63.0 | 58.1 | 54.49 |
| **Conductor (7B)** | **99.4** | **83.93** | **93.3** | **87.5** | **77.27** |

### Comparison with Multi-Agent Baselines (Constrained Setting: 4K tokens / Minimum Inference)

| Method | MATH500 | MMLU | RLPR | LCB | Average |
|--------|---------|------|------|-----|---------|
| MoA | 83.10 | 88.46 | 38.37 | 38.57 | 62.13 |
| MASRouter | 80.60 | 86.28 | 32.80 | 27.86 | 56.89 |
| RouterDC | 59.25 | 87.52 | 27.53 | 35.33 | 52.41 |
| 5× Self-Reflection (GPT-5) | 76.93 | 91.79 | 31.80 | 57.57 | 64.52 |
| **Conductor** | **89.33** | **93.14** | **42.63** | **64.29** | **72.35** |

### Ablation Study

| Configuration | MATH500 | LiveCodeBench | Notes |
|---------------|---------|--------------|-------|
| Conductor (full) | 89.33 | 64.29 | OOD few-shot + subtasks |
| w/o subtasks | 88.50 | 58.62 | Removing prompt engineering → LCB drops 5.7% |
| w/o few-shot | 82.00 | 54.86 | Removing few-shot examples → overall decline |
| All GPT-5 workers | 93.33 | — | Fixed workers → loss of heterogeneous complementarity |

### Key Findings
- The 7B Conductor outperforms GPT-5 by 2.5% on AIME25 and by 5.2% on GPQA-D—margins comparable to an entire generational improvement.
- The Conductor uses an average of only 3 workflow steps (well below the 5-step limit), compared to 4–5 steps for MASRouter, indicating greater efficiency.
- Emergent behavior: 2-step workflows for simpler MMLU tasks and 3–4-step workflows for more complex LiveCodeBench tasks, reflecting automatic difficulty-adaptive compute allocation.
- When restricted to open-source workers only (R1-Distill/Gemma/Qwen), the Conductor still outperforms Claude Sonnet 4 by approximately 10%.
- Recursive topologies yield an additional +2.2% on BigCodeBench and +1% on GPQA-D, establishing a new test-time scaling axis.
- The 3B Conductor selects the same model distribution as the 7B variant, but the 7B achieves additional gains through superior prompt engineering, indicating that model scale directly translates to coordination capability.

## Highlights & Insights
- **Paradigm Innovation**: This is the first work to learn agent coordination strategies end-to-end via RL—prompt engineering, verification, debate, and task decomposition all emerge naturally from reward maximization, requiring no human priors.
- **Small Model Orchestrating Large Models**: A 7B Conductor coordinating frontier models more than 100× its size achieves a new frontier of collective intelligence, with training costs on the order of 2× H100 for a few days.
- **Natural Language as a Universal Workflow Language**: The Conductor's output is not a discrete selection but full natural-language instructions, with expressiveness equivalent to any scaffold a human prompt engineer could write.
- **Counter-Intuitive Finding on OOD Few-Shot**: Using successful coordination strategies from out-of-domain tasks as few-shot examples outperforms using in-domain examples, as it avoids lazy exploitation.

## Limitations & Future Work
- Reliance on expensive closed-source APIs (GPT-5, Claude, Gemini) makes evaluation costly and unpredictable.
- Training data comprises only 960 questions; generalization to domains beyond math, code, and science remains to be verified.
- Recursion depth is manually capped; automatic discovery of optimal recursive strategies has not been explored.
- Failure modes of the Conductor—when it incorrectly assigns models or produces poor prompts—have not been analyzed.
- The worker pool is fixed at 7 models; scaling efficiency and combinatorial explosion with larger pools are open questions.

## Related Work & Insights
- **vs. MoA**: MoA employs a fixed layer + aggregator topology; having 7 candidate responses may confuse the aggregator when correct and incorrect answers are mixed (especially in large solution-space tasks such as LiveCodeBench). The Conductor learns targeted subtask assignment, avoiding this issue.
- **vs. MASRouter**: MASRouter trains a routing classifier to select from predefined topologies, limiting expressiveness. The Conductor freely constructs any topology via natural language.
- **vs. Self-Reflection**: Five rounds of self-correction approach the ceiling of single-model improvement (GPT-5: 57.57, with no significant further gain). The Conductor breaks through this ceiling via cross-model coordination (64.29).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Paradigm innovation in RL-based agent coordination learning; recursive topologies establish a new test-time scaling axis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Seven benchmarks, comprehensive multi-agent baselines, ablation studies, scale analysis, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Compelling analysis of emergent behaviors; thorough discussion of design decisions.
- **Value**: ⭐⭐⭐⭐⭐ Foundational work that defines a new paradigm for training RL-based agent coordinators.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](../../AAAI2026/reinforcement_learning/hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)
- [\[ACL 2026\] Breaking the Impasse: Dual-Scale Evolutionary Policy Training for Social Language Agents](../../ACL2026/reinforcement_learning/breaking_the_impasse_dual-scale_evolutionary_policy_training_for_social_language.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICML 2026\] Randomized Advantage Transformation (RAT): Computing Natural Policy Gradients via Direct Backpropagation](../../ICML2026/reinforcement_learning/randomized_advantage_transformation_rat_computing_natural_policy_gradients_via_d.md)
- [\[ICLR 2026\] Towards Strategic Persuasion with Language Models](towards_strategic_persuasion_with_language_models.md)

</div>

<!-- RELATED:END -->
