---
title: >-
  [Paper Note] Training Software Engineering Agents and Verifiers with SWE-Gym
description: >-
  [ICML 2025][Code Intelligence][Software Engineering Agents] This paper proposes SWE-Gym, the first environment designed for training software engineering (SWE) agents, containing 2,438 real-world task instances from 11 open-source Python repositories. By leveraging rejection sampling fine-tuning on SWE-Gym to train SWE agents and verifiers, it achieves resolve rates of $32.0\%$ on SWE-Bench Verified and $26.0\%$ on SWE-Bench Lite, setting a new SOTA for open-weight SWE agents…
tags:
  - "ICML 2025"
  - "Code Intelligence"
  - "Software Engineering Agents"
  - "SWE-Bench"
  - "Training Environments"
  - "Verifier"
  - "Inference-time scaling"
date: 2026-05-08
content_hash: d95b3ae98308d298
---

# Training Software Engineering Agents and Verifiers with SWE-Gym

**Conference**: ICML 2025  
**arXiv**: [2412.21139](https://arxiv.org/abs/2412.21139)  
**Code**: Yes (Publicly released SWE-Gym environments, models, and trajectory data)  
**Area**: Code Intelligence  
**Keywords**: Software Engineering Agents, SWE-Bench, Training Environments, Verifier, Inference-time scaling

## TL;DR
This paper proposes SWE-Gym, the first environment designed for training software engineering (SWE) agents, containing 2,438 real-world task instances from 11 open-source Python repositories. By leveraging rejection sampling fine-tuning on SWE-Gym to train SWE agents and verifiers, it achieves resolve rates of $32.0\%$ on SWE-Bench Verified and $26.0\%$ on SWE-Bench Lite, setting a new SOTA for open-weight SWE agents.

## Background & Motivation
**Background**: LLM-based SWE agents have demonstrated immense potential in automatically resolving GitHub issues, with SWE-Bench becoming the standard evaluation benchmark. However, the current state-of-the-art (SOTA) SWE agents rely heavily on closed-source models (such as GPT-4 and Claude), while the performance of open-source models lags far behind.

**Limitations of Prior Work**: Unlike fields such as mathematical reasoning or dialogue, which enjoy rich training datasets and environments, the software engineering domain severely lacks usable training environments. The training set of SWE-Bench only provides code patches (git diff) without step-by-step developer trajectories or executable environments.

**Key Challenge**: Real-world software engineering tasks require interaction with executable runtimes, configuring software dependencies, and reproducible test suites—building such a training environment is extremely challenging. Existing datasets are either synthetic tasks (e.g., R2E) or isolated coding problems (e.g., APPS, HumanEval), which fail to represent realistic repository-level programming.

**Goal**: To construct the first real-world software engineering training environment to support the training and enhancement of open-source SWE agents through reinforcement learning and supervised fine-tuning.

**Key Insight**: Systematically extract task instances from GitHub PRs, manually configure executable environments and unit tests, and build a comprehensive environment suitable for training policy models and verifiers.

**Core Idea**: With a training environment capable of executing tests, a simple approach of "sampling successful trajectories + fine-tuning" can significantly boost open-source SWE agents, while also enabling the training of verifiers for inference-time scaling.

## Method

### Overall Architecture
SWE-Gym environment $\rightarrow$ sample agent trajectories $\rightarrow$ two major application directions: (1) **Policy Training**: perform rejection sampling fine-tuning on successful trajectories to enhance base agent capabilities; (2) **Inference-Time Scaling**: train a verifier (Outcome-Supervised Reward Model, ORM) to rerank multiple candidate trajectories and select the optimal solution.

### Key Designs

1. **SWE-Gym Environment Construction**:

    - **Function**: Build 2,438 real SWE task instances from GitHub PRs, each with an executable environment and unit tests.
    - **Mechanism**:
        - Extract 64,689 raw task instances from 358 Python repositories (SWE-Gym Raw).
        - Filter down to 11 high-quality repositories and manually configure dependency environments for each version.
        - Utilize SWE-Bench execution verification scripts to ensure gold patches pass more tests.
        - Provide a subset of 230 simpler instances, named SWE-Gym Lite, for rapid prototyping.
    - **Design Motivation**: Built with distinct repositories from SWE-Bench to avoid data contamination. It required approximately 200 manual annotation hours and 10,000 CPU core hours to construct, resulting in the public release of 6TB Docker images.
    - **Novelty**: The only dataset that concurrently exhibits four characteristics: authentic repository-level tasks, executable environments, real task descriptions, and designated training sets.

2. **Agent Policy Training (General Prompting Mode)**:

    - **Function**: Use OpenHands (a ReAct-based general agent framework) to sample successful trajectories and fine-tune Qwen-2.5-Coder-Instruct.
    - **Mechanism**: Rejection sampling fine-tuning—apply supervised training strictly on successful trajectories that pass unit tests.
    - **Key Data**: Utilized only 491 successful trajectories (sampled using GPT-4o and Claude-3.5-Sonnet), with an average of roughly 19 interaction turns and 19,000 tokens per trajectory.
    - **Gain**: The 32B model improved from $3.0\%$ to $15.3\%$ ($+12.3\%$) on SWE-Bench Lite, and from $7.0\%$ to $20.6\%$ ($+13.6\%$) on SWE-Bench Verified.
    - **Extra Findings**: Fine-tuning significantly reduces model behaviors of getting "stuck in a loop" (repeating the same action more than three times), with the looping rate of the 7B model dropping from $47.0\%$ to $31.0\%$.

3. **Agent Policy Training (Specialized Workflow Mode)**:

    - **Function**: Conduct self-improvement training using MoatlessTools (a predefined workflow-based agent framework).
    - **Mechanism**: Iterative rejection sampling fine-tuning—generate 30 high-temperature sample trajectories per round $\rightarrow$ filter successful trajectories $\rightarrow$ fine-tune the policy.
    - **Key Innovation—Per-Instance Capping**: Limit each task to a maximum of 2 trajectories to prevent the dataset from being biased toward simpler tasks.
    - **Gain**: The 7B model improved from $7.0\%$ to $10.0\%$ (after two iterations), and the 32B model improved from $19.0\%$ to $19.7\%$.

4. **Verifier Training and Inference-Time Scaling**:

    - **Function**: Train an Outcome-Supervised Reward Model (ORM) to estimate the success probability of a trajectory.
    - **Mechanism**: Fine-tune 32B Qwen2.5-Coder-Instruct, with the input being the trajectory (issue description + agent action sequence + git diff), to predict `<YES>` or `<NO>`.
    - **Key Data**: 2,636 trajectories (evenly split between success and failure), obtained from off-policy and on-policy sampling.
    - **Inference**: Sample multiple candidate trajectories for each task, take the log probability of `<YES>` from the verifier as the score, and choose the trajectory with the highest score.
    - **Gain**: Performance on SWE-Bench Verified scales from $20.6\%$ to $32.0\%$ ($+11.4\%$), demonstrating an empirical log-linear inference-time scaling law.

### Loss & Training

**Policy Model Training**: Standard supervised fine-tuning (SFT) to minimize the language modeling loss on successful trajectories.

**Verifier Training**: Classification loss—success trajectories are labeled as `<YES>` and failures as `<NO>`, next-token probability is utilized for fine-tuning.

**Per-Instance Capping**: Limit each task to a maximum of $\text{cap}=2$ trajectories, prioritizing trajectories with fewer interaction turns to balance distributional bias and dataset scale.

## Key Experimental Results

### Main Results (OpenHands Framework)

| Model Size | Benchmark | Zero-shot Resolve Rate | Fine-tuned Resolve Rate | Gain |
|-------|------|----------------|-------------|------|
| 7B | SWE-Bench Lite | $1.0\%$ | $10.0\%$ | $+9.0\%$ |
| 14B | SWE-Bench Lite | $2.7\%$ | $12.7\%$ | $+10.0\%$ |
| 32B | SWE-Bench Lite | $3.0\%$ | $15.3\%$ | $+12.3\%$ |
| 7B | SWE-Bench Verified | $1.8\%$ | $10.6\%$ | $+8.8\%$ |
| 14B | SWE-Bench Verified | $4.0\%$ | $16.4\%$ | $+12.4\%$ |
| 32B | SWE-Bench Verified | $7.0\%$ | $20.6\%$ | $+13.6\%$ |

### Verifier Inference-time Scaling

| Configuration | SWE-Bench Verified | SWE-Bench Lite | Description |
|------|-------------------|----------------|------|
| Fine-tuned Agent ($t=0$) | $20.6\%$ | $15.3\%$ | Baseline |
| + Verifier Reranking | **$32.0\%$** | **$26.0\%$** | $+11.4\% / +10.7\%$ |

### MoatlessTools Self-Improvement

| Setting | 7B Resolve Rate | 32B Resolve Rate | Description |
|------|------------|-------------|------|
| Zero-shot | $7.0\%$ | $19.0\%$ | Specialized workflow baseline is higher than general prompting |
| Iteration 1 | $9.0\%$ | $19.7\%$ | Self-improvement is effective |
| Iteration 2 | $10.0\%$ | $19.7\%$ | 32B model saturates |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Per-Instance Cap=1 | Performance degradation | Dataset is too small |
| Per-Instance Cap=2 | Optimal | Balances bias and data volume |
| No Cap | Slightly below Cap=2 | Biased towards simpler tasks |
| Self-Improvement (OpenHands 32B) | $15.3\% \rightarrow 8.7\%$ | Self-improvement is ineffective under general frameworks |
| Training Data Scaling | Linear improvement | 491 trajectories not saturated; performance is constrained by sampling budget |

### Key Findings
- Just 491 successful trajectories yield an absolute improvement of $+12\text{--}14\%$, and performance scales linearly with the number of training trajectories without bottlenecking.
- Fine-tuning significantly mitigates the behavior of getting "stuck in a loop," which is a key bottleneck when deploying open-source models as agents.
- Inference-time scaling displays a roughly log-linear growth trend, proving the effectiveness of the verifier.
- Self-improvement is ineffective in the general prompting mode (performance actually drops), but is highly effective in the specialized workflow mode.
- Specialized workflows are more amenable to open-source models: the 32B model achieves a $19.0\%$ zero-shot resolve rate under MoatlessTools, compared to only $3.0\%$ under OpenHands.

## Highlights & Insights
- Fills the vacancy of SWE agent training environments with massive engineering contributions (200 manual annotation hours + 10,000 CPU core hours + 6TB Docker images).
- A simple rejection sampling fine-tuning brings huge improvements, demonstrating that "having an execution environment" itself is the main bottleneck.
- Per-Instance Capping is a simple yet effective optimization to rejection sampling fine-tuning.
- The log-linear trend of verifier inference-time scaling aligns directly with findings in mathematical reasoning, indicating the universality of this paradigm.
- The unsaturated training curve implies that a larger computing budget (more sampling) could yield greater gains.

## Limitations & Future Work
- Limited only to Python repositories, excluding languages like Java or TypeScript.
- Self-improvement fails under general prompting frameworks; more advanced policy optimization methods (such as PPO) are required.
- Training data is predominantly sourced from GPT-4o and Claude (off-policy), resulting in limited pure self-improvement capability.
- The 32B model saturates after two iterations under specialized workflows, restricted by the action-space constraints of the workflows.
- High construction costs of the environment make it difficult to scale automatically to more repositories.

## Related Work & Insights
- Comparable with the training paradigm shift in mathematical reasoning: only with the advent of training datasets and reward signals like GSM8K/MATH did mathematical reasoning experience explosive growth; SWE-Gym aims to replicate this trajectory in software engineering.
- The verifier concept originates from Outcome Reward Models (ORM) in mathematical reasoning and is successfully adapted to the agent domain.
- Complementary to rather than a replacement for SWE-Bench: SWE-Gym uses distinct repositories and focuses on training rather than evaluation.
- Inspiration: Other agent tasks requiring environmental interaction (e.g., web navigation, OS operations) can adopt a similar methodology to build interactive training environments.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](../../ICLR2026/code_intelligence/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)
- [\[ICLR 2026\] SWE-RM: Execution-Free Feedback for Software Engineering Agents](../../ICLR2026/code_intelligence/swe-rm_execution-free_feedback_for_software_engineering_agents.md)
- [\[ACL 2026\] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](../../ACL2026/code_intelligence/eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)
- [\[ICLR 2026\] Kimi-Dev: Agentless Training as Skill Prior for SWE-agents](../../ICLR2026/code_intelligence/kimi-dev_agentless_training_as_skill_prior_for_swe-agents.md)

</div>

<!-- RELATED:END -->
