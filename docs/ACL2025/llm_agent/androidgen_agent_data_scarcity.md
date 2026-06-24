---
title: >-
  [Paper Note] AndroidGen: Building an Android Language Agent under Data Scarcity
description: >-
  [LLM Agent] This paper proposes the AndroidGen framework, which enhances LLM capabilities for Android operations under conditions of high-quality training data scarcity using four modules: Experience Search (ExpSearch), Reflection Planning (ReflectPlan), Automatic Checking (AutoCheck), and Step-level Critic (StepCritic). It successfully trains open-source mobile agents without manual annotation by automatically generating trajectory data.
tags:
  - "LLM Agent"
date: 2026-05-08
content_hash: d9a81801840287d2
---

# AndroidGen: Building an Android Language Agent under Data Scarcity

| Information | Content |
|------|------|
| Conference | ACL 2025 |
| arXiv | [2504.19298](https://arxiv.org/abs/2504.19298) |
| Code | [GitHub](https://github.com/THUDM/AndroidGen) |
| Area | LLM Agent / Mobile Agent |
| Keywords | Android agent, data scarcity, trajectory generation, automatic evaluation, open-source agent |

## TL;DR

This paper proposes the AndroidGen framework, which enhances LLM capabilities for Android operations under conditions of high-quality training data scarcity using four modules: Experience Search (ExpSearch), Reflection Planning (ReflectPlan), Automatic Checking (AutoCheck), and Step-level Critic (StepCritic). It successfully trains open-source mobile agents without manual annotation by automatically generating trajectory data.

## Background & Motivation

- **Problem Definition**: Utilizing LLMs as agents to complete user tasks (e.g., setting alarms, sending messages, searching maps) on real mobile devices is an important but under-realized goal. The core bottleneck lies in the **scarcity of high-quality trajectory data**.
- **Data Collection Difficulty**: (1) **Scenario Diversity**—huge variations across different applications require broad coverage; (2) **High Annotation Costs for Complex Tasks**—multi-step tasks require precise execution and planning; (3) **Challenging Quality Control**—verifying whether each operation perfectly meets the task requirements is both time-consuming and labor-intensive.
- **Limitations of Prior Work**: Manual annotation is time-consuming and costly, whereas automated methods (using GPT-4, etc., to complete tasks automatically) suffer from extremely low success rates (e.g., M3A+GPT-4o achieves only 27.7% on AndroidWorld) and lack effective automatic quality filtering strategies.
- **Core Motivation**: There is an urgent need for a complete pipeline that not only provides an inference framework to improve agent performance but also automatically generates high-quality training data to train open-source models.

## Method

### Overall Architecture

AndroidGen consists of three phases: Preliminary → Task Execution → Update, featuring four core modules:

### Key Designs

1. **ExpSearch (Experience Search)**:
    - Leverages the in-context learning capabilities of LLMs to retrieve the most similar completed tasks from a historical trajectory database as exemplars.
    - Uses Contriever to encode instructions for similarity calculations and selects the top-1 result.
    - Re-evaluates and updates the database using StepCritic after each completed task, achieving **iterative self-improvement** and **generalization from easy to difficult tasks**.

2. **ReflectPlan (Reflection Planning)**:
    - Initial step: Analyzes the task and environment to generate a step-by-step plan.
    - Subsequent steps: Reflects on current progress, updates the plan state, and dynamically corrects the plan when encountering failures or loops.
    - Resolves the issue of traditional planning being overly optimistic about execution outcomes.

3. **AutoCheck (Automatic Checking)**:
    - Proactively validates execution correctness before each operation (e.g., verifying element ID existence, type compliance, scrolling completion).
    - Employs rule-based strategies rather than LLM self-checking to prevent false positives caused by inconsistent self-evaluation standards.
    - Aborts execution when issues are detected and provides feedback in the subsequent round.

4. **StepCritic (Step-level Critic)**:
    - Decomposes tasks into sub-goals and provides fine-grained evaluation based on the complete sequence of actions and the final device state.
    - Annotates whether each sub-goal is completed along with its corresponding step index (-1 indicates uncompleted).
    - Supports trajectory augmentation: partially completed trajectories can be truncated at the completed sub-goal step to produce multiple valid training instances.

### Loss & Training

LoRA fine-tuning is conducted using the standard language modeling loss. The planning steps and execution steps are mixed during training to imbue the model with both planning and execution capabilities.

## Key Experimental Results

### Main Results: AndroidWorld Success Rate

| Agent | Model | Average Success Rate |
|-------|------|-----------|
| SeeAct | GPT-4o | 15.9% |
| M3A | GPT-4o | 27.7% |
| AndroidGen | GLM-4-9B* | 29.2% |
| AndroidGen | Llama-3-70B* | 35.3% |
| **AndroidGen** | **GPT-4o** | **46.8%** |

### AitW Benchmark Comparison

| Method | General | Web Shopping |
|------|---------|-------------|
| AppAgent (GPT-4o) | 16.7 | 8.3 |
| DigiRL (RL Trained)* | 71.9 | 67.2 |
| AndroidGen (GLM-4-9B*) | 65.6 | 59.4 |
| AndroidGen (Llama-3-70B*) | 74.0 | 79.2 |
| **AndroidGen (GPT-4o)** | **85.4** | **81.3** |

### Ablation Study

| Method | Easy | Medium | Hard | Average |
|------|------|--------|------|-----|
| Base Agent | 35.0 | 5.9 | 0.0 | 20.7 |
| +ReflectPlan | 51.7 | 14.7 | 0.0 | 32.4 |
| +AutoCheck | 53.3 | 17.6 | 0.0 | 34.2 |
| +ExpSearch | **65.0** | **32.4** | **11.8** | **46.8** |

### Key Findings

1. **AndroidGen significantly outperforms baselines**: The GPT-4o-powered version achieves 46.8% on AndroidWorld (compared to SeeAct's 15.9% and M3A's 27.7%).
2. **ExpSearch contributes the most**: Elevating performance from 34.2% to 46.8%, it is the only module capable of solving Hard tasks (11.8%).
3. **Effective training of open-source models**: Without manual annotation, Llama-3-70B* surpasses the RL-trained DigiRL on AitW (74.0% vs. 71.9%).
4. **StepCritic outperforms binary evaluation**: It achieves a trajectory-level accuracy of 87.9%, outperforming the Captioner+GPT-4 baseline (84.6%), while providing fine-grained sub-goal labels.
5. **Trajectory augmentation strategy is effective**: Utilizing truncation-based augmentation of partially completed trajectories maximizes data utilization.
6. **Popular apps evaluation**: Achieves a 65% success rate across 8 real-world applications including Google Maps, YouTube, and Spotify.

## Highlights & Insights

- A complete "framework + data + model" closed loop: from inference framework to automatic data generation and open-source model training, forming a reproducible pipeline.
- StepCritic's sub-goal level evaluation provides much richer training signals compared to simplistic binary success/failure judgments.
- Adopting rules rather than LLM self-checking in AutoCheck is a pragmatic design choice, effectively avoiding false positives due to inconsistent self-evaluation standards.
- The iterative self-improvement mechanism of ExpSearch enables **generalization from easy to difficult tasks without human intervention**.
- The trajectory augmentation algorithm ingeniously leverages partially completed trajectories, significantly mitigating data scarcity issues.

## Limitations & Future Work

- Task success rate on AndroidWorld is still under 50% (Hard tasks at only 11.8%), leaving a gap for practical deployment.
- Reliance on GPT-4o as the StepCritic evaluator introduces potential evaluation bias and high API costs.
- Environment observations are based solely on the XML accessibility tree, lacking visual perception capabilities (no screenshots are used).
- Evaluated only in English-language environments, failing to cover mobile scenarios in other languages.
- The retrieval quality of ExpSearch is bounded by database scale, potentially lacking similar tasks in the initial stages.

## Related Work & Insights

- **AI Scientist / AppAgent / Mobile-Agent**: Various agent frameworks serve as comparisons and references.
- **DigiRL** (Bai et al., 2024): An RL-based offline-to-online training method; AndroidGen's training data pipeline can be complementary to it.
- **ReAct** (Yao et al., 2022): The foundation of the reason-and-act paradigm; ReflectPlan builds upon it by adding dynamic plan updating.
- Offers general inspiration for the paradigm of "automatically generating training data to train open-source models".

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[ACL 2025\] Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks](agents_under_siege_breaking_pragmatic_multi-agent_llm_systems_with_optimized_pro.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](../../ACL2026/llm_agent/anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ICML 2025\] From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?](../../ICML2025/llm_agent/from_passive_to_active_reasoning_can_large_language_models_ask_the_right_questio.md)

</div>

<!-- RELATED:END -->
