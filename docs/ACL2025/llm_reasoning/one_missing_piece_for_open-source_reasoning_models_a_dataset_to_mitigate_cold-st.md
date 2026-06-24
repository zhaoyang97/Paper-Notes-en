---
title: >-
  [Paper Note] One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL
description: >-
  [ACL 2025][Reasoning][Long Chain-of-Thought] This work introduces the Long CoT Collection—a 100K long chain-of-thought reasoning dataset annotated by a short-CoT LLM (such as GPT-4o). By leveraging reasoning flows extracted from o1 as guidance, it enables short-CoT LLMs to generate long-CoT data, thereby addressing the cold-start problem in reinforcement learning. Models initialized on this data achieve a 2-3x performance gain in subsequent RL training.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Long Chain-of-Thought"
  - "Cold-Start"
  - "Reinforcement Learning"
  - "Reasoning Models"
  - "Thinking Budget Control"
date: 2026-05-08
content_hash: 3b670093358bef4a
---

# One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL

**Conference**: ACL 2025  
**arXiv**: [2506.02338](https://arxiv.org/abs/2506.02338)  
**Code**: Yes (code, dataset, and models are publicly released)  
**Area**: LLM Reasoning  
**Keywords**: Long Chain-of-Thought, Cold-Start, Reinforcement Learning, Reasoning Models, Thinking Budget Control

## TL;DR

This work introduces the Long CoT Collection—a 100K long chain-of-thought reasoning dataset annotated by a short-CoT LLM (such as GPT-4o). By leveraging reasoning flows extracted from o1 as guidance, it enables short-CoT LLMs to generate long-CoT data, thereby addressing the cold-start problem in reinforcement learning. Models initialized on this data achieve a 2-3x performance gain in subsequent RL training.

## Background & Motivation

### Background

Large reasoning models (LRMs) like OpenAI o1 have achieved breakthroughs in reasoning tasks through test-time scaling (generating long chain-of-thought reasoning). DeepSeek-R1 has open-sourced its construction method, where key findings include:

**Cold-start problem**: Directly applying reinforcement learning (RLVR) to short-CoT LLMs yields poor results due to weak learning signals.

**Key step**: Supervised Fine-Tuning (SFT) on long-CoT data is required beforehand to help the model learn the reasoning structure.

**Limitations of Prior Work**: Current methods rely heavily on collecting outputs from existing LRMs like R1 as training data.

### Core Problem

The methodology for constructing long-CoT data remains unclear—existing works either rely on distillation from closed-source LRMs (o1/R1) or suffer from insufficient dataset scale and quality. This constitutes a key bottleneck for the development of open-source reasoning models.

### Design Motivation

**Can short-CoT LLMs be leveraged to construct long-CoT datasets?** If possible, this would eliminate dependency on existing LRMs, enabling autonomous LRM development. Furthermore, this data construction process naturally supports thinking budget control, potentially mitigating the "overthinking" problem in LRMs (e.g., QwQ-32B generating approximately 1500 tokens for "1+1+3?").

## Method

### Overall Architecture

The data construction pipeline consists of four steps:
1. Collect a 1K seed dataset containing o1's reasoning flows and thinking budgets.
2. Retrieve similar reasoning flows as demonstrations for new questions.
3. Guide the short-CoT LLM to generate reasoning flows for new questions.
4. Generate step-by-step long CoT reasoning along the generated reasoning flows.

### Key Designs

1. **Seed Dataset Collection (1K demonstrations)**

    - Manually collect the **reasoning flows** of o1 on 1K reasoning questions from the ChatGPT interface.
    - A reasoning flow is defined as a sequence of reasoning step outlines $S = \{s_1, s_2, ..., s_n\}$.
    - The **thinking budget** $b_{ref}$ (number of thought tokens = total completion tokens - response tokens) is recorded simultaneously.
    - Source questions are selected from the magpie-reasoning-V1 dataset.
    - Function: Capture advanced reasoning strategies of LRMs (e.g., verification, backtracking, multi-path exploration).

2. **Reasoning Flow Retrieval**

    - Dynamically retrieve similar examples from the seed dataset as In-Context Learning (ICL) demonstrations for new questions.
    - Two retrieval dimensions:
        - **Domain Matching**: Questions in the same or similar domains typically share reasoning patterns.
        - **Thinking Budget Control**: Match reasoning flows of similar lengths using the similarity function $1 - |\frac{\min(x,y)}{\max(x,y)} - 1|$.
    - **Design Motivation**: Ensure that the retrieved demonstrations match the target question in terms of reasoning strategies and depth.

3. **Reasoning Flow Generation**

    - Guided by the retrieved demonstrations, GPT-4o is prompted to predict the number of outline steps $|S|$ and generate the sequence of reasoning outlines.
    - **Key Findings**: Without demonstrations, LLMs tend to think linearly; with demonstrations, they can mimic advanced strategies like verification and backtracking.
    - This constitutes an "indirect knowledge transfer" from LRMs to short-CoT LLMs—communicated strictly through reasoning flow templates.

4. **Step-by-step Long CoT Generation**

    - For each step $\hat{s}_i$ in the reasoning flow, generate a detailed reasoning process based on the existing reasoning history, the current step, and the next step.
    - Generate the final answer after all steps are completed.
    - Finally, aggregate the steps into a complete reasoning chain.
    - **Design Motivation**: Step-by-step generation addresses the coherence collapse issue when short-CoT LLMs directly generate long reasoning paths.

5. **Correctness Filtering**

    - Verify answer consistency using GPT-4o.
    - Retain 76% of the instances after filtering.
    - Ensure that training does not occur on erroneous reasoning paths.

### Thinking Budget Control

Control over reasoning length is achieved by adjusting the number of outline steps:
- 100% Budget: Complete reasoning flow.
- 50% Budget: Compressed reasoning.
- 25% Budget: Highly simplified reasoning.

This provides a practical means to address the "overthinking" problem.

## Key Experimental Results

### Data Quality Evaluation (vs R1, evaluated by o3-mini)

Head-to-head comparison on 100 samples:
- **Reasoning Flow**: Long CoT Collection **outperforms** R1.
- **Reasoning Strategy**: Marginally lower than R1, yet highly competitive.
- **Correctness**: Slightly lower than R1, overall comparable.

### Main Results

| Model | Params | GPQA Diamond | MMLU-Pro |
|------|------|-------------|----------|
| o1-mini | - | 60.0 | 80.3 |
| R1 | 671B | 71.5 | 84.0 |
| Bespoke-7B (R1 Distilled) | 7B | 38.9 | - |
| Llama-3.1-8B-Instruct | 8B | 22.7 | 43.7 |
| **Llama-3.1-8B-LC** | **8B** | **36.4** | **44.5** |
| Qwen-2.5-7B-Instruct | 7B | 37.6 | 49.9 |
| **Qwen-2.5-7B-LC** | **7B** | **39.9** | **51.4** |

On GPQA, Llama-8B achieves a 60% gain (22.7 → 36.4), and Qwen-2.5-7B-LC outperforms Bespoke-7B (a model directly distilled from R1).

### Key Experimental Results — Mitigating RL Cold-Start (Figure 1)

| Setting | MATH-500 RL Gain | GPQA RL Gain |
|------|----------------|------------|
| Qwen-0.5B → RLVR | Baseline Gain | Baseline Gain |
| **Qwen-0.5B-LC → RLVR** | **2-3x Gain** | **2-3x Gain** |

Aligning reinforcement learning after initializing on the Long CoT Collection yields **2-3x** the performance gain of applying direct RL.

### Ablation Study

| Budget Ratio | MATH-500 |
|---------|----------|
| 100% | 66.6 |
| 50% | 60.7 |
| 25% | 57.6 |

### Best-of-N Sampling Analysis

On Llama-3.1-8B:
- The LC model consistently outperforms the baseline from Pass@1 to Pass@32.
- Qwen-2.5-7B-LC demonstrates more pronounced advantages at larger $N$ (16, 32).
- This indicates that after SFT, the model not only achieves higher accuracy but also explores more diverse reasoning paths.

### Key Findings

1. **Short-CoT LLMs can generate high-quality long-CoTs**: Guided by reasoning flows, GPT-4o produces reasoning data comparable in quality to R1.
2. **Significant mitigation of the cold-start problem**: The 2-3x RL performance gain constitutes a core contribution.
3. **Effective transfer of reasoning strategies**: The data contains rich reasoning trigger words (such as "Wait", "To verify"), promoting diverse reasoning trajectories.
4. **Controllable thinking budget**: Reasoning length is precisely managed by adjusting the number of outline items.
5. **Generalization to broad reasoning tasks**: Performance gains are observed not only in math, but also in GPQA and MMLU-Pro.
6. **Excessive budget constraints are detrimental**: A 25% budget leads to over-compression of information, resulting in chaotic reasoning.

## Highlights & Insights

- **The pipeline, rather than the data itself, is the core contribution**: It reveals a viable path for building long-CoT leveraging short-CoT LLMs guided by minimal LRM demonstrations.
- **Reasoning flow as an intermediate representation**: Decomposes long-CoTs into "planning first, execution later", guaranteeing coherence through hierarchical generation.
- **Best-of-N as a pre-evaluation of RL potential**: Utilizing BoN sampling to estimate the ceiling performance of models in RL serves as a practical evaluation methodology.
- **Solutions to overthinking**: Budget control is not only a technique for data construction but also a practical tool for deployment.
- **Indirect knowledge distillation**: Only the "how to think" (reasoning flows) is transferred, rather than the "what to think" (specific reasoning contents).

## Limitations & Future Work

1. **Seed data dependency on o1**: Collecting 1K o1 reasoning flows is still required, meaning dependency on existing LRMs is not entirely eliminated.
2. **Limited RL experiment scale**: Due to GPU constraints, the verification of RL efficacy is confined to 0.5B models; validation on larger models remains to be explored.
3. **Annotation cost of GPT-4o**: The construction cost for the 100K dataset is not disclosed in detail.
4. **Domain limitations**: Evaluation is primarily focused on math and general reasoning, with expert domains such as coding and science left for future exploration.
5. **Only considering o1 as the reference LRM**: The framework can be extended to other partially transparent reasoning models.

## Related Work & Insights

- **Relation to DeepSeek-R1**: While R1 highlights the cold-start problem and the importance of long-CoT SFT, this work provides a framework for autonomous data generation.
- **Comparison with Sky-T1 and Bespoke-7B**: These models directly distill outputs from LRMs parentally, whereas this work utilizes indirect generation via short-CoT LLMs.
- **Inspirations**: The decoupling approach of reasoning flows can be generalized to other scenarios requiring long-sequence reasoning (such as planning and code generation).

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | ⭐⭐⭐⭐ | The paradigm of using short-CoT LLMs to generate long-CoT is novel and highly practical. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Multi-perspective verification via BoN, RL, and general reasoning, though the scale of RL experiments is constrained. |
| Writing Quality | ⭐⭐⭐⭐ | Clear motivation of the problem, with detailed descriptions of the pipeline layout. |
| Value | ⭐⭐⭐⭐⭐ | Addresses key bottlenecks for open-source reasoning models. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [\[ACL 2025\] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](glore_long_cot_representation.md)
- [\[ACL 2025\] Improve Vision Language Model Chain-of-thought Reasoning](improve_vlm_cot_reasoning.md)
- [\[ACL 2025\] Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](dcot_diverse_cot_refinement.md)
- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)

</div>

<!-- RELATED:END -->
