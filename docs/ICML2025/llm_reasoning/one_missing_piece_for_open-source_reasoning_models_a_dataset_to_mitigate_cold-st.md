---
title: >-
  [Paper Note] One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL
description: >-
  [Reasoning] This paper introduces the Long CoT Collection—a 100K long-chain reasoning dataset annotated by short CoT LLMs (e.g., GPT-4o). By extracting reasoning flows from o1 as indirect guidance, short CoT models are enabled to generate high-quality long reasoning chains. This effectively mitigates the cold-start problem of open-source reasoning models during the reinforcement learning phase, yielding a 2 to 3-fold performance improvement in RLVR for the initialized models.
tags:
  - "Reasoning"
date: 2026-05-08
content_hash: 032cd02534745a00
---

# One Missing Piece for Open-Source Reasoning Models: A Dataset to Mitigate Cold-Starting Short CoT LLMs in RL

## Basic Information

- **Conference**: ICML 2025
- **arXiv**: [2506.02338](https://arxiv.org/abs/2506.02338)
- **Code**: Mentioned as publicly available in the paper (codes, datasets, models)
- **Area**: LLM Reasoning
- **Keywords**: Long CoT, Cold-start, Reinforcement Learning, Reasoning Models, Test-time Computing Budget, Dataset Construction

## TL;DR

This paper introduces the Long CoT Collection—a 100K long-chain reasoning dataset annotated by short CoT LLMs (e.g., GPT-4o). By extracting reasoning flows from o1 as indirect guidance, short CoT models are enabled to generate high-quality long reasoning chains. This effectively mitigates the cold-start problem of open-source reasoning models during the reinforcement learning phase, yielding a 2 to 3-fold performance improvement in RLVR for the initialized models.

## Background & Motivation

- **The Rise of LRMs and Closed-Source Dilemma**: Large Reasoning Models (LRMs) represented by o1/R1 have demonstrated breakthrough reasoning capabilities through test-time scaling, but their closed-source nature limits academic research and practical applications.
- **Cold-Start Problem**: DeepSeek-R1 revealed that RLVR training requires SFT on Long CoT data prior to RL to stabilize training, but the construction of this cold-start data has remained opaque. Existing approaches heavily rely on distilling outputs from existing LRMs like R1, which is essentially "borrowing an egg to hatch a chicken."
- **Core Problem**: Can standard LLMs capable of only short-chain reasoning (e.g., GPT-4o) be used to construct high-quality Long CoT datasets? This would eliminate dependence on existing LRM outputs and enable independent development of reasoning models.
- **Overthinking Issue**: LRMs tend to generate excessively long reasoning chains even for simple questions (e.g., QwQ-32B generating ~1500 tokens for "1+1+3?"), necessitating controllable thinking budget mechanisms.

## Method

### Overall Architecture

The data construction is divided into two major stages: (1) Collecting a 1K seed dataset (extracting reasoning flows from o1); (2) Leveraging seed data to guide short CoT LLMs in scaling up to generate 100K long CoT data.

### Stage 1: Seed Data Collection (1K Teacher Demonstrations)

- **Reasoning Flow Annotation**: Reasoning flows $S_{ref} = \{s_1, s_2, ..., s_n\}$ of o1 on 1K questions from the magpie-reasoning-V1 dataset were manually collected from the ChatGPT web interface, where each $s_i$ represents a summary description of a reasoning step.
- **Thinking Budget Recording**: The thinking token count $b_{ref}$ of o1 is calculated via the OpenAI API (total completion tokens minus the response tokens returned).
- **Seed Dataset**: $\mathcal{D}_{ref} = \{q, S_{ref}, b_{ref}\}$, consisting of triplets of questions, reasoning flows, and thinking budgets.

### Stage 2: 100K Data Expansion (Short CoT LLM Annotation)

#### Step 1: Reasoning Flow Retrieval

For a new question $q$, relevant demonstrations are dynamically retrieved from the seed dataset, considering two factors:

- **Domain Matching**: Questions in the same or similar domains tend to share reasoning processes (e.g., o1 tends to verify calculations in arithmetic reasoning). Matching scores are calculated using the main-domain and sub-domain of the magpie dataset.
- **Thinking Budget Control**: Reasoning flows of similar lengths are retrieved. The similarity formula is $1 - \left|\frac{\min(x,y)}{\max(x,y)} - 1\right|$ to align the generated reasoning chain length with LRMs.

#### Step 2: Reasoning Flow Generation

Based on the retrieved demonstrations, GPT-4o generates a reasoning flow $\hat{S}$ for the new question: first predicting the required number of steps $|S|$, then generating a series of reasoning outlines. **Key Findings**: Without demonstration guidance, LLMs tend to only reason linearly and fail to utilize strategies unique to LRMs, such as verification and multi-path exploration.

#### Step 3: Step-by-step Long CoT Generation

Guided by the generated reasoning flow $\hat{S}$, the LLM generates a long reasoning chain step-by-step. For each step $\hat{s}_i$, reasoning content $r_i$ is generated based on the prior reasoning $\{r_k\}_0^{i-1}$, the current step $\hat{s}_i$, and the next step $\hat{s}_{i+1}$. Once all steps are exhausted, the final answer is generated, which is then aggregated into a complete sequence.

#### Step 4: Correctness Filtering

GPT-4o is used to verify the consistency between the generated answer and the ground-truth answer, filtering out incorrect reasoning chains. The retention rate is approximately 76%.

### Thinking Budget Control Mechanism

By controlling the number of reasoning outlines to adjust the length of the generated reasoning chains, three versions were constructed: 100%, 50%, and 25% budget Long CoT Collections, providing a flexible approach to resolve the overthinking issue.

## Experiments

### Experimental Setup

- **Base Models**: Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Qwen-2.5-0.5B
- **Evaluation Benchmarks**: MATH-500, AIME24, GPQA Diamond, MMLU-Pro
- **RL Method**: GRPO, trained on 10K integer-answer samples filtered from NuminaMATH, with a maximum sequence length of 16K

### Main Results I: Best-of-N Sampling (RL Starting Point Quality Evaluation)

| Model | MATH-500 Pass@1 | MATH-500 Pass@32 | AIME24 Pass@1 | AIME24 Pass@32 |
|------|:---:|:---:|:---:|:---:|
| Llama-3.1-8B-Instruct | ~48 | ~78 | ~3 | ~20 |
| Llama-3.1-8B-LC (Ours) | ~55 | ~88 | ~7 | ~33 |
| Qwen-2.5-7B-Instruct | ~72 | ~88 | ~10 | ~33 |
| Qwen-2.5-7B-LC (Ours) | ~72 | ~93 | ~10 | ~43 |

- On Llama-8B, BoN results after training on the Long CoT Collection significantly improve across all values of N.
- Qwen-7B LC shows clear improvements at larger N values (Pass@16/32), indicating that the SFT-trained model can explore more diverse reasoning paths.

### Main Results II: General Reasoning Capability (Table 1)

| Model | Size | GPQA Diamond | MMLU-Pro |
|------|:---:|:---:|:---:|
| o1-mini | N/A | 60.0 | 80.3 |
| o1 | N/A | 77.3 | - |
| R1 | 671B | 71.5 | 84.0 |
| QwQ-32B | 32B | 65.2 | 71.0 |
| Sky-T1 | 32B | 56.8 | 69.2 |
| Bespoke-7B | 7B | 38.9 | - |
| OpenThinker-7B | 7B | 42.4 | - |
| Llama-3.1-8B-Instruct | 8B | 22.7 | 43.7 |
| **Llama-3.1-8B-LC (Ours)** | 8B | **36.4** | **44.5** |
| Qwen-2.5-7B-Instruct | 7B | 37.6 | 49.9 |
| **Qwen-2.5-7B-LC (Ours)** | 7B | **39.9** | **51.4** |

- Llama-8B-LC gains **+13.7** (22.7→36.4) on GPQA, presenting a significant improvement.
- Qwen-7B-LC slightly outperforms Bespoke-7B on GPQA (39.9 vs 38.9), where the latter is an R1 distillation model.
- Modest improvements are also observed on MMLU-Pro, indicating that reasoning strategies can transfer to general domains.

### Ablation Study: Impact of Thinking Budget on Performance (Table 2)

| Training Data Budget Ratio | MATH-500 |
|:---:|:---:|
| 100% | 66.6 |
| 50% | 60.7 |
| 25% | 57.6 |

- A more sufficient budget leads to stronger downstream mathematical reasoning capabilities.
- Overly compressing to a 25% budget causes reasoning confusion (forcing information into too few outlines).

### Key Findings

1. **2-3x RL Gain**: On Qwen-0.5B, conducting RLVR after initializing with the Long CoT Collection yields 2-3 times the performance gain on MATH-500 and GPQA compared to running RL directly from the base model.
2. **Data Quality Nearing R1**: In head-to-head comparisons using o3-mini as an evaluator, the Long CoT Collection exceeds R1's output in reasoning flow quality; it is slightly weaker in reasoning strategy and correctness but remains competitive.
3. **Reasonable Thinking Token Allocation**: Compared to R1, the token distribution of the Long CoT Collection is more compact and closer to o1-mini, effectively avoiding overthinking.
4. **Abundant Reasoning Triggers**: The generated reasoning chains contain reasoning trigger words such as "Wait" and "To verify," helping to explore diverse reasoning paths.

## Highlights & Insights

- 🔑 First to demonstrate that short CoT LLMs (e.g., GPT-4o) can generate high-quality Long CoT data when guided by reasoning flows, without relying on distillation from LRM outputs.
- 🔑 Proposes a new paradigm of using reasoning flows as "indirect guidance": first extracting a macro-level reasoning framework, and then letting the short CoT model fill in the details.
- 🔑 The thinking budget control mechanism provides a practical solution to address the overthinking problem.
- 🔑 The clear two-stage experimental design of SFT+RL validates the value of the dataset for the cold-start problem (yielding a 2-3x RL gain).

## Limitations & Future Work

- SFT was only conducted on 7-8B models, and RL on 0.5B models (due to GPU resource constraints, 16×A100 40GB). The effects on larger models have not been verified.
- Only o1 was used as the reference LRM. Other LRMs (e.g., Gemini Thinking) were not explored.
- Not validated in expert domains (e.g., code, science).
- The seed data requires manual collection of o1's reasoning flows from the ChatGPT web interface, leading to limited automation.
- A 76% correctness filtering rate indicates that 24% of the data is discarded, suggesting further room to optimize data utilization efficiency.

## Related Work & Insights

- **DeepSeek-R1** (2025): An open-source reasoning model that revealed the training paradigm of SFT cold-start + RLVR.
- **Sky-T1, Bespoke-Stratos**: Representative works for constructing training data by distilling LRM outputs.
- **GRPO** (Shao et al., 2024): The policy optimization algorithm used in our RL phase.
- **Yeo et al. (2025)**: A detailed analysis of the role of the RLVR stage after SFT.

## Rating

⭐⭐⭐⭐ (4/5)

**Reason**: This paper proposes an important and practical research question (constructing cold-start data while eliminating reliance on LRMs). The method is simple yet effective (reasoning flow guidance + step-by-step generation), and the experimental design is solid and validates the core hypothesis (2-3x RL gain). The shortcomings lie in the limitation of computational resources, as RL experiments were only conducted on 0.5B models, and the method still requires o1's reasoning flows as seeds, meaning it has not completely decoupled from LRM dependencies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](../../ACL2025/llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [\[NeurIPS 2025\] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](../../NeurIPS2025/llm_reasoning/beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)
- [\[ICML 2025\] PENCIL: Long Thoughts with Short Memory](pencil_long_thoughts_with_short_memory.md)
- [\[ICML 2025\] MARGE: Improving Math Reasoning for LLMs with Guided Exploration](marge_improving_math_reasoning_for_llms_with_guided_exploration.md)
- [\[ICML 2025\] Putnam-AXIOM: A Functional & Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs](putnam-axiom_a_functional_and_static_benchmark_for_measuring_higher_level_mathem.md)

</div>

<!-- RELATED:END -->
