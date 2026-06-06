---
title: >-
  [Paper Note] Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning
description: >-
  [ACL 2026][LLM Reasoning][Test-time reinforcement learning] This paper systematically analyzes the sources and amplification mechanisms of spurious signals in Test-Time Reinforcement Learning (TTRL). It identifies that a…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Test-time reinforcement learning"
  - "pseudo-label noise"
  - "GRPO bias"
  - "denoising and debiasing"
  - "mathematical reasoning"
date: 2026-05-08
content_hash: 50298cae131defac
---

# Understanding and Mitigating Spurious Signal Amplification in Test-Time Reinforcement Learning for Math Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.21327](https://arxiv.org/abs/2604.21327)  
**Code**: [https://github.com/yuyongcan/DDRL](https://github.com/yuyongcan/DDRL)  
**Area**: Image Restoration  
**Keywords**: Test-time reinforcement learning, pseudo-label noise, GRPO bias, denoising and debiasing, mathematical reasoning

## TL;DR

This paper systematically analyzes the sources and amplification mechanisms of spurious signals in Test-Time Reinforcement Learning (TTRL). It identifies that ambiguous regions formed by mid-frequency answers are the primary noise sources and that the group-relative normalization in GRPO amplifies these signals. The proposed DDRL framework mitigates these issues through three components: balanced sampling, fixed advantage values, and consensus offline refinement, achieving a 15.3% relative improvement on Qwen2.5-Math-1.5B.

## Background & Motivation

**Background**: TTRL adapts models to distribution shifts at test time by constructing pseudo-labels through multiple sampling and majority voting, followed by unsupervised RL using GRPO. It operates under completely unsupervised conditions where reward signals are derived entirely from the model's own outputs.

**Limitations of Prior Work**: TTRL is susceptible to spurious reward signals, where incorrect answers may be wrongly rewarded and correct answers may be punished. However, the specific sources and propagation mechanisms of these spurious signals have not been systematically analyzed.

**Key Challenge**: (1) Source level—the relationship between answer frequency and reliability is non-linear: high-frequency answers are mostly correct, low-frequency answers are mostly incorrect, and mid-frequency answers are highly ambiguous with wildly fluctuating accuracy. Standard TTRL treats all sampled rollouts equally regardless of these differences. (2) Amplification level—the group-relative normalization of GRPO assigns extremely high advantage values when positive samples are scarce. While reasonable in supervised RL (where rare positives represent valuable signals), in TTRL, a small number of positive samples indicates low consensus and high uncertainty. GRPO thus assigns the highest weights to the most unreliable samples.

**Goal**: Systematically understand the source and amplification mechanism of spurious signals in TTRL and design effective mitigation strategies.

**Key Insight**: Analyze pseudo-label reliability based on answer sampling frequency and analyze signal amplification through the mathematical properties of GRPO advantage estimation.

**Core Idea**: (1) Balanced confidence sampling: exclude mid-frequency ambiguous samples to maintain a balance between reliable positive and negative samples. (2) Debiased advantage estimation: replace group-relative normalization with fixed advantage values $A_i = \mathbb{I}(y=y^*) - \mathbb{I}(y \neq y^*)$ to eliminate the amplification effect. (3) Consensus offline refinement: perform efficient and stable optimization using rejection sampling datasets after the RL phase.

## Method

### Overall Architecture

DDRL consists of three stages: (1) Balanced confidence sampling, which selects reliable positive and negative samples based on answer frequency and excludes ambiguous mid-frequency regions; (2) Debiased advantage estimation, which replaces GRPO's group normalization with fixed label-dependent advantage values; (3) Consensus offline refinement, which constructs a rejection sampling dataset for stable SFT-based optimization after RL.

### Key Designs

1.  **Spurious Signal Source Analysis and Balanced Sampling**:
    - **Function**: Reduce pseudo-label noise at the source.
    - **Mechanism**: After $N$ samples are generated, they are analyzed by answer frequency. High-frequency answers are typically correct (reliable positives), low-frequency answers are typically incorrect (reliable negatives), and mid-frequency answers exhibit high variance in correctness (noise sources). The balanced sampling strategy selects the top-$K^+$ highest frequency samples as positives (capped at $\lfloor K/2 \rfloor$), selects $K^-$ lowest frequency samples as negatives, and discards the mid-frequency ambiguous region entirely.
    - **Design Motivation**: Mid-frequency samples are neither consistently right nor wrong; using them as pseudo-labels for RL introduces substantial noise. The 50% positive sample cap prevents positive signals from dominating the training.

2.  **Debiased Advantage Estimation**:
    - **Function**: Eliminate the amplification of spurious signals caused by GRPO normalization.
    - **Mechanism**: Advantage values are fixed to $A_i = +1$ for positive samples and $A_i = -1$ for negative samples, bypassing group-level normalization. This eliminates the vicious cycle where fewer positive samples lead to larger advantages for the most uncertain samples. Preliminary experiments in Table 1 show that removing normalization alone improves AIME2024 performance from 15.8% to 20.6%.
    - **Design Motivation**: The assumption in GRPO that "rare positive samples equal valuable signals" holds in supervised settings but fails in TTRL, where "rare positive samples equal low consensus and high uncertainty."

3.  **Consensus Offline Refinement**:
    - **Function**: Provide stable optimization following the RL stage.
    - **Mechanism**: A rejection sampling dataset is constructed from high-consistency answers to perform SFT on the post-RL model. This stage leverages high-consensus samples to provide clean supervision and stabilize potential fluctuations from RL training.
    - **Design Motivation**: RL training can be inherently unstable; offline refinement serves as a concluding step to ensure overall stability.

### Loss & Training

The RL stage utilizes a modified GRPO (fixed advantages + balanced sampling), while the refinement stage uses standard SFT loss. Evaluations are conducted using Qwen2.5-Math-1.5B/3B and LLaMA-3.1-8B-Instruct on benchmarks including MATH-500 and AIME2024.

## Key Experimental Results

### Main Results

| Model/Method | AIME2024 | MATH-500 | Gain |
| :--- | :--- | :--- | :--- |
| Qwen2.5-Math-1.5B + TTRL | 15.8 | 73.0 | - |
| Qwen2.5-Math-1.5B + DDRL | **18.2** | **84.2** | **+15.3%** |
| LLaMA-3.1-8B + TTRL | - | - | - |
| LLaMA-3.1-8B + DDRL | - | - | **+12.7%** |

### Ablation Study

| Configuration | AIME2024 | MATH | Description |
| :--- | :--- | :--- | :--- |
| GRPO (Standard Normalization) | 15.8 | 73.0 | Amplifies spurious signals |
| GRPO (No Normalization) | 20.6 | 75.0 | Improvement through debiasing alone |
| + Balanced Sampling | Further Improv. | Further Improv. | Denoising |
| + Offline Refinement | Optimal | Optimal | Full DDRL |

### Key Findings

- Mid-frequency answers are the primary source of spurious signals, as their accuracy variance is extremely high, making them unreliable as pseudo-labels.
- GRPO normalization systematically amplifies spurious signals in low-consensus scenarios; removing normalization significantly improves performance.
- The three components of DDRL provide independent and stackable gains.
- The 50% positive sample cap in balanced sampling is critical for training stability.
- Improvements are consistent across three different scales of LLMs.

## Highlights & Insights

- **Thorough Analysis of the "Frequency-Reliability" Relationship**: By categorizing answer frequencies into high, medium, and low zones, the study clearly identifies the noise source (mid-frequency), providing direct guidance for sampling strategies.
- **Deep Theoretical Analysis of GRPO Bias**: The work reveals the core contradiction that "reasonable assumptions in supervised RL are violated in unsupervised TTRL." This insight is valuable for any unsupervised method utilizing GRPO.
- **Simplicity of Fixed Advantage Values**: Replacing complex group normalization with a simple $+1/-1$ fixed advantage yields better results, demonstrating the principle that simplicity is more robust in noisy environments.

## Limitations & Future Work

- Validated only on mathematical reasoning; other reasoning tasks (e.g., code, logic) have not been tested.
- Frequency thresholds for distinguishing between high, medium, and low frequency may require manual tuning for different tasks.
- The offline refinement stage introduces additional computational costs.
- Effectiveness may be limited when the base model is extremely weak (i.e., when majority voting itself is unreliable).

## Related Work & Insights

- **vs Standard TTRL**: TTRL treats all samples equally and uses standard GRPO, leading to amplified spurious signals. DDRL addresses this via denoising (sampling) and debiasing (advantage estimation).
- **vs EMPO/STILL (Unsupervised RL)**: While these methods also explore unsupervised RL, they do not analyze the underlying mechanism of spurious signals. DDRL provides a systematic analysis and targeted solutions.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically analyzing spurious signals provides great insight, though the technical implementation (fixed advantages + sampling filters) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Uses three models, multiple benchmarks, and step-by-step ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain for analyzing the problem (frequency-reliability + GRPO bias) is exceptionally clear.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](../../CVPR2026/llm_reasoning/understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)

</div>

<!-- RELATED:END -->
