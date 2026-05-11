---
title: >-
  [Paper Note] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning
description: >-
  [ICLR 2026][Code Intelligence][Reinforcement Learning] This paper proposes Supervised Reinforcement Learning (SRL), which reframes problem solving as a step-wise action generation process. By leveraging dense rewards bas…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "Reinforcement Learning"
  - "Supervised Learning"
  - "Step-wise Reasoning"
  - "Sequence Similarity Reward"
  - "Hard Problem Learning"
date: 2026-05-08
content_hash: bee9ea6306f8b202
---

# Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning

**Conference**: ICLR 2026
**arXiv**: [2510.25992](https://arxiv.org/abs/2510.25992)
**Code**: None
**Area**: Code Intelligence
**Keywords**: Reinforcement Learning, Supervised Learning, Step-wise Reasoning, Sequence Similarity Reward, Hard Problem Learning

## TL;DR

This paper proposes Supervised Reinforcement Learning (SRL), which reframes problem solving as a step-wise action generation process. By leveraging dense rewards based on sequence similarity, SRL enables small models to learn from expert trajectories on difficult reasoning problems that neither SFT nor RLVR can effectively handle.

## Background & Motivation

Large language models face a fundamental dilemma in multi-step reasoning tasks:

**Limitations of Prior Work — RLVR**: Reinforcement learning based on final-answer correctness (e.g., GRPO) relies on the model's ability to sample correct solutions within a limited number of rollouts. For small models (e.g., 7B), pass@k approaches zero on hard problems, resulting in extremely sparse reward signals from which no meaningful policy can be learned. Methods such as DAPO partially address this by filtering out all-wrong or all-correct samples, but fundamentally abandon these hard problems.

**Limitations of Prior Work — SFT**: Supervised fine-tuning enforces token-level imitation of expert trajectories. For long and complex reasoning chains, this rigid imitation tends to cause overfitting and shallow reasoning behavior. Experiments show that direct SFT on the s1K dataset actually degrades performance (see Figure 1).

**Key Challenge**: Hard problems have limited data and complex reasoning chains, making SFT ineffective; yet the model cannot sample correct solutions, making RLVR equally ineffective. This is especially pronounced when training small open-source models.

The authors define this problem regime as $\mathcal{D}_{\text{hard}}$—the set of problems on which the model's success rate across $k$ samples approaches zero. The goal of SRL is to provide effective learning signals within this hard regime.

## Method

### Overall Architecture

SRL reformulates problem solving as a sequential decision-making process. The core idea is to train the model to generate one "action" (i.e., a reasoning step) at each step, rather than producing a complete solution in one pass or imitating an expert token by token. Rewards are computed based on the similarity between the generated action and the corresponding expert action.

The overall pipeline consists of three stages:
1. Extract step-wise action sequences from expert trajectories
2. Construct step-wise training data
3. Perform RL training using sequence similarity rewards

### Key Designs

1. **Action-based Problem Formulation**: Given an expert solution trajectory $\mathbf{y}$, it is decomposed into a sequence of action tuples $\mathbf{y} = \{\mathbf{y}_{\text{step}}^n\}_{n=1}^N$. Each step represents a logical action—an algebraic operation in mathematical reasoning, or a terminal command in software engineering. This formulation is domain-agnostic.

   → Mechanism: The continuous reasoning process is discretized into comparable atomic operations.
   → Design Motivation: Fine-grained decomposition reduces the learning difficulty at each step and enables the model to receive meaningful feedback locally.

2. **Step-wise Training Data Construction**: From a complete solution of $N$ steps, $N-1$ partial trajectories are constructed. For step $k$, the input is $\mathbf{x}_{\text{step}}^k = [\mathbf{x}, \mathbf{y}_{\text{step}}^1, \ldots, \mathbf{y}_{\text{step}}^{k-1}]$, and the task is to predict the next step $\mathbf{y}_{\text{step}}^k$.

   → Mechanism: A single expert solution is converted into multiple training instances, each corresponding to a different intermediate state.
   → Design Motivation: This substantially increases the volume of training data while teaching the model to continue reasoning from various intermediate states.

3. **Sequence Similarity Reward and Inner Monologue**: The model first generates an internal reasoning process $\mathbf{y}'_{\text{think}}$ wrapped in `<think>` tags, then outputs the action $\mathbf{y}'^k_{\text{step}}$. The reward is computed solely based on the sequence similarity between the generated action and the expert action:

   $R(\mathbf{y}'^k_{\text{step}}, \mathbf{y}^k_{\text{step}}) = \frac{2M}{T}$

   where $T$ is the total number of elements in both sequences and $M$ is the total number of elements in all non-overlapping matching blocks. This is implemented using Python's `difflib.SequenceMatcher`. If the output format is incorrect, a reward of $-1$ is assigned.

   → Mechanism: Comparison is performed at the action level rather than the token level, preserving the model's freedom in its internal reasoning.
   → Design Motivation: The reward is dense ($r \in [0,1]$) rather than a sparse binary signal, providing gradient information even when no rollout produces a fully correct solution.

4. **Dynamic Sampling Strategy**: Samples whose rollout reward variance is below a threshold $\epsilon$ are filtered out, as their advantage estimates approach zero and provide negligible learning signal. Sampling and filtering continue until the batch is filled.

   → Mechanism: This generalizes the filtering strategy of DAPO—originally designed for binary rewards—to continuous reward settings.
   → Design Motivation: Avoids wasting computation on samples that are either already mastered or completely indistinguishable.

### Loss & Training

The GRPO objective is used in conjunction with the sequence similarity reward described above. Key hyperparameters:
- Batch size: 512 (SRL), 128 (GRPO, due to high filtering rate)
- Learning rate: 5e-7
- Number of rollouts: 8
- KL loss coefficient: 0 (no KL constraint)
- Maximum 30 epochs of training; best checkpoint selected on validation set

Training can be performed with SRL alone or as a two-stage curriculum: SRL → RLVR.

## Key Experimental Results

### Main Results

**Mathematical Reasoning** (Base model: Qwen2.5-7B-Instruct; Training data: s1K-1.1, 1,000 hard problems)

| Method | AMC23 Avg@32 | AIME24 Avg@32 | AIME25 Avg@32 | Minerva Math | Avg |
|--------|-------------|---------------|---------------|-------------|-----|
| Base Model | 49.3 | 10.5 | 7.5 | 34.9 | 24.6 |
| SFT (R1 reasoning) | 26.8 | 3.9 | 5.4 | 20.2 | 16.6 |
| RLVR (GRPO) | 52.0 | 11.1 | 7.4 | 33.8 | 24.5 |
| **SRL** | **51.5** | **13.2** | **7.1** | **36.4** | **27.6** |
| **SRL → RLVR** | **52.1** | **13.3** | **8.6** | **36.4** | **28.3** |

Key observations: SFT suffers severe performance degradation on hard data (−8 points); RLVR yields marginal improvement; SRL provides significant gains (+3.0%); SRL→RLVR achieves the best overall performance (+3.7%).

**Software Engineering** (Base model: Qwen2.5-Coder-7B-Instruct; 5,000 expert trajectories)

| Method | Oracle File Edit | End-to-End |
|--------|-----------------|------------|
| Base Model | 5.8 | 3.2 |
| SWE-Gym-7B (SFT) | 8.4 | 4.2 |
| **SRL** | **14.8** | **8.6** |

SRL improves over SWE-Gym-7B by 74% in the oracle setting and doubles end-to-end performance.

### Ablation Study

| Configuration | Avg Performance | Note |
|---------------|----------------|------|
| SRL w/o dynamic sampling | 24.7 | Filtering low-variance samples yields +2.9% |
| SRL w/ dynamic sampling | 27.6 | Confirms the importance of the filtering strategy |
| Final-answer reward (RLVR) | 24.5 | Sparse reward provides limited benefit |
| Global sequence similarity (single-step) | 25.9 | Some improvement but inferior to multi-step |
| Multi-step sequence similarity (SRL) | 27.6 | Fine-grained guidance achieves best performance |

### Key Findings

1. **Reasoning length does not increase significantly**: The reasoning length distribution of SRL-trained models closely matches that of the base model, indicating that performance gains stem from improved reasoning quality rather than longer outputs.
2. **Emergent interleaved reasoning patterns**: The SRL→RLVR model exhibits distinctive reasoning behaviors—(1) upfront planning, (2) dynamic adjustment during the process, and (3) reflective verification—none of which are observed in conventional models.
3. **Cross-domain generalization**: SRL proves effective not only in mathematical reasoning but also in software engineering agent tasks, demonstrating the generality of the framework.

## Highlights & Insights

1. **Filling an important gap**: SRL offers an elegant middle ground between SFT overfitting and sparse RLVR rewards. By combining step-wise decomposition with sequence similarity rewards, it retains expert guidance while preserving the model's reasoning freedom.
2. **Elegant reward function design**: Computing similarity only on actions while leaving the internal reasoning process unconstrained allows the model to develop its own reasoning style. The use of `difflib.SequenceMatcher` makes reward computation both fast and stable.
3. **Curriculum learning strategy**: The SRL→RLVR combination treats SRL as a superior initialization method—first establishing foundational reasoning capabilities through fine-grained expert guidance, then further optimizing through free exploration.
4. **High practical utility**: No additional reward model training is required, no complex process reward annotation is needed, and training signals can be constructed directly from existing SFT data.

## Limitations & Future Work

1. **Dependence on structured expert trajectory formats**: SRL requires solution trajectories with explicit step delimiters (e.g., the numbered step format used by DeepSeek R1), a condition not universally satisfied.
2. **Student models must possess basic instruction-following capability**: If the base model cannot generate properly formatted outputs, initial rollouts will not provide useful learning signals.
3. **Sequence similarity reward may lack semantic precision**: String-matching-based similarity may fail to distinguish between mathematically equivalent steps expressed in different forms.
4. **Large models not explored**: Experiments are conducted solely on 7B models; the marginal benefit of SRL at larger scales remains unclear.
5. **Integration with process reward models**: Combining SRL with PRMs could provide more semantically grounded step-level rewards than sequence similarity alone.

## Related Work & Insights

- **DeepSeek-R1** and **GRPO** are representative RLVR works; SRL builds upon the GRPO optimization framework.
- **s1K**, **LIMO**, and related works demonstrate that a small amount of high-quality data can effectively distill reasoning capabilities; SRL achieves further gains on the same data.
- **SWE-Gym** and **SWE-Smith** provide SFT data for software engineering tasks; SRL substantially outperforms SFT baselines trained on the same data.
- Key insight: By combining ideas from RL and imitation learning, SRL identifies a principled balance between "what to imitate" and "what to explore freely."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — An elegant fusion of SFT and RLVR that fills an important gap
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across mathematics and software engineering with thorough ablations, but limited to 7B models
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, method description is precise, and figures are intuitive
- Value: ⭐⭐⭐⭐⭐ — Provides a practical new paradigm for training small models on hard problems

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [\[NeurIPS 2025\] Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](../../NeurIPS2025/code_intelligence/co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
