---
title: >-
  [Paper Note] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models
description: >-
  [NeurIPS 2025][LLM Alignment][reinforcement-learning] RL fine-tuning of LLMs updates only 5%–30% of parameters in practice (sparse subnetworks)…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "reinforcement-learning"
  - "sparse subnetwork"
  - "parameter update sparsity"
  - "LLM finetuning"
  - "lottery ticket hypothesis"
date: 2026-05-08
content_hash: 9caba12b9b56b4ce
---

# Reinforcement Learning Finetunes Small Subnetworks in Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.11711](https://arxiv.org/abs/2505.11711)  
**Code**: [GitHub](https://github.com/SagnikMukherjee/sparsity_in_rl)  
**Area**: LLM Alignment
**Keywords**: reinforcement-learning, sparse subnetwork, parameter update sparsity, LLM finetuning, lottery ticket hypothesis

## TL;DR

RL fine-tuning of LLMs updates only 5%–30% of parameters in practice (sparse subnetworks), and these subnetworks exhibit high consistency across different random seeds, datasets, and algorithms. Fine-tuning only the identified subnetwork can reproduce both the performance and the parameter values of full fine-tuning.

## Background & Motivation

- Reinforcement learning (RL) is a critical stage in LLM post-training, used for improving reasoning capabilities and aligning models with human values.
- The prevailing assumption holds that RL requires substantial modification of model parameters to achieve target behaviors, motivating the widespread adoption of full-parameter fine-tuning.
- However, **does RL actually update all parameters?** This paper provides a negative answer.
- This phenomenon emerges naturally without any explicit sparsity regularization, architectural constraints, or parameter-efficient training methods.
- Prior work on the Lottery Ticket Hypothesis (LTH) focuses on performance recovery; this paper further demonstrates that nearly identical parameter values can also be recovered.

## Method

### Overall Architecture

Rather than proposing a new algorithm, this paper systematically analyzes the parameter update sparsity phenomenon in RL fine-tuning. The core pipeline is:

1. **Measuring update sparsity**: Comparing parameters before and after RL fine-tuning, defining $\text{sparsity}(\theta^0, \theta^1) = 1 - \|\theta^1 - \theta^0\|_0 / n$
2. **Extracting subnetworks**: Defining a binary mask $m \in \{0,1\}^{|\theta|}$, where $m_i = 1$ when $(\theta_{\text{init}} - \theta_{\text{full}})_i \neq 0$
3. **Subnetwork fine-tuning validation**: Restricting gradient updates via $m \odot \nabla_\theta \mathcal{L}(\theta)$, training only the subnetwork
4. **Cross-condition consistency analysis**: Comparing subnetwork overlap across different seeds, datasets, and algorithms

### Key Designs

**Definition and measurement of parameter update sparsity**:

- bfloat16 precision is used; absolute differences $\leq 10^{-5}$ are treated as equal (consistent with PyTorch default tolerance).
- Seven RL algorithms are covered: PPO, GRPO, DPO, ORPO, KTO, SimPO, and PRIME.
- Ten LLMs from different model families are evaluated.

**Subnetwork overlap metrics**:

$$o_1 = \frac{|\mathcal{I}_1 \cap \mathcal{I}_2|}{|\mathcal{I}_1|}, \quad o_2 = \frac{|\mathcal{I}_1 \cap \mathcal{I}_2|}{|\mathcal{I}_2|}$$

where $\mathcal{I}_1, \mathcal{I}_2$ are the index sets of updated parameters from two separate training runs.

**Core Conjecture (Conjecture 1)**: Under the same data and hyperparameters, subnetwork fine-tuning yields $\theta_{\text{sub}} \approx \theta_{\text{full}}$—that is, not only performance but also parameter values are nearly identical.

### Analysis of Update Sparsity Origins

The paper systematically investigates multiple potential contributing factors:

| Factor | Impact |
|--------|--------|
| Gradient clipping | Limited (sparsity similar with/without clipping: 69.8% vs. 68.8%) |
| KL regularization | Limited (SimPO without KL remains sparse) |
| SFT pre-training | Not necessary (DeepSeek-R1-Zero without SFT still 86% sparse) |
| Training steps | Large impact early on; converges over time |
| **In-distribution data training** | **Primary driver** |

Core finding: **Training on in-distribution data is the primary driver of sparsity.** Computing gradients over sequences already assigned high probability by the policy requires almost no parameter updates.

## Key Experimental Results

### Main Results: RL Update Sparsity

| Algorithm | Model | Update Sparsity |
|-----------|-------|----------------|
| DPO | Tulu-3-8B | 81.4% |
| DPO | Tulu-3-70B | 95.2% |
| GRPO | DeepSeek-Math-7B | 68.5% |
| GRPO | DeepSeek-R1-Zero | 86.0% |
| KTO | Eurus-7B | 96.0% |
| PPO | Math-Shepherd-Mistral-7B | 80.8% |
| SimPO | Llama-3-8B-SimPO | 86.5% |
| PRIME | Eurus-2-7B | 77.0% |

**68.5%–96.0% of parameters remain unchanged across all RL fine-tuned models.** In contrast, SFT exhibits only 6%–15% sparsity.

### Subnetwork Fine-Tuning Validation

| Task | $\theta_{\text{full}}$ | $\theta_{\text{sub}}$ | Gain |
|------|----------------------|---------------------|------|
| AGIEval LSAT-AR (DPO) | 21.3 | 24.8 | +3.5 |
| AGIEval LSAT-LR (DPO) | 53.1 | 54.7 | +1.6 |
| MMLU Pro Math (DPO) | 50.8 | 51.6 | +0.8 |
| MATH500 Overall (PRIME) | 69.8 | 72.2 | +2.4 |
| MATH500 Lvl5 (PRIME) | 40.3 | 45.5 | +5.2 |

**Subnetwork fine-tuning not only recovers the performance of the full model but outperforms full fine-tuning on all tasks.** At tolerance $10^{-4}$, $\theta_{\text{full}}$ and $\theta_{\text{sub}}$ are 100% identical.

### Ablation Study: Cross-Condition Subnetwork Overlap

| Varying Factor | Random Baseline | RL Subnetwork Overlap |
|----------------|-----------------|-----------------------|
| Different seeds | 36.7% | 60.5% |
| Different data | 14.6% / 36.7% | 26.7% / 67.1% |
| Seeds + data + algorithm | 23.0% / 12.9% | 59.1% / 33.2% |

### Key Findings

- The rank of update matrices is nearly full (99.2%–99.8%), indicating that RL updates are "sparse but full-rank."
- Updates are not concentrated in specific layers—nearly all parameter matrices exhibit similarly sparse updates (except LayerNorm).
- In PRIME, approximately 72% of parameters are never updated, 8% receive mutually canceling gradients, and 20% constitute the effective subnetwork.
- In-distribution SFT (e.g., rejection sampling) also yields sparse updates (~90% sparsity), whereas out-of-distribution DPO produces only ~7% sparsity.

## Highlights & Insights

1. **Beyond LTH**: Not only can subnetwork performance be reproduced, but parameter values are nearly identical—a stronger conclusion than the Lottery Ticket Hypothesis.
2. **Sparse but full-rank**: This stands in sharp contrast to the low-rank assumption underlying LoRA; RL updates select a small fraction of parameters yet span nearly the full column space of the parameter matrices.
3. **In-distribution training is key**: This provides a unified explanation for why both on-policy RL and off-policy RL following SFT produce sparse updates.
4. **Practical implications**: The findings provide a theoretical basis for efficient RL training—training only the subnetwork can substantially reduce computational cost.
5. **Transferable structure in pretrained models**: The high overlap of subnetworks across conditions suggests that pretrained models contain inherent substructures that are naturally amenable to RL fine-tuning.

## Limitations & Future Work

- Due to RL computational costs, only one factor is varied at a time, potentially overlooking interaction effects between factors.
- Some experiments rely on publicly available checkpoints rather than fully controlled training runs.
- The analysis is limited to language models; multimodal and diffusion models remain unexplored.
- Methods for early identification of subnetworks during training are not investigated—how can subnetworks be discovered at the onset of training?
- A rigorous theoretical analysis of the mathematical nature of update sparsity is lacking.
- Certain counterexamples (e.g., prolonged RL training) exist but are not thoroughly discussed.

## Related Work & Insights

- **Challenge to LoRA**: RL updates are sparse but full-rank, which differs from LoRA's low-rank assumption, suggesting that LoRA may not be the optimal strategy for RL fine-tuning.
- **Implications for efficient training**: Early identification of subnetworks during training could substantially reduce the computational overhead of RL.
- **Understanding RL vs. SFT**: The superior preservation of pretrained capabilities under RL compared to SFT may stem precisely from updating fewer parameters.
- **Cross-algorithm subnetwork reuse**: A cheaper algorithm (e.g., DPO) could be used to identify subnetworks, which are then applied to more expensive algorithms (e.g., PPO).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic revelation of parameter update sparsity in RL fine-tuning, with findings that surpass LTH
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 7 algorithms and 10 models with systematic ablation of contributing factors
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rigorous argumentation, though some evidence relies on publicly available checkpoints
- Value: ⭐⭐⭐⭐⭐ Deep insights into the nature of RL fine-tuning with significant theoretical and practical implications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](strategyproof_reinforcement_learning_from_human_feedback.md)
- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)
- [\[NeurIPS 2025\] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)

</div>

<!-- RELATED:END -->
