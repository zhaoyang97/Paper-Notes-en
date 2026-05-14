---
title: >-
  [Paper Note] First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training
description: >-
  [NeurIPS 2025][Multimodal VLM][unsupervised post-training] This paper proposes MM-UPT, a framework that introduces a third-stage "unsupervised post-training" phase following SFT and RL. By combining majority voting as a…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "unsupervised post-training"
  - "multimodal reasoning"
  - "GRPO"
  - "majority voting"
  - "self-improvement"
date: 2026-05-08
content_hash: 67652b0796285e67
---

# First SFT, Second RL, Third UPT: Continual Improving Multi-Modal LLM Reasoning via Unsupervised Post-Training

**Conference**: NeurIPS 2025
**arXiv**: [2505.22453](https://arxiv.org/abs/2505.22453)
**Code**: [https://github.com/waltonfuture/MM-UPT](https://github.com/waltonfuture/MM-UPT)
**Area**: Multimodal VLM / LLM Reasoning
**Keywords**: unsupervised post-training, multimodal reasoning, GRPO, majority voting, self-improvement

## TL;DR
This paper proposes MM-UPT, a framework that introduces a third-stage "unsupervised post-training" phase following SFT and RL. By combining majority voting as a pseudo-reward signal with GRPO, MM-UPT enables self-improvement of MLLMs, boosting Qwen2.5-VL-7B from 66.3% to 72.9% on MathVista.

## Background & Motivation

**Background**: Current MLLM post-training primarily relies on two stages — SFT (requiring annotated data) and RL (requiring verifiable ground truth or human preferences). Both stages demand large quantities of high-quality annotated multimodal data.

**Limitations of Prior Work**: As task complexity and volume grow, large-scale annotation of multimodal data becomes unsustainable. Multimodal reasoning tasks in particular require precise process-level annotations, incurring prohibitively high costs.

**Key Challenge**: Models require continual improvement, yet high-quality annotated data will eventually be exhausted. Prior unsupervised methods (e.g., Genixer, STIC) involve complex pipelines that are difficult to scale iteratively.

**Goal**: How can MLLMs continuously improve their reasoning capabilities without any external supervision?

**Key Insight**: Majority voting provides a simple yet effective signal — if the majority of sampled responses to a given question agree on an answer, that answer is likely correct and can serve as a pseudo-label.

**Core Idea**: Leverage GRPO with majority-voting pseudo-rewards to realize fully unsupervised multimodal post-training.

## Method

### Overall Architecture
MM-UPT operates on MLLMs that have already undergone SFT and RL. The input consists of unlabeled multimodal data $\{(I_i, q_i)\}$ (image–question pairs without answers). For each question, $G$ responses are sampled; majority voting determines the pseudo-label, and GRPO is applied to update the model.

### Key Designs

1. **Majority-Voting Pseudo-Reward Mechanism**:

    - Function: Generate training signals for unlabeled data.
    - Mechanism: For question $q$, sample $G=10$ responses $\{o_i\}$, extract answers via rule-based parsing $\hat{Y} = E(O)$, and select the most frequent answer $y^* = \arg\max_y \sum \mathbb{I}[y=\hat{y}_i]$. Responses consistent with $y^*$ receive reward $r_i=1$; others receive $r_i=0$.
    - Design Motivation: Majority voting has been demonstrated to effectively improve model performance without relying on external labels, encouraging the model to converge toward high-confidence, consistent answers.

2. **GRPO-Based Online Reinforcement Learning**:

    - Function: Update model parameters using pseudo-rewards.
    - Mechanism: Standard GRPO objective with group-normalized advantage estimates $\hat{A}_i = \frac{r_i - \text{mean}}{\text{std}}$, regularized by a KL divergence constraint to prevent excessive deviation from the reference model. The coefficient $\beta=0.01$ controls the KL penalty.
    - Design Motivation: GRPO requires no additional value model, offering high computational efficiency; the KL constraint prevents the model from over-fitting to inaccurate pseudo-labels.

3. **Synthetic Data Strategy**:

    - Function: Enable the model to generate its own training questions, further scaling data volume.
    - Mechanism: Two strategies are employed — (1) **Contextual synthesis**: given an original (image, question, answer) triplet, the model generates semantically distinct but related new questions; (2) **Direct synthesis**: given only an image, the model freely generates questions.
    - Design Motivation: Once human-curated data is exhausted, the model can autonomously generate new data to sustain continual training.

### Loss & Training
- Training for 15 episodes; learning rate $1 \times 10^{-6}$; rollout temperature 0.7.
- 10 responses sampled per data point ($G=10$).
- The visual encoder participates in training (not frozen).

## Key Experimental Results

### Main Results (Scenario 1: Standard Datasets with Labels Removed)

| Method | MathVision | MathVerse | MathVista | We-Math | Avg |
|--------|------------|-----------|-----------|---------|-----|
| Qwen2.5-VL-7B (base) | 24.87 | 43.83 | 66.30 | 62.87 | 49.47 |
| + GRPO (supervised, MMR1) | 29.01 | 45.03 | 71.40 | 67.24 | 53.17 |
| + SRLM (unsupervised) | 25.33 | 45.08 | 67.00 | 64.66 | 50.52 |
| + Genixer (unsupervised) | 23.68 | 43.30 | 65.50 | 64.66 | 49.29 |
| + **MM-UPT (unsupervised, MMR1)** | **26.15** | **44.87** | **72.90** | **68.74** | **53.17** |

As an unsupervised method, MM-UPT matches supervised GRPO (both 53.17 avg) and substantially outperforms other unsupervised baselines.

### Ablation Study: Different Base Models

| Model | Base Avg | + MM-UPT Avg | Gain |
|-------|----------|--------------|------|
| Qwen2.5-VL-3B | 39.00 | 41.72 | +7.4% |
| Qwen2.5-VL-7B | 49.47 | 51.23 | +3.6% |
| MM-Eureka-7B | 53.10 | 53.78 | +1.3% |
| ThinkLite-VL-7B | 52.63 | 54.07 | +2.8% |

### Key Findings
- **Synthetic data is effective**: The direct synthesis strategy even surpasses results obtained with original questions on Geo3K and GeoQA (+5.8% vs. +3.6%), owing to greater question diversity.
- **Compatible with supervised GRPO**: Models already trained with supervised GRPO (MM-Eureka, ThinkLite) continue to benefit from MM-UPT.
- **Trade-offs exist**: MM-UPT improves accuracy but reduces response diversity, and requires the base model to possess sufficient initial capability — otherwise majority voting may amplify errors.

## Highlights & Insights
- **Minimal yet effective**: The core idea reduces to "GRPO + majority voting," with no complex data filtering, multi-stage pipelines, or external tools. This simplicity facilitates reproducibility and iterative development.
- **On par with supervised methods**: Unsupervised MM-UPT matches supervised GRPO on the MMR1 dataset (both 53.17 avg), demonstrating that exploiting the internal self-consistency of a sufficiently capable model can substitute for external labels.
- **A forward-looking three-stage post-training paradigm**: The SFT → RL → UPT formulation is prescient, opening a new direction for post-training research.

## Limitations & Future Work
- **Majority voting accuracy depends on model capability**: When the model's initial accuracy on a problem type is low, majority voting locks in incorrect answers, creating a negative feedback loop that reinforces errors.
- **Validated only on mathematical reasoning**: The approach has not been evaluated on open-ended generation, dialogue, or perception tasks, where majority voting may not be directly applicable.
- **Occasional degradation on MathVerse**: Under certain configurations, MathVerse scores decrease, indicating that the method is not universally beneficial.
- **Scalability concerns**: Sampling 10 responses per data point incurs substantial computational overhead.

## Related Work & Insights
- **vs. TTRL**: The concurrent work TTRL also applies majority voting with GRPO for LLM training; MM-UPT extends this idea to the multimodal domain.
- **vs. Genixer/STIC**: These methods employ SFT/DPO for unsupervised improvement through complex pipelines; MM-UPT adopts online RL for a simpler and more effective approach.
- **vs. Supervised GRPO**: MM-UPT achieves comparable performance without labels, underscoring that the model's pre-existing internal knowledge constitutes an important source of training signal.

## Rating
- Novelty: ⭐⭐⭐ The core idea (majority voting + GRPO) is relatively straightforward, and TTRL is a concurrent work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models, datasets, and scenarios are evaluated with comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and the three-stage paradigm is well-positioned.
- Value: ⭐⭐⭐⭐ The work identifies unsupervised post-training as an important direction and offers a simple, practical method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[NeurIPS 2025\] Enhancing Outcome Reward-Based RL Training of MLLMs with Self-Consistency Sampling](enhancing_the_outcome_reward-based_rl_training_of_mllms_with_self-consistency_sa.md)
- [\[NeurIPS 2025\] Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs](towards_comprehensive_scene_understanding_integrating_first_and_third-person_vie.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](continual_multimodal_contrastive_learning.md)
- [\[NeurIPS 2025\] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)

</div>

<!-- RELATED:END -->
