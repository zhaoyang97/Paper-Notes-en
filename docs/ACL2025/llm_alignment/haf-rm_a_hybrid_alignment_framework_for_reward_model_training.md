---
title: >-
  [Paper Note] HAF-RM: A Hybrid Alignment Framework for Reward Model Training
description: >-
  [ACL 2025][LLM Alignment][Reward Model Training] This paper proposes HAF-RM, a Hybrid Alignment Framework that retains the policy layer during reward model training. By simultaneously optimizing sequence-level reward loss and token-level policy loss to jointly supervise the shared internal preference model, HAF-RM consistently outperforms standard Baselines and DPO methods across 5 datasets.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Reward Model Training"
  - "Hybrid Alignment Framework"
  - "Policy Loss Regularization"
  - "DPO"
  - "RLHF"
  - "Bradley-Terry Model"
date: 2026-05-08
content_hash: c29cace0bf16ac9c
---

# HAF-RM: A Hybrid Alignment Framework for Reward Model Training

**Conference**: ACL 2025  
**arXiv**: [2407.04185](https://arxiv.org/abs/2407.04185)  
**Code**: [https://haf-rm-anonymized.github.io](https://haf-rm-anonymized.github.io)  
**Authors**: Shujun Liu, Xiaoyu Shen, Yuhang Lai, Siyuan Wang, Shengbin Yue, Zengfeng Huang, Xuanjing Huang, Zhongyu Wei  
**Institutions**: Fudan University, Eastern Institute of Technology (Ningbo), University of Southern California  
**Area**: LLM Alignment / Reward Model  
**Keywords**: Reward Model Training, Hybrid Alignment Framework, Policy Loss Regularization, DPO, RLHF, Bradley-Terry Model  

## TL;DR

This paper proposes HAF-RM, a Hybrid Alignment Framework that retains the policy layer during reward model training. By simultaneously optimizing sequence-level reward loss and token-level policy loss to jointly supervise the shared internal preference model, HAF-RM consistently outperforms standard Baselines and DPO methods across 5 datasets.

## Background & Motivation

**Background**: The Reward Model (RM) is a core component of LLM alignment, widely used in scenarios such as RLHF training, Best-of-N sampling, and data construction. Standard training frameworks decouple the reward model into an internal preference model (the core Transformer body) and a reward prediction layer (linear projection), trained end-to-end using the Bradley-Terry preference loss.

**Limitations of Prior Work**:
   - Most reward models are proprietary commercial products, limiting further training and transferability.
   - Incorrect and ambiguous preference annotations exist in the training data.
   - **The standard training paradigm may lead to insufficient supervision of the internal preference model**—the core Transformer body is only indirectly supervised through sequence-level reward signals, leaving token-level preference information unutilized.
   - Although DPO can implicitly generate reward values, its generalization ability on out-of-distribution (OOD) data is extremely poor.

**Design Motivation**: Reward models and policy models are highly similar in structure—both share the same Transformer backbone (internal preference model) and only differ in their output layers (reward layer vs. policy layer). HAF leverages this structural similarity to directly supervise the internal preference model at the token level by introducing an additional policy loss, while optimizing the reward mapping layer at the sequence level.

## Method

### Overall Architecture

HAF retains both the reward layer $F$ and the policy layer $K$ (see Figure 1), which share the internal preference model $\phi$:
- Reward model output: $\boldsymbol{r}(x, y) = F \circ \phi(x, y)$ (sequence-level scalar reward)
- Policy model output: $\boldsymbol{\pi}(x, y) = K \circ \phi(x, y)$ (token-level generation probability)

### Loss & Training

**Reward Loss** (sequence-level): Standard Bradley-Terry preference loss

$$\mathcal{L}_s = \mathbb{E}_{(x,y,y') \sim \mathcal{D}} [-\log\sigma(\boldsymbol{r}(x,y) - \boldsymbol{r}(x,y'))]$$

**Policy Loss** (token-level): Formulated as DPO loss

$$\mathcal{L}_P = \mathbb{E}_{(x,y,y') \sim \mathcal{D}} [-\log\sigma(\tau(pd_{win} - pd_{lose}))]$$

where $pd_{win} = \log \frac{\boldsymbol{\pi}(x,y)}{\boldsymbol{\pi}_{ref}(x,y)}$ represents the log-probability ratio relative to the reference policy, and $\tau=0.1$ is a hyperparameter.

**Hybrid Alignment Loss**:

$$\mathcal{L}_H = \mathbb{E}_d [D_1(\boldsymbol{r}(d), \boldsymbol{r}^*(d)) + \alpha \cdot D_2(\boldsymbol{\pi}(d), \boldsymbol{\pi}^*(d))]$$

where $\alpha$ is a balancing hyperparameter, and the shared internal preference model $\phi$ simultaneously receives gradients from both loss terms.

### Why is HAF Better?

The paper provides two intuitive explanations:

1. **Claim 1**: The model learned via joint calibration loss outperforms the model using only the standard reward loss—because the additional policy constraint restricts the search space of the preference space, reducing overfitting.
2. **Claim 2**: The policy loss acts as a **regularization term** preventing internal representation degradation—standard reward training may cause the representation space of the internal preference model to collapse, whereas the token-level policy loss maintains the richness of the representations.

**Empirical Validation** (Figure 3): Generations from the policy model with shared parameters are scored higher by the reward model, indicating that both indeed learn similar preferences.

### Model Architecture

Based on a standard decoder-only LLM, the original policy projection layer (the linear layer before softmax) is retained, while a reward projection layer (which outputs a scalar) is added. Both share the Transformer backbone.

## Key Experimental Results

### Datasets

5 public preference datasets:
- HH-Harmless (12,915), HH-Helpful (13,543), Beaver Safe (47,625), Alpaca Human Pref (8,722), Chatbot Arena (19,466)

### Base Models

- Phi-2-2.7B (full-parameter fine-tuning)
- Mistral-7B-base-v0.3 (LoRA)
- Mistral-7B-Instruct-v0.2 (LoRA)

### Main Results: Internal Preference Classification Accuracy

| Method | Helpful | Harmless | CA | BS | AHP | Average |
|------|---------|----------|-----|-----|-----|------|
| DPO (Mistral) | 74.29 | 70.30 | 81.90 | **92.70** | 60.30 | 75.90 |
| Baseline (Mistral) | 76.20 | 72.70 | 79.80 | 80.80 | 56.30 | 73.16 |
| **HAF (Mistral)** | 75.80 | **73.10** | 81.90 | 88.70 | **63.10** | **76.52** |
| DPO (Phi-2) | 69.70 | 66.30 | 66.80 | 87.80 | 52.60 | 68.64 |
| Baseline (Phi-2) | 64.30 | 69.50 | 79.30 | 76.00 | 58.40 | 69.50 |
| **HAF (Phi-2)** | **76.40** | **70.40** | 79.00 | 84.00 | **60.80** | **74.12** |

**Key Findings**:
- HAF consistently outperforms Baseline and DPO in average accuracy across all three backbone models.
- DPO performs best on the BS dataset (where data distribution is highly concentrated) but remains unstable on other datasets.
- DPO and Baseline learn different features, and HAF effectively integrates both.

### Training on Mixed Data

Training after uniformly mixing the 5 datasets:
- HAF achieves the best overall performance across all models, indicating it is better at learning the diversity of mixed preference distributions.
- DPO's performance on CA and Helpful drops significantly on mixed data—it tends to fit the dominant features of the data distribution.

### Out-of-Distribution (OOD) Generalization

The datasets are divided into Safety categories (BS, Harmless) and Chat categories (AHP, CA, Helpful) for cross-category evaluation:

**In-Domain Evaluation** (different datasets within the same category):
- HAF (Mistral) achieves an average of 70.30%, outperforming Baseline by 9.07% and DPO by 2.26%.

**Out-of-Domain Evaluation** (RewardBench):
- HAF (Mistral) averages 81.95%, outperforming Baseline by 28.47% and DPO by 7.80%.
- DPO's OOD test results are mostly around 50%—completely losing its modeling capability.

### Downstream Task: Best-of-N Sampling

Using each reward model to select the best response from multiple candidates generated by the policy model:
- HAF achieves the highest win rate in GPT-4 evaluation (Phi-2: 52.0% vs. Baseline 27.4%; Mistral: 51.1%).
- HAF-selected Top-1 responses also show the highest alignment with GPT-4 rankings (Phi-2: 33.77%, Mistral: 18.19%).

### Downstream Task: RLHF

Using each reward model for PPO training:
- Policies trained with the HAF reward model secure the highest win rates in most scenarios under GPT-4 evaluation.
- RLHF policies trained with the DPO reward model show unstable performance.

## Highlights & Insights

1. **Clever Exploitation of Structural Similarity**: The insight that the reward model and policy model essentially share the internal preference backbone is core to this paper; the hybrid supervision idea is simple yet elegant.
2. **OOD Vulnerability of DPO**: The experiments clearly reveal that DPO as a reward model almost completely fails in out-of-distribution (OOD) scenarios (with accuracy near 50%), which is driven by its strong preference for specific linguistic styles.
3. **Regularization Perspective**: The explanation that policy loss acts as a regularization to prevent internal representation degradation is persuasive and backed by empirical evidence.
4. **Comprehensive Evaluation**: The experiments cover key application scenarios of reward models, ranging from intrinsic evaluation to OOD generalization, and further to extrinsic evaluation via Best-of-N and RLHF.

## Limitations & Future Work

1. **Limited Scale of Backbone Models**: The largest model evaluated is Mistral-7B; it has not been validated on larger scales (e.g., 70B models).
2. **Choice of Policy Loss**: Only DPO is used as the implementation of policy loss, leaving other policy optimization methods (e.g., KTO, IPO, etc.) unexplored.
3. **Sensitivity of Hyperparameter $\alpha$**: The balancing coefficient in the hybrid loss may require tuning for different datasets or architectures.
4. **Weak Theoretical Analysis**: Claim 1 and Claim 2 are mostly intuitive explanations and lack rigorous theoretical proofs.
5. **DPO Still Dominates on the BS Dataset**: HAF does not completely outperform DPO on datasets with concentrated distributions, suggesting that the hybrid framework might introduce noise in certain scenarios.

## Related Work & Insights

- **RLHF and Reward Models**: Christiano et al. (2017) and Ouyang et al. (2022) established the standard training framework.
- **DPO**: Direct Preference Optimization by Rafailov et al. (2023) implicitly converts a policy model into a reward model; this paper reveals its OOD limitations.
- **Data Augmentation Directions**: Methods like Li et al. (2023a) and Wu et al (2023) improve reward models from a data perspective, which is complementary to the training framework improvements in this paper.
- **Fine-Grained Reward Signals**: Cao et al. (2024) and Lai et al. (2024) leverage fine-grained signals to improve reward models; HAF's token-level policy loss can be seen as a complementary approach.
- **Bradley-Terry Model**: The standard preference modeling framework that optimizes the model by translating reward differences into probabilities.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The hybrid supervision idea is simple yet effective, offering a novel perspective by leveraging structural similarity.
- **Experimental Thoroughness** ⭐⭐⭐⭐⭐: Comprehensive evaluation featuring 5 datasets × 3 backbone models × joint training + OOD + Best-of-N + RLHF.
- **Theoretical Depth** ⭐⭐⭐: The intuitive explanations are convincing but lack rigorous proofs.
- **Value** ⭐⭐⭐⭐: Simple and plug-and-play enhancement, offering direct practical references for reward model training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](../../ICML2025/llm_alignment/on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[ACL 2025\] Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization](rethinking_reward_model_evaluation_through_the_lens_of_reward_overoptimization.md)
- [\[ICLR 2026\] Reward Model Routing in Alignment](../../ICLR2026/llm_alignment/reward_model_routing_in_alignment.md)
- [\[ACL 2025\] Model Extrapolation Expedites Alignment](expo_model_extrapolation.md)
- [\[ACL 2025\] From Lists to Emojis: How Format Bias Affects Model Alignment](from_lists_to_emojis_how_format_bias_affects_model_alignment.md)

</div>

<!-- RELATED:END -->
