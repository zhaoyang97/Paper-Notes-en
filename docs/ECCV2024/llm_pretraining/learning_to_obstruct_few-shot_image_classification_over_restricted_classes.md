---
title: >-
  [Paper Note] Learning to Obstruct Few-Shot Image Classification over Restricted Classes
description: >-
  [ECCV 2024][LLM Pretraining][Few-Shot Learning] The Learning to Obstruct (LTO) algorithm is proposed, which modifies pre-trained backbone parameters via a MAML-like meta-learning approach to make them a "bad initialization" for specific restricted classes. This hinders the fine-tuning performance of few-shot classification methods on restricted classes while maintaining normal performance on other classes.
tags:
  - "ECCV 2024"
  - "LLM Pretraining"
  - "Few-Shot Learning"
  - "Model Security"
  - "Meta-Learning"
  - "Pre-trained Model Protection"
  - "Class Restriction"
date: 2026-05-08
content_hash: 679fcd9b2afc15a0
---

# Learning to Obstruct Few-Shot Image Classification over Restricted Classes

**Conference**: ECCV 2024  
**arXiv**: [2409.19210](https://arxiv.org/abs/2409.19210)  
**Code**: Unreleased  
**Area**: LLM Pre-training  
**Keywords**: Few-Shot Learning, Model Security, Meta-Learning, Pre-trained Model Protection, Class Restriction

## TL;DR

The Learning to Obstruct (LTO) algorithm is proposed, which modifies pre-trained backbone parameters via a MAML-like meta-learning approach to make them a "bad initialization" for specific restricted classes. This hinders the fine-tuning performance of few-shot classification methods on restricted classes while maintaining normal performance on other classes.

## Background & Motivation

Open-source pre-trained models significantly lower the barrier to building computer vision systems, but they also bring security risks: malicious users can leverage few-shot fine-tuning to quickly develop harmful applications (e.g., face classification in privacy-violating scenarios).

**Core Problem**: Can a pre-trained model be developed such that it is difficult to fine-tune on specific "restricted classes" while maintaining normal fine-tuning capability on other classes?

This is a novel problem setting. The differences from existing works are:
- **Machine Unlearning**: The goal is to "remove" acquired class knowledge; LTO aims to "prevent learning" new restricted classes.
- **Data Poisoning**: Modifies training data to disrupt the model; LTO modifies the model weights themselves.
- **MAML**: Learns a "good initialization" to quickly adapt the model to new tasks; LTO does the opposite, learning a "bad initialization".

## Method

### Overall Architecture

Given a pre-trained backbone $\theta^p$, a restricted class set $\mathcal{R}$, and a few-shot classification (FSC) algorithm $\bm{F}$, the LTO algorithm $\bm{A}$ modifies the pre-trained weights to $\bm{A}(\theta^p)$ such that the FSC method performs poorly on $\mathcal{R}$ after adaptation, while performing normally on other classes $\mathcal{R}'$.

### Key Designs

1. **LTO Optimization Objective — Learning a Bad Initialization**:

Inspired by MAML, LTO splits the data into $\mathcal{D}_{obs}$ (to evaluate obstruction quality) and $\mathcal{D}_{fsc}$ (for FSC training). The optimization problem is formulated as:

$$\min_\theta \mathbb{E}_{\mathcal{T}^{(t)}} \left[\mathcal{L}_{\mathcal{R}'}([\tilde{\theta}, \tilde{\phi}], \mathcal{D}_{obs}^{(t)}) - \mathcal{L}_{\mathcal{R}}([\tilde{\theta}, \tilde{\phi}], \mathcal{D}_{obs}^{(t)})\right]$$

$$\text{s.t.} \quad \tilde{\theta}, \tilde{\phi} = F([\theta, \phi], \mathcal{D}_{fsc}^{(t)})$$

- $\mathcal{L}_{\mathcal{R}'}$: Loss on other classes (minimized $\to$ maintain performance)
- $-\mathcal{L}_{\mathcal{R}}$: Negative loss on restricted classes (maximized $\to$ disrupt performance)
- Key constraint: $\tilde{\theta}$ represents the parameters updated by the FSC learner $F$, which means the model first "tries to learn" before its obstruction effect is evaluated.

2. **Bi-level Optimization**:

A bi-level optimization similar to MAML is used:
- **Inner loop**: Updates parameters $\theta \to \tilde{\theta}$ using the FSC learner $F$ on $\mathcal{D}_{fsc}$.
- **Outer loop**: Calculates the gradient $\Delta\theta^{(t)} = \nabla_\theta[\mathcal{L}_{\mathcal{R}'} - \mathcal{L}_{\mathcal{R}}]$ on $\mathcal{D}_{obs}$ and backpropagates using a first-order approximation.
- The parameters are restored to the starting state of each epoch at the end of every epoch, ensuring that different tasks within a batch share the same starting point.

3. **Restricted Class Selection Strategy**: Classes are grouped semantically into superclasses (e.g., "birds", "electronic devices"). An entire superclass is chosen as $\mathcal{R}$ to simulate real-world scenarios where certain categories of applications are restricted.

4. **Extension to Attribute Learning**: For CelebA attribute classification, each attribute is treated as an independent binary classification task. $\mathcal{R}$ is the set of restricted attributes. CLIP binary classifiers are constructed for each attribute, and LTO is applied.

5. **CLIP-based FSC Adaptation**: For CLIP-based methods (CoOp, TipAdapter), due to GPU memory constraints, a resampling strategy is adopted—randomly sampling a subset of all prompts and classes every few steps to compute gradients, providing an unbiased estimator.

### Loss & Training

The outer optimization objective of LTO is $\mathcal{L}_{\mathcal{R}'} - \mathcal{L}_{\mathcal{R}}$, using mini-batch gradient descent. The obstruction learning is conducted for 200 steps, with a batch size of 20, and the inner loop FSC undergoes 20 learning steps.

Two baselines for comparison:
- **Only$\mathcal{R}$**: Directly maximizes the loss on restricted classes (ignoring other classes).
- **NoF**: Directly optimizes without passing through the FSC learner (ignoring the fine-tuning process).

## Key Experimental Results

### Main Results

DropRatio@2% (Δ@2, higher is better for obstruction) of classic FSC on ImageNet:

| FSC Method | Setting | Only$\mathcal{R}$ | NoF | **LTO (Ours)** |
|----------|------|----------|-----|---------|
| ProtoNet | 1-shot | 1.10 | 3.77 | **4.42** |
| ProtoNet | 5-shot | 1.10 | 2.00 | **2.40** |
| MetaOptNet | 1-shot | 1.95 | 8.65 | **8.85** |
| MetaOptNet | 5-shot | 1.94 | 10.11 | **13.40** |

CLIP-based FSC on CIFAR100/ImageNet Δ@2 (Average):

| FSC Method | CIFAR100 Only$\mathcal{R}$ | CIFAR100 Ours | ImageNet Only$\mathcal{R}$ | ImageNet Ours |
|----------|----------|------|----------|------|
| CE | 1.48 | **11.99** | 2.80 | **7.65** |
| CoOp | 1.80 | **6.73** | 1.19 | **4.58** |
| TipAdapter | 2.08 | **10.16** | 2.16 | **5.86** |

### Ablation Study

Data efficiency analysis (ImageNet superclass id=1, increasing FSC training data):

| FSC Method | 1× (5-shot) | 2× | 3× | 4× |
|----------|------------|-----|-----|-----|
| CE | 9.93 | 6.34 | 2.76 | 2.82 |
| CoOp | 6.15 | 2.46 | 2.26 | 2.32 |
| TipAdapter | 5.92 | 3.09 | 3.06 | 3.76 |

Cross-method transferability (Δ@2 when training with $\bm{F}$ and evaluating with $\bm{F}'$):

| $\bm{F}$ \ $\bm{F}'$ | CE | CoOp | TipAdapter | Average |
|------|------|------|-----------|------|
| CE | 9.93 | 4.71 | 7.33 | 7.32 |
| CoOp | 4.79 | 6.15 | 4.34 | 5.09 |
| TipAdapter | 4.16 | 7.75 | 5.92 | 5.94 |

### Key Findings

- LTO effectively obstructs learning on restricted classes across all FSC methods, yielding a Δ@2 significantly greater than 1 (meaning the performance drop on restricted classes is much larger than on other classes).
- Considering the inner-loop optimization of the FSC learner $F$ is crucial: the NoF baseline (which does not simulate the fine-tuning process) performs significantly worse than the full LTO.
- Increasing the FSC training data or training duration only partially recovers the obstructed performance, indicating that the obstruction effect has a certain level of robustness.
- LTO exhibits cross-method transferability: obstruction learned using one FSC method remains effective against other unseen FSC methods.
- Comparison with the machine unlearning method SSD indicates that SSD's obstruction effect is far weaker than LTO's (CIFAR100 average Δ@2: SSD 1.48 vs. LTO 11.99), as SSD merely unlearns rather than preventing relearning.
- In attribute learning, some attributes are easier to obstruct (Pale_Skin: 31.64%), while others are harder (Gray_Hair: 0.08%). Semantic correlation among attributes can lead to collateral obstruction.

## Highlights & Insights

1. **Novel and Practical Problem Formulation**: A brand-new perspective is presented in open-source model security—instead of preventing model leakage, it makes the model "unusable" for specific tasks from the source.
2. **Duality with MAML**: While MAML learns a good initialization, LTO learns a bad initialization, which is conceptually neat and elegant.
3. **Impressive Cross-Method Transferability**: Obstruction trained on one FSC method is also effective against other unseen FSC methods, indicating that LTO indeed modifies the feature representations in the backbone associated with the restricted classes.
4. **Well-designed Evaluation Metrics**: DropRatio@β simultaneously considers both the magnitude of reduction in restricted classes and the level of preservation on public classes.

## Limitations & Future Work

1. LTO assumes that the obstructer knows the FSC algorithm. Although experiments display cross-method transferability, the effect on completely unknown fine-tuning strategies remains unverified.
2. Adding a sufficient amount of data (4×) can partially counteract the obstruction effect, indicating limited protection in data-abundant scenarios.
3. Evaluated only on image classification, its applicability to other downstream tasks (e.g., detection, segmentation) is unknown.
4. In attribute learning, unintended collateral obstruction occurs due to semantic correlation (e.g., obstructing "blond hair" inadvertently affects "brown hair"). Precise single-attribute obstruction remains a challenge.
5. Obstruction over 200 steps requires repeating the inner-loop FSC training (20 steps × 20 batch), which incurs a non-trivial computational overhead.

## Related Work & Insights

- **MAML [Finn et al.]**: The methodological foundation of this paper, with LTO being its "reverse version."
- **ProtoNet / MetaOptNet**: Classic FSC methods used as target benchmarks for obstruction.
- **CoOp / TipAdapter**: CLIP-based FSC methods, which validate the efficacy of LTO on foundation models.
- **SSD [Foster et al.]**: A machine unlearning method, where experimental comparison shows that "forgetting" is not equivalent to "blocking relearning."
- **Data Poisoning**: Modifies the data rather than the model, which is complementary to LTO.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — A completely new problem formulation with a very clever design that is dual to MAML.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Validated across four FSC methods, three datasets, and two task types, along with deep analyses on data/time efficiency and cross-method transferability.
- Writing Quality: ⭐⭐⭐⭐ — Clear introduction to the problem and complete formulation derivations. Although tables are dense, they are highly informative.
- Value: ⭐⭐⭐⭐ — Pioneers a new direction of "obstructing learning", offering important insights for the AI safety community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning](../../ICML2025/llm_pretraining/revisiting_continuity_of_image_tokens_for_cross-domain_few-shot_learning.md)
- [\[ECCV 2024\] Prompting Language-Informed Distribution for Compositional Zero-Shot Learning](prompting_language-informed_distribution_for_compositional_zero-shot_learning.md)
- [\[ECCV 2024\] DragAPart: Learning a Part-Level Motion Prior for Articulated Objects](dragapart_learning_a_part-level_motion_prior_for_articulated_objects.md)
- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)
- [\[ICML 2026\] XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge](../../ICML2026/llm_pretraining/xtransfer_modality-agnostic_few-shot_model_transfer_for_human_sensing_at_the_edg.md)

</div>

<!-- RELATED:END -->
