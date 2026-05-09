---
title: >-
  [Paper Note] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation
description: >-
  [NeurIPS 2025][LLM Safety][machine unlearning] This paper models the machine unlearning evaluation problem as a cryptographic game (the unlearning sample inference game), quantifies unlearning quality via the adversary's "advantage," and addresses multiple shortcomings of traditional MIA accuracy as an evaluation metric—namely, the lack of a retrain-as-zero baseline, sensitivity to data partitioning, and sensitivity to the choice of MIA. A SWAP test is further proposed as an efficient practical approximation.
tags:
  - NeurIPS 2025
  - LLM Safety
  - machine unlearning
  - evaluation metric
  - cryptographic game
  - membership inference attack
  - SWAP test
date: 2026-05-08
content_hash: dab0960e9923d1c7
---

# A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation

**Conference**: NeurIPS 2025
**arXiv**: [2404.11577](https://arxiv.org/abs/2404.11577)
**Code**: None
**Area**: AI Safety / Machine Unlearning Evaluation / Privacy Protection
**Keywords**: machine unlearning, evaluation metric, cryptographic game, membership inference attack, SWAP test

## TL;DR
This paper models the machine unlearning evaluation problem as a cryptographic game (the unlearning sample inference game), quantifies unlearning quality via the adversary's "advantage," and addresses multiple shortcomings of traditional MIA accuracy as an evaluation metric—namely, the lack of a retrain-as-zero baseline, sensitivity to data partitioning, and sensitivity to the choice of MIA. A SWAP test is further proposed as an efficient practical approximation.

## Background & Motivation
Machine unlearning (MU) is motivated by the "right to be forgotten" enshrined in data protection regulations such as GDPR, which grants users the right to request that organizations remove their personal data from trained models. The naive solution of full retraining is computationally prohibitive, giving rise to a large body of approximate unlearning algorithms. However, **how to reliably evaluate the data removal effectiveness of these algorithms** remains a critical open problem, as also highlighted by the NeurIPS 2023 Machine Unlearning Competition.

Existing evaluation approaches fall into three categories:

- **Retraining comparison** (parameter or posterior divergence): highly sensitive to training stochasticity and therefore unreliable.
- **Theoretical guarantees** (certified removal): relies on strong assumptions such as convexity, limiting practical applicability.
- **Attack-based evaluation** (MIA): the most common approach but also the most problematic—MIA accuracy is uncalibrated, retraining does not guarantee the lowest possible MIA accuracy, and results depend heavily on the choice of MIA algorithm and data partitioning.

## Core Problem
How can one design a theoretically sound and practically applicable evaluation metric for machine unlearning? Specifically, a good metric should satisfy: (1) retraining, as the ideal unlearning method, should receive a perfect score; (2) it should be consistent with theoretical guarantees such as certified removal; and (3) it should be robust to the choice of MIA algorithm. Existing MIA accuracy/AUC metrics fail to satisfy all three properties simultaneously.

## Method

### Overall Architecture
The core idea draws from cryptographic security games: the paper defines an **Unlearning Sample Inference Game** between a challenger (the unlearning algorithm) and an adversary (a MIA attacker), and quantifies unlearning quality via the adversary's advantage. This advantage is carefully designed to possess provably desirable theoretical properties.

The game $\mathcal{G} = (\text{Ul}, \mathcal{A}, \mathcal{D}, \mathbb{P}_\mathcal{D}, \alpha)$ proceeds in three phases:

### Key Designs

1. **Initialization Phase**: The dataset $\mathcal{D}$ is randomly partitioned into a retain set $\mathcal{R}$, a forget set $\mathcal{F}$, and a test set $\mathcal{T}$, subject to the forgetting ratio $\alpha = |\mathcal{F}|/(|\mathcal{R} \cup \mathcal{F}|)$ and $|\mathcal{F}| = |\mathcal{T}|$. A random oracle $\mathcal{O}_s(b)$ is constructed, sampling from $\mathcal{F}$ when $b=0$ and from $\mathcal{T}$ when $b=1$. A sensitivity distribution $\mathbb{P}_\mathcal{D}$ is introduced to assign higher sampling probability to more sensitive data.

2. **Challenger Phase**: The unlearning algorithm Ul performs unlearning of $\mathcal{F}$ on a model trained on $\mathcal{R} \cup \mathcal{F}$, producing the unlearned model $m = \text{Ul}(\text{Lr}(\mathcal{R} \cup \mathcal{F}), \mathcal{F})$.

3. **Adversary Phase**: The adversary $\mathcal{A}$ has access only to the unlearned model $m$ and the oracle $\mathcal{O}$, and must guess $b \in \{0, 1\}$—i.e., determine whether the data provided by the oracle originates from $\mathcal{F}$ or $\mathcal{T}$.

4. **Advantage Definition**: The advantage is computed by averaging over all possible data partitions and taking the absolute value:
$$\text{Adv}(\mathcal{A}, \text{Ul}) = \frac{1}{|\mathcal{S}_\alpha|} \left| \sum_{s \in \mathcal{S}_\alpha} \Pr(\mathcal{A}^{\mathcal{O}_s(0)}(m)=1) - \sum_{s \in \mathcal{S}_\alpha} \Pr(\mathcal{A}^{\mathcal{O}_s(1)}(m)=1) \right|$$
**Unlearning Quality** is then defined as $\mathcal{Q}(\text{Ul}) = 1 - \sup_\mathcal{A} \text{Adv}(\mathcal{A}, \text{Ul})$.

5. **SWAP Test (Core Practical Tool)**: Full enumeration of all partitions is infeasible. The SWAP test uses only two symmetric partitions $s = (\mathcal{R}, \mathcal{F}, \mathcal{T})$ and $s' = (\mathcal{R}, \mathcal{T}, \mathcal{F})$ (swapping the forget and test sets), and approximates the full metric using the average advantage over this swap pair. The key insight is that swap pairs preserve the symmetry of the original definition and thus retain properties such as the zero baseline. By contrast, randomly selecting two independent partitions with overlap allows a trivial table-lookup attack to achieve advantage = 1, rendering the metric completely degenerate.

6. **Weak Adversary Assumption**: Since state-of-the-art MIAs make independent decisions for each data point, the adversary is restricted in practice to a single interaction with the oracle (weak adversary), with the supremum approximated by selecting the strongest among multiple SOTA MIAs.

### Core Theorems

- **Zero-Baseline Theorem (Theorem 3.3)**: For any adversary $\mathcal{A}$, $\text{Adv}(\mathcal{A}, \text{Retrain}) = 0$, i.e., $\mathcal{Q}(\text{Retrain}) = 1$. The proof relies on symmetry—each data point appears in $\mathcal{F}$ and $\mathcal{T}$ the same number of times across all partitions, causing any MIA bias to cancel out.
- **Certified Removal Guarantee (Theorem 3.5)**: If Ul is an $(\epsilon, \delta)$-certified removal algorithm, then $\text{Adv}(\mathcal{A}, \text{Ul}) \leq 2(1 - \frac{2 - 2\delta}{e^\epsilon + 1})$. This establishes a quantitative connection between the proposed metric and theoretical guarantees.

## Key Experimental Results

Experiments are conducted on CIFAR-10 with ResNet-20, comparing seven unlearning methods: Retrain, Fisher, Ftfinal, Retrfinal, NegGrad, SalUn, and SSD.

| Setting | Key Finding |
|---------|------------|
| Dataset size ablation | Method rankings remain consistent across dataset sizes; small-scale experiments reliably predict large-scale behavior |
| $\alpha$ ablation | Rankings remain consistent across different forgetting ratios |
| DP budget validation | $\mathcal{Q} \approx 1$ for Retrain (validates zero baseline); $\mathcal{Q}$ negatively correlates with $\epsilon$ (validates Theorem 3.5) |
| vs. IC Score | IC Score fails to produce a negative correlation with DP budget and yields inconsistent rankings |
| vs. MIA AUC | MIA AUC has poor discriminability; most methods cluster in the range 0.4–0.5, making differentiation difficult |
| More datasets | Rankings are consistent across CIFAR-100, MNIST, and SST5, and reflect dataset difficulty ($\mathcal{Q}$ higher on MNIST, lower on CIFAR-100) |
| Model architecture | Rankings are consistent across ResNet-20/56/110 |

### Ablation Study
- SSD consistently and significantly outperforms all other approximate unlearning methods across all settings, consistent with its SOTA status.
- The SWAP test exhibits substantially lower variance compared to random-partition methods, demonstrating greater stability.
- The weak adversary restriction does not impair practical utility, as current SOTA MIAs are all weak adversaries.

## Highlights & Insights
- **Elegant transfer of the cryptographic perspective**: The paper imports the concept of advantage from security games into unlearning evaluation, naturally resolving the retrain zero-baseline problem through symmetry rather than artificial offset correction.
- **Simplicity of the SWAP test design**: Only two training-and-unlearning runs are needed to approximate the full metric, while preserving the core theoretical properties—making the approach highly practical.
- **Unified framework**: The formulation accommodates black-box/white-box and weak/strong adversaries, offering strong theoretical generality.
- **Bridging certified removal and empirical evaluation**: Theorem 3.5 establishes a quantitative link between the empirical metric and theoretical guarantees, a rare contribution that bridges theory and practice.

## Limitations & Future Work
- **Bottleneck of the weak adversary**: Even the None (no unlearning) baseline achieves $\mathcal{Q}$ as high as 0.587, indicating that current SOTA MIAs are not sufficiently powerful. Stronger MIAs in the future would improve the metric's discriminability.
- **Restricted to i.i.d. unlearning scenarios**: Uniform partitioning limits applicability to non-i.i.d. unlearning settings (e.g., class-wise forgetting); the authors acknowledge that extending to non-uniform partitions would break the existing theoretical guarantees.
- **High experimental computational cost**: Although the SWAP test simplifies the procedure, the total cost of training multiple models (including DP models) remains substantial (approximately 6 GPU-days for the variance ablation).
- **Limited dataset and task coverage**: Validation is primarily on image classification; evaluation in generative models and large language models remains absent.

## Related Work & Insights
- **vs. Triantafillou & Kairouz (2023)**: That work uses hypothesis testing combined with MIA to estimate privacy budgets, which is theoretically more rigorous but computationally prohibitive (requiring a separate MIA to be trained per forget sample); the proposed method is substantially more practical.
- **vs. Goel et al. (2023) IC Test**: The IC Test evaluates unlearning indirectly by confusing class labels, but is insensitive to DP budget changes and produces inconsistent rankings; the proposed method directly measures privacy leakage.
- **vs. MIA AUC (Golatkar et al. 2021)**: MIA AUC does not satisfy the zero baseline, is sensitive to data partitioning, and has poor discriminability; the proposed method overcomes all of these shortcomings.
- **vs. Brimhall et al. (2025) Computational Unlearning**: A similarly cryptography-inspired approach that compares retrained and unlearned models in an indistinguishability game, but relies on different technical assumptions.

## Inspirations & Connections
- The symmetry-based design of the SWAP test is transferable to other comparative evaluation settings, such as assessing the effectiveness of data augmentation.
- Extending this framework to machine unlearning evaluation for large language models would require resolving the theoretical generalization to non-i.i.d. unlearning scenarios, which represents a potentially important research direction.

## Rating
- Novelty: ⭐⭐⭐⭐ — Evaluating unlearning through a cryptographic game lens is novel, though the concept of advantage is well-established in cryptography.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-dataset, multi-architecture, extensive ablations, and comparisons against multiple baselines; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theorems are clearly stated, proofs are complete, and discussions are thorough; writing is of a high standard.
- Value: ⭐⭐⭐⭐ — Provides a more reliable tool for unlearning evaluation, though utility is bounded by the capability of current SOTA MIAs and the i.i.d. assumption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](simu_selective_influence_machine_unlearning.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](../../ICLR2026/llm_safety/ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[NeurIPS 2025\] PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [\[NeurIPS 2025\] On the Empirical Power of Goodness-of-Fit Tests in Watermark Detection](on_the_empirical_power_of_goodness-of-fit_tests_in_watermark_detection.md)
- [\[NeurIPS 2025\] Music Arena: Live Evaluation for Text-to-Music](music_arena_live_evaluation_for_text-to-music.md)

</div>

<!-- RELATED:END -->
