---
title: >-
  [Paper Note] MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction
description: >-
  [ICLR 2026][LLM Pretraining][material property prediction] MoMa is a modular material property prediction framework that trains task-specific modules across multiple tasks and stores them centrally in a MoMa Hub, then applies a training-free Adaptive Module Composition (AMC) algorithm driven by representation similarity to assemble customized models for downstream tasks, achieving an average improvement of 14% over the strongest baseline across 17 datasets.
tags:
  - ICLR 2026
  - LLM Pretraining
  - material property prediction
  - modular learning
  - adaptive combination
  - transfer learning
  - graph neural networks
date: 2026-05-08
content_hash: 73c370979d9c5ec4
---

# MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction

**Conference**: ICLR 2026
**arXiv**: [2502.15483](https://arxiv.org/abs/2502.15483)
**Code**: [https://github.com/GenSI-THUAIR/MoMa](https://github.com/GenSI-THUAIR/MoMa)
**Area**: Materials Science / Modular Deep Learning
**Keywords**: material property prediction, modular learning, adaptive combination, transfer learning, graph neural networks

## TL;DR
MoMa is a modular material property prediction framework that trains task-specific modules across multiple tasks and stores them centrally in a MoMa Hub, then applies a training-free Adaptive Module Composition (AMC) algorithm driven by representation similarity to assemble customized models for downstream tasks, achieving an average improvement of 14% over the strongest baseline across 17 datasets.

## Background & Motivation

**State of the Field**: Deep learning for material property prediction follows two main paradigms: (1) task-specific models trained from scratch (e.g., CGCNN); (2) pre-train-then-fine-tune, particularly large-scale force-field models (e.g., JMP) pre-trained on potential energy surface data and fine-tuned to downstream tasks. The latter has achieved notable success across multiple tasks.

**Limitations of Prior Work**: (1) **Diversity**: Material tasks span diverse systems (crystals, organic molecules) and properties (thermal stability, electronic behavior, mechanical properties); force-field models are pre-trained only on potential-energy-surface-related properties, limiting generalization. (2) **Heterogeneity**: Different material properties are governed by different physical laws, so joint multi-task training introduces knowledge conflicts that a single model struggles to reconcile.

**Root Cause**: The pre-training paradigm pursues a "one model for all problems" objective, yet the intrinsic heterogeneity of material tasks means that joint training introduces negative transfer.

**Paper Goals**: (1) How can knowledge from diverse material data be leveraged while avoiding inter-task conflicts? (2) How can downstream adaptation be performed efficiently under data scarcity?

**Starting Point**: Modular learning — encapsulating each task as an independently trained module to prevent interference, then adaptively selecting and composing the most synergistic modules for downstream tasks.

**Core Idea**: Train diverse task-specific modules and store them in MoMa Hub; apply training-free adaptive module composition via kNN representation-space estimation combined with convex-optimization weight solving, followed by fine-tuning for downstream adaptation.

## Method

### Overall Architecture
**Stage 1 (Module Training & Centralization)**: Independently train modules (Full or Adapter) on 18 high-resource material property prediction datasets and store them in MoMa Hub. **Stage 2 (AMC & Fine-tuning)**: For a new task, the AMC algorithm proceeds in three steps — (1) kNN-based prediction estimates the affinity of each module → (2) convex optimization solves for optimal composition weights → (3) weight-space module merging → fine-tuning.

### Key Designs

1. **MoMa Hub Module Training & Storage**:

    - **Function**: Train an independent module for each high-resource material task and manage them centrally.
    - **Mechanism**: Using pre-trained JMP as the backbone, two module forms are provided — Full Module (full model fine-tuned as a module) and Adapter Module (backbone frozen, only inserted adapter layers trained, parameter-efficient). The module set is $\mathcal{H} = \{g_1, g_2, \ldots, g_N\}$, currently containing 18 task modules sourced from Matminer.
    - **Design Motivation**: Independent training avoids inter-task knowledge conflicts; the Hub design protects data privacy (only model parameters are shared, not data) and supports community contributions.

2. **Adaptive Module Composition (AMC)**:

    - **Function**: Adaptively select and weight-combine optimal modules for a downstream task.
    - **Mechanism**: Three-step pipeline — ① For each module $g_j$, encode downstream data to obtain representations $\mathcal{X}^j$ and estimate predictions via leave-one-out kNN label propagation: $\hat{y}_i^j = \sum_{k \in \mathcal{N}_i} \frac{f_d(\mathbf{x}_i^j, \mathbf{x}_k^j)}{Z_i^j} y_k$; ② minimize the ensemble surrogate error $E_\mathcal{D}(\mathbf{w}) = \frac{1}{M}\|\sum_j w_j \hat{\mathbf{y}}^j - \mathbf{y}\|_2^2$ subject to $\sum w_j = 1, w_j \geq 0$ and solve via convex optimization; ③ weight-space merging $g_\mathcal{D} = \sum_j w_j^* g_j$.
    - **Design Motivation**: Search-based methods yield noisy error signals when modules are highly heterogeneous; routing networks overfit under data scarcity. AMC uses representation quality rather than prediction error as the supervision signal, requires no additional trainable parameters, and converges within 30 seconds.

3. **Weight-Space Module Merging & Fine-tuning**:

    - **Function**: Merge weighted modules into a single initialization model, then fine-tune on downstream data.
    - **Mechanism**: Leverages linear mode connectivity: since all modules share the same pre-trained initialization, their parameter spaces are compatible. After merging, a task-specific head is appended and the model is fine-tuned to convergence on $\mathcal{D}$.
    - **Design Motivation**: Direct weighted averaging is simple and efficient, avoiding the computational overhead of loading multiple modules at inference time.

### Loss & Training
Module training uses the standard MAE loss. The AMC stage is training-free (weights are solved via convex optimization); the Pearson correlation between the surrogate error and final fine-tuning MAE exceeds 0.6 (empirically validated). Fine-tuning follows the standard setup for each downstream task. Experiments are conducted on 17 low-data material property prediction tasks, with 5 data splits × 5 random seeds per task.

## Key Experimental Results

### Main Results (17 Material Property Prediction Tasks)

| Method | Avg. Rank | Best/17 | Representative Task (3D Band Gap MAE) |
|--------|-----------|---------|---------------------------------------|
| MoMa (Full) | 1.35 | 14 | 0.200 |
| MoMa (Adapter) | 2.59 | 2 | 0.245 |
| JMP-FT | 3.12 | 1 | 0.249 |
| JMP-MT | 4.53 | 0 | 0.423 |
| UMA | 4.53 | 0 | 0.268 |
| MoE-(18) | 4.71 | 0 | 0.361 |
| CGCNN | 6.88 | 0 | 0.492 |

### Ablation Study (AMC Composition Strategy Comparison)

| Configuration | vs. AMC (MAE Increase) | Notes |
|---------------|------------------------|-------|
| AMC (Full) | — | Best |
| Select Average | +11.0% | Retains AMC-selected modules but averages uniformly |
| All Average (Model Soup) | +18.0% | Averages all modules |
| Random Selection | +20.2% | Randomly selects an equal number of modules |

### Key Findings
- MoMa (Full) ranks in the top two on 16/17 tasks, with an average improvement of 14% over JMP-FT and 24.8% over JMP-MT.
- Advantages are more pronounced in few-shot settings: at 10-shot, MoMa achieves a normalized MAE of 0.550 vs. 0.700 for JMP-FT (−21%).
- Hub scalability: normalized MAE decreases monotonically from 0.204 (5 modules) to 0.176 (30 modules).
- AMC weights provide insights into material property relationships: e.g., dielectric constant prediction assigns high weight to the band gap module, consistent with physical intuition.
- Effectiveness is also demonstrated on the Orb-v2 backbone (improvement on 13/17 tasks, average +6.1%), confirming backbone-agnostic applicability.

## Highlights & Insights
- The modular paradigm elegantly resolves the "diversity vs. heterogeneity" tension in materials science: independent training avoids conflicts while adaptive composition exploits synergies, achieving both objectives simultaneously.
- The representation-driven, training-free design of AMC is particularly well-suited to data-scarce materials science scenarios; both the theoretical justification and empirical validation of the surrogate error are compelling.

## Limitations & Future Work
- MoMa Hub currently covers only 18 crystal tasks and 12 QM9 molecular tasks, leaving substantial coverage gaps.
- Weight-space merging relies on the linear mode connectivity assumption, which may break down when modules are highly divergent.
- The kNN estimation in AMC may become unstable under extremely low-data regimes (fewer than 10 samples).

## Related Work & Insights
- **vs. JMP**: JMP employs joint multi-task pre-training (MT); MoMa demonstrates that independent training combined with adaptive composition outperforms MT on 16/17 tasks, suggesting that multi-task loss introduces severe negative transfer across heterogeneous material tasks.
- **vs. Model Soup**: Uniformly averaging all modules performs poorly (+18% MAE), underscoring the importance of AMC's adaptive weight optimization.
- **vs. MoE-(18)**: The routing-network-based approach ranks 4.71 across 17 tasks; AMC's representation-driven strategy proves more stable.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing modular learning to materials science represents a new paradigm; the AMC algorithm is well-motivated and soundly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 17 tasks × 5 splits × 5 seeds, with full coverage of ablation, few-shot, scalability, and multi-backbone analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Framework diagrams are clear, experimental descriptions are thorough, and motivation is well-established.
- **Value**: ⭐⭐⭐⭐ The open-source platform design has the potential to advance modular knowledge sharing in the materials science community.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_pretraining/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[AAAI 2026\] Learning Time in Static Classifiers](../../AAAI2026/llm_pretraining/learning_time_in_static_classifiers.md)
- [\[CVPR 2026\] Watch and Learn: Learning to Use Computers from Online Videos](../../CVPR2026/llm_pretraining/watch_and_learn_computer_use_from_videos.md)

<!-- RELATED:END -->
