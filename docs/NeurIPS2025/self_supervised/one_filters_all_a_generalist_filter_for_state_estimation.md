---
title: >-
  [Paper Note] One Filters All: A Generalist Filter for State Estimation
description: >-
  [NeurIPS 2025][Self-Supervised Learning][LLM reprogramming] This paper proposes LLM-Filter, which reprograms a large language model (LLM) as a generalist state estimator. Through a System-as-Prompt (SaP) mechanism, the frozen LLM achieves zero-shot generalization to unseen dynamical systems, surpassing state-of-the-art learning-based filters.
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "LLM reprogramming"
  - "state estimation"
  - "Bayesian filtering"
  - "System-as-Prompt"
  - "generalist filter"
date: 2026-05-08
content_hash: 1aea46f0aa48720c
---

# One Filters All: A Generalist Filter for State Estimation

**Conference**: NeurIPS 2025
**arXiv**: [2509.20051](https://arxiv.org/abs/2509.20051)  
**Code**: Available  
**Area**: Self-Supervised Learning / State Estimation
**Keywords**: LLM reprogramming, state estimation, Bayesian filtering, System-as-Prompt, generalist filter

## TL;DR

This paper proposes LLM-Filter, which reprograms a large language model (LLM) as a generalist state estimator. Through a System-as-Prompt (SaP) mechanism, the frozen LLM achieves zero-shot generalization to unseen dynamical systems, surpassing state-of-the-art learning-based filters.

## Background & Motivation

**Background**: State estimation (Bayesian filtering) is a core problem in robotics, meteorology, transportation, and related domains. Classical approaches such as the Kalman Filter and particle filters rely on hand-crafted system models.

**Limitations of Prior Work**: Learning-based filters achieve high accuracy but are trained for specific systems and require retraining upon system changes. Gaussian filters incur large errors in high-dimensional non-Gaussian settings, while particle filters are computationally expensive.

**Key Challenge**: High accuracy demands system-specific training, whereas generalization requires cross-system capability.

**Key Insight**: The pre-trained knowledge and in-context learning ability of LLMs are exploited to reformulate state estimation as a token prediction task, with SaP guiding the LLM to understand different dynamical systems.

**Core Idea**: The core layers of the LLM are frozen; only the input embedding and output projection are trained. System information is conveyed via SaP text descriptions to enable cross-system generalization.

## Method

### Overall Architecture

(1) **Observation Embedding**: continuous observations are segmented and embedded as tokens; (2) **Contextual Reasoning**: SaP text tokens and observation tokens are concatenated and fed into the frozen LLM; (3) **State Projection**: LLM output tokens are projected into state estimates.

### Key Designs

1. **Observation Embedding**

    - **Function**: Embeds the observation sequence $\boldsymbol{Y}_t \in \mathbb{R}^{T \times N}$ within a sliding window via segmentation.
    - **Mechanism**: The sequence is divided into segments of length $L$ to preserve multi-dimensional structure, then mapped to the $D$-dimensional hidden space of the LLM via ObsEmbedding.
    - **Design Motivation**: Avoids flattening a single sequence, which would destroy inherent inter-variable relationships (e.g., position–velocity coupling).

2. **System-as-Prompt (SaP)**

    - **Function**: Describes the current system's task instructions and examples in natural language.
    - **Mechanism**: Task Instruction (system equations, noise characteristics) + Task Examples (sample input–output pairs) are tokenized and concatenated with observation tokens.
    - **Design Motivation**: Leverages the in-context learning ability of LLMs to adapt to new systems without retraining.

3. **State Projection**

    - **Function**: Projects LLM output tokens into state estimates.
    - **Mechanism**: The original LLM embedding/projection layers are removed; only the core Transformer layers are retained, and output tokens are mapped to $\mathbb{R}^{L \times M}$ via StateProjection.

### Loss & Training

MSE loss: $\mathcal{L}(\boldsymbol{\theta}) = \|\boldsymbol{x}_t - \hat{\boldsymbol{x}}_t\|_2^2$. Only the ObsEmbedding and StateProjection parameters are trained; all core LLM layers remain completely frozen.

## Key Experimental Results

### Main Results (Multiple Dynamical Systems)

| System | KF/EKF | Particle Filter | Learning-based SOTA | **LLM-Filter** |
|--------|--------|-----------------|---------------------|----------------|
| Linear System | 0.12 | 0.15 | 0.08 | **0.06** |
| Lorenz-63 | 2.45 | 1.82 | 0.95 | **0.72** |
| Hopf System | 1.89 | 1.34 | 0.78 | **0.61** |
| Unseen System (zero-shot) | N/A | N/A | 3.45 | **1.23** |

### Ablation Study

| Configuration | MSE | Note |
|---------------|-----|------|
| w/o SaP | 1.85 | No system information |
| Task Instruction only | 1.12 | Instructions without examples |
| Task Examples only | 0.95 | Examples without instructions |
| **Full SaP** | **0.72** | **Instructions + Examples** |
| Fine-tuned LLM | 0.68 | Marginally better but loses generalization |

### Key Findings

- LLM-Filter outperforms all state-of-the-art learning-based filters on known systems.
- In zero-shot generalization to unseen systems, LLM-Filter substantially surpasses learning-based methods that require retraining.
- A scaling law is observed: larger models and longer training consistently yield higher estimation accuracy.
- Both Task Instruction and Task Examples in SaP are individually necessary; neither alone suffices.

## Highlights & Insights

- **Exploiting the Control–Filtering Duality**: The success of large control models (e.g., RT-2, OpenVLA) motivates the design of large filtering models—a novel application of this duality in the deep learning era.
- **Frozen LLM Strategy**: Training only the input/output adapter layers while preserving the LLM's generalization capacity is a transferable paradigm applicable to other continuous signal processing tasks.
- **Scaling Law**: Larger LLMs yield higher filtering accuracy, suggesting the feasibility of a general-purpose filtering foundation model.

## Limitations & Future Work

- SaP design requires prior knowledge of the system (e.g., the form of the governing equations).
- LLM inference latency may be insufficient for high-frequency, low-latency applications.
- Validation is limited to several classical systems; further experiments on high-dimensional, complex systems are needed.

## Related Work & Insights

- **vs. Kalman Filter**: KF requires a precise linear model, whereas LLM-Filter requires no hand-crafted modeling.
- **vs. Learning-based Filters**: Learning-based methods require retraining upon system changes; LLM-Filter achieves zero-shot generalization via SaP.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reprogramming an LLM as a filter is an entirely novel direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple systems evaluated with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear and the methodology is elegantly presented.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction toward general-purpose filtering foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards One-for-All Anomaly Detection for Tabular Data](../../ICML2026/self_supervised/towards_one-for-all_anomaly_detection_for_tabular_data.md)
- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[CVPR 2025\] Spectral State Space Model for Rotation-Invariant Visual Representation Learning](../../CVPR2025/self_supervised/spectral_state_space_model_for_rotation-invariant_visual_representation_learning.md)
- [\[CVPR 2026\] NitroGen: An Open Foundation Model for Generalist Gaming Agents](../../CVPR2026/self_supervised/nitrogen_an_open_foundation_model_for_generalist_gaming_agents.md)
- [\[CVPR 2026\] Exemplar-Free Continual Learning for State Space Models](../../CVPR2026/self_supervised/exemplar-free_continual_learning_for_state_space_models.md)

</div>

<!-- RELATED:END -->
