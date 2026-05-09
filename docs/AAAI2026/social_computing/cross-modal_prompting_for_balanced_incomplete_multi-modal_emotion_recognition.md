---
title: >-
  [Paper Note] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition
description: >-
  [AAAI 2026][Social Computing][Incomplete multi-modal] This paper proposes Cross-modal Prompting (ComP), which addresses the modality imbalance problem in incomplete multi-modal emotion recognition (IMER) via progressive prompt generation, cross-modal knowledge propagation, and a dynamic scheduler, achieving state-of-the-art performance across 4 datasets and 7 missing rates.
tags:
  - AAAI 2026
  - Social Computing
  - Incomplete multi-modal
  - emotion recognition
  - prompt learning
  - modality balance
  - knowledge propagation
date: 2026-05-08
content_hash: 98b77aa18fd25d3f
---

# Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition

**Conference**: AAAI 2026
**arXiv**: [2512.11239](https://arxiv.org/abs/2512.11239)
**Code**: [GitHub](https://github.com/WenjueHE/2026-AAAI-ComP)
**Area**: Social Computing
**Keywords**: Incomplete multi-modal, emotion recognition, prompt learning, modality balance, knowledge propagation

## TL;DR
This paper proposes Cross-modal Prompting (ComP), which addresses the modality imbalance problem in incomplete multi-modal emotion recognition (IMER) via progressive prompt generation, cross-modal knowledge propagation, and a dynamic scheduler, achieving state-of-the-art performance across 4 datasets and 7 missing rates.

## Background & Motivation

**State of the Field**: Multi-modal Emotion Recognition (MER) leverages multi-source information such as audio, text, and video, but modality missing is common in real-world scenarios (e.g., audio unavailability due to background noise, or speech recognition failure).

**Limitations of Prior Work**:
   - **Modality performance gap**: Recognition capabilities vary substantially across modalities.
   - **Modality under-optimization**: Some modalities perform worse after joint multi-modal training than when trained independently.
   - Existing IMER methods either reconstruct missing data (costly and not necessarily helpful) or learn unified representations while ignoring modality imbalance.

**Root Cause**: How to simultaneously handle modality incompleteness and modality imbalance under missing-modality conditions?

**Paper Goals**: Jointly address missing-modality handling and modality balance through cross-modal prompt learning.

**Starting Point**: Rather than recovering missing data, each modality's information is compressed into prompts that are propagated to other modalities to enhance task-relevant consistent information, while dynamic scheduling achieves balance.

**Core Idea**: Progressive prompt generation compresses cross-modal consistent information; a knowledge propagation module enhances task-relevant features of each modality — augmenting rather than reconstructing.

## Method

### Overall Architecture
Two-stage training:
1. **Stage 1**: Each modality independently trains its encoder and classifier.
2. **Stage 2**: Prompt generation → knowledge propagation → multi-modal collaborative fusion.

### Key Designs

1. **Progressive Prompt Generation (PG)**:

    - **Function**: Compresses each modality's features into representative and consistent prompts.
    - **Mechanism**: Global inputs are compressed into a small number of prototypes; a dynamic gradient modulator prevents prototypes from being dominated by easy samples; context features are then fused to produce low-dimensional prompts.
    - **Design Motivation**: Prompts should capture cross-modally consistent emotional information rather than modality-specific noise.

2. **Cross-modal Knowledge Propagation (KP)**:

    - **Function**: Injects prompts from other modalities into the current modality to enhance task-relevant features.
    - **Mechanism**: Modality features $\mathbf{Z}_l^u$ are concatenated with two cross-modal prompts $\mathbf{P}_l^{vu}, \mathbf{P}_l^{wu}$, then linearly projected for compression, enhanced via multi-head self-attention, and projected back to the original space.
    - **Novelty**: Missing instances are masked in MSA, yet their information is naturally "reconstructed" through the propagation of cross-modal prompts, requiring no dedicated missing-data recovery module.

3. **Multi-modal Coordinator**:

    - Dynamically re-weights the outputs of each modality as a complementary balancing strategy.

### Loss & Training
$$\mathcal{L} = \sum_{u} \mathcal{L}_{enc}(\mathbf{Z}^u, \hat{\mathbf{X}}^u) + \sum_{u} \mathcal{L}_{task}(\mathbf{Y}^u, \mathbf{Y})$$

## Key Experimental Results

### Main Results (IEMOCAP 4-class, varying missing rates)

| Method | 0.1 ACC | 0.3 ACC | 0.5 ACC | 0.7 ACC |
|--------|---------|---------|---------|---------|
| GCNet | 74.82 | 74.49 | 72.67 | 71.00 |
| MoMKE | 76.70 | 73.47 | 69.73 | 66.52 |
| SDR-GNN | 78.48 | 78.22 | 75.47 | 70.52 |
| **ComP** | **80.66** | **78.37** | **75.62** | **73.41** |

ComP achieves the best performance at all missing rates, with more pronounced advantages at high missing rates (0.7, +2.89%).

### Ablation Study / Key Findings
- In baseline methods, Video and Text modalities suffer performance degradation after joint multi-modal training (modality under-optimization); ComP enables all modalities to benefit from multi-modal learning.
- ComP consistently outperforms 7 SOTA methods across 4 datasets and 7 missing rates.
- The gradient modulator is critical for prompt quality — it prevents easy samples from dominating prototype learning.

## Highlights & Insights
- The **"augment rather than recover"** paradigm is elegant: instead of reconstructing missing data, cross-modal prompts enhance existing modalities — more efficient and a natural solution to the missing-modality problem.
- **Visualization of the modality imbalance problem** (Fig. 1) intuitively demonstrates both the problem and the effectiveness of the proposed solution.
- **Missing data information is naturally reconstructed during knowledge propagation** — no additional missing-data handling module is required.

## Limitations & Future Work
- Prompt generation and knowledge propagation increase model complexity.
- Validation is limited to emotion recognition; applicability to other incomplete multi-modal tasks remains unexplored.
- The design motivation and theoretical analysis of the dynamic gradient modulator could be further elaborated.

## Related Work & Insights
- **vs. MMIN**: MMIN uses cascaded autoencoders to recover missing data, which is costly and not always informative; ComP achieves more efficient enhancement via prompts.
- **vs. MoMKE**: MoMKE adopts a Mixture-of-Experts strategy without balancing individual modalities, leading to significant performance degradation at high missing rates; ComP demonstrates superior stability.
- **vs. MMPareto**: MMPareto balances gradients via Pareto efficiency but does not account for missing-modality scenarios; ComP unifies modality balancing and missing-modality handling within a single framework.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified framework integrating prompt learning, modality balance, and missing-modality handling is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 datasets × 7 missing rates × 7 baselines — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear and method descriptions are detailed.
- Value: ⭐⭐⭐⭐ Provides a practical solution for incomplete multi-modal scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)
- [\[AAAI 2026\] From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks](from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)
- [\[AAAI 2026\] SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](../../ICLR2026/social_computing/sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)

<!-- RELATED:END -->
