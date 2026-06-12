---
title: >-
  [Paper Note] The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations
description: >-
  [AAAI 2026][Multimodal VLM][neural network representation comparison] This paper proposes the *Triangle of Similarity* framework, which integrates three complementary perspectives — static representational similarity (CK…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "neural network representation comparison"
  - "CKA"
  - "linear mode connectivity"
  - "network pruning"
  - "model similarity"
date: 2026-05-08
content_hash: 6208ed94d48a6e99
---

# The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations

**Conference**: AAAI 2026
**arXiv**: [2601.17093](https://arxiv.org/abs/2601.17093)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: neural network representation comparison, CKA, linear mode connectivity, network pruning, model similarity

## TL;DR

This paper proposes the *Triangle of Similarity* framework, which integrates three complementary perspectives — static representational similarity (CKA/Procrustes), functional similarity (linear mode connectivity/predictive distribution similarity), and sparsity similarity (pruning robustness) — to comprehensively compare neural networks. Key findings include that architectural family is the primary determinant of representational similarity, and that a model's representational structure is more robust to pruning than its task accuracy.

## Background & Motivation

### State of the Field
As deep learning becomes increasingly prevalent in scientific discovery (medical imaging, particle physics, etc.), a central interpretability question grows in importance: **do different models learn similar underlying concepts?** Answering this question is essential for understanding the nature of learned representations and for guiding model selection and transfer learning.

Existing model comparison methods each focus on different aspects:
- **CKA (Centered Kernel Alignment)**: compares the geometric structure of representation spaces — "what representations look like"
- **Linear Mode Connectivity (LMC)**: probes whether two models reside in the same functional basin of the loss landscape — "whether models are equivalent in weight space"
- **Network Pruning**: tests model robustness through parameter removal — "whether similarity holds under stress"

### Limitations of Prior Work

**Single-perspective limitations**: Each method provides only a limited view. CKA measures geometric alignment but ignores functional equivalence; LMC probes weight space but does not directly reveal representational structure; pruning experiments are typically used for model compression rather than comparative analysis.

**Potential inconsistency across metrics**: Different metrics may yield contradictory conclusions for the same model pair (e.g., high CKA but low Procrustes), and relying on a single metric carries the risk of drawing incorrect conclusions.

**Lack of a systematic integrative framework**: Although each method is individually mature, no unified framework exists to integrate them and provide a comprehensive portrait of model relationships.

### Root Cause
Every existing comparison method has blind spots — **static** methods ignore functional equivalence, **functional** methods ignore representational geometry, and both treat models as static entities while ignoring **dynamic behavior under stress**. How can multiple perspectives be synthesized to yield more reliable conclusions?

### Core Idea
Analogous to triangulation — using three complementary "observation stations" (static, functional, sparsity) to cross-validate model similarity. A particularly novel contribution is the repurposing of network pruning as a "stress-testing tool" rather than a compression tool, enabling observation of how model relationships evolve as parameters are removed.

## Method

### Overall Architecture
For any model pair $(M_A, M_B)$, a three-panel analysis is constructed:
- **Panel 1 (Static Representation View)**: Extract layer-wise activations and compute full similarity matrices using CKA and Procrustes.
- **Panel 2 (Functional View)**: Apply LMC for same-architecture pairs; apply predictive distribution similarity (JSD) for cross-architecture pairs.
- **Panel 3 (Sparsity View)**: Apply progressive pruning and track changes in accuracy and cross-model similarity across sparsity levels.

### Key Designs

1. **Panel 1 — Static Representation View**:

    - **CKA (Centered Kernel Alignment)**: Measures relational geometric similarity; invariant to orthogonal transformations. Standard method from Kornblith et al. 2019.
    - **Procrustes Analysis**: Measures geometric alignment by finding the optimal orthogonal transformation to align two sets of representations. Complementary to CKA — CKA focuses on relational patterns while Procrustes focuses on geometric shape.
    - **Evaluation Data**: 5,000 images from the CIFAR-10 test set (upsampled to model input resolution), serving the dual purpose of testing the robustness of architectural clustering on out-of-distribution data and enabling rapid prototyping.
    - **Design Motivation**: Establishes a baseline similarity map of model layer hierarchies.

2. **Panel 2 — Functional View**:

    - **Same Architecture (LMC)**: Computes accuracy along the linear interpolation path $\theta(\alpha) = (1-\alpha)\theta_A + \alpha\theta_B$. A low-error path implies functionally equivalent solutions.
    - **Cross-Architecture (Predictive Similarity)**: Computes the Jensen-Shannon Divergence (JSD) between softmax prediction distributions of the two models on the test set. JSD ≈ 0 implies similar predictions.
    - **Design Motivation**: Geometric similarity in representation space does not imply functional equivalence — two models may have similar intermediate representations yet produce very different outputs.

3. **Panel 3 — Sparsity View (Most Novel Contribution)**:

    - **Pruning Method**: Global Magnitude Pruning applied progressively at varying sparsity levels $s$.
    - **Two Key Trends Tracked**: (1) Per-model accuracy at different sparsity levels $\text{acc}(M_A^{(s)})$; (2) Cross-model similarity as a function of sparsity $\text{sim}(M_A^{(s)}, M_B^{(s)})$.
    - **Information Revealed**: If models share a robust sparse computational core, similarity should remain stable under pruning; if similarity is merely a byproduct of overparameterization, it should collapse rapidly under pruning.
    - **Design Motivation**: Redefines pruning from a "compression tool" to a "comparative analysis tool" — inferring the nature of shared representations by observing model behavior under stress.

### Model Coverage
- **CNN family**: ResNet18, ResNet50 (including ImageNet/CIFAR-10 trained and randomly initialized variants)
- **ViT family**: ViT-Tiny-Patch16, DeiT-Tiny-Patch16
- **VLM family**: DINOv2-Base, CLIP-ViT-B/32, BLIP-ViT-B/16, LLaVA-1.5-7B (vision tower only)

## Key Experimental Results

### Main Results (Cross-View Statistical Validation)

| Dimension | Metric / Finding | Value |
|-----------|-----------------|-------|
| Static–Sparsity Correlation | Pearson correlation coefficient | r = 0.882 (p < 0.0001) |
| Metric Disagreement Rate | High-divergence pairs (CKA vs. Procrustes) | 4/21 pairs (19.0%) |
| Divergence Threshold | CKA–Procrustes difference > 0.15 | Definition of high divergence |

### Ablation Study (Effect of Pruning on Accuracy vs. Self-Similarity)

| Model | CKA Self-Similarity at 40% Sparsity | Accuracy Drop at 40% Sparsity | Notes |
|-------|-------------------------------------|-------------------------------|-------|
| ResNet18 | > 0.85 | Moderate | Robust representations, mild accuracy drop |
| ResNet50 | > 0.85 | Moderate | CNNs exhibit similar behavior overall |
| DeiT-TP16 | > 0.80 | **Sharp drop** | Accuracy collapses while CKA remains high |
| ViT-TP16 | > 0.70 | **Sharp drop** | Transformers more fragile |
| LMC Barrier (ViT) | > 40% accuracy drop | CKA > 0.7 | Functional equivalence far more fragile than representational structure |

### Key Findings

1. **Architectural family dominates representational similarity**: CKA and Procrustes heatmaps reveal clear block structure — Transformer-family models (ViT, DeiT, DINO, CLIP, BLIP, LLaVA) form highly cohesive clusters, CNNs form a separate but looser cluster, and cross-family similarity is consistently lower than within-family similarity.
2. **Clustering remains robust on OOD data**: Architectural clustering patterns remain stable even when evaluated on CIFAR-10 (out-of-distribution data), indicating this is an intrinsic architectural property rather than a data-driven artifact.
3. **Accuracy is more fragile than representational structure**: At 40% sparsity, CKA self-similarity typically remains above 0.8, while accuracy (especially for ViTs) may drop sharply. This implies that many weights are critical for final accuracy, but the model's core representational structure is more distributed and more robust.
4. **Functional equivalence is extremely fragile**: The LMC barrier begins to rise sharply at sparsity levels far below those at which CKA degrades — functional solutions (specific high-performance weight configurations) are confined to extremely narrow basins, whereas representational structure is a more robust, distributed property.
5. **Static similarity strongly correlates with sparsity robustness**: Pearson r = 0.882 indicates that model pairs with higher initial representational similarity are also more likely to remain similar under pruning pressure, suggesting they may share a deeper computational core.
6. **Single metrics are unreliable**: 19% of model pairs exhibit significant divergence (> 0.15) between CKA and Procrustes, and relying on a single metric may lead to erroneous conclusions.

## Highlights & Insights

1. **Framework thinking over methodological novelty**: Rather than proposing new similarity metrics, this work creatively integrates three existing methods into a complementary analytical framework — the "triangulation" analogy is elegant and effective.
2. **Novel use of pruning as an analytical tool**: Network pruning is repurposed from its traditional role in model compression to serve as a stress-testing instrument for model comparison, representing a genuinely fresh perspective.
3. **Profound insight into functional fragility vs. representational robustness**: The finding that *what a model does* is far more fragile than *what its internal representations look like* offers meaningful insight into the fundamental nature of neural networks.
4. **Practical value of empirical findings**: High static similarity combined with low sparsity robustness may indicate that a model is poorly suited for transfer — a practically actionable guideline for model selection.
5. **Broad model coverage**: Spanning 8 architectures from CNNs to ViTs to VLMs, the study provides a useful initial baseline across diverse model families.

## Limitations & Future Work

1. **High computational cost**: CKA scales as $O(N^2)$ in the number of samples; LMC requires multiple evaluations along high-dimensional paths, making both methods unfriendly to large foundation models.
2. **Exclusive use of global magnitude pruning**: Structured pruning, lottery-ticket rewinding, and learned sparsity patterns may reveal different similarity dynamics.
3. **Limited validation on large-scale models**: Experiments are conducted on relatively small models (the largest being the vision tower of LLaVA-1.5-7B); validation on large-scale LLMs such as GPT or LLaMA is absent.
4. **Limited functional comparison across architectures**: Cross-architecture comparison is restricted to prediction-level JSD, discarding information about weight space.
5. **Correlation does not imply causation**: The observed correlation (r = 0.882) does not necessarily reflect a causal relationship; deeper theoretical analysis is lacking.
6. **Static evaluation data selection**: Evaluating models originally trained on ImageNet using CIFAR-10 data is justified in the paper, but fully in-distribution evaluation might yield different results.

## Related Work & Insights

- **CKA (Kornblith et al. 2019)**: The standard method for representational similarity analysis, invariant to orthogonal transformations and the core tool of Panel 1.
- **Linear Mode Connectivity (Draxler et al. 2018)**: Demonstrates that independently trained networks of the same architecture can typically be connected in weight space via high-accuracy paths.
- **Lottery Ticket Hypothesis (Frankle & Carlin 2019)**: Reveals the existence of trainable sparse subnetworks within dense networks; this paper extends the idea to comparative analysis.
- **Platonic Representation Hypothesis (Huh et al. 2024)**: Proposes that different models may converge to a shared "Platonic" representation; the sparsity view in this paper suggests that pruning may expose such shared computational cores.
- **Insights**: (1) Synthesizing multiple complementary perspectives is more reliable than relying on a single metric — a principle generalizable to other model analysis tasks; (2) Repurposing existing tools across domains (e.g., pruning → analytical instrument) is an effective strategy for generating novel insights; (3) Representational robustness may be a more fundamental model property than accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Relational Visual Similarity](../../CVPR2026/multimodal_vlm/relational_visual_similarity.md)
- [\[CVPR 2026\] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion](../../CVPR2026/multimodal_vlm/videofusion_a_spatio-temporal_collaborative_network_for_multi-modal_video_fusion.md)
- [\[AAAI 2026\] See, Symbolize, Act: Grounding VLMs with Spatial Representations for Better Gameplay](see_symbolize_act_grounding_vlms_with_spatial_representations_for_better_gamepla.md)
- [\[NeurIPS 2025\] Hierarchical Self-Attention: Generalizing Neural Attention Mechanics to Multi-Scale Problems](../../NeurIPS2025/multimodal_vlm/hierarchical_self-attention_generalizing_neural_attention_mechanics_to_multi-sca.md)
- [\[AAAI 2026\] Exploring LLMs for Scientific Information Extraction using the SciEx Framework](exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)

</div>

<!-- RELATED:END -->
