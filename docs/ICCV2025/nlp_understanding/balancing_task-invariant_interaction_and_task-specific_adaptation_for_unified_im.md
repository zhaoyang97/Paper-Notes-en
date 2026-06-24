---
title: >-
  [Paper Note] Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion
description: >-
  [ICCV 2025][NLP Understanding][Unified Image Fusion] TITA proposes a unified image fusion framework that requires no task identifier at inference. It employs an Interaction-enhanced Pixel Attention (IPA) module to explore task-invariant complementary information extraction, an Operation-based Adaptive Fusion (OAF) module to dynamically adapt to task-specific requirements, and the FAMO strategy to mitigate multi-task gradient conflicts.
tags:
  - "ICCV 2025"
  - "NLP Understanding"
  - "Unified Image Fusion"
  - "Multi-task Learning"
  - "Pixel Attention"
  - "Adaptive Fusion"
  - "Gradient Conflict"
date: 2026-05-08
content_hash: c8cb96047b82b069
---

# Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion

**Conference**: ICCV 2025
**arXiv**: [2504.05164](https://arxiv.org/abs/2504.05164)  
**Code**: [github.com/huxingyuabc/TITA](https://github.com/huxingyuabc/TITA)  
**Area**: Image Fusion
**Keywords**: Unified Image Fusion, Multi-task Learning, Pixel Attention, Adaptive Fusion, Gradient Conflict

## TL;DR

TITA proposes a unified image fusion framework that requires no task identifier at inference. It employs an Interaction-enhanced Pixel Attention (IPA) module to explore task-invariant complementary information extraction, an Operation-based Adaptive Fusion (OAF) module to dynamically adapt to task-specific requirements, and the FAMO strategy to mitigate multi-task gradient conflicts.

## Background & Motivation

Image fusion aims to integrate complementary information from multi-source images to enhance image quality, covering diverse scenarios such as infrared-visible fusion (IVF), multi-exposure fusion (MEF), and multi-focus fusion (MFF). Existing methods face three core challenges:

**Unified methods neglect task specificity**: Existing unified fusion methods (e.g., U2Fusion, PMGI) treat different fusion tasks as a single problem using shared architectures and unified objective functions. Although they achieve task-invariant knowledge sharing, they ignore the distinct physical characteristics of each task (e.g., IVF emphasizes thermal infrared saliency, MEF balances luminance, MFF extracts in-focus regions), thereby limiting overall performance.

**General methods rely on task identifiers**: General fusion methods (e.g., SwinFusion, TC-MoA) introduce task-specific adaptation but require explicit task identifiers at inference to select corresponding model branches or loss functions, which limits generalization to unseen tasks.

**Multi-task gradient conflicts**: The optimization directions of different fusion tasks may be mutually contradictory, and naively averaging gradients can degrade performance on certain tasks.

This paper seeks to simultaneously address all three challenges: **how to leverage the commonality across fusion tasks (complementary information extraction) while adapting to task-specific characteristics, without relying on task identifiers?**

## Method

### Overall Architecture

TITA is built upon the SwinFusion architecture and consists of two stages:
1. **Task-invariant interaction stage**: $L$ stacked Interaction-enhanced SwinFusion (ISF) modules, each containing one IPA module.
2. **Task-specific adaptation stage**: The Operation-based Adaptive Fusion (OAF) module.

### Key Designs

1. **Interaction-enhanced Pixel Attention (IPA) Module**:

    - An improvement over the Pixel Attention (PA) mechanism, where PA dynamically modulates the weights of self-attention and cross-attention via a relation discriminator $\phi_{\theta_s}(\cdot)$.
    - Two key modifications in IPA:
        - **Removal of direct noise injection on V**: The noise injection on Value in PA may cause irreversible information loss.
        - **Reinforced cross-attention preference**: When the relation discriminator score is high (i.e., the two sources are more uncorrelated), the cross-attention weight is explicitly increased. The key construction becomes: $K_1 = [(N_1 + X_{i,1} - X_{i,1}\phi_{\theta_s})W_K, (N_2 + X_{i,1} + X_{i,1}\phi_{\theta_s})W_K]$
    - Design motivation: A higher uncorrelatedness score implies more complementary information to be extracted via cross-attention.
    - All intra-domain fusion blocks in SwinFusion are replaced with inter-domain fusion (ISF), further increasing the number of cross-attention operations.

2. **Operation-based Adaptive Fusion (OAF) Module**:

    - Three parallel operation branches:
        - **HPF branch**: Spatially varying high-pass filtering to capture high-frequency textures and edges.
        - **ADD branch**: Residual addition for overall information enhancement.
        - **MUL branch**: Element-wise multiplication to facilitate nonlinear feature interaction.
    - The operands (e.g., convolution kernels) of each branch are predicted from input features by a hypernetwork (2-layer MLP).
    - The dynamic weights $W$ for the three branches are predicted by a separate weight prediction network from dual-source features $(X_1, X_2)$.
    - Output: $X_f = \sum_{o \in \{h,a,m\}} W_1 \cdot \hat{X}_{o,1} + W_2 \cdot \hat{X}_{o,2}$
    - Design motivation: Different fusion tasks have different demands for high-frequency preservation, overall enhancement, and nonlinear combination. Dynamic weights enable automatic adaptation to task characteristics without explicit task identifiers.

3. **FAMO Multi-objective Optimization Strategy**:

    - Experiments reveal severe gradient conflicts (large angular and magnitude discrepancies) when directly averaging multi-task gradients.
    - FAMO is adopted: learnable logits $\xi_t$ generate softmax weights to dynamically adjust the loss weight of each task.
    - FAMO equalizes the loss reduction rates across tasks to achieve fair optimization.
    - Design motivation: The introduction of the TA module exacerbates gradient conflicts; FAMO effectively alleviates this issue (ablation studies confirm that TA benefits most from MO).

### Loss & Training

Task-specific objectives are used (task identifiers are required during training but not at inference):

$$\ell = \lambda_1 \ell_{ssim} + \lambda_2 \ell_{text} + \lambda_3 \ell_{int}$$

where $\ell_{text}$ (texture loss) applies maximum gradient constraints, and $\ell_{int}$ (intensity loss) uses task-specific aggregation (max for IVF/MFF, mean for MEF). Training configuration: Adam optimizer (lr=2e-5), batch size 8, 20,000 iterations, with uniform sampling across tasks.

## Key Experimental Results

### Main Results — Quantitative Comparison on Three Fusion Tasks

**IVF Task (LLVIP dataset)**:

| Method | Type | MI ↑ | FMI ↑ | Qabf ↑ | VIF ↑ |
|--------|------|------|-------|--------|-------|
| SwinFusion | General | 3.873 | 0.889 | 0.650 | 0.907 |
| TC-MoA | General | 3.606 | 0.886 | 0.600 | 0.925 |
| Text-IF | Dedicated | 3.322 | 0.892 | **0.684** | **0.932** |
| CCF | Unified | 2.789 | 0.881 | 0.499 | 0.719 |
| **TITA** | **Unified** | **4.176** | **0.896** | 0.679 | 0.926 |

**MFF Task (Lytro+MFFW+MFI-WHU)**:

| Method | Type | MI ↑ | FMI ↑ | Qabf ↑ | SSIM ↑ |
|--------|------|------|-------|--------|--------|
| IFCNN | Dedicated (MFF) | 6.495 | 0.882 | 0.658 | 0.991 |
| SwinFusion | General | 6.261 | 0.881 | 0.687 | 0.991 |
| **TITA** | **Unified** | 6.546 | **0.885** | **0.697** | **0.993** |

Without using task identifiers, TITA surpasses dedicated and general methods on multiple metrics.

### Ablation Study — Contribution of Three Components (IVF)

| TI | TA | MO | MI ↑ | FMI ↑ | Qabf ↑ | VIF ↑ |
|----|----|----|------|-------|--------|-------|
| ✗ | ✗ | ✗ | 3.612 | 0.889 | 0.646 | 0.845 |
| ✓ | ✗ | ✗ | 3.882 | 0.892 | 0.664 | 0.904 |
| ✗ | ✓ | ✗ | 3.685 | 0.891 | 0.662 | 0.853 |
| ✗ | ✗ | ✓ | 3.680 | 0.891 | 0.662 | 0.853 |
| ✓ | ✓ | ✗ | 3.883 | 0.893 | 0.666 | 0.906 |
| ✓ | ✗ | ✓ | 4.122 | 0.895 | 0.676 | 0.919 |
| ✓ | ✓ | ✓ | **4.176** | **0.896** | **0.680** | **0.926** |

All three components are mutually reinforcing; MO yields the largest gain for TA, confirming that TA introduces gradient conflicts that must be mitigated by MO.

### Key Findings

1. **Generalization to unseen tasks**: TITA performs well on unseen tasks such as medical image fusion (MIF) and pansharpening (PAN), while FusionDN and CCF completely collapse on unseen tasks.
2. **More cross-attention is better**: IeSF (all inter-domain) > SF (original) > IrSF (all intra-domain), validating the importance of multi-source interaction.
3. **The MUL branch is the most critical in OAF**: Removing the MUL branch causes the largest performance drop, as image fusion inherently involves extensive nonlinear operations.
4. **Dynamic weight visualization**: OAF automatically assigns different operation weight distributions to different fusion tasks (e.g., HPF weights are smaller but indispensable in the MFF task).

## Highlights & Insights

- The paper accurately identifies the three-fold challenge of unified fusion frameworks (task invariance, task specificity, gradient conflict) and provides a systematic solution.
- The causal design in IPA — "higher uncorrelatedness → larger cross-attention weight" — is intuitively sound: regions with stronger complementarity require more cross-modal interaction.
- Only 1.39M parameters, making the framework lightweight and efficient.
- The design requiring no task identifier at inference allows the framework to be directly applied to any new fusion task.

## Limitations & Future Work

- Only three operation branches (HPF, ADD, MUL) are included in OAF, which may be insufficient to cover all fusion task requirements.
- Training data is imbalanced across tasks (IVF: 12,025 pairs vs. MFF: 800 pairs); although uniform sampling is applied, its effectiveness remains limited.
- Compared to recent methods leveraging diffusion models or large language models (e.g., DDFM, Text-IF), there remains a gap in perceptual quality.

## Related Work & Insights

- **Relation to SwinFusion**: TITA inherits its architecture but extends the general method into a unified framework that requires no task identifier.
- **Relation to TC-MoA**: TC-MoA uses a mixture of experts to adapt to different tasks but requires task identifiers; TITA's OAF implicitly achieves similar functionality through dynamic weights.
- FAMO, as a multi-task optimization strategy, is generalizable and can be extended to other multi-task vision systems.

## Rating

- Novelty: ⭐⭐⭐⭐ (Systematic solution to three-fold challenges; well-motivated component designs)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three tasks + two unseen tasks + detailed ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; motivations are thoroughly articulated)
- Value: ⭐⭐⭐⭐ (Advances the state of unified image fusion; convincing generalization validation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] QQSUM: A Novel Task and Model of Quantitative Query-Focused Summarization for Review-based Product Question Answering](../../ACL2025/nlp_understanding/qqsum_a_novel_task_and_model_of_quantitative_query-focused_summarization_for_rev.md)
- [\[ACL 2026\] MetFuse: Figurative Fusion between Metonymy and Metaphor](../../ACL2026/nlp_understanding/metfuse_figurative_fusion_between_metonymy_and_metaphor.md)
- [\[ACL 2025\] CaLMQA: Exploring Culturally Specific Long-Form Question Answering across 23 Languages](../../ACL2025/nlp_understanding/calmqa_cultural_multilingual_qa.md)
- [\[ACL 2026\] Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation](../../ACL2026/nlp_understanding/knowledge-driven_augmentation_and_retrieval_for_integrative_temporal_adaptation.md)
- [\[ACL 2025\] End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems](../../ACL2025/nlp_understanding/end-to-end_dialog_neural_coreference_resolution_balancing_efficiency_and_accurac.md)

</div>

<!-- RELATED:END -->
