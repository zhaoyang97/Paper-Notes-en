---
title: >-
  [Paper Note] Self-supervised Multiplex Consensus Mamba for General Image Fusion
description: >-
  [AAAI 2026][Medical Imaging][General Image Fusion] This paper proposes the SMC-Mamba framework, which achieves general image fusion across infrared-visible, medical, multi-focus…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "General Image Fusion"
  - "Mamba"
  - "Mixture of Experts"
  - "Contrastive Learning"
  - "High-Frequency Preservation"
date: 2026-05-08
content_hash: 6a9280a7fe378de1
---

# Self-supervised Multiplex Consensus Mamba for General Image Fusion

**Conference**: AAAI 2026
**arXiv**: [2512.20921](https://arxiv.org/abs/2512.20921)
**Code**: N/A
**Area**: Medical Imaging / Image Fusion
**Keywords**: General Image Fusion, Mamba, Mixture of Experts, Contrastive Learning, High-Frequency Preservation

## TL;DR

This paper proposes the SMC-Mamba framework, which achieves general image fusion across infrared-visible, medical, multi-focus, and multi-exposure tasks through **Modality-Agnostic Feature Enhancement (MAFE)**, **Multiplex Consensus Cross-modal Mamba (MCCM)**, and **Bi-level Self-supervised Contrastive Learning loss (BSCL)**, comprehensively surpassing state-of-the-art methods.

## Background & Motivation

1. **Background**: Image fusion integrates complementary information from different modalities to generate high-quality fused images, enhancing downstream tasks such as object detection and semantic segmentation. The primary domains include infrared-visible (IVIF), medical (MDIF), multi-focus (MFIF), and multi-exposure (MEIF) fusion.
2. **Limitations of Prior Work**: (a) Most existing methods are designed for single specific tasks with poor generalization; (b) CNNs are constrained by local receptive fields, while Transformers incur prohibitive computational complexity ($O(n^2)$); (c) deep learning methods inherently favor low-frequency content, making it difficult to accurately capture high-frequency textures and structural details.
3. **Key Challenge**: General fusion requires a dynamically adaptive architecture that accommodates heterogeneous modality characteristics without increasing computational complexity. Existing Mamba-based fusion methods focus solely on spatial scanning or single-modality scenarios, neglecting spatial-channel interaction and cross-modal dependencies.
4. **Goal**: Design an efficient general fusion framework capable of handling all four fusion tasks while preserving high-frequency details without increasing model complexity.
5. **Key Insight**: Combine the linear-complexity global modeling capability of Mamba with the dynamic adaptability of Mixture of Experts (MoE), and employ self-supervised contrastive learning to constrain high-frequency information.
6. **Core Idea**: Dynamically select and aggregate cross-modal experts via a MoE mechanism, while enforcing high-frequency preservation at both the feature and pixel levels through bi-level contrastive learning.

## Method

### Overall Architecture

Given two modality inputs $I_{m1}, I_{m2}$, the MAFE module first enhances single-modality features. The MCCM module then fuses complementary cross-modal information through multi-expert cross-modal Mamba. Finally, the BSCL loss constrains high-frequency information at both the feature and pixel levels. The overall structure follows an encoder–fusion–decoder design.

### Key Designs

1. **Modality-Agnostic Feature Enhancement Module (MAFE)**

   - **Function**: Enhances single-modality representations by simultaneously capturing local details and global context.
   - **Mechanism**: Consists of a local branch and a global branch. The local branch partitions features into patches and adaptively extracts fine-grained spatial features using 3×3 depthwise convolution with a gating mechanism ($F_L = \text{Gate}(\text{Conv}_{1 \times 1}(F_{sk}^{j\_dw})) \odot F_{sk}^{j\_dw}$). The global branch contains two parallel SSMs: (a) a **Spatial-Channel SSM** that employs SC-Scan to capture spatial-channel correlations; and (b) a **Frequency-Rotation SSM** that transforms features to the frequency domain via DFT, applies FR-Scan separately to the amplitude and phase, and reconstructs spatial features via IDFT, achieving global enhancement in the frequency domain (a single modification in the frequency domain affects all spatial features). Local and global features are ultimately concatenated.
   - **Design Motivation**: SSMs excel at global modeling but lose local details, necessitating a local branch as a complement. Frequency-domain processing inherently exerts global influence, compensating for the limitations of spatial Mamba.

2. **Multiplex Consensus Cross-modal Mamba Module (MCCM)**

   - **Function**: Dynamically fuses complementary cross-modal information via a MoE mechanism, balancing expert diversity and consensus.
   - **Mechanism**: Comprises $N=4$ cross-modal Mamba experts $\{CM_1, ..., CM_4\}$, each independently performing cross-modal fusion. A gating network extracts global features via GAP and GMP, then computes TopK ($k=2$) expert weights. Cross-Modal Scanning (CM-Scan) alternates between the two modalities in both spatial and channel dimensions with forward and backward scans. Three auxiliary losses jointly regulate training: a **load balancing loss** $\mathcal{L}_{wb}$ prevents gating collapse; an **expert diversity loss** $\mathcal{L}_{div}$ promotes heterogeneous behavior via cosine similarity minimization; and a **consensus loss** $\mathcal{L}_{cons}$ encourages experts to converge toward a unified representation. A temporal decay weight $\lambda(t) = \cos(t/T \cdot \pi/2)$ encourages diversity in early training and emphasizes consensus in later stages.
   - **Design Motivation**: Different fusion tasks have distinct objectives (IVIF preserves thermal targets; MFIF preserves sharp regions), and MoE enables dynamic adaptation. The dynamic diversity–consensus balance ensures both exploration and convergence.

3. **Bi-level Self-supervised Contrastive Learning Loss (BSCL)**

   - **Function**: Reinforces high-frequency information preservation without increasing model complexity, while improving downstream task performance.
   - **Mechanism**: The Haar wavelet lifting scheme decomposes features and images into high-frequency and low-frequency components. At the **feature level**, the high-frequency component of the fused feature $F_{mf}^h$ is pulled toward the high-frequency components of the input modalities $F_{mc}^h$ and pushed away from their low-frequency components $F_{mc}^l$: $\mathcal{L}_{fcl} = \|F_{mf}^h - F_{mc}^h\|_1^2 / \|F_{mf}^h - F_{mc}^l\|_1^2 + ...$. The same contrastive constraint is applied at the **pixel level** as $\mathcal{L}_{pcl}$.
   - **Design Motivation**: The inherent frequency bias of deep networks leads to low-frequency dominance, whereas high-frequency textures and edges are critical for fusion quality and downstream tasks. The self-supervised formulation introduces no additional annotation cost.

### Loss & Training

$$\mathcal{L}_{total} = 0.8 \mathcal{L}_{fcl} + 0.4 \mathcal{L}_{pcl} + \mathcal{L}_{mccm} + \mathcal{L}_{ssim} + \mathcal{L}_{int}$$

Adam optimizer with an initial learning rate of $2 \times 10^{-4}$; cosine annealing halves the rate every 1,000 iterations; batch size 1; trained on a single RTX 3090.

## Key Experimental Results

### Main Results

Partial results on the MSRS dataset (IVIF task):

| Method | Type | MI↑ | SF↑ | VIF↑ | Qabf↑ | MS_SSIM↑ |
|--------|------|-----|-----|------|-------|----------|
| CDDFuse | Task-specific | 3.657 | 12.083 | 0.819 | 0.548 | 0.459 |
| Fusionmamba1 | General | 4.121 | 10.955 | 0.974 | 0.652 | 0.511 |
| TC-MoA | General | 3.251 | 9.370 | 0.811 | 0.565 | 0.515 |
| **SMC-Mamba** | General | **4.490** | **12.211** | **0.991** | **0.658** | **0.522** |

### Ablation Study

| Configuration | Description | Effect |
|---------------|-------------|--------|
| w/o MAFE | Remove modality enhancement | Loss of global and local features; performance degrades |
| w/o Frequency SSM | Remove frequency branch | Weakened global representation |
| w/o MoE | Single expert substitute | Reduced task adaptability |
| w/o BSCL | Remove contrastive loss | Significant loss of high-frequency details |
| w/o Consensus Loss | Remove consensus constraint | Inconsistent expert outputs |
| w/o Diversity Loss | Remove diversity constraint | Expert homogenization |

### Key Findings

- SMC-Mamba comprehensively surpasses existing general-purpose and task-specific methods across all four fusion tasks (IVIF, MDIF, MFIF, MEIF).
- BSCL contributes significantly to high-frequency detail preservation without increasing inference cost, as it is used only during training.
- The temporal decay weight strategy effectively balances expert diversity and consensus convergence.
- Cross-modal scanning substantially improves cross-modal feature interaction quality compared to single-modality Mamba scanning.

## Highlights & Insights

- **Self-supervised Contrastive Learning for High-Frequency Constraint**: By decomposing features via Haar wavelets and treating high/low-frequency components as positive/negative samples, the proposed contrastive loss enforces high-frequency preservation—an elegant application of contrastive learning to low-level vision tasks.
- **Dynamic Diversity–Consensus Balance in MoE**: The temporal decay strategy that encourages diversity early and enforces consensus later is a generalizable design principle for other MoE applications.
- **Frequency-Domain Mamba**: Applying Mamba scanning to frequency-domain amplitude and phase components represents a novel perspective.

## Limitations & Future Work

- Training is conducted on a single RTX 3090; efficiency and scalability analyses are insufficient.
- The number of experts (4) and the Top-2 selection are fixed; adaptive expert count configurations remain unexplored.
- The choice of Haar wavelets in BSCL is relatively simple; whether more sophisticated frequency decompositions could yield further improvements warrants investigation.
- Downstream task validation is limited to detection and segmentation; other tasks such as tracking are not addressed.

## Related Work & Insights

- **vs. Fusionmamba1/2**: These methods perform only single-modality spatial Mamba scanning; SMC-Mamba introduces cross-modal scanning and MoE for substantial improvement.
- **vs. TC-MoA**: TC-MoA also employs MoE but lacks the diversity–consensus balancing mechanism.
- **vs. CDDFuse**: Task-specific methods may excel on their target tasks but cannot generalize to other fusion scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of frequency-domain Mamba, MoE consensus mechanism, and bi-level contrastive learning is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four fusion tasks with extensive comparison methods and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, though the large number of modules adds some complexity.
- Value: ⭐⭐⭐⭐ A meaningful advancement in the direction of general image fusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization](spa_achieving_consensus_in_llm_alignment_via_self-priority_optimization.md)
- [\[AAAI 2026\] Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis](vascular_anatomy-aware_self-supervised_pre-training_for_x-ray_angiogram_analysis.md)
- [\[AAAI 2026\] Virtual Multiplex Staining for Histological Images Using a Marker-wise Conditioned Diffusion Model](virtual_multiplex_staining_for_histological_images_using_a_marker-wise_condition.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](../../CVPR2026/medical_imaging/decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[AAAI 2026\] ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling](propl_universal_semi-supervised_ultrasound_image_segmentation_via_prompt-guided_.md)

</div>

<!-- RELATED:END -->
