---
title: >-
  [Paper Note] EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection
description: >-
  [CVPR 2026][Segmentation][Camouflaged Object Detection] This paper proposes EReCu, a unified unsupervised camouflaged object detection framework consisting of three synergistic modules — Multi-cue Native Perception (MNP), Pseudo-label Evolution Fusion (PEF), and Local Pseudo-label Refinement (LPR) — achieving boundary-accurate and detail-rich camouflaged object segmentation without any manual annotations.
tags:
  - CVPR 2026
  - Segmentation
  - Camouflaged Object Detection
  - Unsupervised Segmentation
  - Pseudo-label Learning
  - Teacher-Student Framework
  - Attention Fusion
date: 2026-05-08
content_hash: af41eb4396df4b7e
---

# EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection

**Conference**: CVPR 2026
**arXiv**: [2603.11521](https://arxiv.org/abs/2603.11521)
**Code**: [GitHub](https://github.com/JSLiam94/EReCu)
**Area**: Segmentation
**Keywords**: Camouflaged Object Detection, Unsupervised Segmentation, Pseudo-label Learning, Teacher-Student Framework, Attention Fusion

## TL;DR

This paper proposes EReCu, a unified unsupervised camouflaged object detection framework consisting of three synergistic modules — Multi-cue Native Perception (MNP), Pseudo-label Evolution Fusion (PEF), and Local Pseudo-label Refinement (LPR) — achieving boundary-accurate and detail-rich camouflaged object segmentation without any manual annotations.

## Background & Motivation

Camouflaged Object Detection (COD) aims to segment concealed targets that are highly similar to their backgrounds, representing an extremely challenging visual task. Although fully supervised methods achieve strong performance, they rely on costly and ambiguous pixel-level annotations, limiting data scale and ecological diversity. Unsupervised COD (UCOD) has thus emerged as an important research direction.

Existing UCOD methods fall into two paradigms, each with inherent bottlenecks:

**Pseudo-label guided**: Methods such as UCOS-DA employ static pseudo-labels, while UCOD-DPL introduces teacher-student dynamic fusion. However, they over-rely on high-dimensional embeddings while neglecting perceptual cues from the raw image, leading to **boundary overflow** and semantic drift.

**Feature learning based**: Methods such as SdalsNet decouple foreground/background features via attention mechanisms, and EASE introduces environmental prototype retrieval. However, the absence of explicit pseudo-label supervision results in **boundary ambiguity** and loss of fine details.

The core insight is that semantic reliability and texture fidelity should not be optimized in isolation, but should co-evolve through mutual feedback loops. EReCu is built on this principle, allowing native perceptual cues to continuously guide pseudo-label evolution while perceptual learning benefits from progressively denoised supervision.

## Method

### Overall Architecture

EReCu is built upon a DINO-pretrained teacher-student architecture. The teacher branch provides stable semantic guidance, while the student branch progressively learns refined camouflage masks under the supervision of evolved pseudo-labels. Three core modules form a synergistic pipeline:
- **MNP** extracts native texture and semantic cues $F_{\text{MNP}}$ and a quality metric $S_{\text{mc}}$
- **PEF** leverages these cues to evolve global pseudo-labels through teacher-student interaction and spectral tensor attention fusion
- **LPR** utilizes native cues to generate local pseudo-labels from high-confidence regions, repairing boundary and texture details missed by global predictions

### Key Designs

1. **Multi-cue Native Perception Module (MNP)**: MNP serves as the cornerstone of the entire framework, providing native perceptual guidance to both PEF and LPR. The core idea is that, despite camouflage arising from high visual similarity with the background, subtle yet discriminative texture variations still exist in the raw image. MNP fuses low-level texture features (LBP, DoG) with mid-level semantic features (frozen ResNet-18) to construct a multi-cue representation: $F_{\text{MNP}} = \mathcal{C}(F_{\text{text}}, F_{\text{sem}})$. A multi-cue quality metric $S_{\text{mc}}$ is further proposed, which partitions the image into three regions based on the predicted mask — interior $R_i$, boundary $R_s$, and exterior $R_o$ — and evaluates foreground-background separability via cosine similarity computed on randomly sampled patches. $S_{\text{mc}} = (D_{\text{io}} + D_{\text{is}} + S_{\text{so}})/3$, where larger values indicate stronger foreground-background separation.

2. **Pseudo-label Evolution Fusion Module (PEF)**: Comprises two complementary sub-modules:

    - **Evolving Pseudo-label Learning (EPL)**: Enables interaction between shallow student features and deep teacher features through depthwise separable convolutions (DSC). DSC decomposes standard convolutions into depthwise and pointwise operations, reducing computational cost while enhancing fine-grained texture and boundary structure. The iterative optimization jointly employs Dice loss (teacher-student consistency) and $\mathcal{L}_{\text{MNP}}$ (native cue regularization), enabling pseudo-labels to continuously evolve under dual semantic and perceptual guidance.
    - **Spectral Tensor Attention Fusion (STAF)**: Stacks three attention maps from different layers of the student network into a third-order tensor $\mathcal{T}_s \in \mathbb{R}^{3 \times C \times HW}$, captures inter-layer, inter-channel, and spatial correlations via Tucker decomposition, retains principal energy components and filters noise via truncated SVD, and projects the result into a fused prediction $M_s^{\text{fu}}$. Complexity is $\mathcal{O}(r^2 d)$, where $r \ll d$.

3. **Local Pseudo-label Refinement Module (LPR)**: Exploits the spatial diversity of DINO's multi-head self-attention (MHSA) to refine pseudo-labels.

    - **Target-Aware Attention Selection (TAS)**: Jointly filters focused and semantically consistent attention heads using attention entropy $E_k$ and multi-cue metric $S_{\text{mc}}$: $\mathcal{A}_{\text{sel}} = \{A_k \mid E_k < \tau_e \wedge S_{\text{mc}}(\hat{A}_k, F_{\text{MNP}}) > \tau_s\}$.
    - **Local Pseudo-label Generation (LPG)**: Generates local pseudo-labels from high-confidence regions of selected attention heads (adaptive threshold $\tau_k = \mu_{A_k} + \alpha \cdot \sigma_{A_k}$), guiding student network refinement via a joint Dice + CE loss.

### Loss & Training

- **EPL loss**: Dual Dice loss (student-teacher alignment) + $\mathcal{L}_{\text{MNP}}$ (native cue regularization), applied iteratively
- **LPR loss**: $\mathcal{L}_{\text{LPR}} = \mathcal{L}_D(M_s^{\text{fu}}, \bigcup_k P_k) + \mathcal{L}_{\text{CE}}(M_s^{\text{fu}}, \bigcup_k P_k)$
- Teacher model updated via EMA ($\eta=0.99$); trained for 25 epochs, batch size 32, AdamW + cosine annealing, AMP mixed precision
- Encoder: DINO-ViT-S/8; texture extractor: LBP and DoG

## Key Experimental Results

### Main Results

| Dataset | Metric | EReCu | Prev. SOTA (UCOD-DPL) | Gain |
|--------|------|-------|---------------------|------|
| CHAMELEON | $S_m\uparrow$ | .7321 | .7287 | +0.34% |
| CAMO | $S_m\uparrow$ | .7027 | .7013 | +0.14% |
| COD10K | $S_m\uparrow$ | .7221 | .7090 | +1.31% |
| COD10K | $F_\omega^\beta\uparrow$ | .5628 | .5481 | +1.47% |
| COD10K | $M\downarrow$ | .0613 | .0601 | -0.12% |
| NC4K | $S_m\uparrow$ | .7583 | .7538 | +0.45% |
| NC4K | $E_m^\phi\uparrow$ | .8498 | .8447 | +0.51% |

EReCu comprehensively outperforms all UOS and UCOD baseline methods across all four datasets.

### Ablation Study

| Configuration | CAMO $S_m\uparrow$ | COD10K $S_m\uparrow$ | Note |
|------|---------------------|----------------------|------|
| MNP+EPL+STAF+LPR (Full) | .7027 | .7221 | Best |
| w/o MNP | .6887 | .7111 | Localization degradation due to missing texture cues |
| w/o EPL | .6758 | .7038 | Significant drop in structural consistency |
| w/o STAF | .6815 | .7179 | Regional inconsistencies emerge |
| w/o LPR | .6895 | .7109 | Weakened local detail recovery |
| DINO-ViT-S/8 only | .6376 | .6400 | Baseline |

### Key Findings

- Each module contributes positively and complementarily; combining three or more modules significantly outperforms any two-module combination
- The MNP + EPL pairing yields the most substantial improvement, validating the synergy between native cue alignment and pseudo-label learning
- In challenging scenarios (depth artifacts, extreme texture suppression), EReCu produces clearer boundaries and more complete structures

## Highlights & Insights

1. **Paradigm innovation**: EReCu is the first to unify pseudo-label evolution and native perceptual learning through a self-evolving teacher-student mechanism, bridging the gap between "pseudo-label refinement" and "feature learning" paradigms
2. **Multi-cue quality metric**: $S_{\text{mc}}$ evaluates foreground-background separability via cosine similarity of randomly sampled patches, providing an elegant, robust, and transferable tool for pseudo-label quality assessment
3. **Spectral tensor fusion**: Fusing multi-layer attention maps via Tucker decomposition + truncated SVD better preserves semantic and structural information than simple weighted aggregation, while remaining computationally efficient

## Limitations & Future Work

- Performance gains are relatively modest on certain datasets/metrics (e.g., $S_m$ improves by only 0.34% on CHAMELEON), suggesting that UCOD performance may be approaching a ceiling in some scenarios
- The framework's complexity is considerable (three synergistic modules + teacher-student architecture), raising concerns about inference efficiency and deployment cost
- Validation is limited to standard COD benchmarks; more complex real-world ecological scenarios and cross-domain generalization have not been explored
- The texture extractors (LBP, DoG) are hand-crafted; learnable alternatives for low-level feature extraction warrant investigation

## Related Work & Insights

- EReCu shares the teacher-student framework with UCOD-DPL, but introduces native cue regularization to avoid boundary overflow caused by purely semantic guidance
- SdalsNet's attention-based decoupling is complementary to EReCu's LPR — the former handles global separation while the latter performs local refinement
- The spectral fusion strategy of STAF is generalizable to other tasks requiring multi-layer feature map fusion
- The interior-boundary-exterior three-region partition in $S_{\text{mc}}$ can inspire quality assessment strategies in other segmentation tasks

## Rating

- Novelty: ⭐⭐⭐⭐ The unified three-module synergistic framework design is innovative, and the concept of native perception-guided pseudo-label evolution is clearly motivated
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, comprehensive ablation studies, visual analysis, and broad baseline coverage
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams and visualizations are clear, inter-module relationships are well articulated, and mathematical derivations are complete
- Value: ⭐⭐⭐⭐ A significant advance in unsupervised camouflage detection; the native perception-guided pseudo-label paradigm offers broader methodological inspiration

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)
- [\[CVPR 2026\] Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning](love_me_love_my_label_rethinking_the_role_of_labels_in_prompt_retrieval_for_visu.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)

</div>

<!-- RELATED:END -->
