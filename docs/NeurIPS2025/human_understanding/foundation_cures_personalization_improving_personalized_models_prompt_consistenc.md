---
title: >-
  [Paper Note] Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency via Hidden Foundation Knowledge
description: >-
  [NeurIPS 2025][Human Understanding][Face Personalization] FreeCure reveals that identity embeddings in face personalization models suppress but do not destroy the prompt control capability of the foundation model. Based…
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "Face Personalization"
  - "Diffusion Models"
  - "Prompt Consistency"
  - "Training-Free"
  - "Self-Attention"
date: 2026-05-08
content_hash: c9de68b66fc816a6
---

# Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency via Hidden Foundation Knowledge

**Conference**: NeurIPS 2025
**arXiv**: [2411.15277](https://arxiv.org/abs/2411.15277)  
**Code**: [Project Page](https://yiyangcai.github.io/freecure-aigc.github.io/)  
**Area**: Human Understanding
**Keywords**: Face Personalization, Diffusion Models, Prompt Consistency, Training-Free, Self-Attention

## TL;DR
FreeCure reveals that identity embeddings in face personalization models suppress but do not destroy the prompt control capability of the foundation model. Based on this insight, the paper proposes a training-free framework that injects attribute information from the foundation model into the personalized generation process via Foundation-Aware Self-Attention (FASA). The method substantially improves prompt consistency while preserving identity fidelity, and can be seamlessly integrated into mainstream architectures including SD, SDXL, and FLUX.

## Background & Motivation

**Background** Face personalization models (FastComposer, PhotoMaker, PuLID, InfiniteYou, etc.) generate identity-preserving images by fusing identity embeddings into cross-attention layers, yet balancing identity fidelity and prompt consistency remains a core challenge.

**Limitations of Prior Work** Identity embeddings dominate the cross-attention mechanism, effectively "suppressing" the normal expression of other attribute tokens (e.g., hair color, expression, accessories), causing generated results to fail to accurately reflect facial attributes specified in the prompt.

**Key Challenge** Identity embeddings are indispensable for identity preservation, yet they are precisely the source of degraded prompt consistency. Directly modifying cross-attention disrupts identity extraction capability.

**Goal** Restore facial attribute control suppressed by identity embeddings without modifying the cross-attention modules of the personalization model, thereby preserving identity capability.

**Key Insight** The observation that removing identity embeddings from a personalization model recovers the high prompt consistency of the foundation model—indicating that foundation knowledge is "suppressed" but not "destroyed" and can be leveraged through self-attention layers.

**Core Idea** A dual-inference paradigm is used to extract correct attribute information from the foundation model, which is then locally injected via FASA at the self-attention layers.

## Method

### Overall Architecture
FreeCure employs a dual-inference paradigm: PD (with identity embeddings) and FD (without identity embeddings / replaced with zero tensors). PD preserves identity but yields weak attribute expression, while FD produces accurate attributes but lacks identity. The FASA module injects correct attributes from FD into PD at the self-attention layers, with segmentation masks constraining the injection region to protect identity.

### Key Designs

1. **Foundation-Aware Self-Attention (FASA)**:
    - **Function**: Fuses information from PD and FD in the self-attention layers.
    - **Mechanism**: Concatenates the K/V of FD after those of PD: $\hat{K} = [K_p, K_f], \hat{V} = [V_p, V_f]$, and computes attention using Q from PD: $\text{FASA} = \text{Softmax}(\frac{[\mathbf{1}, \omega\mathcal{M}] \odot Q_p\hat{K}^T}{\sqrt{d}})\hat{V}$, where $\mathcal{M}$ is the attribute mask and $\omega$ is a scaling factor.
    - **Design Motivation**: Cross-attention layers are highly sensitive—minor modifications disrupt identity; self-attention layers retain foundation model knowledge and serve as a safe intervention point.

2. **Fine-Grained Attribute Mask Control**:
    - **Function**: Restricts attribute injection to occur only within the target facial region.
    - **Mechanism**: A face parsing model (BiSeNet/SAM) extracts binary masks $M_i$ for target attributes (hairstyle, accessories, eye color, etc.) from FD outputs and merges them as $\mathcal{M} = \bigcup\{M_i\}$. The mask ensures FASA injects FD information only in attribute regions, leaving identity information in non-attribute regions undisturbed.
    - **Design Motivation**: Without masking, FASA introduces substantial irrelevant FD features that severely harm identity fidelity.

3. **Asymmetric Prompt Guidance (APG)**:
    - **Function**: Recovers abstract attributes (e.g., expression).
    - **Mechanism**: Performs DDIM inversion on the FASA-processed image using a template prompt without attribute descriptions, then denoises from an intermediate timestep using a prompt containing full attribute descriptions. Denoising begins from $\hat{z_{\gamma T}}$ (with $\gamma=0.5$), preserving high-level identity information.
    - **Design Motivation**: FASA relies on spatial masks and is suited for spatially localizable attributes (hairstyle, glasses); global attributes such as expression lack clear spatial boundaries and require a different strategy.

### FLUX Adaptation
In FLUX's full-attention DiT, the FASA mask is applied only to the interaction between PD visual queries and FD visual keys, preserving the original cross-modal attention patterns.

## Key Experimental Results

### Main Results — Prompt Consistency (PC) and Identity Fidelity (IF)

| Method | PC% ↑ | IF% ↑ | PC×IF (hMean) ↑ |
|--------|-------|-------|-----------------|
| InstantID | 21.89 | 63.94 | 32.61 |
| + FreeCure | **23.62** (+7.9%) | 62.01 (−3.0%) | **34.21** (+4.9%) |
| PuLID (FLUX) | 22.42 | 74.97 | 34.52 |
| + FreeCure | **24.78** (+10.5%) | 72.61 (−3.2%) | **36.95** (+7.0%) |
| InfiniteYou | 23.77 | 79.71 | 36.62 |
| + FreeCure | **25.25** (+6.2%) | 77.13 (−3.2%) | **38.05** (+3.9%) |

### Multi-Attribute Prompt Performance

| # Attributes | Baseline PC (SDv1.5) | +FreeCure |
|--------------|----------------------|-----------|
| 1 attribute | 21.01 | 22.70 (+8.0%) |
| 2 attributes | 20.34 | 22.34 (+9.9%) |
| 3 attributes | 18.49 | 20.49 (+10.8%) |

### Ablation Study

| Configuration | Effect | Remarks |
|---------------|--------|---------|
| FASA without mask | Severe identity loss | FD features fully override PD |
| FASA with mask | PC ↑, IF minimally reduced | Precise injection of target attributes |
| FASA + APG | Optimal | Both spatial and abstract attributes recovered |
| Cross-attention interpolation | Rapid identity loss | Confirms cross-attention should not be modified |

### Key Findings
- FreeCure improves the joint PC×IF metric across all 8 baseline models.
- Improvements from FreeCure become more pronounced as the number of attributes increases (from 8% to 10.8%), indicating greater value in complex scenarios.
- IF degradation is controlled within approximately 3%, largely attributable to a positive increase in facial diversity.

## Highlights & Insights
- The finding that "identity embeddings suppress rather than destroy foundation knowledge" offers a new conceptual perspective for the personalization community.
- The FASA design is elegant: precise local attribute injection is achieved via K/V concatenation with masking, without touching the sensitive cross-attention layers.
- Generalizability across three generations of foundation models (SD / SDXL / FLUX) demonstrates the architecture-agnostic nature of the approach.

## Limitations & Future Work
- Running an additional face parsing model to extract masks incurs extra inference time.
- The dual-inference paradigm doubles inference cost.
- Control over extremely fine-grained attributes (e.g., iris color, earring shape) leaves room for further improvement.

## Related Work & Insights
- **vs. PhotoSwap/MasaCtrl**: These methods leverage self-attention for editing but target generic objects; FreeCure is specifically designed for face personalization.
- **vs. InstantID/PuLID**: These serve as baseline methods enhanced by FreeCure, which operates as a plug-in.
- **vs. ControlNet**: ControlNet controls generation via additional conditions; FreeCure exploits knowledge intrinsic to the model and requires no additional training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The insight that "foundation knowledge is suppressed but not destroyed" is original, and the FASA design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Large-scale evaluation covering 8 baseline models, 3 generations of foundation models, and 50 identities × 20 prompts.
- **Writing Quality**: ⭐⭐⭐⭐ In-depth and intuitive analysis with rich visualizations.
- **Value**: ⭐⭐⭐⭐ Training-free and plug-and-play; directly applicable to face personalization tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mechanistic Interpretability of RNNs Emulating Hidden Markov Models](mechanistic_interpretability_of_rnns_emulating_hidden_markov_models.md)
- [\[NeurIPS 2025\] K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning](k-decore_facilitating_knowledge_transfer_in_continual_structured_knowledge_reaso.md)
- [\[CVPR 2026\] UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](../../CVPR2026/human_understanding/unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)
- [\[NeurIPS 2025\] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)
- [\[ICCV 2025\] PHD: Personalized 3D Human Body Fitting with Point Diffusion](../../ICCV2025/human_understanding/phd_personalized_3d_human_body_fitting_with_point_diffusion.md)

</div>

<!-- RELATED:END -->
