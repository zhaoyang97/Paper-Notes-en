---
title: >-
  [Paper Note] Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration
description: >-
  [CVPR 2025][LLM (Other)][Old Film Restoration] This paper proposes the MambaOFR framework to address the complex compound degradations unique to old films. It designs degradation-aware prompts to guide the Mamba model in dynamically adjusting restoration modes, incorporates a flow-guided masked deformable alignment module to prevent the propagation of structural defects, and introduces the first benchmark dataset for old film restoration containing both synthetic and real-wor…
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Old Film Restoration"
  - "Mamba"
  - "Degradation-aware"
  - "Optical Flow Alignment"
  - "State Space Model"
date: 2026-05-08
content_hash: 9ee4ab3f0ccde363
---

# Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration

**Conference**: CVPR 2025  
**Code**: None  
**Area**: Video Restoration / Old Film Restoration  
**Keywords**: Old Film Restoration, Mamba, Degradation-aware, Optical Flow Alignment, State Space Model

## TL;DR

This paper proposes the MambaOFR framework to address the complex compound degradations unique to old films. It designs degradation-aware prompts to guide the Mamba model in dynamically adjusting restoration modes, incorporates a flow-guided masked deformable alignment module to prevent the propagation of structural defects, and introduces the first benchmark dataset for old film restoration containing both synthetic and real-world data.

## Background & Motivation

### Background

**Background**: Old film restoration aims to restore film quality from the analog era, which requires handling unique degradations such as scratches, flicker, noise, color fading, and film grain. In recent years, deep-learning-based video restoration methods (e.g., BasicVSR++, RVRT) have performed exceptionally well in modern video denoising and super-resolution tasks.

**Limitations of Prior Work**: (1) Degradations are highly specific and compound: Old film degradation differs from that of digital videos (e.g., JPEG compression, motion blur), encompassing physical aging of film (scratches, emulsion cracks), chemical changes (color shifting, fading), and digitization artifacts, often co-occurring. (2) Dedicated methods underperform compared to general ones: Existing specialized restoration methods for old films (e.g., detection-restoration pipelines targeting scratches) perform worse than general video restoration methods when handling compound degradations. (3) Difficulty in temporal feature propagation: Structural defects like scratches display spatial consistency across video frames; simple temporal attention or alignment tends to propagate these defect features into adjacent frames.

**Key Challenge**: Old film restoration demands long-range temporal modeling (leveraging multi-frame information for propagation) alongside degradation-adaptive processing (tailoring strategies for different degradation types). However, long-range modeling risks propagating structural defects, whereas static restoration strategies fail to adapt to complex and variable compound degradations.

**Goal**: How to design an old film restoration framework that adaptively handles diverse compound degradations while preventing the temporal propagation of defects?

**Key Insight**: Leverage Mamba (State Space Models) for efficient long-sequence modeling of the video temporal dimension, dynamically adjust restoration behaviors via degradation-aware prompts, and design a specialized masked alignment to suppress defect propagation.

**Core Idea**: Degradation-aware prompts driving Mamba-adaptive restoration + masked deformable alignment blocking defect propagation + the first comprehensive benchmark dataset.

## Method

### Overall Architecture

MambaOFR adopts an encoder-decoder architecture with multi-frame input, centered around a Mamba-based spatiotemporal processing backbone. Given continuous video frames: (1) A degradation estimation module analyzes the degradation type and severity of each frame to generate degradation-aware prompts. (2) Guided by these prompts, the Mamba backbone extracts spatiotemporal features, where the flow-guided masked deformable alignment module handles inter-frame alignment. (3) The decoder reconstructs the restored frames.

### Key Designs

1. **Degradation-aware Prompt Generation**:
    - **Function**: Dynamically adjusts the restoration policy of the network based on different degradation types.
    - **Mechanism**: A lightweight degradation estimation network is designed to extract input frame features and predict a degradation representation vector (encoding degradation type and severity). This representation undergoes an affine transformation to generate scale and shift parameters, which are injected as prompts into various layers of the Mamba backbone to dynamically modulate the feature processing. $\gamma, \beta = \text{MLP}(z_{deg})$, $\hat{f} = \gamma \cdot f + \beta$
    - **Design Motivation**: Scratch restoration demands robust spatial inpainting, whereas color fading restoration requires global color mapping. A static parameter network cannot simultaneously optimize both strategies.

2. **Flow-Guided Mask Deformable Alignment**:
    - **Function**: Halts the propagation of structural defects when leveraging multi-frame information.
    - **Mechanism**: First, the inter-frame optical flow is estimated while a defect detection branch is trained to predict binary masks of structural defects (e.g., scratches and emulsion cracks) on the frames. During deformable alignment, the defect mask is used to downweight features within the defect region, ensuring that areas covered by scratches in adjacent frames are not propagated. In practice, the mask is multiplied by the deformable attention weights to perform soft-masking.
    - **Design Motivation**: Conventional optical flow alignment propagates incorrect features from the scratch locations of reference frames to the current frame. The masking mechanism ensures that only information from "clean" regions is propagated.

3. **Comprehensive Benchmark Dataset**:
    - **Function**: Represents the first standardized evaluation suite for old film restoration comprising both synthetic and real-world degradations.
    - **Mechanism**: The synthetic part utilizes a compound degradation simulation pipeline (randomly overlaying scratches, noise, flicker, color fading, etc.) on high-quality digital videos to generate paired training data. The real-world part collects authentic old film clips for qualitative evaluation, covering degradation patterns of different eras and film types.
    - **Design Motivation**: Existing methods lack a unified evaluation platform, and varying test sets across different papers make results incomparable.

### Loss & Training
The model is trained end-to-end, with the optimization objective balancing task-specific loss functions and regularization terms.


## Key Experimental Results

### Key Findings

- MambaOFR consistently outperforms existing specialized old film restoration methods and general-purpose video restoration methods in PSNR/SSIM on synthetic test datasets.
- The degradation-aware prompts improve the model's adaptive capacity across varied degradations by approximately 1.5–2 dB PSNR.
- The flow-guided masked deformable alignment shows the most significant improvement in restoration quality on frames containing scratches.
- Ablation studies confirm that all three components (prompts, masked alignment, and Mamba backbone) contribute independently to the performance.
- Qualitative restoration results on real old films exhibit visibly superior visual quality compared to competing approaches.

## Highlights & Insights

- **Problem-driven Design**: Each module is tailored directly to the specific limitations associated with old film restoration.
- **Appropriate Application of Mamba**: Capitalizes on the long-sequence modeling advantages of SSMs to process the temporal dimension of videos, achieving superior computational efficiency compared to Transformers.
- **Benchmark Contribution**: Fills the gap of the absence of standardized evaluations in old film restoration.

## Limitations & Future Work

- Extremely severe physical damages (large-area missing content) remain difficult to restore.
- There remains a domain gap between synthetic degradations and real-world old film degradations.
- Color restoration relies heavily on the color distribution of the training data, which may lead to sub-optimal results for films with specific artistic color styles.
- Future work could integrate generative models (e.g., Diffusion Models) to perform semantic reconstruction on corrupted regions.


## Related Work & Insights
- **vs. Representative Methods in the Same Field**: This method provides unique structural contributions and complements existing methods.
- **vs. Traditional Methods**: Compared to traditional pipelines, the proposed approach achieves significant improvements on key evaluation metrics.
- **Insights**: The technical pipeline of this work serves as a valuable reference for future research in old film restoration.


## Rating
- Novelty: ⭐⭐⭐⭐ Significant and unique methodological contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated thoroughly across multiple datasets.
- Writing Quality: ⭐⭐⭐⭐ Extremely well-structured and clear.
- Value: ⭐⭐⭐⭐ Highly promising for advancing the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](../../AAAI2026/llm_nlp/an_invariant_latent_space_perspective_on_language_model_inve.md)
- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](../../ACL2025/llm_nlp/biased_llms_can_influence_political_decision-making.md)
- [\[ICML 2026\] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](../../ICML2026/llm_nlp/a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)
- [\[NeurIPS 2025\] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](../../NeurIPS2025/llm_nlp/unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)
- [\[ACL 2025\] Making FETCH! Happen: Finding Emergent Dog Whistles Through Common Habitats](../../ACL2025/llm_nlp/making_fetch_happen_finding_emergent_dog_whistles_through_common_habitats.md)

</div>

<!-- RELATED:END -->
