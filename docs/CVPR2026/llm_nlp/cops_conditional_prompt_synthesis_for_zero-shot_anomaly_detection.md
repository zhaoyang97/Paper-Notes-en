---
title: >-
  [Paper Note] CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection
description: >-
  [CVPR 2026][LLM/NLP][zero-shot anomaly detection] CoPS is a framework that dynamically generates prompts through two visual conditioning mechanisms — Explicit State Token Synthesis (ESTS) and Implicit Category Token Sampling (ICTS) — combined with Spatially-Aware Global-local Alignment (SAGA), achieving zero-shot anomaly detection SOTA across 13 industrial and medical datasets.
tags:
  - CVPR 2026
  - LLM/NLP
  - zero-shot anomaly detection
  - conditional prompt synthesis
  - CLIP
  - vision-language model
  - industrial defect
date: 2026-05-08
content_hash: 3a2c5165ad8ec4b3
---

# CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection

**Conference**: CVPR 2026  
**arXiv**: [2508.03447](https://arxiv.org/abs/2508.03447)  
**Code**: [https://github.com/cqylunlun/CoPS](https://github.com/cqylunlun/CoPS)  
**Area**: LLM / NLP (Other)  
**Keywords**: zero-shot anomaly detection, conditional prompt synthesis, CLIP, vision-language model, industrial defect

## TL;DR
CoPS is a framework that dynamically generates prompts through two visual conditioning mechanisms — Explicit State Token Synthesis (ESTS) and Implicit Category Token Sampling (ICTS) — combined with Spatially-Aware Global-local Alignment (SAGA), achieving zero-shot anomaly detection SOTA across 13 industrial and medical datasets.

## Background & Motivation
1. **Background**: Large-scale pre-trained vision-language models demonstrate strong cross-category generalization for zero-shot anomaly detection (ZSAD). Existing methods achieve cross-category anomaly detection by fine-tuning on a single auxiliary dataset.
2. **Limitations of Prior Work**: (i) Static learnable tokens fail to capture the continuous and diverse patterns of normal and anomalous states, limiting generalization to unseen categories; (ii) Fixed text labels provide overly sparse category information, causing the model to overfit to specific semantic subspaces.
3. **Key Challenge**: While prompt learning eliminates the need for manual prompt design, its static nature and sparsity become generalization bottlenecks — normal/anomalous states are continuously variable, and the category label space itself is highly sparse.
4. **Goal**: Design a visual feature-conditioned dynamic prompt synthesis framework enabling prompts to adaptively model the state and category information of input images.
5. **Key Insight**: Decompose prompts into three parts — context words, state words, and category words — where context words can be shared while the latter two need to be dynamically generated based on visual features.
6. **Core Idea**: Inject normal/anomalous prototypes from local features into state words (explicit), sample from global features via VAE into category words (implicit), achieving visually-conditioned dynamic prompt synthesis.

## Method

### Overall Architecture
Built on pre-trained CLIP, the input image passes through a frozen visual encoder to extract global features $\mathbf{g}$ and local features $\mathbf{F}$. ESTS extracts normal/anomalous prototypes from local features and injects them into state words; ICTS samples diversified category tokens from global features via VAE; finally, a learnable text encoder and SAGA module enable image-level and pixel-level anomaly detection.

### Key Designs

1. **Explicit State Token Synthesis (ESTS)**:
    - Function: Extracts representative normal and anomalous prototypes from fine-grained local features and explicitly injects them into prompt state words
    - Mechanism: Uses consistency self-attention (V-V attention) to extract fine-grained local features $\mathbf{F}$ from the frozen visual encoder, then prototype extractor $\mathcal{P}_\theta$ generates $M$ normal prototypes $\mathbf{P}_n$ and anomalous prototypes $\mathbf{P}_a$ under center constraints, assembling them as dynamic state tokens replacing static learnable tokens
    - Design Motivation: Fixed state words (e.g., "good"/"damaged") cannot capture continuously diverse normal/anomalous patterns. Extracting prototypes from actual image local features enables adaptive modeling of the current image's state, enhancing generalization

2. **Implicit Category Token Sampling (ICTS)**:
    - Function: Models semantic global features via VAE and generates diversified category tokens through sampling
    - Mechanism: A variational autoencoder $\mathcal{E}_\psi$ parameterizes the latent distribution of global features $\mathbf{g}$, drawing $R$ decoded samples $\mathbf{S} \in \mathbb{R}^{R \times C}$ as dense category tokens. Each input image thus generates $R$ complete sets of normal/anomalous prompts
    - Design Motivation: Fixed text labels are too sparse to provide rich category semantic information. VAE sampling implicitly augments category representation diversity, preventing model overfitting to a single semantic subspace

3. **Spatially-Aware Global-local Alignment (SAGA)**:
    - Function: Combines distance-aware spatial attention for fine-grained image-text alignment
    - Mechanism: Approximates anomaly state using the distance between query features and nearest prototypes, introducing a distance-aware spatial attention mechanism to refine pixel-level text-image alignment. Global-local (glocal) similarity interaction enhances image-level alignment. Outputs image-level anomaly score $s_{\text{cls}}$ and pixel-level anomaly map $\mathcal{S}_{\text{seg}}$
    - Design Motivation: Standard global alignment ignores local spatial information, while anomaly detection inherently requires precise spatial localization

### Loss & Training
Binary focal loss for image-level classification; Dice loss and binary cross-entropy loss for pixel-level segmentation. Fine-tuned on a single auxiliary training set (e.g., MVTec AD), directly applied to unseen categories at test time.

## Key Experimental Results

### Main Results

| Dataset | Metric | CoPS | Prev. SOTA | Gain |
|---------|--------|------|------------|------|
| 13-dataset avg. | Cls AUROC | SOTA | - | +1.4% |
| 13-dataset avg. | Seg AUROC | SOTA | - | +1.9% |
| MVTec AD | Cls AUROC | Best | AnomalyCLIP, etc. | Significant |
| VisA | Seg AUROC | Best | - | Clear advantage |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Full CoPS | Best | Complete model |
| w/o ESTS | Decreased | Removing explicit state synthesis has the largest impact |
| w/o ICTS | Decreased | Removing implicit category sampling also has notable impact |
| w/o SAGA | Decreased | Spatial-aware alignment is especially important for segmentation |
| Static prompt baseline | Significantly below CoPS | Validates the necessity of dynamic prompts |

### Key Findings
- ESTS contributes the most, indicating that adaptive state modeling is the core challenge in zero-shot anomaly detection
- ICTS's VAE sampling effectively mitigates category label sparsity, especially in cross-domain scenarios (industrial → medical)
- Distance-aware spatial attention significantly improves pixel-level segmentation quality but has less impact on image-level classification

## Highlights & Insights
- **Prompt decomposition design philosophy** is elegant: shared context words + explicitly injected state words + implicitly sampled category words, each serving its purpose
- **VAE implicit augmentation** is an elegant trick: replacing fixed labels with sampling naturally increases category representation diversity
- The use of consistency self-attention (V-V) avoids introducing additional adaptation modules, preserving CLIP features' original semantics

## Limitations & Future Work
- Relies on CLIP's pre-trained feature space; may have limited effectiveness for visual domains not covered by CLIP (e.g., specialized industrial scenarios)
- Prototype count M and sampling count R require manual tuning
- Future work could explore adaptive prototype count determination or replacing CLIP with stronger visual foundation models

## Related Work & Insights
- **vs AnomalyCLIP**: AnomalyCLIP uses static learnable tokens without visual conditioning; this work overcomes that limitation through explicit/implicit injection
- **vs AdaCLIP**: AdaCLIP relies on hand-designed template sets; this work eliminates manual design through end-to-end learning
- **vs VCP-CLIP**: VCP-CLIP directly embeds image features into category words; this work provides richer semantic diversity through VAE sampling

## Rating
- Novelty: ⭐⭐⭐⭐ The explicit + implicit dual-pathway dynamic prompt synthesis is a novel combination
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across 13 datasets with complete ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-explained methods
- Value: ⭐⭐⭐⭐ Practical advancement in zero-shot anomaly detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts](../../AAAI2026/llm_nlp/promptmoe_generalizable_zero-shot_anomaly_detection_via_visually-guided_prompt_m.md)
- [\[CVPR 2026\] Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)
- [\[CVPR 2026\] Composing Concepts from Images and Videos via Concept-prompt Binding](composing_concepts_from_images_and_videos_via_concept-prompt_binding.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[CVPR 2026\] Sign Language Recognition in the Age of LLMs](sign_language_recognition_llms.md)

</div>

<!-- RELATED:END -->
