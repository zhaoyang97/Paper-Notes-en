---
title: >-
  [Paper Note] CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation
description: >-
  [CVPR 2025][Image Generation][CLIP Bias Analysis] This work systematically reveals two types of bias in CLIP within multi-object scenarios: text encoders bias toward earlier-mentioned objects, and image encoders bias toward larger objects. It traces the origin of these biases to the statistical pattern in contrastive training data where larger objects tend to be mentioned first.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "CLIP Bias Analysis"
  - "Multi-Object Representation"
  - "Text Encoder Bias"
  - "Contrastive Learning"
  - "ComCO Dataset"
date: 2026-05-08
content_hash: a3ebe616f256a19e
---

# CLIP Under the Microscope: A Fine-Grained Analysis of Multi-Object Representation

**Conference**: CVPR 2025  
**arXiv**: [2502.19842](https://arxiv.org/abs/2502.19842)  
**Code**: [https://clip-oscope.github.io/](https://clip-oscope.github.io/)  
**Area**: Image Generation  
**Keywords**: CLIP Bias Analysis, Multi-Object Representation, Text Encoder Bias, Contrastive Learning, ComCO Dataset

## TL;DR
This work systematically reveals two types of bias in CLIP within multi-object scenarios: text encoders bias toward earlier-mentioned objects, and image encoders bias toward larger objects. It traces the origin of these biases to the statistical pattern in contrastive training data where larger objects tend to be mentioned first.

## Background & Motivation

**Background**: CLIP is widely used in tasks such as image-text alignment, retrieval, and generation guidance. Although CLIP's limitations in compositional understanding are well-known, its specific bias patterns in multi-object representation have not been systematically analyzed.

**Limitations of Prior Work**: (1) Do CLIP's text embeddings faithfully encode all objects in multi-object scenarios? (2) Do image embeddings represent objects of different sizes uniformly? (3) Where do these biases originate? Prior works only focus on general phenomena in two-object scenarios, lacking granular quantification and root-cause analysis.

**Key Challenge**: CLIP should learn general vision-language alignment. However, its training data contains a statistical bias where "larger objects are mentioned first." Contrastive learning then propagates the size bias of the image encoder into the positional bias of the text encoder.

**Goal**: To quantitatively refine CLIP's multi-object biases using a controllable synthetic dataset (ComCO), trace the root causes of these biases, and suggest mitigation strategies.

**Key Insight**: Render multi-object scenes (72 COCO objects) with controllable size, position, and quantity in Blender, and design two metric frameworks—Text-based Object Retrieval (TOR) and Image-based Object Retrieval (IOR)—to quantify the biases of the text and image encoders, respectively.

**Core Idea**: Use controllable synthetic data to reveal the "first-in, first-coded" bias of CLIP's text encoder and the "size-dominates" bias of the image encoder, and trace them back to the statistical correlation between object size and mention order in the training data.

## Method

### Overall Architecture
ComCO dataset (Blender rendering of scenes with 2–5 objects) $\rightarrow$ TOR evaluation (cosine similarity ranking of multi-object text vs. individual object texts) $\rightarrow$ IOR evaluation (cosine similarity ranking of multi-object images vs. individual object images) $\rightarrow$ LAION data analysis to trace training biases.

### Key Designs

1. **Text-based Object Retrieval (TOR)**:

    - **Function**: Quantifying the text encoder's attention to objects at different structural positions.
    - **Mechanism**: Given a multi-object description "a horse, a dog, a cat", compute the cosine similarity between its embedding and the individual embeddings of "a horse", "a dog", and "a cat". If the encoder is unbiased, each object should receive equal similarity.
    - **Design Motivation**: In the CLIP LAION model, the first object achieves a $63.96\%$ retrieval probability, while the fourth receives only $3.76\%$—indicating a severe "first-in" bias.

2. **Image-based Object Retrieval (IOR)**:

    - **Function**: Quantifying the image encoder's attention to objects of different sizes.
    - **Mechanism**: Given an image containing one large and three small objects, compute the cosine similarity between its embedding and each individual single-object image embedding.
    - **Design Motivation**: The large object obtains an $85.45\%$ retrieval probability, while the three small objects obtain only $6.36\%$, $5.45\%$, and $2.73\%$, respectively—showing that CLS token attention is heavily biased toward large objects.

3. **Bias Origin Tracing**:

    - **Function**: Explaining why CLIP develops these biases.
    - **Mechanism**: Analysis of the LAION training dataset reveals that: (a) objects with larger surface areas are mentioned earlier in captions (statistically significant); (b) the loss function of contrastive learning theoretically allows convergence to an incomplete representation that only encodes a subset of objects (mathematical proof provided); (c) during training, the bias intensifies as training steps progress (validated via 5 checkpoints).
    - **Design Motivation**: SBERT/SimCSE are used as control experiments—since they do not undergo contrastive image-text training, they exhibit the opposite bias (tending toward the last object), proving that the bias indeed originates from CLIP's contrastive training process.

### Loss & Training
This is an analysis-only paper without training. Hard negative training methods such as NegCLIP and SugarCrepe can mitigate but not eliminate the biases.

## Key Experimental Results

### Main Results

| Model | TOR 1st Object | TOR 4th Object | IOR Large Object | IOR Small Object |
|------|-----------|-----------|----------|----------|
| CLIP LAION | 63.96% | 3.76% | 85.45% | 6.36% |
| CLIP OpenAI | 50.31% | 6.79% | - | - |
| NegCLIP | - | - | 61.67% | 15.00% |
| SugarCrepe | 44.29% | 6.66% | - | - |

### Ablation Study

| Baseline | Bias Pattern | Description |
|------|---------|------|
| SBERT | Biased toward last object | No contrastive image-text training |
| SimCSE | Biased toward last object | No contrastive image-text training |
| NegCLIP | Bias mitigated | Hard negative training |
| Training $2\text{B}\rightarrow 10\text{B}$ steps | Bias aggravated | Deteriorates as training progresses |

### Key Findings
- **Text Bias 16:1**: The retrieval probability for the 1st object is 17 times higher than that for the 4th object (CLIP LAION).
- **Image Bias 13:1**: Large objects achieve a retrieval probability that is 31 times higher than that of the smallest objects.
- **Bias is inherent to contrastive learning**: Mathematically proved that CLIP's InfoNCE loss can converge to incomplete text representations.
- **Hard negative training mitigates but does not cure the issue**: SugarCrepe reduces the bias ratio from 17:1 to 6.7:1.

## Highlights & Insights
- **First detailed quantification of CLIP's multi-object bias**: Transitioning from "knowing there is a bias" to "precisely measuring how severe the bias is and where it comes from."
- **The bias propagation chain (Training Data $\rightarrow$ Image Encoder $\rightarrow$ Text Encoder)** is a key discovery—implying that modifying the description habits of the training data might be the most fundamental solution.
- **The ComCO dataset** provides a standardized multi-object evaluation toolkit for future research.

## Limitations & Future Work
- ComCO uses synthetic data, which exhibits a domain gap with real-world multi-object scenes.
- Only object size and mentioning order are analyzed; other factors (e.g., object salience, category frequency) remain unexplored.
- No concrete de-biasing training scheme has been proposed.

## Related Work & Insights
- **vs Winoground / ARO**: These benchmarks evaluate compositional understanding but do not distinguish the sources of biases. ComCO's controllable design allows precise attribution.
- **vs NegCLIP**: Hard negative training can mitigate the biases, but this paper reveals that the root cause lies in the training data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to systematically reveal the root-cause chain of CLIP's multi-object biases.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model, multi-dimensional, controlled experiments, training trajectory validation, and mathematical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Decisive and rigorous logic from phenomenon to attribution.
- Value: ⭐⭐⭐⭐ Serves as an important warning for all downstream tasks dependent on CLIP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs](finelip_clip_long_text_fine_grained.md)
- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [\[CVPR 2025\] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fade_fine_grained_erasure_diffusion.md)
- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[CVPR 2025\] Multi-Group Proportional Representation for Text-to-Image Models](multi-group_proportional_representations_for_text-to-image_models.md)

</div>

<!-- RELATED:END -->
