---
title: >-
  [Paper Note] The Double-Ellipsoid Geometry of CLIP
description: >-
  [ICML 2025][LLM Pretraining][CLIP] Through data-driven analysis, it is discovered that CLIP's primary embeddings before L2 normalization-exhibit a double-ellipsoidal shell geometry—where image and text embeddings reside on linearly separable ellipsoidal shells shifted away from the origin. The concept of conformity is introduced to explain how this structure helps mitigate false negatives and accounts for the emergence of the modality gap.
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "CLIP"
  - "modality gap"
  - "ellipsoid"
  - "contrastive learning"
  - "conformity"
  - "thin-shell"
date: 2026-05-08
content_hash: 181dee23205705e8
---

# The Double-Ellipsoid Geometry of CLIP

**Conference**: ICML 2025  
**arXiv**: [2411.14517](https://arxiv.org/abs/2411.14517)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: CLIP, modality gap, ellipsoid, contrastive learning, conformity, thin-shell

## TL;DR

Through data-driven analysis, it is discovered that CLIP's primary embeddings before L2 normalization-exhibit a double-ellipsoidal shell geometry—where image and text embeddings reside on linearly separable ellipsoidal shells shifted away from the origin. The concept of conformity is introduced to explain how this structure helps mitigate false negatives and accounts for the emergence of the modality gap.

## Background & Motivation

**Background**: CLIP, as a representative of multimodal contrastive learning, is widely applied in tasks such as image generation, classification, and segmentation, yet the geometric structure of its latent space remains poorly understood. Existing research focuses on phenomena such as alignment/uniformity and the modality gap.

**Limitations of Prior Work**: Analysis is usually conducted on the unit sphere after L2 normalization. However, normalization is a dimensionality reduction process that discards the semantic information carried by the norms. On MS-COCO, embeddings with the largest norms correspond to the most unusual content. There lacks a unified geometric explanation for the modality gap and the narrow cone effect.

**Key Challenge**: L2 normalization projects all vectors onto a sphere, artificially "flattening" the original geometry. Consequently, the norm information is lost, making structural properties difficult to analyze.

**Goal**: (1) Reveal the complete geometric structure before normalization; (2) Explain the benefits of this structure to contrastive learning; (3) Explain the rationality of the modality gap.

**Key Insight**: Analyze primary embeddings (before normalization) and conduct a purely data-driven analysis using the MS-COCO validation set.

**Core Idea**: The image and text embeddings of CLIP are located on ellipsoidal shells shifted away from the origin. This eccentric structure naturally mitigates false negatives by controlling the distance to the mean to achieve "semantic blurring".

## Method

### Overall Architecture

Statistical analysis is performed on the MS-COCO validation set within the 512-dimensional space of CLIP ViT-B/32 to establish six geometric properties. Subsequently, the optimality of this geometry is explained starting from the NT-Xent loss, and finally, applications are demonstrated by introducing the concept of conformity.

### Key Designs

1. **Six Geometric Properties**:

    - Function: Establish a complete geometric picture of CLIP's latent space.
    - Key Findings:
        - **Property 1 (Linear Separability)**: Image and text can be 100% linearly separated using only 2 features (the 93rd and 134th dimensions). Nine features act as modality "labels".
        - **Property 2 (Thin Shell)**: After de-meaning, the distribution of $\|\tilde{v}\|$ is concentrated in a narrow range, with $\mu_{norm}^2 = 57.57 \gg \text{var}(y) = 0.19$.
        - **Property 3 (Ellipsoidal)**: The variance of each dimension exhibits a long-tail distribution, forming an inhomogeneous sphere.
        - **Property 4 (Tilt)**: The covariance matrix exhibits significant off-diagonal dominance, indicating strong correlations among features.
        - **Property 5 (Offset from Origin)**: $\|m_i\|/\|\sigma_i\| = 0.94$, $\|m_t\|/\|\sigma_t\| = 1.03$.
        - **Property 6 (Loss Optimality)**: When $\alpha=0$ (the current CLIP position), the NT-Xent loss achieves the optimal balance between alignment and uniformity.

2. **Conformity**:

    - Function: Quantify the "typicality" of a sample with respect to the overall distribution.
    - Mechanism: Define $C(v^j) = \mathbb{E}_{v^k}[\cos(v^j, v^k)]$. Key theorem: Under the thin-shell assumption, $\hat{C}(v^j) = a \cdot \cos(m, v^j) + b$, which means estimation requires only the cosine similarity with the mean vector. The Pearson correlation on MS-COCO is 0.9998. This substitutes $O(N^2)$ with $O(N)$.
    - Design Motivation: Rigorously define "typicality", and the equivalent form using the cosine of the mean allows for highly efficient computation.

3. **Eccentric Ellipsoid and False Negative Mitigation**:

    - Function: Explain the advantages of the geometric structure offset from the origin.
    - Mechanism: For an origin-centered sphere, the distribution of cosine similarity is similar across all vectors, failing to distinguish between common and rare samples. On an eccentric sphere, being closer to the mean leads to higher cosine similarity, resulting in a "semantic blurring" effect (thus mitigating false negatives), while being far from the mean leads to sharp contrasts. Common concepts (which suffer from more false negatives) are naturally embedded closer to the mean.
    - Design Motivation: Since CLIP uses the standard NT-Xent loss and does not explicitly address false negatives, the eccentric geometry is an implicitly learned solution.

### Modality Gap Explanation

Images and text exhibit different conformity distributions—in a single image-text pair, the image might be common while the text is unique (or vice versa). The separated double-ellipsoid allows each modality to independently control its conformity distribution. At $\alpha=0$, the KL divergence between the two modality conformity distributions is minimized ($\approx 0.14$).

## Key Experimental Results

### Conformity Validation

| Modality | Pearson Correlation (C vs Ĉ) | a | b |
|------|---------------------|---|---|
| Image | 0.9998 | 1.461 | -0.002 |
| Text | 0.9998 | 1.411 | -0.008 |

### Generative Model Evaluation

| Method | Conformity | Interpretation |
|------|-----------|------|
| Glide (Image) | High | Generates common images, lacking details |
| unCLIP (Image) | Close to real | More details and diversity |
| ClipCap (Text) | High | Generates common descriptions |
| Caption Reward (Text) | Low (super-human) | Generates unique descriptions |

### Key Findings

- The linear relationship between conformity and the cosine similarity to the mean consistently holds across different architectures and datasets.
- Low conformity samples have larger CLIP norms, corresponding to more unique/rare content.
- vSLERP leverages the ellipsoidal geometry to achieve optimization-free semantic editing: adjusting $\alpha$ controls the editing magnitude while maintaining the identity of the individual.
- The optimality of the eccentric ellipsoid is directly verified through loss vs $\alpha$ experiments.

## Highlights & Insights

- The six properties form a complete picture—progressing step-by-step from separability, to thin shell, to ellipsoid, to tilt, to eccentric, and finally to loss optimality.
- The concept of conformity is simple yet powerful—simplifying $O(N^2)$ to $O(N)$, backed by rigorous mathematical derivations.
- Explaining the eccentric structure from the perspective of false negatives is the most creative contribution—linking an unaddressed problem to the emergence of geometric structure.
- Conformity can serve directly as a metric for evaluating generative model diversity.

## Limitations & Future Work

- The analysis is mainly based on ViT-B/32 and MS-COCO; generalization to larger models and datasets remains to be validated.
- The ellipsoidal structure is an observation of learning outcomes rather than a theoretical derivation, and causal relationships are not rigorously proven.
- The approximation accuracy of conformity may degrade for extreme samples.
- There is a lack of practical experiments leveraging the geometric structure to improve downstream tasks.
- The applicability to other contrastive learning models (e.g., ALIGN, SigLIP) has not been verified.

## Related Work & Insights

- **vs Liang et al. (2022)**: First discovered the modality gap and narrow cone. This work provides a deeper geometric explanation at the primary embedding level.
- **vs Schrodi et al. (2024)**: Also discussed linear separability and entropy. The conformity concept in this study is more precise.
- **vs Wang & Isola (2020)**: Alignment and uniformity decomposition. This study demonstrates how the eccentric ellipsoid optimally balances both.
- **Insight**: Conformity can be generalized as a dataset quality assessment tool and a metric for generative model diversity.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel perspective, where the six properties, conformity, and modality gap explanation form a complete story.
- Experimental Thoroughness: ⭐⭐⭐ Primarily analytical, lacking experiments that utilize geometry for downstream improvements.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely fluent narrative transitioning from data-driven observation, to theoretical explanation, to application.
- Value: ⭐⭐⭐⭐ Deepens the understanding of CLIP's representation space, and the concept of conformity has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Whitened CLIP as a Likelihood Surrogate of Images and Captions](whitened_clip_as_a_likelihood_surrogate_of_images_and_captions.md)
- [\[ICML 2025\] Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry](algebra_unveils_deep_learning_--_an_invitation_to_neuroalgebraic_geometry.md)
- [\[CVPR 2026\] Reconstructing CLIP for Open-Vocabulary Dense Perception](../../CVPR2026/llm_pretraining/reconstructing_clip_for_open-vocabulary_dense_perception.md)
- [\[NeurIPS 2025\] Predict Training Data Quality via Its Geometry in Metric Space](../../NeurIPS2025/llm_pretraining/predict_training_data_quality_via_its_geometry_in_metric_space.md)
- [\[CVPR 2025\] Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](../../CVPR2025/llm_pretraining/seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)

</div>

<!-- RELATED:END -->
