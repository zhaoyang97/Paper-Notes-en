---
title: >-
  [Paper Note] Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation
description: >-
  [NeurIPS 2025][LLM Safety][Sign language translation] Geo-Sign projects skeleton features into a Poincaré ball model of hyperbolic space and regularizes an mT5 language model via a hyperbolic contrastive loss…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "Sign language translation"
  - "hyperbolic geometry"
  - "Poincaré ball"
  - "contrastive learning"
  - "skeleton representation"
date: 2026-05-08
content_hash: 8e2c2ca5fa9250e4
---

# Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation

**Conference**: NeurIPS 2025
**arXiv**: [2506.00129](https://arxiv.org/abs/2506.00129)
**Code**: [GitHub](https://github.com/ed-fish/geo-sign)
**Area**: AI Safety
**Keywords**: Sign language translation, hyperbolic geometry, Poincaré ball, contrastive learning, skeleton representation

## TL;DR
Geo-Sign projects skeleton features into a Poincaré ball model of hyperbolic space and regularizes an mT5 language model via a hyperbolic contrastive loss, enabling the model to perceive the hierarchical structure of sign language motion. Using only skeleton data, the method surpasses RGB-based SOTA on CSL-Daily (BLEU-4 +1.81, ROUGE-L +3.03).

## Background & Motivation

**Background**: Sign language translation (SLT) has recently shifted toward leveraging large language models (e.g., T5 variants) to process visual features. Most SOTA methods rely on RGB video input and large visual encoders (e.g., DINO-ViT), incurring high computational costs and raising privacy concerns.

**Limitations of Prior Work**: Skeleton representations extracted via spatial-temporal graph convolution (ST-GCN) are projected into Euclidean space for processing by language models. However, in Euclidean space, large-scale arm movements dominate the embedding norm, compressing the discriminability of fine-grained motions such as finger joint articulations. For example, the ASL sign for "water" requires distinguishing a finger W-shape touching the chin (leaf-node motion) from arm abduction (branch-node motion), yet in flat space these embeddings become conflated.

**Key Challenge**: Sign language motion possesses a natural tree-like hierarchical structure (torso → arm → wrist → fingers), yet the polynomial volume growth of Euclidean space cannot effectively encode such hierarchies.

**Goal**: Enhance the geometric properties of skeleton representations so that they naturally respect the kinematic hierarchy of sign language.

**Key Insight**: Hyperbolic space exhibits exponential volume growth $V_H(r) \propto e^{(d-1)r}$, making it naturally suited for encoding tree-like hierarchies — distances near the boundary amplify fine-grained distinctions, while the near-origin region approximates Euclidean space for semantic-level representations.

**Core Idea**: Project skeleton part features into a Poincaré ball with learnable curvature, align pose–text embeddings via a hyperbolic contrastive loss, and regularize the language model to perceive motion hierarchy.

## Method

### Overall Architecture

Input 2D skeleton keypoints (extracted by RTM-Pose) → grouped by body part (body / left hand / right hand / face) → ST-GCN extracts per-part features → two branches: (1) concatenation and projection to the mT5 encoder (Euclidean main branch); (2) temporal average pooling followed by projection to the Poincaré ball (hyperbolic regularization branch). Final loss: $\alpha \cdot \mathcal{L}_{CE} + (1-\alpha) \cdot \mathcal{L}_{hyp\_reg}$.

### Key Designs

1. **Hyperbolic Projection Layer**:

   - Function: Maps Euclidean part features $\bar{\mathbf{f}}_p$ onto the Poincaré ball.
   - Core formula: $\mathbf{h}_p = \exp_{\mathbf{0}}^c(s_p \mathbf{W}^p \bar{\mathbf{f}}_p)$, where the exponential map is $\exp_{\mathbf{0}}^c(\mathbf{v}) = \tanh(\frac{\sqrt{c}\|\mathbf{v}\|_2}{2}) \frac{\mathbf{v}}{\frac{\sqrt{c}}{2}\|\mathbf{v}\|_2}$.
   - A learnable scaling scalar $s_p$ controls the "depth" of each part in hyperbolic space — high-amplitude arm motions remain near the origin, while fine-grained finger motions are pushed toward the boundary.
   - Design Motivation: Exploit the distance-amplification effect of the boundary region in hyperbolic space to better discriminate fine-grained finger motions.

2. **Weighted Fréchet Mean Aggregation (Pooled Strategy)**:

   - Function: Aggregates multiple part embeddings $\{\mathbf{h}_p\}$ into a global pose embedding $\boldsymbol{\mu}_\text{pose}$.
   - Mechanism: Weights $w_p \propto \exp(d_{\mathbb{B}_c}(\mathbf{0}, \mathbf{h}_p))$ assign higher importance to parts farther from the origin (i.e., closer to the boundary, encoding finer-grained information).
   - Iterative algorithm: Compute a weighted sum of logarithmic maps in the tangent space, then map back to the manifold via the exponential map.
   - Design Motivation: The Fréchet mean is the geometrically correct averaging operation in hyperbolic space, outperforming Euclidean averaging in fidelity.

3. **Hyperbolic Attention Alignment (Token Strategy)**:

   - Function: Each pose part $\mathbf{h}_p$ serves as a Query attending over text token embeddings via hyperbolic attention, producing part-specific context vectors $\mathbf{c}_p$.
   - Key transformation: $\mathbf{k}_t = (\mathbf{M} \otimes \mathbf{v}_t) \oplus \mathbf{b}$ (Möbius affine transformation).
   - Attention score: $s_{pt} = -d_{\mathbb{B}_c}(\mathbf{h}_p, \mathbf{k}_t)$ (negative geodesic distance).
   - Context vector: $\mathbf{c}_p = \mu_{\mathcal{B}_c}(\{\mathbf{v}_t\}, \{\alpha_{pt}\})$ (hyperbolic weighted midpoint).
   - Design Motivation: Allows different body parts to attend to different text tokens, enabling finer-grained pose–semantics alignment.

4. **Hyperbolic Contrastive Loss (Geometric Contrastive Loss)**:

   - Core formula: $\mathcal{L}_\text{hyp\_pair}(\mathbf{p}_i, \mathbf{t}_i) = -\log \frac{\exp(-d_{\mathbb{B}_c}(\mathbf{p}_i, \mathbf{t}_i)/\tau)}{\sum_{j=1}^B \exp(-d_{\mathbb{B}_c}(\mathbf{p}_i, \mathbf{t}_j)/\tau + m \cdot \mathbb{I}(i \neq j))}$
   - Learnable temperature $\tau$ and learnable margin $m$.
   - Geodesic distance: $d_{\mathbb{B}_c}(\mathbf{u}, \mathbf{v}) = \frac{2}{\sqrt{c}} \text{artanh}(\sqrt{c}\|(-\mathbf{u}) \oplus_c \mathbf{v}\|_2)$.

### Loss & Training
- Total loss: $\mathcal{L}_\text{total} = \alpha \cdot \mathcal{L}_{CE} + (1-\alpha) \cdot \mathcal{L}_{hyp\_reg}$, where $\alpha$ is dynamically adjusted during training, with greater emphasis on hyperbolic regularization in early stages.
- Euclidean parameters are optimized with AdamW; hyperbolic parameters (curvature $c$, manifold parameters) are optimized with Riemannian Adam.
- Curvature $c$ is learned end-to-end (initialized at $c=1.5$) and optimized in log space.

## Key Experimental Results

### Main Results: CSL-Daily Sign Language Translation

| Method | Modality | B-1 | B-4 | ROUGE-L |
|--------|----------|-----|-----|---------|
| Uni-Sign (Pose) | Skeleton | 53.86 | 25.61 | 54.92 |
| Uni-Sign (Pose+RGB) | Skeleton+RGB | 55.08 | 26.36 | 56.51 |
| Geo-Sign (Euclidean Token) | Skeleton | 54.02 | 25.98 | 53.93 |
| Geo-Sign (Hyperbolic Pooled) | Skeleton | 55.80 | 27.17 | 57.75 |
| **Geo-Sign (Hyperbolic Token)** | **Skeleton** | **55.89** | **27.42** | **57.95** |
| CV-SLT (Gloss-based) | RGB | 58.29 | 28.94 | 57.06 |

### Ablation Study: Contrastive Strategy and Geometric Space

| Configuration | B-4 | ROUGE-L | Note |
|---------------|-----|---------|------|
| Uni-Sign (Pose, no regularization) | 25.61 | 54.92 | Baseline |
| Euclidean Pooled | 25.72 | 55.57 | Euclidean contrastive is helpful |
| Euclidean Token | 25.98 | 53.93 | Token alignment improves BLEU |
| Hyperbolic Pooled | 27.17 | 57.75 | Hyperbolic vs. Euclidean: +1.45/+2.18 |
| **Hyperbolic Token** | **27.42** | **57.95** | Best configuration |

### Key Findings
- Hyperbolic vs. Euclidean space: Hyperbolic Token outperforms Euclidean Token by +1.44 on B-4 and +4.02 on ROUGE-L, demonstrating a substantial contribution from hyperbolic geometry.
- Skeleton-only Geo-Sign surpasses Uni-Sign (Pose+RGB) and, on ROUGE-L, becomes the first skeleton-based method to exceed the SOTA gloss-based approach CV-SLT (57.95 vs. 57.06).
- Consistent improvements are also observed on How2Sign (ASL) and WLASL2000 (isolated sign language recognition), demonstrating language-agnostic generalizability.
- The Token strategy outperforms the Pooled strategy (finer part–text alignment), though Pooled is more computationally efficient.

## Highlights & Insights
- **Elegant alignment between geometry and linguistics**: The tree-like kinematic hierarchy of sign language naturally matches the exponential volume growth of hyperbolic space — a textbook case of choosing the right mathematical tool.
- **Learnable curvature**: End-to-end optimization of $c$ allows the model to self-adaptively tune the "magnification factor" — a more negative curvature $\kappa = -c$ places greater emphasis on fine-grained distinction.
- **Skeleton surpassing RGB**: The method achieves performance exceeding RGB-based approaches while using only skeleton data — which is inherently de-identified — carrying significant implications for privacy-preserving deployment (e.g., sign language translation in public spaces).

## Limitations & Future Work
- Hyperbolic operations (exponential map, logarithmic map, Möbius addition) require high-precision floating point (float32) for numerical stability, potentially affecting training efficiency.
- The memory overhead of the Token strategy scales substantially with sequence length, which may be limiting for very long sign language videos.
- Evaluation is limited to two sign language datasets (CSL-Daily, How2Sign); broader assessment across more languages and datasets is needed.
- The choice of the initial curvature value ($c = 1.5$) and the hyperbolic dimensionality ($d_\text{hyp}=256$) lacks theoretical justification.

## Related Work & Insights
- **vs. Uni-Sign (Gong et al.)**: Uni-Sign serves as the backbone architecture (shared ST-GCN pretraining); Geo-Sign augments it solely with a hyperbolic regularization branch, demonstrating the value of geometric priors.
- **vs. Sign2GPT / FLa-LLM**: These methods rely on large visual encoders coupled with LLMs, whereas Geo-Sign achieves competitive performance using only lightweight skeleton input at higher efficiency.
- **vs. Hyperbolic Action Recognition (Franco et al.)**: Prior work has applied hyperbolic space to general action recognition, but Geo-Sign is the first to apply it to end-to-end sign language translation, introducing hyperbolic attention and Fréchet mean aggregation as novel components.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The application of hyperbolic geometry to sign language translation is well-motivated and original; the hyperbolic attention in the Token strategy constitutes an independently novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations convincingly demonstrate the contribution of hyperbolic space, though dataset coverage is limited.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous with intuitive figures, though the hyperbolic geometry preliminaries are dense.
- Value: ⭐⭐⭐⭐ Directly valuable to the sign language translation community; the hyperbolic regularization paradigm is transferable to other hierarchical action understanding tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Lost in Translation? A Comparative Study on the Cross-Lingual Transfer of Composite Harms](../../AAAI2026/llm_safety/lost_in_translation_a_comparative_study_on_the_cross-lingual_transfer_of_composi.md)
- [\[ICCV 2025\] Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation](../../ICCV2025/llm_safety/forgetting_through_transforming_enabling_federated_unlearning_via_class-aware_re.md)
- [\[AAAI 2026\] StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak](../../AAAI2026/llm_safety/stylebreak_revealing_alignment_vulnerabilities_in_large_audio-language_models_vi.md)
- [\[NeurIPS 2025\] Self-Refining Language Model Anonymizers via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](demystifying_language_model_forgetting_with_low-rank_example_associations.md)

</div>

<!-- RELATED:END -->
