---
title: >-
  [Paper Note] Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition
description: >-
  [ECCV 2024][Image Generation][Idempotent generative models] This paper proposes Idempotent Generative Models (IGM), theoretically establishing the equivalence between generative models and maximum entropy coding (spectral contrastive learning). By imposing idempotent constraints on the feature space of skeleton data, the features of the generative model become more compact and suitable for recognition tasks, improving the accuracy on NTU 60 xsub from 84.6% to 86.2%.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Idempotent generative models"
  - "skeleton-based action recognition"
  - "self-supervised learning"
  - "contrastive learning"
  - "diffusion models"
date: 2026-05-08
content_hash: 3e0d08e4666b161b
---

# Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition

**Conference**: ECCV 2024  
**arXiv**: [2410.20349](https://arxiv.org/abs/2410.20349)  
**Code**: [GitHub](https://github.com/LanglandsLin/IGM)  
**Area**: Image Generation  
**Keywords**: Idempotent generative models, skeleton-based action recognition, self-supervised learning, contrastive learning, diffusion models

## TL;DR

This paper proposes Idempotent Generative Models (IGM), theoretically establishing the equivalence between generative models and maximum entropy coding (spectral contrastive learning). By imposing idempotent constraints on the feature space of skeleton data, the features of the generative model become more compact and suitable for recognition tasks, improving the accuracy on NTU 60 xsub from 84.6% to 86.2%.

## Background & Motivation

Skeleton data represents human motion using 3D joint coordinates. Compared to RGB videos, skeleton data is compact and efficient, making it widely used for action recognition tasks. Existing self-supervised pre-training methods fall into two main categories:

**Generative Learning** (e.g., MAE, MAMP): These methods learn spatiotemporal correlations by predicting or reconstructing masked skeleton data. However, they preserve excessive appearance information irrelevant to recognition, which contradicts the natural characteristics of skeleton data being "spatially sparse and temporally consistent."

**Contrastive Learning** (e.g., AimCLR, CMD): These methods construct positive sample pairs through data augmentation to maintain consistency in the embedding space. However, the augmentation process often discards fine-grained motion details.

These two paradigms offer complementary advantages, but prior work has typically explored them in isolation. The core problem is: **Can the advantages of generative models and contrastive learning be unified?**

The authors identify a theoretical path starting from an information theory perspective:
- Generative models are equivalent to Maximum Entropy Coding.
- Once idempotent constraints are imposed on the generative model, its loss becomes equivalent to Spectral Contrastive Learning.
- This provides a theoretical basis for incorporating contrastive learning within a generative framework.

## Method

### Overall Architecture

IGM consists of three core components:

1. **Encoder** $f(\cdot)$: Extracts conditional features $\mathbf{z}$ from augmented skeleton data.
2. **Generator** $g(\cdot)$: A conditional denoising generator based on diffusion models, reconstructing the skeleton conditioned on $\mathbf{z}$.
3. **Adapter** $h(\cdot)$: Projects and fuses the high-level semantic features from the encoder into the feature space of the generator.

Training utilizes two types of losses: diffusion noise prediction loss + idempotent feature constraints. During inference, only the encoder $f(\cdot)$ is used for downstream recognition tasks.

### Key Designs

#### 1. Theoretical Foundation: Generative Model = Maximum Entropy Coding

The reconstruction loss of self-conditional generative models, $\mathcal{L} = H(\mathbf{x}|\mathbf{z})$, essentially maximizes the mutual information $I(\mathbf{z}; \mathbf{x})$. Since the encoding process is deterministic ($H(\mathbf{z}|\mathbf{x}) \to 0$), maximizing mutual information is equivalent to maximizing the entropy of the feature space $H(\mathbf{z})$.

By using the rate-distortion (lossy coding length) as a proxy for the entropy of continuous random variables and applying Taylor expansion, it is shown that generative models primarily act to **reduce the similarity between data points in the feature space**:

$$L = -\frac{\mu\lambda^2}{2}\sum_{i,j}(\mathbf{z}_i^T\mathbf{z}_j)^2 - \mathbf{R}$$

#### 2. Idempotent Generative Model = Spectral Contrastive Learning

Idempotency refers to the stability of re-encoding: $f(\hat{\mathbf{x}}) = \mathbf{z}$, meaning that re-encoding generated data should yield the same representation.

The idempotent loss is structured as $\mathcal{L}_{\text{ide}} = \|f(\hat{\mathbf{x}}) - \mathbf{z}\|^2 = 2 - 2f(\hat{\mathbf{x}})^Tf(\mathbf{x})$.

Combining the idempotent loss with the entropy maximization objective leads to:

$$\mathcal{L} = \|\mathbf{A} - \mathbf{F}^T\mathbf{F}\|_F^2 + \mathbf{R} + \mathbf{C}$$

where $\mathbf{A}$ is the adjacency matrix defined by the data generation process. This matches the exact loss formulation of **Spectral Contrastive Learning**! Furthermore, compared to standard spectral contrastive learning, IGM additionally optimizes the high-order residual term $\mathbf{R}$.

#### 3. Relationship with MAE

MAE implicitly maximizes feature similarity between different masked instances of the same data via a random masking process $M(\cdot)$. However, the transformed data may deviate from the true distribution. In contrast, the idempotent generative model achieves a similar objective via the generative process $G(\cdot)$, producing data closer to the true distribution.

#### 4. Downstream Task Error Bound

According to spectral contrastive learning theory, the error rate of downstream linear evaluation is bounded by:

$$P_e \le c_1\sum_{i=d+1}^{m}\lambda_i^2 + c_2\alpha$$

where $\alpha$ is a term related to cluster purity. This implies a need to increase the diversity of generated data (reducing small singular values of the adjacency matrix) while preserving motion semantics (maintaining cluster purity). The noise sampling process of the diffusion model naturally provides this diversity.

#### 5. Manifold Decoupled Feature Fusion Module (MDFF)

While recognition tasks focus on high-frequency motion details, generative tasks primarily optimize the principal component space (low-frequency information), operating in different feature subspaces. The adapter decouples these components via high-frequency extraction:

$$\hat{\mathbf{z}} \Leftarrow (1+\eta)\mathbf{z} - \eta\text{SoftMax}(\mathbf{z}^T\mathbf{z})\mathbf{z}$$

This is equivalent to the gradient update of the uniformity loss in contrastive learning, filtering out low-frequency information (such as the sequence mean) and keeping the semantics most crucial for recognition. High-frequency conditions are then injected into the generator via Adaptive LayerNorm (AdaLN).

### Loss & Training

The total loss consists of two parts:

**1. Noise Prediction Loss:**

$$\mathcal{L}_{\text{gen}} = \|g(\mathbf{x}_t, h(\mathbf{z}), t) - \varepsilon\|^2$$

**2. Idempotent Constraints (Dual Constraints):**

**(a) Feature Idempotent Constraint** — Ensures that the features of re-encoded generated data match the original features:

$$\mathcal{L}_{\text{ide\_feat}} = -f(\mathbf{x})^T f(\mathbf{x}_0, \mathbf{z}_{t'}, t, t')$$

where $\mathbf{x}_0$ is the generated data estimated via one-step denoising. Since generated data can be noisy, the noise feature and timestep are passed as auxiliary information.

**(b) Distribution Idempotent Constraint** — Ensures that the feature manifold structure of the generated data matches that of the original data:

$$\mathcal{L}_{\text{ide\_dist}} = \mathcal{D}(\mathcal{P}(\mathbf{x}_0), \mathcal{P}(\mathbf{x}))$$

where $\mathcal{P}(\mathbf{x}) = f(\mathbf{x})^Tf(\mathbf{X})$ represents the similarity structure among features. This links not only different generative variations of the same data, but also different data with similar characteristics to build tighter clusters.

## Key Experimental Results

### Main Results

Comparison with unsupervised methods on the NTU RGB+D dataset:

| Method | Architecture | NTU 60 xview | NTU 60 xsub | NTU 120 xset | NTU 120 xsub |
|------|------|:---:|:---:|:---:|:---:|
| 3s-AimCLR (Contrastive) | GCN | 83.4 | 77.8 | 66.7 | 67.9 |
| 3s-CMD (Contrastive) | GRU | 90.9 | 84.1 | 76.1 | 74.7 |
| MAMP (Generative) | Transformer | 89.1 | 84.9 | 79.1 | 78.6 |
| PCM3 (Hybrid) | GRU | 90.4 | 83.9 | 77.5 | 76.3 |
| **IGM (Ours)** | Transformer | **91.2** | **86.2** | **81.4** | **80.0** |

The proposed model achieves state-of-the-art results across all four evaluation protocols, improving the accuracy on NTU 60 xsub from the previous best of 84.9% to 86.2% (+1.3%), and on NTU 120 xsub from 78.6% to 80.0% (+1.4%).

### Ablation Study

KNN evaluation (NTU 60 dataset):

| Method | xview | xsub |
|------|:---:|:---:|
| IGM w/o $\mathcal{L}_{\text{ide}}$ | 67.2 | 64.7 |
| IGM w/ $\mathcal{L}_{\text{ide\_feat}}$ | 70.7 | 68.4 |
| IGM w/ $\mathcal{L}_{\text{ide\_dist}}$ | 72.1 | 69.0 |
| **IGM (Full)** | **72.6** | **69.3** |

### Key Findings

1. **Idempotent constraints are crucial**: Disabling the idempotent constraint drops KNN xsub from 69.3 to 64.7 (-4.6%), validating the theoretical analysis.
2. **Distribution idempotency outperforms feature idempotency**: The distribution constraint (69.0) is more effective than the feature constraint (68.4) as it captures richer structural information.
3. **Dual idempotent constraints are complementary**: Incorporating both constraints yields the best performance (69.3), showing that feature-level and distribution-level constraints focus on different aspects of consistency.
4. **IGM excels in zero-shot adaptation**: Reasonable performance is achieved even on unseen, previously unrecognizable scenarios.
5. **Unified framework outperforms isolated paradigms**: The proposed unified framework outperforms pure contrastive or pure generative approaches across all datasets.

## Highlights & Insights

1. **Prominent theoretical contribution**: This work is the first to rigorously prove the equivalence between generative models (with idempotent constraints) and spectral contrastive learning, establishing a theoretical foundation to unify these two fields.
2. **Clever exploitation of noise sampling in diffusion**: It addresses the tension of insufficient diversity in self-conditional generation—where standard generation is constrained by the distance to original data, limiting diversity—while the noise sampling process in diffusion inherently provides diversity.
3. **Manifold decoupling design**: By applying high-pass filtering to extract high-frequency details key to recognition, it mitigates the dimensional collapse issues where generative features lean heavily towards the principal component space.
4. **Complete pipeline from theory to application**: Beginning with information-theoretic analysis, the work derives the necessity of idempotent constraints and designs specific feature- and distribution-level loss terms.

## Limitations & Future Work

1. **Confined to skeleton modality**: Although the theoretical framework is generalizable, experiments were only validated on skeleton data, leaving its effectiveness on other modalities like RGB video unverified.
2. **Computational overhead of diffusion sampling**: The training stage requires a diffusion process to generate diverse data, increasing the training cost.
3. **NTU dataset limitation**: Evaluations are mainly restricted to NTU and PKUMMD datasets, lacking validation on larger-scale datasets.
4. The practical contribution of the high-order residual term $\mathbf{R}$ has not been explicitly quantified in the experiments.
5. Future work can explore extending this framework to other self-supervised learning domains (e.g., video understanding, point cloud analysis).

## Related Work & Insights

- **MAE / MAMP**: Typical representations of generative pre-training. IGM incorporates idempotent constraints onto these structures to compensate for deficiencies in recognition ability.
- **MCR² (Maximal Coding Rate Reduction)**: Provides the theoretical tool of lossy coding length as an entropy proxy.
- **Spectral Contrastive Learning**: IGM proves that idempotent generative models are equivalent to it, bringing together two seemingly distinct paradigms.
- **Insights**: In self-supervised learning for other modalities (e.g., image, video), imposing idempotent constraints on generative models could be investigated to enhance representation performance.

## Rating

- **Novelty**: ★★★★★ — Outstanding theoretical contributions, bridging generative models and contrastive learning.
- **Experimental Thoroughness**: ★★★★☆ — Extensive ablation studies, but evaluated on limited datasets.
- **Writing Quality**: ★★★★☆ — Clear derivation process, though mathematical notations are dense in parts.
- **Value**: ★★★★☆ — Directly valuable in the field of skeleton-based action recognition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion](macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)
- [\[ICCV 2025\] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition](../../ICCV2025/image_generation/bridging_the_skeleton_text_modality_gap_diffusion_powered_modality_alignment_for.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning](diffusion-driven_data_replay_a_novel_approach_to_combat_forgetting_in_federated_.md)
- [\[ECCV 2024\] LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning](lego_learning_egocentric_action_frame_generation_via_vi.md)

</div>

<!-- RELATED:END -->
