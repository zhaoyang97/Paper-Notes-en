---
title: >-
  [Paper Note] Precise, Fast, and Low-cost Concept Erasure in Value Space: Orthogonal Complement Matters
description: >-
  [CVPR 2025][Image Generation][Concept Erasure] This paper proposes AdaVD (Adaptive Value Decomposer), a training-free concept erasure method for T2I diffusion models. By projecting the original prompt onto the orthogonal complement space of the target concept within the value space of the cross-attention, and introducing an adaptive shift factor, it achieves precise erasure of the target concept while minimally affecting non-target content.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Concept Erasure"
  - "Diffusion Model Safety"
  - "Orthogonal Complement Space"
  - "Training-Free Method"
  - "Prior Preservation"
date: 2026-05-08
content_hash: c907ae17ed5e3397
---

# Precise, Fast, and Low-cost Concept Erasure in Value Space: Orthogonal Complement Matters

**Conference**: CVPR 2025  
**arXiv**: [2412.06143](https://arxiv.org/abs/2412.06143)  
**Code**: [GitHub](https://github.com/WYuan1001/AdaVD)  
**Area**: Image Generation  
**Keywords**: Concept Erasure, Diffusion Model Safety, Orthogonal Complement Space, Training-Free Method, Prior Preservation

## TL;DR

This paper proposes AdaVD (Adaptive Value Decomposer), a training-free concept erasure method for T2I diffusion models. By projecting the original prompt onto the orthogonal complement space of the target concept within the value space of the cross-attention, and introducing an adaptive shift factor, it achieves precise erasure of the target concept while minimally affecting non-target content.

## Background & Motivation

- **Safety risks of T2I models**: Text-to-image diffusion models may generate copyrighted, offensive, or unsafe content due to noisy internet annotations in the training data. Standard retraining from scratch is prohibitively expensive, which necessitates low-cost concept erasure techniques.
- **Trade-off between erasure efficacy and prior preservation**: Concept erasure requires balancing two objectives—(1) precisely erasing the target concept (erasure efficacy) and (2) minimizing the impact on non-target content (prior preservation).
- **Limitations of training-based methods**: Methods like ESD and UCE require separate fine-tuning for each new concept, which is unsuitable for real-time scenarios (such as online platforms needing to instantly block newly emerging copyrighted concepts). Furthermore, regularization makes it difficult to simultaneously ensure both erasure efficacy and prior preservation.
- **Limitations of existing training-free methods**: Negative prompts exhibit weak erasure efficacy; SLD severely damages non-target priors; SuppressEOT requires manual specification of target token positions, which lacks automation.
- **Key Insight**: In cross-attention, the Key controls "where" (spatial layout) while the Value controls "what" (content). Since concept erasure essentially modifies the visual content, operations should be performed within the Value space. Projecting the value vector of the original prompt onto the orthogonal complement space of the target concept's value vector can precisely remove the target semantics.

## Method

### Overall Architecture

AdaVD performs three main operations in each cross-attention layer of the UNet:
1. **Token-wise target embedding preprocessing**: Duplicates key token embeddings of the target concept to enhance target semantic signals.
2. **Orthogonal value decomposition**: Projects the value vector of the original prompt onto the orthogonal complement space of the target value.
3. **Adaptive shift adjustment**: Dynamically regulates the erasure strength based on the alignment intensity between each token and the target concept.

### Key Designs

**1. Token-wise target embedding preprocessing**
- **Function**: Enhances the semantic representation of the target concept in the value space, making orthogonal decomposition more precise.
- **Mechanism**: Duplicates the last subject token of the target concept (e.g., "snoopy" or "gogh") to all positions except [SOT]. Leveraging CLIP's causal attention mechanism, the last subject token has already aggregated the complete prompt information.
- **Design Motivation**: The original target embedding contains many [EOT] padding tokens, which dilutes the semantic signal; copying the key token strengthens the semantics and improves the accuracy of the orthogonal decomposition.

**2. Orthogonal complement projection (Core Operation)**
- **Function**: Precisely removes the semantic components of the target concept from the visual content of the original prompt.
- **Mechanism**: For each token position $j$, the original value vector $\boldsymbol{v}^j$ is projected onto the orthogonal complement space of the target value vector $\boldsymbol{v}_t^j$: $\boldsymbol{v}_r^j = \boldsymbol{v}^j - \frac{\boldsymbol{v}_t^{j\top}\boldsymbol{v}^j}{\boldsymbol{v}_t^{j\top}\boldsymbol{v}_t^j}\boldsymbol{v}_t^j$. For multi-concept erasure, Gram-Schmidt orthogonalization is first applied to multiple target values before projecting onto the joint orthogonal complement space.
- **Design Motivation**: Orthogonal complement projection mathematically guarantees retaining the maximum amount of information after removing the component in the target direction, serving as the optimal semantic separation operation.

**3. Adaptive Shift Factor**
- **Function**: Distinguishes between strong and weak alignment of each token with the target concept, preventing over-erasure of non-target content.
- **Mechanism**: Computes the projection coefficient $\alpha_j = \frac{\boldsymbol{v}_t^{j\top}\boldsymbol{v}^j}{\boldsymbol{v}_t^{j\top}\boldsymbol{v}_t^j}$ for each token, which measures its alignment with the target concept. Strongly aligned tokens (e.g., prompt words directly containing the target concept) are fully erased, while weakly aligned tokens (generic semantics like "[EOT]") retain more of their original information.
- **Design Motivation**: All tokens align to some degree with the target concept (especially generic tokens like [EOT]). Completely projecting every token would degrade prior knowledge. The adaptive shift factor achieves a fine-grained "tiered erasure".

### Loss & Training

AdaVD is a training-free method and does not involve loss function optimization. The core operation directly modifies the cross-attention value matrix during inference.

## Key Experimental Results

### Main Results: Comparison of Single-Concept Erasure Performance

| Method | Type | Erasure Efficacy | Prior Preservation (CLIP-T↑) | Runtime |
|------|------|:---:|:---:|:---:|
| ESD | Training | High | Low | Slow |
| UCE | Training | High | Medium | Slow |
| SLD | Training-Free | Medium | Low | Fast |
| **AdaVD** | **Training-Free** | **High/Near-Optimal** | **Optimal (2-10x↑)** | **Fast** |

### Multi-Concept Erasure

| Method | Efficacy of Simultaneous 5-Concept Erasure | Prior Preservation |
|------|:---:|:---:|
| ESD | Medium | Poor |
| SLD | Poor | Poor |
| **AdaVD** | **High** | **Optimal** |

### Key Findings

- AdaVD improves prior preservation by **2-10 times** compared to the runner-up while maintaining optimal or near-optimal erasure efficacy.
- This is the first time a training-free method achieves or even surpasses the erasure performance of training-based methods.
- The shift factor is crucial for prior preservation—removing it significantly degrades the generation quality of non-target concepts.
- The method can be directly transferred to different T2I models, such as SDXL, DreamShaper, and Chilloutmix.
- It supports a wide range of erasure scenarios, including copyrighted concepts (Snoopy), artistic style (Van Gogh), and unsafe concepts (NSFW).

## Highlights & Insights

1. **Mathematical Elegance**: Orthogonal complement projection is one of the most fundamental operations in linear algebra, yet its application in concept erasure is exceptionally effective—optimally separating target and non-target semantics.
2. **Training-free + High Precision**: For the first time, a training-free framework achieves prior preservation capabilities that surpass training-based methods.
3. **Plug-and-play**: Requires no fine-tuning of model parameters, enabling real-time erasure of any new concept, which is highly suitable for online deployment.
4. **Scalable Multi-concept Erasure**: Naturally scales to the simultaneous erasure of multiple concepts using Gram-Schmidt orthogonalization.

## Limitations & Future Work

- The erasure efficacy relies heavily on the semantic representation quality of target concepts provided by the CLIP text encoder.
- Orthogonal complement projection is a linear operation in the value space and may not fully handle non-linear entanglement between concepts.
- In extreme cases (where target and non-target concepts have highly overlapping semantics), leakage may still occur.
- Future work could incorporate spatial information from attention maps to achieve finer-grained, region-level erasure.

## Related Work & Insights

- **ESD (Erased Stable Diffusion)**: A pioneer in training-based erasure, but does not consider prior preservation.
- **SLD (Safe Latent Diffusion)**: A pioneer in training-free erasure, but exhibits poor prior preservation.
- **SAFREE**: A concurrent work that also employs orthogonal decomposition, but operates in the text embedding space.
- **Cross-attention Analysis**: The understanding that Key=Where and Value=What serves as the theoretical foundation for this method's design.
- **Insight**: Combining classic mathematical tools (orthogonal projection) with the internal mechanisms of deep models (attention values) can yield elegant and highly effective solutions.

## Rating

⭐⭐⭐⭐ — The method design is elegant and concise; the core concept of "orthogonal complement projection" is intuitively clear and mathematically rigorous. Prior preservation performance is substantially ahead of baselines. The training-free nature makes it highly practical. The adaptive design of the shift factor successfully addresses the critical issue of over-erasure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Orthogonal Concept Erasure for Diffusion Models](../../ICML2026/image_generation/orthogonal_concept_erasure_for_diffusion_models.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[CVPR 2025\] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fade_fine_grained_erasure_diffusion.md)
- [\[CVPR 2026\] MapRoute: Semantic Routing for Precise Concept Erasure with Mapper](../../CVPR2026/image_generation/maproute_semantic_routing_concept_erasure.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
