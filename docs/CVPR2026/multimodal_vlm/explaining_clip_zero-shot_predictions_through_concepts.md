---
title: >-
  [Paper Note] Explaining CLIP Zero-shot Predictions Through Concepts
description: >-
  [CVPR 2026][Multimodal VLM][CLIP] This paper proposes EZPC, which maps CLIP image-text embeddings into an interpretable concept space by learning a linear projection matrix. While maintaining almost no loss in zero-shot classification accuracy (H-mean gap of only ~1% on CIFAR-100/CUB/ImageNet-100), it provides faithful explanations based on human-understandable concepts for CLIP predictions with a negligible inference overhead increase of about 0.1ms.
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "CLIP"
  - "Zero-shot Classification"
  - "Concept Bottleneck Models"
  - "Interpretability"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: d4f46eecc3c252a1
---

# Explaining CLIP Zero-shot Predictions Through Concepts

**Conference**: CVPR 2026  
**arXiv**: [2603.28211](https://arxiv.org/abs/2603.28211)  
**Code**: [https://github.com/oonat/ezpc](https://github.com/oonat/ezpc)  
**Area**: Information Retrieval  
**Keywords**: CLIP, Zero-shot Classification, Concept Bottleneck Models, Interpretability, Vision-Language Models

## TL;DR
This paper proposes EZPC, which maps CLIP image-text embeddings into an interpretable concept space by learning a linear projection matrix. While maintaining almost no loss in zero-shot classification accuracy (H-mean gap of only ~1% on CIFAR-100/CUB/ImageNet-100), it provides faithful explanations based on human-understandable concepts for CLIP predictions with a negligible inference overhead increase of about 0.1ms.

## Background & Motivation

1. **Background**: Vision-Language Models (VLMs) like CLIP have achieved great success in zero-shot image recognition by aligning images and text into a shared semantic space, enabling the recognition of arbitrary categories without task-specific training. Simultaneously, Concept Bottleneck Models (CBMs) provide interpretable reasoning via an intermediate layer of human-defined concepts but rely on concept annotations and fail to generalize to unseen categories.

2. **Limitations of Prior Work**: The high-dimensional embeddings of CLIP are entangled black boxes—users cannot understand why a model associates an image with a specific label. While CBMs are interpretable, they require concept supervision and are restricted to a closed world (fixed set of categories). SpLiCE decomposes CLIP embeddings into concept compositions but requires image-wise optimization (59x slower than CLIP), and Z-CBM requires large concept libraries and expensive regressions.

3. **Key Challenge**: Interpretability and open-world generalization capability seem mutually exclusive—CBMs offer interpretability but lack generalization, while CLIP generalizes but lacks interpretability.

4. **Goal**: How can CLIP's zero-shot capabilities be maintained while making its predictions explainable through human-understandable concepts?

5. **Key Insight**: The internal representations of CLIP may already implicitly encode human-understandable semantic structures, requiring only an appropriate projection to "decode" them.

6. **Core Idea**: Learn a single linear projection matrix $A$ to jointly map CLIP image-text embeddings into a predefined concept space, maintaining interpretability with a matching loss and semantic faithfulness with a reconstruction loss.

## Method

### Overall Architecture
The objective of EZPC is straightforward: decompose every zero-shot judgment of CLIP into "which human-readable concepts were activated by this image" without modifying CLIP itself or slowing down inference. It first prepares a set of $m$ textually described concepts (e.g., "has feathers," "made of metal"). Then, it trains only one linear projection matrix $A \in \mathbb{R}^{d \times m}$ to project both the $d$-dimensional CLIP image embedding $v_x$ and text embeddings into this $m$-dimensional concept space. The image becomes a concept activation vector $c_x = v_x A$, and each category also becomes a concept vector $c_k$. Classification is performed by finding the nearest category in the concept space: $\hat{y} = \arg\max_k \langle c_x, c_k \rangle$. Since this is a dot product, the category score is naturally the sum of individual concept contributions $\langle c_x, c_k \rangle = \sum_{j=1}^{m} s_{x,k}^{(j)}$, where $s_{x,k} = c_x \odot c_k$. Thus, "why it was classified as this category" can be read directly by identifying which dimensions in the element-wise product are the largest. Training only optimizes the single matrix $A$, using two losses to ensure interpretability and faithfulness respectively.

### Key Designs

**1. Shared Linear Concept Projection: Moving images and text into concept space via one matrix**

CLIP's high-dimensional embeddings are entangled black boxes that cannot tell users which semantics a match is based on. Instead of per-image solving (the root cause of SpLiCE and Z-CBM being dozens of times slower), EZPC learns a globally shared $A$, passing both the image and all category text through the same projection: $c_x = v_x A$, $C_\mathcal{Y} = T A$. The insistence on a **linear** projection rather than a powerful non-linear mapping is because linearity ensures "explanation" and "decision" are strictly the same thing—the category logit $\langle c_x, c_k\rangle$ is simply the sum of concept scores $s_{x,k}^{(j)}$. The contribution of each dimension can be extracted exactly without the approximate bias of post-hoc attribution. This faithfulness-by-construction is more trustworthy than post-hoc methods like saliency maps, and the cost is just a single matrix multiplication, making inference nearly free.

**2. Match Loss: Anchoring projection columns to real concept directions to prevent drift**

If the projection matrix is optimized without constraints, the column vectors might drift toward directions that classify well numerically but no longer correspond to any human concepts, rendering the explanation invalid. EZPC first uses the CLIP text encoder to encode all concept phrases into $\Phi \in \mathbb{R}^{d \times m}$, initializes $A = \Phi$, and then adds an MSE constraint during training to pull $A$ back toward the concept basis:

$$\mathcal{L}_{\text{match}} = \frac{1}{dm}\sum_{i,j}(A_{ij} - \Phi_{ij})^2$$

This acts as a soft anchor for each concept direction—allowing $A$ to fine-tune for downstream classification while preventing it from straying so far that "the column for concept $j$ no longer represents concept $j$," thus balancing flexibility and interpretability.

**3. Reconstruction Loss: Ensuring consistency between concept space decisions and original CLIP**

Interpretability alone is insufficient; if the model's classification tendency after projection differs from the original CLIP, the explanation is for a different model. EZPC uses KL divergence to force the category distribution in the concept space to align with the distribution in the original CLIP embedding space:

$$\mathcal{L}_{\text{recon}} = \frac{1}{B}\sum_{i=1}^{B} \text{KL}\big(\text{softmax}(c_i C_\mathcal{Y}^\top) \,\|\, \text{softmax}(v_i T^\top)\big)$$

The left side represents the similarity distribution calculated in the concept space after projection, and the right side is the original CLIP distribution $v_i T^\top$. Aligning the former with the latter ensures that the model's ranking of judgments for each image remains basically unchanged after adding the interpretable layer. This is key to EZPC keeping accuracy loss under 1% without sacrificing CLIP's zero-shot capability.

### Loss & Training
The total loss combines the two terms, with $\lambda$ balancing interpretability and faithfulness: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{match}} + \lambda \mathcal{L}_{\text{recon}}$. Most datasets use $\lambda = 1$, while CUB and Places365 use $\lambda = 5$. The concept set reuse vocabulary from LF-CBM generated by GPT-3 and incorporates the large concept pool from ImageNet-1k to expand coverage. All experiments use an 80/20 split of seen/unseen categories to test open-world generalization.

## Key Experimental Results

### Main Results

**Generalized Zero-Shot Performance (Harmonic Mean)**:

| Dataset | CLIP | Z-CBM | SpLiCE | EZPC |
|--------|------|-------|--------|------|
| CIFAR-100 | 0.408 | 0.365 | 0.270 | **0.403** |
| ImageNet-100 | 0.693 | 0.585 | 0.389 | **0.682** |
| CUB | 0.474 | 0.189 | 0.070 | **0.465** |
| ImageNet-1k | 0.530 | 0.462 | 0.300 | 0.481 |
| Places365 | 0.362 | 0.357 | 0.282 | 0.352 |

**Inference Efficiency Comparison**:

| Method | Latency (ms/img) | Overhead Factor |
|------|--------------|----------|
| CLIP | 5.77 | 1.0× |
| Z-CBM | 542.34 | 94.0× |
| SpLiCE | 338.51 | 58.7× |
| EZPC | 5.90 | ~1.0× |

### Ablation Study

| $\lambda$ | Zero-shot Seen | Unseen | GZS H-mean |
|-----------|----------------|--------|------------|
| 0.01 | 0.377 | 0.508 | 0.358 |
| 0.1 | 0.654 | 0.820 | 0.630 |
| 1 | 0.699 | 0.851 | 0.682 |
| 10 | 0.707 | 0.859 | 0.695 |
| 100 | 0.706 | 0.857 | 0.692 |

### Key Findings
- **EZPC maintains a performance gap within 1% of CLIP on most datasets** (CIFAR-100: -0.5%, ImageNet-100: -1.1%, CUB: -0.9%), whereas SpLiCE and Z-CBM often lag by 10-15%.
- **A trade-off exists between quantitative and qualitative results via $\lambda$**: Higher $\lambda$ improves quantitative metrics (better preservation of CLIP distribution), but qualitative analysis shows smaller $\lambda$ (e.g., 1) produces more semantically relevant concept activations.
- **Concept space exhibits strong spatial alignment**: On the Indigo Bunting class in CUB, the Pointing Accuracy for the positive concept "blue-gray body" reaches 96.7%, while the negative concept "red face" is near 0.
- **Cross-dataset transfer is effective**: Projection matrices trained on ImageNet-100 can be directly transferred to CIFAR-100 and CUB with performance close to CLIP.

## Highlights & Insights
- **Minimalist Interpretability Solution**: Achieving concept-level explanation with just a linear projection matrix and near-zero inference overhead (0.1ms) makes it suitable for large-scale deployment. This stands in stark contrast to SpLiCE (59x slower) and Z-CBM (94x slower), which require per-image optimization.
- **Guaranteed Faithfulness of Explanations**: Since concept scores directly constitute the prediction logit ($\langle c_x, c_k \rangle = \sum_j s_{x,k}^{(j)}$), the explanation is faithfulness-by-construction rather than post-hoc attribution. This is more reliable than post-hoc explanation methods like saliency maps.
- **Balanced Design of Match-Reconstruction Dual Objectives**: $\mathcal{L}_{\text{match}}$ preserves interpretability, while $\mathcal{L}_{\text{recon}}$ preserves performance. $\lambda$ controls the balance—this design pattern can be transferred to other tasks requiring a trade-off between interpretability and performance.

## Limitations & Future Work
- **Linear Projection Assumption Limits Expressivity**: Highly non-linear semantic relationships may not be fully captured in the concept space.
- **Dependence on Concept Set Quality**: Interpretability depends on the quality and diversity of the concept vocabulary; biases in the concept set affect the faithfulness of the explanation.
- **Limited to Classification Tasks**: The current method focuses on classification; extending it to multimodal reasoning, VQA, and other tasks is an open problem.
- **Larger Performance Gap on ImageNet-1k** (5%): Information loss during concept decomposition is more pronounced in large-scale settings.
- **Future Directions**: Non-linear concept mapping, adaptive concept discovery, and integration with LLMs for dynamic concept vocabulary expansion.

## Related Work & Insights
- **vs SpLiCE**: SpLiCE sparsely decomposes CLIP embeddings into combinations of concept vectors but requires image-wise optimization and is 59x slower; EZPC learns a unified projection with nearly free inference.
- **vs Z-CBM**: Z-CBM reconstructs embeddings from a concept library for zero-shot CBM but requires large libraries and expensive regressions (94x slower); EZPC is significantly more efficient and performs better.
- **vs LF-CBM**: LF-CBM requires concept labels for training and is a closed-set method; EZPC utilizes the concept set of LF-CBM but achieves open-world zero-shot capability.
- The discovery that CLIP internal representations naturally encode human-alignable structures holds theoretical value for understanding the semantic organization of large-scale pretrained models.

## Rating
- Novelty: ⭐⭐⭐⭐ The linear projection + dual loss scheme is concise and elegant, though the technical depth is limited.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive with 5 datasets, multiple qualitative analyses, cross-domain experiments, and efficiency comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, standardized mathematical notation, and intuitive experimental presentation.
- Value: ⭐⭐⭐⭐ Provides a practical and efficient solution for VLM interpretability with real-world deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SOTA: Self-adaptive Optimal Transport for Zero-Shot Classification with Multiple Foundation Models](sota_self-adaptive_optimal_transport_for_zero-shot_classification_with_multiple_.md)
- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)
- [\[CVPR 2026\] From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition](from_weights_to_concepts_data-free_interpretability_of_clip_via_singular_vector_.md)
- [\[ICML 2026\] Contrastive Spectral Rectification: Test-Time Defense towards Zero-shot Adversarial Robustness of CLIP](../../ICML2026/multimodal_vlm/contrastive_spectral_rectification_test-time_defense_towards_zero-shot_adversari.md)
- [\[CVPR 2026\] From Attraction to Equilibrium: Physics-Inspired Semantic Gravitons for Zero-Shot Anomaly Detection](from_attraction_to_equilibrium_physics-inspired_semantic_gravitons_for_zero-shot.md)

</div>

<!-- RELATED:END -->
