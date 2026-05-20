---
title: >-
  [Paper Note] Explaining CLIP Zero-shot Predictions Through Concepts
description: >-
  [CVPR 2026][Information Retrieval & RAG][CLIP] This paper proposes EZPC, which learns a linear projection matrix $A$ to jointly map CLIP image and text embeddings into an interpretable concept space. The method provides…
tags:
  - "CVPR 2026"
  - "Information Retrieval & RAG"
  - "CLIP"
  - "Zero-shot Classification"
  - "Concept Bottleneck Model"
  - "Interpretability"
  - "Vision-Language Model"
date: 2026-05-08
content_hash: 6f1c4464472d5950
---

# Explaining CLIP Zero-shot Predictions Through Concepts

**Conference**: CVPR 2026
**arXiv**: [2603.28211](https://arxiv.org/abs/2603.28211)  
**Code**: [https://github.com/oonat/ezpc](https://github.com/oonat/ezpc)  
**Area**: Information Retrieval
**Keywords**: CLIP, Zero-shot Classification, Concept Bottleneck Model, Interpretability, Vision-Language Model

## TL;DR
This paper proposes EZPC, which learns a linear projection matrix $A$ to jointly map CLIP image and text embeddings into an interpretable concept space. The method provides faithful, human-understandable explanations for CLIP predictions with negligible accuracy loss (H-mean gap of ~1% on CIFAR-100/CUB/ImageNet-100) and an inference overhead of only ~0.1ms.

## Background & Motivation

1. **Background**: Vision-language models (VLMs) such as CLIP have achieved remarkable success in zero-shot image recognition by aligning images and text in a shared semantic space, enabling recognition of arbitrary categories without task-specific training. Concept Bottleneck Models (CBMs), meanwhile, provide interpretable reasoning through a human-defined concept layer, but require concept annotations and cannot generalize to unseen categories.

2. **Limitations of Prior Work**: CLIP's high-dimensional embeddings are entangled black boxes—users cannot understand why the model associates a given image with a particular label. Although CBMs are interpretable, they require concept supervision and are limited to a closed-world setting. SpLiCE decomposes CLIP embeddings into concept combinations but requires per-image optimization (59× slower than CLIP), while Z-CBM demands a large concept vocabulary and expensive regression.

3. **Key Challenge**: Interpretability and open-world generalization appear to be mutually exclusive—CBMs offer interpretability but lack generalization, while CLIP generalizes but is not interpretable.

4. **Goal**: How can CLIP's zero-shot capability be preserved while making its predictions explainable through human-understandable concepts?

5. **Key Insight**: CLIP's internal representations may already implicitly encode human-understandable semantic structure, requiring only an appropriate projection to "decode" it.

6. **Core Idea**: Learn a single linear projection matrix $A$ to jointly map CLIP image and text embeddings into a predefined concept space, using a matching loss to preserve interpretability and a reconstruction loss to maintain semantic fidelity.

## Method

### Overall Architecture
The EZPC pipeline: (1) define a set of $m$ human-understandable concepts (e.g., "has feathers", "made of metal"); (2) learn a projection matrix $A \in \mathbb{R}^{d \times m}$ mapping CLIP's $d$-dimensional embedding space to an $m$-dimensional concept space; (3) perform zero-shot classification in concept space via the dot product between the image concept vector $c_x = v_x A$ and the class concept vector $c_k$; (4) decompose each concept's contribution via the element-wise product $s_{x,k} = c_x \odot c_k$ to provide faithful explanations.

### Key Designs

1. **Shared Concept Projection**:

    - **Function**: Uniformly maps both image and text embeddings into an interpretable concept space.
    - **Mechanism**: A learnable projection matrix $A \in \mathbb{R}^{d \times m}$ is defined, where each column corresponds to a concept direction. Image concept activations are computed as $c_x = v_x A$ and class concept activations as $C_\mathcal{Y} = T A$. Classification is performed in concept space via $\hat{y} = \arg\max_k \langle c_x, c_k \rangle$. Since $\langle c_x, c_k \rangle = \sum_{j=1}^{m} s_{x,k}^{(j)}$, the contribution of each concept dimension to the prediction is directly decomposable.
    - **Design Motivation**: Using a single unified projection rather than per-image optimization ensures high efficiency; the linear projection guarantees explanation faithfulness—concept scores directly constitute classification logits, so explanations and the decision process are fully consistent.

2. **Matching Loss**:

    - **Function**: Ensures the columns of the projection matrix $A$ remain aligned with known concept embeddings, preserving interpretability.
    - **Mechanism**: All concept phrases are encoded with the CLIP text encoder to obtain $\Phi \in \mathbb{R}^{d \times m}$. $A$ is initialized as $\Phi$ and constrained via the MSE loss $\mathcal{L}_{\text{match}} = \frac{1}{dm}\sum_{i,j}(A_{ij} - \Phi_{ij})^2$ to prevent it from drifting away from the concept basis.
    - **Design Motivation**: Without this constraint, $A$ may drift during training toward directions that no longer correspond to interpretable concepts. The anchoring loss strikes a balance between flexibility and interpretability.

3. **Reconstruction Loss**:

    - **Function**: Ensures the similarity distribution in concept space is consistent with that in CLIP's original embedding space.
    - **Mechanism**: KL divergence is used to measure the discrepancy between the concept-space distribution and CLIP's original distribution: $\mathcal{L}_{\text{recon}} = \frac{1}{B}\sum_{i=1}^{B} \text{KL}(\text{softmax}(c_i C_\mathcal{Y}^\top) \| \text{softmax}(v_i T^\top))$
    - **Design Motivation**: Ensures that classification decisions after concept decomposition remain consistent with CLIP's original predictions—i.e., semantic fidelity is maintained without altering the model's core judgments by adding an interpretable layer.

### Loss & Training
- Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{match}} + \lambda \mathcal{L}_{\text{recon}}$
- $\lambda = 1$ for most datasets; $\lambda = 5$ for CUB and Places365.
- The concept set is drawn from the GPT-3-generated concept vocabulary used in LF-CBM, augmented with a large ImageNet-1k concept pool for broader coverage.
- All experiments use an 80/20 seen/unseen class split.

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

| Method | Latency (ms/img) | Overhead |
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
- **EZPC closes within 1% of CLIP on most datasets** (CIFAR-100: −0.5%, ImageNet-100: −1.1%, CUB: −0.9%), while SpLiCE and Z-CBM frequently lag by 10–15%.
- **A quantitative–qualitative trade-off exists for $\lambda$**: larger $\lambda$ improves quantitative metrics (better preservation of CLIP's distribution), but qualitative analysis shows that smaller $\lambda$ (e.g., 1) yields more semantically relevant concept activations.
- **The concept space exhibits good spatial alignment**: on the Indigo Bunting class in CUB, the Pointing Accuracy for the positive concept "blue-grey body" reaches 96.7%, while the negative concept "red face" scores near zero.
- **Cross-dataset transfer is effective**: a projection matrix trained on ImageNet-100 transfers directly to CIFAR-100 and CUB with performance close to CLIP.

## Highlights & Insights
- **A minimalist interpretability solution**: a single linear projection matrix delivers concept-level explanations with near-zero inference overhead (~0.1ms), making it suitable for large-scale deployment—in stark contrast to SpLiCE (59× slower) and Z-CBM (94× slower), both of which require per-image optimization.
- **Guaranteed faithfulness of explanations**: since concept scores directly constitute prediction logits ($\langle c_x, c_k \rangle = \sum_j s_{x,k}^{(j)}$), explanations are constructively faithful rather than post-hoc attributions—a stronger guarantee than saliency-map-based methods.
- **Balanced dual-objective design**: $\mathcal{L}_{\text{match}}$ preserves interpretability, $\mathcal{L}_{\text{recon}}$ preserves performance, and $\lambda$ controls the trade-off. This design pattern is transferable to other tasks requiring a balance between interpretability and accuracy.

## Limitations & Future Work
- **The linear projection assumption limits expressive capacity**: highly nonlinear semantic relationships may not be fully captured in the concept space.
- **Dependence on concept set quality**: interpretability relies on the quality and diversity of the concept vocabulary; biases in the concept set affect explanation fidelity.
- **Restricted to classification tasks**: the current method focuses on classification; extending it to multimodal reasoning, VQA, and related tasks remains an open problem.
- **Larger performance gap on ImageNet-1k** (5%): information loss from concept decomposition becomes more pronounced at scale.
- **Future directions**: nonlinear concept mappings, adaptive concept discovery, and integration with LLMs for dynamic concept vocabulary expansion.

## Related Work & Insights
- **vs. SpLiCE**: SpLiCE sparsely decomposes CLIP embeddings into combinations of concept vectors but requires per-image optimization, making it 59× slower. EZPC learns a unified projection with near-free inference.
- **vs. Z-CBM**: Z-CBM reconstructs embeddings from a concept library for zero-shot CBM but requires a large concept pool and expensive regression (94× slower). EZPC is substantially more efficient and achieves better performance.
- **vs. LF-CBM**: LF-CBM requires concept-annotated training and is a closed-set method. EZPC leverages LF-CBM's concept set while enabling open-world zero-shot generalization.
- The finding that CLIP's internal representations naturally encode human-alignable structure has theoretical value for understanding the semantic organization of large-scale pretrained models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The linear projection + dual-loss formulation is elegant and concise, though limited in technical depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five datasets, multiple qualitative analyses, cross-domain experiments, and efficiency comparisons constitute a comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical structure, rigorous mathematical notation, and intuitive experimental presentation.
- **Value**: ⭐⭐⭐⭐ Provides a practical and efficient solution for VLM interpretability with real deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](../../ICLR2026/information_retrieval/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)
- [\[AAAI 2026\] OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description](../../AAAI2026/information_retrieval/oad-promoter_enhancing_zero-shot_vqa_using_large_language_models_with_object_att.md)
- [\[NeurIPS 2025\] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](../../NeurIPS2025/information_retrieval/worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)
- [\[NeurIPS 2025\] SuperCLIP: CLIP with Simple Classification Supervision](../../NeurIPS2025/information_retrieval/superclip_clip_with_simple_classification_supervision.md)
- [\[ICCV 2025\] External Knowledge Injection for CLIP-Based Class-Incremental Learning](../../ICCV2025/information_retrieval/external_knowledge_injection_for_clip-based_class-incremental_learning.md)

</div>

<!-- RELATED:END -->
