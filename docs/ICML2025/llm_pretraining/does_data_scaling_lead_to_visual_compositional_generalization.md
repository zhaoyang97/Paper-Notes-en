---
title: >-
  [Paper Note] Does Data Scaling Lead to Visual Compositional Generalization?
description: >-
  [ICML 2025][LLM Pretraining][Compositional Generalization] This paper systematically investigates the impact of data scale and data diversity on the compositional generalization of visual models through controlled experiments. The authors find that data diversity, rather than data volume, is the key driver of compositional generalization. They also prove that when representations exhibit a linearly factored structure, only 2 compositional samples per concept value are require…
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Compositional Generalization"
  - "Data Diversity"
  - "Linearly Factored Representations"
  - "Visual Reasoning"
  - "Pre-trained Models"
date: 2026-05-08
content_hash: 7970487efd02df6c
---

# Does Data Scaling Lead to Visual Compositional Generalization?

**Conference**: ICML 2025  
**arXiv**: [2507.07102](https://arxiv.org/abs/2507.07102)  
**Code**: [https://github.com/oshapio/visual-compositional-generalization](https://github.com/oshapio/visual-compositional-generalization)  
**Area**: LLM Pre-training  
**Keywords**: Compositional Generalization, Data Diversity, Linearly Factored Representations, Visual Reasoning, Pre-trained Models

## TL;DR
This paper systematically investigates the impact of data scale and data diversity on the compositional generalization of visual models through controlled experiments. The authors find that data diversity, rather than data volume, is the key driver of compositional generalization. They also prove that when representations exhibit a linearly factored structure, only 2 compositional samples per concept value are required for perfect generalization.

## Background & Motivation
**Background**: The dominant paradigm asserts that "scaling up data and model size will boost performance." However, compositional understanding—understanding novel scenes by composing known simple concepts—is a cornerstone of human intelligence, yet existing vision models perform poorly.  

**Limitations of Prior Work**: Even in large-scale datasets like LAION-400M, there is a severe sparsity in concept composition coverage. The compositional explosion results in most possible combinations being missing from the training set.  

**Key Challenge**: Scaling laws predict that scaling up data can consistently improve performance, but the concept composition space grows exponentially, meaning that simply increasing data volume may not resolve compositional sparsity.  

**Goal**: To precisely answer "Can vision models generalize compositionally? Under what conditions?"  

**Key Insight**: Designing an $(n, k)$ framework to parameterize and control concept space complexity and training data diversity.  

**Core Idea**: Compositional generalization is not solved by "scaling data volume," but by increasing the diversity of concept compositions to force the model to discover a linearly factored representation structure.

## Method

### Overall Architecture
Two setups: (1) training a ResNet-50 from scratch while controlling $(n, k)$ parameters; (2) evaluating the compositional generalization of pre-trained foundation models (e.g., DINO, CLIP). Three metrics: generalization accuracy (ID vs OOD), representation structure (linearity $R^2$, orthogonality, decodability), and theoretical guarantees.

### Key Designs

1. **(n,k) Experimental Framework**: 

    - **Function**: Parameterize and control concept space complexity and compositional diversity.
    - **Mechanism**: $n =$ number of possible values per concept, $k =$ number of training combinations in which each concept value appears. Among $n^2$ possible combinations, only $nk$ are used for training.
    - **Design Motivation**: Independently control $n$ and $k$ to precisely disentangle the contributions of "data volume" and "data diversity."

2. **Linearly Factored Embeddings**: 

    - **Function**: Detect whether the model has learned a "concept additive" structure.
    - **Mechanism**: Composite concept representation = vector sum of individual concept representations: $u_c = u_{c1} + u_{c2}$.
    - **Measure**: $R^2$ coefficient measuring the fit between actual representations and linear reconstructions.

3. **Three-stage Feature Learning Dynamics**: 

    - Phase 1 (0-10% coverage): Spurious features, decoded accuracy $< 80\%$, zero-shot generalization at random levels.
    - Phase 2 (25-75% coverage): Discriminative but non-linearly factored, zero-shot accuracy $60\% \text{--} 80\%$.
    - Phase 3 (75-100% coverage): Strongly linear ($R^2 > 0.8$) and orthogonal, zero-shot accuracy $> 90\%$.

4. **Minimum Compositional Learning Proposition (Proposition 4.1)**: 

    - Proving that under ideal linearly factored representations, $k=2$ is sufficient for perfect generalization to all unseen combinations.
    - Condition: The joint span of concept representations has a dimension of $2n-1$.

### Loss & Training
- Training from scratch: ResNet-50 + bilinear classification head, cross-entropy loss predicting both concepts simultaneously.
- Pre-training evaluation: Frozen features + MLP probe, oracle model selection.
- Datasets: DSprites, 3DShapes, PUG, Colored-MNIST, FSprites.

## Key Experimental Results

### Main Results

| Dataset | Setup (n,k) | ID Accuracy | OOD Accuracy | Accuracy Drop |
|--------|-----------|--------|---------|---------|
| CMNIST | (3,2) | ~100% | ~22% | -78% |
| FSprites | (3,2) | ~100% | Varies | Large variation across concepts |
| DSprites | (3,2) | ~100% | Varies | Varies from 3% to 40% |
| Shapes3D | (3,2) | ~100% | Varies | Some only show -17% |

| Pre-trained Model | Strong Concepts | Weak Concepts |
|-----------|---------|---------|
| CLIP-ViT-L/14 | Color-based concepts (best) | Shape-based (weaker) |
| DINOv2-ViT-L/14 | Shape/scale/orientation (best) | Color-based (weaker) |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Increase ID data volume by 4x | No improvement in OOD accuracy | Proves "scaling volume" is ineffective |
| Increase $n$ (more concept values) | OOD accuracy improves | Diversity drives generalization |
| Increase $k$ (more combinations) | OOD accuracy improves | Compositional diversity is also effective |
| ViT vs ResNet | ViT has no advantage | Not an architectural issue |

### Key Findings
- A 4x increase in ID data volume yields almost no improvement in OOD generalization.
- Linearly factored structures only emerge naturally under high-diversity training.
- The three-stage learning dynamics reveal a new understanding of simplicity bias.
- Pre-trained models exhibit partial linear factorization but are far from perfect.
- CLIP excels at color while DINOv2 excels at shape—compositional ability exhibits "concept selectivity."

## Highlights & Insights
- Carefully designed controlled experiments enable causal claims (rather than mere correlation analysis).
- Proposition 4.1: $k=2$ is sufficient (under ideal conditions), yielding an elegant theoretical result.
- The three-stage feature learning dynamics draw an interesting analogy to grokking and phase transitions.
- Offers a strong counterexample to the popular belief that "Scaling Laws can solve everything."

## Limitations & Future Work
- Only considers the combination of two concepts; combining more concepts might be more challenging.
- Datasets are mostly synthetic data.
- Proposition 4.1's assumptions might not hold when the number of concepts is large.
- Lacks deep analysis of compositional generalization in natural images.

## Related Work & Insights
- Trager et al. (2023) and Stein et al. (2024) discovered partial linearly factored structures in large models.
- The simplicity bias work by Geirhos et al. (2020) explains the models' tendency to learn shortcut/spurious features.
- Insight: Dataset curation is more important than dataset scale.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](bayesian_neural_scaling_law_extrapolation_with_prior-data_fitted_networks.md)
- [\[NeurIPS 2025\] Differentiable Hierarchical Visual Tokenization](../../NeurIPS2025/llm_pretraining/differentiable_hierarchical_visual_tokenization.md)
- [\[ICML 2025\] Scaling Inference-Efficient Language Models](scaling_inference-efficient_language_models.md)
- [\[ICML 2026\] Explaining Data Mixing Scaling Laws](../../ICML2026/llm_pretraining/explaining_data_mixing_scaling_laws.md)
- [\[NeurIPS 2025\] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?](../../NeurIPS2025/llm_pretraining/does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)

</div>

<!-- RELATED:END -->
