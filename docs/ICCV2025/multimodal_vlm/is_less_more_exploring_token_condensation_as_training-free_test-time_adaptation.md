---
title: >-
  [Paper Note] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation
description: >-
  [ICCV 2025][Multimodal VLM][Test-time adaptation] This paper proposes Token Condensation as Adaptation (TCA), a training-free test-time adaptation method that leverages a Domain-aware Token Reservoir (DTR) to guide cross…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Test-time adaptation"
  - "Token pruning and merging"
  - "CLIP"
  - "Training-free"
  - "Vision-language models"
date: 2026-05-08
content_hash: 03e907eea0ff0df0
---

# Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation

**Conference**: ICCV 2025
**arXiv**: [2410.14729](https://arxiv.org/abs/2410.14729)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: Test-time adaptation, Token pruning and merging, CLIP, Training-free, Vision-language models

## TL;DR

This paper proposes Token Condensation as Adaptation (TCA), a training-free test-time adaptation method that leverages a Domain-aware Token Reservoir (DTR) to guide cross-head token pruning/merging and logits self-correction. Without modifying model parameters, TCA improves cross-dataset performance of CLIP/SigLIP variants by up to 21.4% while reducing GFLOPs by 12.2%–48.9%.

## Background & Motivation

Vision-language models (VLMs) such as CLIP demonstrate strong zero-shot generalization, yet often suffer performance degradation on specific downstream datasets. Existing test-time adaptation (TTA) methods exhibit the following limitations:

**Conventional TTA is computationally expensive**: Methods such as Tent and SAR require updating model parameters (e.g., batch normalization layers) and rely on large batch sizes (e.g., 256) to stabilize the adaptation process, making them impractical for the large parameter sets of VLMs.

**Test-time prompt tuning (TPT) also has limitations**: TPT refines text-visual feature alignment by learning compact task-specific context prompts, but focuses primarily on text input refinement while neglecting visual distribution shift. It also depends on external source data or extensive data augmentation (e.g., 60× AugMix), causing GFLOPs to spike from 17.59 to 1108.61.

**A key observation**: The authors find that selectively pruning low-attention tokens not only preserves performance but can also enhance performance on unseen datasets. This is attributed to two types of tokens that introduce visual-text misalignment: (1) category-irrelevant background tokens that mislead the model toward non-essential regions; and (2) category-ambiguous object tokens (e.g., animal fur textures) that overlap across classes and disperse visual embeddings.

**Limitations of existing token reduction methods**: Although methods such as EViT and ToME improve efficiency, they tend to sacrifice in-distribution (e.g., ImageNet-1K) performance when reducing tokens and cannot achieve "free lunch" adaptation.

## Method

### Overall Architecture

TCA is a training-free online adaptation framework consisting of three core components:

1. **Domain-aware Token Reservoir (DTR)**: Maintains representative domain-anchor tokens as stable references for adaptation.
2. **Domain-aware Cross-head Token Reduction**: Selectively prunes/merges low-information tokens guided by domain-anchor tokens.
3. **Logits Self-correction**: Refines model predictions using stored domain-anchor tokens.

### Key Designs

1. **Domain-aware Token Reservoir (DTR)**:

    - Function: Maintains a class-organized priority queue $\mathfrak{R} = \{\mathfrak{R}_c\}_{c=1}^C$, storing domain-anchor tokens from all $L$ layers (the \<cls\> token in CLIP, and the pooled vector in SigLIP).
    - Mechanism: Each class buffer $\mathfrak{R}_c$ retains $M$ most reliable domain-anchor tokens, ranked by entropy score: $\mathbf{H}_c(\mathbf{z}_t, \mathbf{t}_c) = -\mathbf{p}_{t,c} \log \mathbf{p}_{t,c}$. Updates occur only when $\arg\max(\mathbf{p}_{t,c}) = c$, ensuring only semantically consistent samples are retained. When the buffer is full, the sample with the highest entropy is replaced.
    - Design Motivation: Empirical evidence shows that over time, low-entropy \<cls\> tokens exhibit increasingly better alignment with text embeddings, making them suitable as domain-level adaptation reference points.

2. **Domain-aware Cross-head Token Reduction**:

    - Function: Performs token pruning and merging between the multi-head self-attention and feed-forward layers.
    - Mechanism:
        - **Domain-aware Token Scoring**: Samples the best-matching domain-anchor token $\mathbf{A}_{c^*}^{l-1}$ from the DTR, concatenates it with the current \<cls\> token, and computes attention: $\text{Attention}([\mathbf{v}_{\text{cls}}^l; \mathbf{A}_{c^*}^{l-1}]\mathbf{W}_Q^h, [\mathbf{V}^l; \mathbf{A}_{c^*}^{l-1}]\mathbf{W}_K^h)$
        - **Cross-head Ranking**: Computes a cross-head average rank score for each token $\mathbf{S}_i^{\text{head}} = \frac{1}{H}\sum_{h=1}^H \text{rank}_h(i)$ rather than naively averaging attention scores, avoiding undue influence from outlier heads.
        - **Two-stage Reduction**: Low-ranked tokens (category-irrelevant background) are pruned first; mid-ranked tokens (category-ambiguous) are then merged via coreset merging.
    - Design Motivation: (1) The generic \<cls\> token may capture semantics unrelated to the target class; concatenating domain-anchor tokens provides historical contextual alignment. (2) Per-head average attention scores are susceptible to outlier heads, whereas cross-head ranking is more robust.

3. **Logits Self-correction**:

    - Function: Compensates for semantic shift introduced by token reduction and refines classification predictions.
    - Mechanism: Computes cross-layer cosine similarity between the current sample's visual \<cls\> token and domain-anchor tokens stored in the DTR, using it as a token-level classifier to correct the original prediction: $\tilde{\mathbf{p}}_{t,c} = \mathbf{p}_{t,c} + \lambda\mathbf{p}_{t,c}^{\text{token}}$, where $\mathbf{p}_{t,c}^{\text{token}} = \frac{1}{M}\sum_{i=1}^M \cos(\mathbf{V}_t^{\text{cls}}, \mathbf{A}_{i,c}^{\text{cls}}) \cdot \mathbf{P} \cdot \mathbb{1}_c$ and $\mathbf{P} = [\exp(\frac{l}{\beta})]_{l=1}^L$ is a layer-wise exponential scaling factor.
    - Design Motivation: Token reduction may introduce semantic shift; the domain knowledge accumulated in the DTR allows prediction correction from a purely visual perspective without modifying model parameters.

### Loss & Training

TCA is a **fully training-free** method and involves no loss functions or training procedures. All operations are executed online at inference time with a batch size of 1 and require no data augmentation.

## Key Experimental Results

### Main Results

| Method | Aug-free | Aircraft | Caltech | Cars | DTD | EuroSAT | Flower | Food | Pets | SUN | UCF | Avg. | GFLOPs |
|--------|----------|----------|---------|------|-----|---------|--------|------|------|-----|-----|------|--------|
| CLIP | ✓ | 23.22 | 93.55 | 66.11 | 45.04 | 50.42 | 66.99 | 82.86 | 86.92 | 65.63 | 65.16 | 64.59 | 17.59 |
| TDA | ✓ | 23.91 | 94.24 | 67.28 | 47.40 | 58.00 | 71.42 | 86.14 | 88.63 | 67.62 | 70.66 | 67.53 | 17.59 |
| TCA R=0.9 | ✓ | **24.87** | 93.63 | 65.33 | 46.16 | **70.43** | **73.33** | 85.31 | **89.53** | 65.92 | **72.38** | **68.69** | **15.45** |

Compared to the CLIP zero-shot baseline, TCA achieves an average improvement of 4.10% on cross-dataset benchmarks while reducing GFLOPs by 12.2%.

### Ablation Study

| Configuration | Avg. Accuracy | Notes |
|---------------|--------------|-------|
| Without DTR (\<cls\> attention only) | 65.17 | EViT R=0.9 baseline |
| Pruning only (no DTR guidance) | 65.17 | No domain-awareness |
| DTR + Pruning | 67.83 | DTR provides domain context |
| DTR + Pruning + Merging | 68.12 | Merging preserves ambiguous token information |
| DTR + Pruning + Merging + Logits correction | **68.69** | Full model, best performance |
| EViT R=0.7 | 62.02 | Aggressive pruning causes large performance drop |
| ToME R=0.7 | 60.33 | Merging cannot compensate for information loss |
| TCA R=0.7 | 66.64 | Remains competitive; reduces GFLOPs by 48.9% |

### Key Findings

- **Most pronounced improvement on EuroSAT**: accuracy improves from 50.42% (CLIP) to 70.43% (TCA, +20%), as satellite images deviate substantially from pretraining data distributions and token reduction effectively removes misleading background tokens.
- **Training-free + reduced computation = dual benefit**: TCA is the only method that simultaneously improves performance and reduces computational cost.
- On the CIFAR-100-C corruption benchmark, TCA outperforms the strongest baseline by up to 21.4%.
- Among DTR update strategies, entropy-ranked priority queues outperform FIFO and similarity-based alternatives.
- The method readily extends to SigLIP and SigLIP v2 by replacing the \<cls\> token with the pooled feature vector.

## Highlights & Insights

- **Novel perspective**: This work is the first to reframe token reduction from an "efficiency tool" to a "training-free adaptation strategy"—a profound insight, as reducing tokens not only lowers computational cost but also improves alignment quality under distribution shift.
- **Cross-head ranking** over averaged attention scores is a simple yet effective improvement that yields substantially greater robustness to outlier heads.
- **Zero additional parameters**: All operations exploit the attention weights and features of the pretrained model without introducing any new parameters.
- Efficient online domain adaptation is achieved via the DTR without requiring large batch sizes or data augmentation.

## Limitations & Future Work

- Hyperparameters $K$ (the layer at which DTR application begins) and $R$ (the token retention ratio) require tuning for different models and datasets.
- DTR capacity $M$ and update strategy have a non-negligible impact on performance, and optimal settings are scenario-dependent.
- On datasets close to the pretraining distribution, such as Food101, TCA yields limited or marginal improvement.
- Validation is currently restricted to image classification; extension to downstream tasks such as detection and segmentation remains unexplored.
- The layer-wise temperature parameter $\beta$ in logits self-correction requires semantic priors for selection.

## Related Work & Insights

- The token pruning concept from EViT is reinterpreted in this work as a domain adaptation tool rather than an efficiency optimization.
- TDA (Training-free Dynamic Adapter) is the closest training-free baseline, but relies on more hyperparameters and incurs higher inference overhead.
- The DTR design is conceptually similar to TDA's positive/negative cache but is more lightweight; TCA additionally exploits token reduction within the attention layers.
- The in-depth analysis of token roles in VLM visual encoders (background tokens vs. ambiguous object tokens) offers a new perspective for understanding distribution shift.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reinterpreting token reduction as an adaptation strategy is an original and insightful contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across 10 datasets and CIFAR-100-C with multiple VLMs; detection tasks are absent.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and empirical analysis is well-grounded.
- Value: ⭐⭐⭐⭐ High practical utility due to the training-free and computation-reducing nature; current applicability is limited to classification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[AAAI 2026\] Panda: Test-Time Adaptation with Negative Data Augmentation](../../AAAI2026/multimodal_vlm/panda_test-time_adaptation_with_negative_data_augmentation.md)

</div>

<!-- RELATED:END -->
