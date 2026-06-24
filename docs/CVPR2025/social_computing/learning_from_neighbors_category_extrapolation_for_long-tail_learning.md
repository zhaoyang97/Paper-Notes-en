---
title: >-
  [Paper Note] Learning from Neighbors: Category Extrapolation for Long-Tail Learning
description: >-
  [CVPR 2025][Social Computing][Long-tail learning] It is discovered that finer-grained category division naturally mitigates the impact of long-tail imbalance. This paper proposes using LLMs to discover fine-grained auxiliary categories semantically related to existing ones, web crawlers to collect images, and a Neighbor-Silencing Loss to prevent auxiliary classes from dominating. This achieves a 16-percentage-point improvement ($41.4\% \to 57.4\%$) on Few-shot classes in Imag…
tags:
  - "CVPR 2025"
  - "Social Computing"
  - "Long-tail learning"
  - "Category extrapolation"
  - "LLM data augmentation"
  - "Neighboring category discovery"
  - "Classifier masking"
date: 2026-05-08
content_hash: 10c58c15b99631a2
---

# Learning from Neighbors: Category Extrapolation for Long-Tail Learning

**Conference**: CVPR 2025  
**arXiv**: [2410.15980](https://arxiv.org/abs/2410.15980)  
**Code**: To be released  
**Area**: Social Computing  
**Keywords**: Long-tail learning, Category extrapolation, LLM data augmentation, Neighboring category discovery, Classifier masking

## TL;DR
It is discovered that finer-grained category division naturally mitigates the impact of long-tail imbalance. This paper proposes using LLMs to discover fine-grained auxiliary categories semantically related to existing ones, web crawlers to collect images, and a Neighbor-Silencing Loss to prevent auxiliary classes from dominating. This achieves a 16-percentage-point improvement ($41.4\% \to 57.4\%$) on Few-shot classes in ImageNet-LT.

## Background & Motivation

**Background**: In long-tail learning, head classes contain a massive number of samples, while tail classes have extremely few. Existing methods include re-sampling, re-weighting, and decoupled training.

**Limitations of Prior Work**: (1) Re-sampling and re-weighting operate only within the existing category space, failing to increase the intrinsic feature diversity of tail classes. (2) Augmented data randomly crawled from the web may introduce noise or even harm performance (e.g., $60.9 \to 56.8$). (3) Fine-grained classification datasets (e.g., iNat18 with 8,142 classes) suffer less from the impact of imbalance than coarse-grained ones (e.g., ImageNet-LT with 1,000 classes), yet no existing methods leverage this phenomenon.

**Key Challenge**: Tail classes require more data, but randomly adding data does not work—the augmented data must be semantically related fine-grained variants to effectively expand the feature space.

**Goal**: Leverage the phenomenon that "finer granularity leads to less sensitivity to long-tail imbalance" to automatically discover fine-grained auxiliary categories to "extrapolate" existing categories and fill the feature space.

**Key Insight**: Use an LLM (GPT-4) to recommend semantically related fine-grained subcategories based on existing class names $\to$ employ web crawlers to collect images $\to$ filter noise using DINOv2 $\to$ apply a Neighbor-Silencing Loss to prevent auxiliary classes from dominating during training $\to$ directly mask auxiliary class weights during inference.

**Core Idea**: Discover semantically adjacent auxiliary fine-grained categories using an LLM + collect data via web crawlers + train with a Neighbor-Silencing Loss + mask auxiliary classes during inference, thereby mitigating long-tail imbalance by filling the feature space through category extrapolation.

## Method

### Overall Architecture
Class names of the existing long-tail dataset $\to$ GPT-4 recommends fine-grained neighboring classes for each class (via structured prompts and in-context learning) $\to$ web crawlers collect candidate images $\to$ DINOv2 filtering (with cosine similarity thresholds $\gamma_1=0.7$ and $\gamma_2=0.98$) $\to$ training with Neighbor-Silencing Loss $\to$ masking auxiliary class weights during inference.

### Key Designs

1. **LLM-driven Fine-grained Category Discovery**:

    - **Function**: Discover fine-grained subcategories semantically related to existing classes.
    - **Mechanism**: Ask GPT-4 questions like "What are the finer-grained categories related to [Class Name]?" using a structured prompt. In-context learning is utilized with examples to ensure format consistency. Tail classes are allocated more neighboring classes (proportional to the ratio of head-to-tail samples).
    - **Design Motivation**: Tail classes require more samples $\to$ more neighboring classes $\to$ more crawled data. Head classes are already sufficient $\to$ a few neighboring classes are sufficient.

2. **DINOv2 Data Filtering**:

    - **Function**: Filter effective samples from noisy crawled web images.
    - **Mechanism**: Dual-threshold filtering—cosine similarity with the category prototype $> \gamma_1=0.7$ (to ensure semantic relevance) and $< \gamma_2=0.98$ (to exclude near-duplicates). The feature space of DINOv2 is more suitable for fine-grained filtering than CLIP.
    - **Design Motivation**: Raw web-crawled data is highly noisy. Ablation studies show that training without filtering (Random Data, RD) actually degrades performance ($60.9 \to 56.8$).

3. **Neighbor-Silencing CE**:

    - **Function**: Prevent auxiliary classes from suppressing the original classes during training.
    - **Mechanism**: Down-weight the logit interactions between the original class $i$ and its auxiliary neighboring class $j$: $\lambda_s = 0.1$. This prevents the model from overly distinguishing between the original class and its neighboring classes, thereby maintaining the decision boundaries of the original classes.
    - **Design Motivation**: Without NS-CE, auxiliary classes compete with original classes, degrading the accuracy of the original classes.

4. **Classifier Masking during Inference**:

    - **Function**: Remove auxiliary classes so the model only predicts among the original classes.
    - **Mechanism**: Directly set auxiliary class weights to zero/mask them, avoiding classifier retraining. Ablation studies show that simple masking outperforms retraining the classifier.
    - **Design Motivation**: Auxiliary classes serve only as "scaffolding" during training and are not needed during inference.

### Loss & Training
The standard CE is replaced with NS-CE (which downweights interactions of auxiliary classes). Three initializations are supported: training from scratch, CLIP, and DINOv2.

## Key Experimental Results

### Main Results

| Setting | Overall | Many | Med | Few |
|------|---------|------|------|------|
| ImageNet-LT Baseline | 60.9 | 72.9 | 56.8 | 41.4 |
| **+Ours (scratch)** | **68.2** (+7.3) | 74.5 | 66.2 | **57.4** (+16.0) |
| CLIP initialization | 74.0 | - | - | - |
| **+Ours (CLIP)** | **77.3** (+3.5) | - | - | - |
| DINOv2 initialization | 79.6 | - | - | - |
| **+Ours (DINOv2)** | **82.0** (+2.4) | - | - | - |

### Ablation Study

| Configuration | Overall |
|------|---------|
| Baseline | 60.9 |
| +Random Data (RD) | 56.8 (-4.1!) |
| +Neighboring Category Data (SD) | 64.9 |
| **+SD + NS-CE** | **68.2** |
| Inference Masking vs Retraining Classifier | Masking is better |

### Key Findings
- **16-percentage-point improvement in Few-shot classes** ($41.4 \to 57.4$): Tail classes benefit the most, as category extrapolation directly fills the tail feature space.
- **Random data is harmful (-4.1), whereas semantically adjacent data is beneficial (+4.0)**: Data must be semantically related to be effective.
- **Effective across all three initializations**: Scratch, CLIP, and DINOv2 consistently benefit, demonstrating the generalizability of the method.
- **Empirical validation that finer granularity is naturally more balanced** (gap of $7.3\%$ for 20 super-classes vs $20.8\%$ for 100 super-classes) explains the theoretical foundation of the method.

## Highlights & Insights
- **The phenomenon that "finer granularity mitigates long-tail imbalance"** is a valuable theoretical finding, explaining why iNat18 (8,142 classes) is more robust to imbalance than ImageNet-LT (1,000 classes).
- **The LLM data pipeline (category discovery $\to$ crawling $\to$ filtering)** is completely automated and can be generalized to any long-tail scenario.
- **Simple masking during inference outperforms retraining classifier**, which is counter-intuitive yet highly practical.

## Limitations & Future Work
- Reliance on GPT-4 to discover neighboring classes—it might generate low-quality results in specialized fields where GPT-4 is less competent.
- Web crawling is limited by the quality of search engine results.
- The parameter $\lambda_s = 0.1$ in NS-CE is fixed; adaptive adjustment might yield better results.

## Related Work & Insights
- **vs Balanced Softmax / RIDE**: These methods re-balance within the existing category space. Ours fundamentally expands the feature space by extrapolating new categories.
- **vs CLIP for Long-Tail**: Our method can be superimposed on CLIP initialization ($74.0 \to 77.3$), showing complementary and synergistic effects.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of "mitigating long-tail via category extrapolation" is novel and profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets, 3 initializations, detailed ablations, and granularity analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Seamless and rigorous logical flow from phenomenon to methodology.
- Value: ⭐⭐⭐⭐⭐ Opens up a brand new data augmentation perspective for long-tail learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Gradient Extrapolation for Debiased Representation Learning](../../ICCV2025/social_computing/gradient_extrapolation_for_debiased_representation_learning.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[ECCV 2024\] Distribution-Aware Robust Learning from Long-Tailed Data with Noisy Labels](../../ECCV2024/social_computing/distribution-aware_robust_learning_from_long-tailed_data_with_noisy_labels.md)
- [\[ICML 2025\] Learning Survival Distributions with the Asymmetric Laplace Distribution](../../ICML2025/social_computing/learning_survival_distributions_with_the_asymmetric_laplace_distribution.md)
- [\[NeurIPS 2025\] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](../../NeurIPS2025/social_computing/graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)

</div>

<!-- RELATED:END -->
