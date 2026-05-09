---
title: >-
  [Paper Note] BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition
description: >-
  [AAAI 2026][Self-Supervised Learning][Long-tailed recognition] BCE3S is proposed, a binary cross-entropy (BCE)-based tripartite synergistic learning framework that integrates BCE-based joint learning, BCE-based contrastive learning, and BCE-based classifier uniformity learning. By decoupling per-class logits via Sigmoid, it suppresses the imbalance effects inherent to long-tailed distributions, achieving state-of-the-art performance on CIFAR10/100-LT, ImageNet-LT, and iNaturalist2018.
tags:
  - AAAI 2026
  - Self-Supervised Learning
  - Long-tailed recognition
  - binary cross-entropy
  - contrastive learning
  - classifier uniformity
  - neural collapse
date: 2026-05-08
content_hash: eb2834969d04027e
---

# BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition

**Conference**: AAAI 2026  
**arXiv**: [2511.14097](https://arxiv.org/abs/2511.14097)  
**Code**: [https://github.com/wakinghours-github/BCE3S](https://github.com/wakinghours-github/BCE3S)  
**Area**: Self-supervised Learning / Long-tailed Recognition  
**Keywords**: Long-tailed recognition, binary cross-entropy, contrastive learning, classifier uniformity, neural collapse

## TL;DR
BCE3S is proposed, a binary cross-entropy (BCE)-based tripartite synergistic learning framework that integrates BCE-based joint learning, BCE-based contrastive learning, and BCE-based classifier uniformity learning. By decoupling per-class logits via Sigmoid, it suppresses the imbalance effects inherent to long-tailed distributions, achieving state-of-the-art performance on CIFAR10/100-LT, ImageNet-LT, and iNaturalist2018.

## Background & Motivation

**Background**: Long-tailed recognition (LTR) is a fundamental problem, as real-world data distributions are typically highly imbalanced, with head classes containing far more samples than tail classes. Existing methods predominantly build on cross-entropy (CE) loss, supplemented by rebalancing techniques such as resampling, reweighting, and logit adjustment.

**Limitations of Prior Work**: The Softmax denominator in CE loss couples the imbalanced logits $\{\bm{w}_j^T\bm{x}+b_j\}$ across all classes, causing the imbalance effects of head classes to be repeatedly injected into feature learning. Even with auxiliary techniques such as contrastive learning and fixed ETF classifiers, this intrinsic limitation of CE remains fundamentally unaddressed.

**Key Challenge**: LTR requires simultaneously satisfying three objectives: (a) features with high intra-class compactness and inter-class separability, (b) uniformly separable classifier vectors, and (c) proper alignment between features and classifiers. Existing methods lack a unified framework to jointly optimize all three objectives, and the Softmax coupling in CE limits the synergistic effectiveness of individual optimization modules.

**Goal**: Design a unified tripartite synergistic learning (TSL) paradigm that jointly optimizes feature–classifier joint learning, feature contrastive learning, and classifier uniformity learning, while replacing CE (Softmax) with BCE (Sigmoid) to decouple imbalanced logits.

**Key Insight**: Prior work has demonstrated that BCE holds greater potential than CE for LTR (Cui et al. 2019), yet this potential has not been fully explored. The authors provide an in-depth gradient-based explanation of BCE's advantages—Sigmoid decouples each class's logit into an independent channel, avoiding the coupled amplification effect of Softmax.

**Core Idea**: Replace CE (Softmax coupling) with BCE (Sigmoid decoupling) and unify joint learning, contrastive learning, and classifier uniformity learning into a cohesive tripartite framework for long-tailed recognition.

## Method

### Overall Architecture
Given a batch of samples from a long-tailed dataset, features $\bm{x}^{(k)}$ are extracted via a backbone (e.g., ResNet) and jointly trained through three parallel BCE loss branches:
- **BCE Joint Learning** $L_{bce}^{(sc)}$: optimizes alignment between features and classifier vectors
- **BCE Contrastive Learning** $L_{bce}^{(ss)}$: enhances intra-class compactness of features
- **BCE Uniformity Learning** $L_{bce}^{(cc)}$: balances inter-classifier separability

Total loss: $L_{bce}^{(tri)} = \frac{1}{B}\sum L_{bce}^{(sc)} + \frac{\lambda_{ss}}{B}\sum L_{bce}^{(ss)} + \frac{\lambda_{cc}}{K}\sum L_{bce}^{(cc)}$

### Key Designs

1. **BCE Joint Learning $L_{bce}^{(sc)}$**:

    - **Function**: Replaces conventional CE joint learning for feature–classifier co-optimization.
    - **Mechanism**: For each sample feature $\bm{x}^{(k)}$, BCE loss is computed against normalized classifier vectors $\bm{w}_j$ ($\|\bm{w}_j\|=1$). The positive term is $\log(1+\exp(-\bm{w}_k^T\bm{x}^{(k)}-b_k))$ and the negative term is $\sum_{j\neq k}\log(1+\exp(\bm{w}_j^T\bm{x}^{(k)}+b_j))$. Negative samples are randomly drawn using a resampling parameter $r$ to reduce head-class dominance.
    - **Design Motivation**: CE's Softmax couples all class logits in the denominator, repeatedly injecting head-class imbalance into gradients. BCE's Sigmoid restricts each pull/push term to a single classifier vector, thereby decoupling imbalance effects. Gradient analysis confirms that $\text{Act}_{bce}(\bm{w}_j^T\bm{x}^{(k)}) = \sigma(\bm{w}_j^T\bm{x}^{(k)})$ is independent of other classes' logits.

2. **BCE Contrastive Learning $L_{bce}^{(ss)}$**:

    - **Function**: Enhances intra-class compactness and inter-class separability in projection space.
    - **Mechanism**: Features are mapped to $\bm{z}^{(k)}$ via a nonlinear projector $\mathcal{P}$. Positive pairs consist of cosine similarities between same-class features, and negative pairs between different-class features. A memory bank stores representative features $\{\bm{z}_*^{(j)}\}$ per class. The loss is $\log(1+\exp(-\frac{1}{\tau}\cos(\bm{z}^{(k)}, \bm{z}_*^{(k)}))) + \sum_{j\neq k}\log(1+\exp(\frac{1}{\tau}\cos(\bm{z}^{(k)}, \bm{z}_*^{(j)})))$.
    - **Design Motivation**: Unlike Softmax-based contrastive learning (e.g., SupCon), BCE contrastive learning avoids coupled normalization over all negative pairs, making it more favorable for tail classes.

3. **BCE Uniformity Learning $L_{bce}^{(cc)}$**:

    - **Function**: Directly optimizes uniform separability among classifier vectors.
    - **Mechanism**: For each classifier vector $\bm{w}_k$, the separation from all other classifier vectors is maximized: $\sum_{j\neq k}\log(1+\exp(\bm{w}_k^T\bm{w}_j))$. In gradient terms, this manifests as an "interactive term"—each $\bm{w}_k$ receives $K-1$ uniform repulsive forces from other classifier vectors per batch.
    - **Design Motivation**: Joint learning is dominated by head-class samples, causing tail classifier vectors to collapse (a degenerate form of Neural Collapse under class imbalance). Uniformity learning directly, uniformly, and persistently maximizes separability among all classifier vectors, naturally converging toward an ETF structure while remaining co-optimized with features—unlike pre-fixed ETF approaches that suffer from misalignment with learned features.

### Loss & Training
- Loss weights $\lambda_{ss}$ (contrastive) and $\lambda_{cc}$ (uniformity) are determined via hyperparameter search.
- Classifier normalization: only classifier vectors are L2-normalized ($\|\bm{w}_j\|=1$); features are not normalized, which experiments confirm to be optimal.
- Optional two-stage strategy: Stage 1 trains with full BCE3S; Stage 2 freezes the feature extractor and fine-tunes the classifier with class-balanced BCE.

## Key Experimental Results

### Main Results

| Dataset | Metric | BCE3S | Prev. SOTA | Gain |
|--------|------|-------|---------|------|
| CIFAR10-LT (IF=100) | Top-1 Acc | **90.08%** | 89.58% (GLMC+MN) | +0.50 |
| CIFAR100-LT (IF=100) | Top-1 Acc | **59.50%** | 58.41% (GLMC+MN) | +1.09 |
| CIFAR100-LT (IF=50) | Top-1 Acc | **65.23%** | 64.57% (GLMC+MN) | +0.66 |
| CIFAR100-LT (IF=10) | Top-1 Acc | **76.13%** | 74.28% (GLMC+MN) | +1.85 |

On ImageNet-LT, BCE3S with ResNeXt50 achieves 58.54%, surpassing GLMC, ProCo, and other competing methods.

### Ablation Study
CIFAR100-LT (IF=100), ResNet32:

| Configuration | Many | Med. | Few | All |
|------|------|------|-----|-----|
| CE Joint $L_{ce}^{(sc)}$ | 82.29 | 51.37 | 15.67 | 51.48 |
| BCE Joint $L_{bce}^{(sc)}$ | 81.11 | 55.06 | 17.40 | 52.88 |
| BCE Joint + Contrastive | 82.74 | 56.57 | 20.63 | 54.95 |
| BCE Joint + Uniformity | 81.03 | 56.51 | 19.20 | 53.90 |
| **BCE3S (Full)** | **83.34** | **57.09** | **22.80** | **55.99** |
| CE TSL (Full) | 83.97 | 54.54 | 18.87 | 54.14 |

### Key Findings
- BCE joint learning yields significant improvements over CE on Medium/Few subsets (+3.69/+1.73), with only a marginal decrease on Many, resulting in an overall gain.
- BCE uniformity learning reduces the standard deviation of tail-class classifier separability from high variance to just 0.106, far outperforming the CE counterpart.
- Full BCE TSL outperforms full CE TSL by 1.85%, demonstrating that BCE's decoupling advantage is amplified under tripartite synergy.
- t-SNE visualizations reveal that under CE, "cat" and "dog" feature clusters heavily overlap, whereas under BCE3S all 10 classes are fully separated, with notably improved tail-class compactness.
- A critical finding: BCE contrastive and uniformity learning yield limited gains when combined with CE joint learning, confirming that Softmax coupling is the fundamental bottleneck.

## Highlights & Insights
- **Gradient-level analysis of BCE vs. CE**: Rather than empirically asserting "BCE is better," the paper derives $\text{Act}_{bce}$ and $\text{Act}_{ce}$ from gradient formulas to rigorously demonstrate how Softmax couples imbalanced logits while Sigmoid decouples them—a theoretically compelling argument.
- **Elegant design of uniformity learning**: Solely using BCE repulsion among classifier vectors naturally converges toward an ETF structure, without requiring a pre-fixed ETF or explicit orthogonality constraints—elegantly resolving the misalignment problem of pre-fixed ETF methods.
- **Non-decomposable gain from tripartite synergy**: Experiments show that peak performance is achieved only when all three branches are BCE-based; the Softmax bottleneck in CE joint learning suppresses the effectiveness of the other BCE branches.

## Limitations & Future Work
- Backbone validation is limited to the ResNet family (ResNet32/50, ResNeXt50); effectiveness on Transformer architectures such as ViT has not been thoroughly verified (LiVT is mentioned but not compared).
- The negative resampling parameter $r$ requires manual tuning and may need dataset-specific adjustment.
- Uniformity learning assumes all classifier vectors should be equidistant (ETF), which may be suboptimal for semantically similar classes.
- Integration of advanced data augmentation strategies (e.g., MixUp, CutMix) with BCE3S remains unexplored.

## Related Work & Insights
- **vs. GLMC (CVPR 2023)**: GLMC similarly combines contrastive learning with reweighting but operates within a CE framework. BCE3S outperforms GLMC+MN by 1.09% on CIFAR100-LT (IF=100), suggesting that a fundamental change in the loss function is more effective than external rebalancing techniques.
- **vs. ProCo (TPAMI 2024)**: ProCo models the feature space with vMF distributions to alleviate the large-sample requirement of contrastive learning. BCE3S achieves comparable or superior results with a simpler Sigmoid-based decoupling (CIFAR100-LT: 59.50% vs. 52.80%).
- **vs. Pre-fixed ETF methods (NC-DRW, RBL)**: These methods fix the ETF classifier prior to training, leading to poor alignment with learned features. BCE3S's uniformity learning allows classifiers to naturally converge toward an ETF structure while being co-optimized with features.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The TSL framework is conceptually novel; the in-depth analysis of BCE in LTR is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four datasets, detailed ablations, visualizations, and gradient-level theoretical analysis—comprehensive coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, though notation and equation references in the main text are occasionally inconsistent.
- **Value**: ⭐⭐⭐⭐ Provides a unified framework and a new design philosophy for long-tailed recognition (systematic justification of BCE over CE).

BCE3S replaces CE (Softmax) with BCE (Sigmoid) as the unified basis for tripartite synergistic learning (joint learning + contrastive learning + uniformity learning), addressing long-tailed bias arising from Softmax-coupled imbalanced classifier vectors, and achieving state-of-the-art results on CIFAR-LT/ImageNet-LT.

## Background & Motivation

1. **Background**: In LTR, head classes contain far more samples than tail classes. Mainstream approaches include resampling, reweighting, decoupled training, and contrastive learning, most of which are built upon the CE loss framework.

2. **Limitations of Prior Work**: CE's Softmax couples the inner products of $K$ classes in the denominator—$\text{Softmax}(z_i) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$—such that under class imbalance, head-class classifier vectors yield larger logits that dominate the denominator gradient, systematically suppressing feature learning for tail classes.

3. **Key Challenge**: Softmax normalization causes each class's gradient to depend not only on its own logit but also on all other classes' classifier vectors—this coupling amplifies bias under imbalance.

4. **Goal**: Fundamentally eliminate inter-class coupling interference in long-tailed learning at the loss function level.

5. **Key Insight**: Replace CE (Softmax) with BCE (Sigmoid)—Sigmoid computes push/pull forces independently for each class without coupling others.

6. **Core Idea**: BCE's Sigmoid decouples inter-class interactions, making pull/pull forces independent of class frequency, fundamentally mitigating long-tailed bias.

## Method

### Overall Architecture
Tripartite synergistic learning: (1) BCE joint learning—sample–classifier alignment; (2) BCE contrastive learning—sample–sample compactness; (3) BCE uniformity learning—classifier–classifier uniformization. All three branches are unified under BCE loss.

### Key Designs

1. **BCE Joint Learning $L_{\text{bce}}^{(\text{sc})}$**:
    - **Function**: Optimizes alignment between sample features and classifier vectors.
    - **Mechanism**: The similarity between features and L2-normalized classifier vectors is evaluated independently for each class via Sigmoid. A resampling parameter $r$ controls the negative class sampling ratio.
    - **Design Motivation**: L2 normalization prevents head-class classifier vectors from dominating gradients due to their larger magnitudes; Sigmoid evaluates each class independently without coupling.

2. **BCE Contrastive Learning $L_{\text{bce}}^{(\text{ss})}$**:
    - **Function**: Enhances intra-class compactness and inter-class separability.
    - **Mechanism**: In the projection space, BCE is used to attract same-class features and repel different-class features. Unlike Softmax-based contrastive learning (e.g., SupCon), BCE computes each positive/negative pair independently.
    - **Design Motivation**: Avoids the interference of imbalanced sample pairs on positive/negative pair weights in Softmax-based contrastive losses.

3. **BCE Uniformity Learning $L_{\text{bce}}^{(\text{cc})}$**:
    - **Function**: Directly drives classifier vectors toward a uniform distribution (ETF-like structure).
    - **Mechanism**: Each classifier vector receives $K-1$ repulsive forces from all other classifiers, entirely independent of sample distribution, providing constant repulsion per batch.
    - **Design Motivation**: Bypasses sample imbalance entirely—classifier uniformization depends only on inter-classifier relationships, unaffected by sample counts.

### Loss & Training
$L = \frac{1}{B}\sum L_{\text{bce}}^{(\text{sc})} + \frac{\lambda_{ss}}{B}\sum L_{\text{bce}}^{(\text{ss})} + \frac{\lambda_{cc}}{K}\sum L_{\text{bce}}^{(\text{cc})}$

## Key Experimental Results

### Main Results

| Dataset | IF | BCE3S | Prev. SOTA | Gain |
|--------|-----|-------|--------|------|
| CIFAR100-LT | 100 | **59.50** | 58.41 | +1.09 |
| CIFAR100-LT | 10 | **76.13** | 74.28 | +1.85 |
| CIFAR10-LT | 100 | **90.08** | 89.58 | +0.50 |
| ImageNet-LT (RX50) | 256 | **58.54** | 58.00 | +0.54 |

### Ablation Study (CIFAR100-LT, IF=100)

| Configuration | Many | Med | Few | All |
|------|------|-----|-----|-----|
| CE baseline | 82.29 | 51.37 | 15.67 | 51.48 |
| BCE Joint Learning | 81.11 | 55.06 | 17.40 | 52.88 |
| + BCE Contrastive Learning | 82.74 | 56.57 | 20.63 | 54.95 |
| **BCE3S (Full)** | **83.34** | **57.09** | **22.80** | **55.99** |

### Key Findings
- BCE alone improves over CE by 2.4% (51.48→52.88), with the largest gains on tail classes (15.67→17.40).
- Synergistic effects among the three BCE components are substantial—BCE3S (55.99) significantly outperforms CE3S (54.14, i.e., tripartite learning with CE).
- Feature compactness: BCE achieves intra-class similarity mean of 95.47 (std 1.81) vs. CE ~82 (std 5.55).
- Classifier uniformity: BCE achieves a standard deviation of 0.106 (highly uniform).

## Highlights & Insights
- **Analyzing long-tailed bias from the perspective of loss function coupling** is a fresh angle—rather than designing new resampling or reweighting strategies, the paper identifies Softmax itself as the source of bias.
- **BCE uniformity learning** operates directly on classifiers and completely bypasses sample distribution—a principle applicable to any imbalanced learning scenario.

## Limitations & Future Work
- Absolute performance gains are modest (~1–2%), potentially approaching the ceiling of this framework.
- Validation is limited to classification tasks; effectiveness on detection, segmentation, and other tasks remains unknown.
- Hyperparameters $\lambda_{ss}$ and $\lambda_{cc}$ require tuning.

## Related Work & Insights
- **vs. ProCo**: ProCo employs probabilistic contrastive learning; BCE3S achieves comparable or better results more simply via Sigmoid decoupling. BCE3S is marginally superior on ImageNet-LT.
- **vs. GLMC**: GLMC uses mixed augmentation and contrastive learning; BCE3S addresses the problem more fundamentally at the loss function level.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The analysis that "Softmax coupling is the root cause of long-tailed bias" is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, detailed ablations, and feature visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical analysis is clearly presented.
- **Value**: ⭐⭐⭐⭐ Offers a new loss design paradigm for long-tailed learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Long-Tailed Recognition via Information-Preservable Two-Stage Learning](../../NeurIPS2025/self_supervised/long-tailed_recognition_via_information-preservable_two-stage_learning.md)
- [\[ICLR 2026\] Maximizing Incremental Information Entropy for Contrastive Learning](../../ICLR2026/self_supervised/maximizing_incremental_information_entropy_for_contrastive_learning.md)
- [\[AAAI 2026\] MovSemCL: Movement-Semantics Contrastive Learning for Trajectory Similarity (Extension)](movsemcl_movement-semantics_contrastive_learning_for_trajectory_similarity_exten.md)
- [\[AAAI 2026\] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [\[AAAI 2026\] Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision](improving_region_representation_learning_from_urban_imagery_with_noisy_long-capt.md)

</div>

<!-- RELATED:END -->
