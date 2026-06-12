---
title: >-
  [Paper Note] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention
description: >-
  [ICCV 2025][Generalized Category Discovery] This paper identifies a hidden vulnerability in GCD—ViT models exhibit distracted attention (attending to backgrounds rather than foreground targets) when processing unlabeled…
tags:
  - "ICCV 2025"
  - "Generalized Category Discovery"
  - "Distracted Attention"
  - "Token Pruning"
  - "Adaptive Pruning"
  - "Plug-and-Play Module"
date: 2026-05-08
content_hash: 0cd9e453d9a3e5c9
---

# A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention

**Conference**: ICCV 2025
**arXiv**: [2507.14315](https://arxiv.org/abs/2507.14315)  
**Code**: [https://github.com/Afleve/AFGCD](https://github.com/Afleve/AFGCD)  
**Area**: Other
**Keywords**: Generalized Category Discovery, Attention Focusing, Token Pruning, ViT, Fine-Grained Recognition

## TL;DR
This paper identifies a previously overlooked issue in GCD—ViT attention on unlabeled data (especially novel categories) tends to disperse onto background regions (distracted attention)—and proposes an Attention Focusing (AF) module that corrects attention via multi-scale token importance measurement combined with adaptive pruning. As a plug-and-play module on top of SimGCD, AF achieves up to 15.4% performance improvement.

## Background & Motivation

**Background**: Generalized Category Discovery (GCD) aims to leverage labeled known-category knowledge to classify unlabeled data containing both known and novel categories. Mainstream approaches fall into non-parametric methods (contrastive learning + K-means clustering) and parametric methods (e.g., SimGCD, which uses prototype classifiers for joint training).

**Limitations of Prior Work**: Existing methods almost universally overlook a hidden issue—distracted attention. Visualization analysis reveals that the [CLS] token attention of labeled data consistently focuses on foreground objects, whereas attention for unlabeled data (especially novel categories) disperses significantly onto background regions, degrading feature quality.

**Key Challenge**: The fundamental cause lies in the asymmetry of data augmentation. For labeled data, the same category contains images with diverse backgrounds, naturally guiding the model to focus on objects rather than backgrounds. For unlabeled data, augmentation produces only minor background variation, allowing the model to exploit spurious background correlations as shortcuts during self-supervised/unsupervised learning.

**Goal**: How to correct attention dispersion on unlabeled data in GCD models without introducing external models?

**Key Insight**: The problem is approached from the perspective of token pruning—if background tokens irrelevant to the task can be adaptively identified and removed, the model is forced to make decisions based on foreground regions. The key challenge is measuring token importance, since unlabeled data lacks labels.

**Core Idea**: Multi-scale learnable query tokens trained exclusively on labeled data are used to measure the importance of each patch, followed by adaptive pruning of low-importance tokens, compelling the model to attend to foreground objects.

## Method

### Overall Architecture
After an input image is divided into patch tokens by the ViT, a TIME module is inserted into each ViT block (except the last) to measure per-token importance scores. Scores from all layers are aggregated in a multi-scale fashion, and the TAP module adaptively prunes low-importance tokens. The remaining tokens pass through the final ViT block and average pooling before being fed into any GCD head. AF serves as a plug-and-play module that does not alter the head design of the original GCD method.

### Key Designs

1. **Token Importance Measurement (TIME)**:

    - **Function**: Measures the importance of each patch token for the classification task within each ViT block.
    - **Mechanism**: A learnable query vector $\mathbf{Q} \in \mathbb{R}^{1 \times D}$ is introduced to perform cross-attention with input tokens: $\mathbf{s} = \mathbf{Q}\mathbf{K}^T / \sqrt{D}$, yielding an importance score vector. An Aggregator then feeds the score-weighted token representation $\mathbf{r} = \text{Softmax}(\mathbf{s})\mathbf{V}$ into an auxiliary classifier (predicting known categories only), trained with cross-entropy loss $\mathcal{L}_{ce}$.
    - **Design Motivation**: **Trained on labeled data only, yet generalizes to unlabeled data**—because labeled and unlabeled data share similar visual style features, the knowledge of "which tokens matter for classification" learned on labeled data can transfer. The auxiliary classifier is decoupled from the backbone via stop-gradient to avoid gradient conflicts. At inference, the auxiliary classifier is discarded; only $\mathbf{Q}$ is retained.

2. **Token Adaptive Pruning (TAP)**:

    - **Function**: Adaptively prunes non-informative tokens based on multi-scale importance scores.
    - **Mechanism**: Score vectors (excluding [CLS] token scores) output by $L-1$ TIME layers are averaged after per-layer softmax: $\mathbf{s}^m = \frac{1}{L-1}\sum_{l=1}^{L-1}\text{Softmax}(\hat{\mathbf{s}}_l)$. Tokens are sorted by score in ascending order, and those whose cumulative scores sum to at most threshold $\tau$ are removed. The remaining tokens plus [CLS] are passed into the final ViT block.
    - **Design Motivation**: Multi-scale aggregation is more robust than single-layer scores (experiments show multi-scale outperforms using only the penultimate layer by 3–5%). Adaptive thresholding rather than fixed-count pruning accommodates varying degrees of background complexity across images.

3. **Single-View TAP Strategy**:

    - **Function**: TAP is applied to only one of the two augmented views.
    - **Design Motivation**: TAP is essentially equivalent to irregular cropping augmentation. Pruning both views excessively removes information and degrades generalization; single-view pruning preserves complete information from one view while forcing the model to focus on foreground in the pruned view.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{gcd} + \lambda \sum_{l=1}^{L-1} \mathcal{L}_{ce}^l$, where $\mathcal{L}_{gcd}$ is the base GCD method's loss and $\mathcal{L}_{ce}^l$ is the auxiliary classification loss of TIME at each layer.
- The auxiliary classifiers within TIME receive gradients only from labeled data.
- Stop-gradient decoupling is applied between the auxiliary classifiers and the backbone.

## Key Experimental Results

### Main Results

| Dataset | Method | All ACC | Old ACC | New ACC | Gain (All) |
|--------|------|---------|---------|---------|-----------|
| CUB | SimGCD | 60.3 | 65.6 | 57.7 | - |
| CUB | SimGCD+AF | 69.0 | 74.3 | 66.3 | **+8.7** |
| Stanford Cars | SimGCD | 53.8 | 71.9 | 45.0 | - |
| Stanford Cars | SimGCD+AF | 67.0 | 80.7 | 60.4 | **+13.2** |
| FGVC-Aircraft | SimGCD | 54.2 | 59.1 | 51.8 | - |
| FGVC-Aircraft | SimGCD+AF | 59.4 | 68.1 | 55.0 | **+5.2** |
| ImageNet-100 | SimGCD | 83.0 | 93.1 | 77.9 | - |
| ImageNet-100 | SimGCD+AF | 85.4 | 94.6 | 80.8 | **+2.4** |

### Ablation Study

| Configuration | CUB All | Stanford Cars All | Note |
|------|---------|-------------------|------|
| SimGCD (baseline) | 60.3 | 53.8 | Baseline |
| + AF (single-layer TIME) | 65.8 | 61.2 | Query from penultimate layer only |
| + AF (multi-scale TIME) | 69.0 | 67.0 | Multi-layer aggregation, significantly better |
| + AF on CMS | +0.9 | +8.7 | Effective on CMS as well |
| + AF on SelEx | +5.8 | +2.3 | Effective on SelEx as well |
| + AF on GET | +4.0 | +1.2 | Effective on GET as well |

### Key Findings
- AF yields the largest gains on fine-grained datasets with complex backgrounds: Stanford Cars +13.2%, CUB +8.7%; gains are marginal on general-purpose datasets with simple backgrounds such as CIFAR-10 (+0.7%).
- Multi-scale aggregation substantially outperforms single-layer measurement, confirming the complementarity of low-level local texture cues and high-level semantic information across ViT layers.
- Single-view TAP outperforms dual-view TAP—applying pruning to both views over-removes information, degrading generalization.
- AF is lightweight: additional parameters consist mainly of one query vector per layer, a small FFN, and an auxiliary classifier; the auxiliary classifier is discarded at inference.

## Highlights & Insights
- **Discovery and quantitative analysis of distracted attention** constitutes the core contribution: this paper is the first to reveal the phenomenon of attention dispersion in GCD for unlabeled data and to trace its root cause to data augmentation asymmetry. This observation has broad applicability—similar issues may exist in any open-world recognition task grounded in self-supervised learning.
- **The TIME design, trained only on labeled data yet generalizing to unlabeled data**, is elegant—it exploits the prior that labeled and unlabeled data share similar visual styles, circumventing the chicken-and-egg problem of obtaining supervision signals for unlabeled data.
- **Adaptive threshold pruning** outperforms fixed-ratio pruning because the foreground-to-background ratio varies considerably across images. This idea is transferable to any ViT task requiring token selection.

## Limitations & Future Work
- **Limited gains on simple-background datasets**: On CIFAR-10/100 and Herbarium-19, where backgrounds do not constitute significant distractors, AF may even cause marginal drops on novel categories.
- **Does not improve foreground feature extraction**: AF addresses the "where to look" problem but not the "how to look" problem; tasks involving intrinsically difficult fine-grained discrimination of foreground regions require complementary methods.
- **Generalization dependency of query vectors**: The design assumes visual style similarity between labeled and unlabeled data; large distributional gaps may cause the approach to fail.
- Future directions include exploring self-supervised token importance signals for unlabeled data (e.g., reconstruction objectives) and combining AF with fine-grained feature enhancement methods.

## Related Work & Insights
- **vs. SimGCD**: SimGCD is a concise and effective parametric GCD method that entirely overlooks the attention issue; AF as a plug-in significantly improves its performance.
- **vs. Cropr**: Cropr prunes a fixed number of tokens per ViT block, whereas this paper employs multi-scale adaptive pruning for greater flexibility and better results.
- **vs. AptGCD/MOS**: Concurrent competing methods also address background interference but require more complex module designs or external models.

## Rating
- Novelty: ⭐⭐⭐⭐ — The finding of distracted attention is insightful; the method is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 7 datasets, generalization validation across 4 GCD baselines, detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear; visualization analysis is intuitive.
- Value: ⭐⭐⭐⭐ — The plug-and-play module offers practical value to the GCD community with substantial gains in fine-grained scenarios.

**Area**: Category Discovery / Open-World Learning
**Keywords**: Generalized Category Discovery, Distracted Attention, Token Pruning, Adaptive Pruning, Plug-and-Play Module

## TL;DR
This paper identifies a hidden vulnerability in GCD—ViT models exhibit distracted attention (attending to backgrounds rather than foreground targets) when processing unlabeled data—and proposes the Attention Focusing (AF) mechanism, which cascades Token Importance Measurement (TIME) and Token Adaptive Pruning (TAP) to remove irrelevant tokens. AF achieves up to 15.4% improvement on SimGCD with negligible computational overhead.

## Background & Motivation
Generalized Category Discovery (GCD) aims to leverage knowledge from labeled known-category data to classify unlabeled data containing both known and novel categories. Existing GCD methods commonly adopt a pretrained ViT as the feature extraction backbone and perform classification using the [CLS] token embedding.

Existing methods (e.g., SimGCD, SPTNet, CMS) primarily focus on improving performance on unlabeled data through unsupervised/self-supervised learning, while overlooking a latent issue: **Distracted Attention**. Specifically, when processing unlabeled data, the model attends not only to the key target objects in an image but also to task-irrelevant background regions, degrading feature extraction quality.

Through visualizing self-attention maps of SimGCD on the CUB dataset, the authors observe that the [CLS] token of labeled data consistently focuses on foreground targets, whereas the [CLS] token of unlabeled data (especially novel categories) exhibits substantial attention to background regions.

## Core Problem
**Why does distracted attention arise in unlabeled data?** The authors hypothesize that data augmentation is a contributing factor: for labeled data, different images of the same category typically have varied backgrounds, naturally guiding the model to focus on foreground targets. For unlabeled data, augmentation usually applies only minor transformations to a single image with little background variation, making it easy for the model to exploit spurious background correlations as "shortcuts" for self-supervised/unsupervised learning.

The importance of this problem lies in the fact that distracted attention directly degrades the quality of feature representations—if the model is "not looking at the key object," downstream classification and clustering are inevitably imprecise. This problem is particularly severe on fine-grained datasets with complex backgrounds.

## Method

### Overall Architecture
The AF mechanism is inserted into the ViT backbone of an existing GCD model and consists of two cascaded modules:
- **Input**: Token sequences produced after the ViT divides an image into patches.
- **TIME module**: Inserted into each ViT block (except the last), producing a token importance score vector per block.
- **TAP module**: Aggregates multi-scale importance scores from all TIME modules and adaptively prunes unimportant tokens.
- **Output**: Remaining tokens are passed through the final ViT block, average-pooled, and fed into the GCD classification head.

A key point is that AF applies pruning to only one augmented view (single-view TAP), leaving the other view intact, so that pruning acts as a form of irregular cropping augmentation.

### Key Designs

1. **Token Importance Measurement (TIME)**:
    - A learnable query vector $\mathbf{Q}$ is introduced into each ViT block.
    - Input tokens serve as Keys $\mathbf{K}$ and Values $\mathbf{V}$; the importance score of each token is computed via cross-attention: $\mathbf{s}(\mathbf{Q}, \mathbf{K}) = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{D}}$
    - A softmax-weighted aggregation yields the image representation $\mathbf{r} = \text{Softmax}(\mathbf{s})\mathbf{V}$, which is further processed by an FFN.
    - An auxiliary classifier (trained only on labeled data) supervises $\mathbf{Q}$: the objective is to assign higher scores to more informative tokens.
    - **Key**: Stop-gradient decouples the auxiliary classifier from the backbone to prevent gradient conflicts; the auxiliary classifier is discarded at inference, retaining only $\mathbf{Q}$.

2. **Token Adaptive Pruning (TAP)**:
    - Multi-scale scores from all TIME blocks are aggregated: $\mathbf{s}_m = \frac{1}{L-1}\sum_{l=1}^{L-1}\text{Softmax}(\hat{\mathbf{s}}^l)$
    - [CLS] token scores are excluded ([CLS] is always retained).
    - Tokens are sorted by score in ascending order; tokens whose cumulative scores do not exceed threshold $\tau$ are pruned.
    - **Adaptivity** implies that the number of pruned tokens varies per image—images with complex backgrounds lose more tokens, while images with large foreground coverage lose fewer.

3. **Single-View TAP vs. Multi-View TAP**:
    - Pruning is applied to only one augmented view; the other remains intact.
    - Rationale: Single-view TAP is equivalent to irregular cropping augmentation, helping the model focus on key objects; multi-view TAP, while reducing background noise, also weakens generalization ability.

### Loss & Training
The total loss is:

$$\mathcal{L} = \mathcal{L}_{gcd} + \lambda \sum_{l=1}^{L-1}\mathcal{L}_{ce}^l$$

- $\mathcal{L}_{gcd}$: loss of the baseline GCD method (e.g., SimGCD's representation learning and classifier learning losses).
- $\mathcal{L}_{ce}^l$: cross-entropy loss of the auxiliary classifier in each TIME module (computed on labeled data only).
- $\lambda$: balancing coefficient, typically 0.05.
- $\tau$: pruning threshold, adjusted per dataset (CUB: 0.2, Stanford Cars: 0.01, FGVC-Aircraft: 0.01).
- Only the last ViT block and TIME modules are fine-tuned; batch size 128, 200 epochs.

## Key Experimental Results

**Fine-Grained Datasets (Core Results)**:

| Dataset | Metric | SimGCD+AF | SimGCD | Gain | Prev. SOTA |
|--------|------|-----------|--------|------|----------|
| CUB | All ACC | 69.0 | 60.3 | +8.7 | AptGCD 70.3 |
| Stanford Cars | All ACC | 67.0 | 53.8 | +13.2 | MOS 64.6 |
| Stanford Cars | New ACC | 60.4 | 45.0 | **+15.4** | MOS 56.7 |
| FGVC-Aircraft | All ACC | 59.4 | 54.2 | +5.2 | MOS/AptGCD 61.1 |

**General-Purpose Datasets**:

| Dataset | Metric | SimGCD+AF | SimGCD | Gain |
|--------|------|-----------|--------|------|
| CIFAR-10 | All ACC | 97.8 | 97.1 | +0.7 |
| CIFAR-100 | All ACC | 82.2 | 80.1 | +2.1 |
| ImageNet-100 | All ACC | 85.4 | 83.0 | +2.4 |
| Herbarium-19 | All ACC | 45.5 | 44.0 | +1.5 |

**Generalizability of AF to Other GCD Methods**:

| Method | CUB All | Stanford Cars All | Aircraft All |
|------|---------|------------------|-------------|
| CMS → CMS+AF | 67.3→68.2 | 53.1→61.8 (+8.7) | 54.2→57.5 |
| SelEx → SelEx+AF | 73.4→79.2 (+5.8) | 58.9→61.2 | 57.2→62.8 (+5.6) |
| GET → GET+AF | 75.2→77.3 | 78.3→81.5 (+3.2) | 57.4→59.5 |

### Ablation Study Highlights
- **Multi-scale vs. single-scale**: Using only the penultimate layer's query for pruning (AF−) is substantially inferior to multi-scale aggregation (AF), demonstrating large variation in attended patches across ViT layers.
- **Query trained on labeled data only**: Training the query on all data (including unlabeled) significantly degrades performance (e.g., Stanford Cars: 67.0→63.0), as unsupervised signals introduce noise.
- **TAP vs. fixed pruning**: Fixing the number of pruned patches (K=32/64/128) consistently underperforms adaptive TAP—large K discards critical information while small K insufficiently reduces background noise.
- **Average pooling vs. [CLS] token**: After pruning, using average pooling over remaining tokens (+AF) substantially outperforms using [CLS] alone (+AF([CLS])), as the final block's [CLS] token cannot fully aggregate information from individual patches.
- **Computational overhead**: Training parameters increase from 81.82M to 132.21M, but inference parameters remain nearly unchanged (81.82M→81.83M); training time increases by approximately 12% and inference time by approximately 25%.

## Highlights & Insights
- **Precise problem identification**: This is the first work to systematically study distracted attention in GCD, clearly demonstrating the discrepancy between labeled and unlabeled data attention via visualization.
- **Elegant design**: The TIME module uses an independent learnable query for cross-attention (rather than relying on the potentially degraded [CLS] token) and trains exclusively on labeled data—exploiting the GCD-specific property that labeled data is clean.
- **Plug-and-play**: AF consistently improves performance when applied to CMS, SelEx, GET, and other GCD methods, demonstrating strong generality.
- The insight that **single-view TAP = irregular cropping augmentation** is notable, establishing a conceptual link between token pruning and data augmentation.

## Limitations & Future Work
- **Limited gains on simple-background datasets**: On low-resolution datasets with simple backgrounds such as CIFAR-10/100, gains are marginal or even negative (CIFAR-100 New: −1.3), confirming that AF fundamentally addresses only background interference.
- **Cannot improve discriminability of foreground features**: The authors acknowledge in their conclusion that AF effectively suppresses background distraction but cannot enhance the model's ability to extract more discriminative features from key target regions.
- **Manual tuning of threshold $\tau$**: Optimal $\tau$ varies significantly across datasets (0.01–0.2); no automatic determination mechanism exists.
- Future directions include integrating external foreground-aware models (e.g., SAM, DINO) for token selection, and combining AF with methods that improve foreground feature discriminability.

## Related Work & Insights
- **vs. AptGCD/MOS**: These concurrent works also address background interference. AptGCD and MOS achieve comparable performance to AF on some datasets, but AF features a simpler module design without reliance on external models. However, AptGCD/MOS marginally outperform AF on certain datasets (e.g., CUB).
- **vs. SPTNet**: SPTNet optimizes the ViT via spatial prompt tuning with an alternating training strategy, incurring higher computational cost. On Stanford Cars, SPTNet (59.0) is substantially weaker than AF (67.0), though the two are comparable on Aircraft.
- **vs. Token Pruning methods (EViT, ToMe, Cropr)**: Conventional token pruning relies on [CLS] attention weights, but the degraded [CLS] quality in GCD's unlabeled data introduces misleading signals. AF's independent query mechanism avoids this issue. Cropr prunes a fixed number of tokens, whereas AF's adaptive strategy is more flexible.

## Related Work & Insights (Transfer Potential)
- **Cross-domain transfer potential**: The "independent query + auxiliary classifier" design of AF can be generalized to other semi-supervised/open-world tasks (e.g., Open-Set Detection, Novel Class Discovery), as long as labeled data is available to train the query.
- **Augmentation vs. Pruning**: The insight that single-view TAP is equivalent to irregular cropping augmentation suggests a deeper connection between token pruning and data augmentation worth further investigation.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic study of distracted attention in GCD; the finding is novel and the method design is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 7 datasets, generalization experiments across 4 GCD methods, multi-dimensional ablations, and computational efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, persuasive visualizations, and well-articulated motivation.
- Value: ⭐⭐⭐⭐ — Reveals an important blind spot in GCD; the plug-and-play module offers practical utility, though gains are limited in simple-background scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery](generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f.md)
- [\[ICCV 2025\] Membership Inference Attacks with False Discovery Rate Control](membership_inference_attacks_with_false_discovery_rate_control.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](../../NeurIPS2025/others/generalized_linear_mode_connectivity_for_transformers.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](../../NeurIPS2025/others/normalization_in_attention_dynamics.md)

</div>

<!-- RELATED:END -->
