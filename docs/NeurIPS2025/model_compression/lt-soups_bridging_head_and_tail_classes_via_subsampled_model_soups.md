---
title: >-
  [Paper Note] LT-Soups: Bridging Head and Tail Classes via Subsampled Model Soups
description: >-
  [NeurIPS 2025][Model Compression][Long-tail distribution] This paper proposes LT-Soups, a two-stage model merging framework that trains multiple models on subsampled datasets with progressively varying imbalance ratios and aggregates them via weight averaging, achieving balanced performance across head and tail classes over the full long-tail spectrum.
tags:
  - NeurIPS 2025
  - Model Compression
  - Long-tail distribution
  - model merging
  - CLIP fine-tuning
  - class imbalance
  - parameter-efficient fine-tuning
date: 2026-05-08
content_hash: 5761645a06aec098
---

# LT-Soups: Bridging Head and Tail Classes via Subsampled Model Soups

**Conference**: NeurIPS 2025
**arXiv**: [2511.10683](https://arxiv.org/abs/2511.10683)
**Code**: [GitHub](https://github.com/Masseeh/LT-Soups)
**Area**: Model Compression & Efficient Learning
**Keywords**: Long-tail distribution, model merging, CLIP fine-tuning, class imbalance, parameter-efficient fine-tuning

## TL;DR

This paper proposes LT-Soups, a two-stage model merging framework that trains multiple models on subsampled datasets with progressively varying imbalance ratios and aggregates them via weight averaging, achieving balanced performance across head and tail classes over the full long-tail spectrum.

## Background & Motivation

**Background**: Real-world datasets typically follow long-tailed distributions, where a few head classes dominate. Vision-language foundation models such as CLIP, combined with PEFT methods (e.g., LoRA, AdaptFormer) and logit adjustment (LA) loss, achieve state-of-the-art performance; however, PEFT tends to sacrifice head-class accuracy in favor of tail-class performance.

**Limitations of Prior Work**: (1) PEFT performs well in tail-heavy scenarios but degrades under balanced or head-heavy distributions; (2) full fine-tuning offers stronger adaptability but risks forgetting pretrained knowledge; (3) conventional Model Soups, trained on the same imbalanced data, remain biased toward head classes.

**Key Challenge**: There exists a fundamental trade-off between head- and tail-class performance, and no single method consistently performs well across all imbalance configurations.

**Goal**: Design a method that is robust across varying imbalance ratios $\rho$ and head-to-tail ratios $\eta$.

**Key Insight**: The paper introduces the head-to-tail ratio $\eta = H/T$ as an additional dimension for characterizing imbalanced distributions, and analyzes the limitations of existing methods within this two-axis framework.

**Core Idea**: Fine-tune separate models on subsets with different degrees of imbalance, then aggregate them via weight averaging, so that individual models "specialize" in different regions of the imbalance spectrum, and their combination achieves balanced representation across head and tail classes.

## Method

### Overall Architecture

LT-Soups proceeds in two stages: (1) fine-tune multiple models on subsampled datasets with progressively increasing imbalance ratios, followed by recursive weight averaging; (2) freeze the merged backbone and fine-tune only the classification head on the full dataset using a class-balanced loss to recover head-class information.

### Key Designs

1. **Two-Axis Imbalance Characterization**: In addition to the conventional imbalance ratio $\rho = n_K / n_1$ (ratio of the smallest to the largest class size), the paper introduces the head-to-tail ratio:
   $$\eta = \frac{H}{T} = \frac{|\{c \mid n_c > \tau\}|}{|\{c \mid n_c \leq \tau\}|}$$
   By systematically varying $\rho$ and $\eta$ on CIFAR-100, the paper reveals that PEFT is superior in tail-heavy settings while full fine-tuning prevails in head-heavy ones.

2. **Progressive Subsampling Strategy**: A sequence of subsets with exponentially increasing imbalance ratios is constructed:
   $$\{D_{\rho_i} \mid \rho_i = 2^i, \; i \in \{0, 1, 2, \dots, \lceil\log_2(\rho)\rceil\}\}$$
   The top $N$ subsets are retained. For each subset, $M$ bootstrap replicas are trained to reduce variance, yielding $NM$ models in total.

3. **Recursive Weight Interpolation**: Models are sorted in ascending order of imbalance degree and merged recursively:
   $$\theta_n = (1 - \lambda)\theta_n + \lambda\theta_{n-1}$$
   where $\lambda$ controls the degree of retention from the previous (more balanced) model. Compared to uniform averaging, this recursive strategy performs better on datasets that require substantial adaptation.

4. **Classifier Retraining (CR)**: The merged backbone is frozen, and the classification head is fine-tuned on the full dataset using LA loss to recalibrate decision boundaries. While PEFT and conventional Model Soups do not benefit from CR, LT-Soups suffers from incomplete head-class information due to subsampling, and CR effectively compensates for this.

### Loss & Training

- Stage 1 uses the Logit Adjustment (LA) loss:
  $$\ell_{LA}(y, g(\bm{z})) = -\log \frac{\exp(g_y(\bm{z}) + \log\pi_y)}{\sum_{y'} \exp(g_{y'}(\bm{z}) + \log\pi_{y'})}$$
- EMA ($\mu = 0.99$) is applied during training of each model as regularization.
- All Stage 1 models can be trained in full parallel.

## Key Experimental Results

### Main Results

Comparison on synthetic and real long-tailed datasets (CLIP backbone):

| Method | CIFAR100-LT All | CIFAR100-LT Few | Places-LT All | ImageNet-LT All | NIH-CXR-LT All | iNat2018 All |
|---|---|---|---|---|---|---|
| Linear Probing | 70.0 | 60.4 | 48.8 | 74.2 | 17.5 | 60.4 |
| Full-FT | 79.6 | 69.3 | 46.6 | 73.9 | 38.0 | 76.1 |
| PEFT | 81.3 | 77.1 | 51.5 | 77.0 | 38.5 | 79.1 |
| Model Soups | 82.1 | 73.0 | 49.4 | 76.0 | 38.0 | 76.4 |
| **LT-Soups** | **83.5** | **78.0** | **51.7** | **77.4** | **39.3** | 78.2 |

### Ablation Study

Ablation of subsampling strategies on TinyImageNet-LT (Soups trained at different fixed $\rho$):

| Method | All | Head | Tail |
|---|---|---|---|
| Full-FT | 73.2 | 83.4 | 67.7 |
| PEFT | 77.1 | 83.0 | 73.9 |
| Soups-1 (most balanced subset) | 71.7 | 74.6 | 70.1 |
| Soups-100 (full data = conventional Soups) | 77.6 | 85.9 | 73.0 |
| **LT-Soups (cross-spectrum merging)** | **78.6** | 85.0 | **75.2** |

Effect of classifier retraining (CR):

| Method | All | Head | Tail |
|---|---|---|---|
| LT-Soups Stage 1 | 78.1 | 84.9 | 74.5 |
| LT-Soups (+ CR) | **78.6** | **85.0** | **75.2** |

### Key Findings

- Soups trained at different $\rho$ values exhibit distinct preferences for head vs. tail classes (Soups-8 achieves the highest tail accuracy at 75.0; Soups-100 achieves the highest head accuracy at 85.9); LT-Soups' cross-spectrum merging achieves the best overall balance.
- LT-Soups is partially insensitive to the choice of loss function (CE/CB/LA all remain effective), as subsampling and weight averaging provide structural regularization.
- The recursive merging strategy substantially outperforms uniform averaging on datasets requiring large adaptation (e.g., iNaturalist: 78.2 vs. 74.7).
- Averaged across 5 benchmarks, LT-Soups achieves the best performance in all three splits: many, medium, and few.

## Highlights & Insights

- The introduction of the head-to-tail ratio $\eta$ as a second dimension of imbalance characterization reveals the distinct operating regimes in which PEFT and full fine-tuning are respectively optimal.
- The subsampling-and-merging paradigm is conceptually simple yet effective, supports fully parallel training, and incurs manageable computational overhead.
- This work is the first to apply Model Soups to long-tailed classification, addressing the head-class bias of conventional Soups through progressive imbalance subsampling.
- Experiments span 5 benchmarks and diverse imbalance structures, lending strong empirical credibility to the conclusions.

## Limitations & Future Work

- The head-tail split relies on a fixed threshold ($n_c > 100$), which may be an oversimplification; parametric frameworks such as the generalized Pareto distribution could offer more precise characterization.
- Validation is limited to the CLIP ViT backbone; generalizability to other foundation models (e.g., DINOv2) remains unexplored.
- Training $NM + 1$ models increases resource requirements, even if parallelizable.
- Only two values of $\lambda$ (0.3 and 0.7) are evaluated; finer-grained adaptive scheduling may yield further improvements.

## Related Work & Insights

- **Model Soups**: The seminal work on improving robustness via weight averaging; this paper extends it to the long-tailed setting.
- **LIFT (Shi et al.)**: PEFT + LA achieves state-of-the-art results; this paper exposes its limitations in non-tail-heavy scenarios.
- **Ensemble methods (BBN, RIDE, SADE, etc.)**: Require multiple experts at inference time; LT-Soups merges into a single network, yielding greater efficiency.
- This work offers a new methodological perspective on long-tailed learning in the era of foundation models.

## Rating

- Novelty: ⭐⭐⭐⭐ The two-axis imbalance analysis framework is original; the subsampled Model Soups approach is both concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five benchmarks, controlled synthetic experiments, and comprehensive ablation studies; experimental design is rigorous.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived; the progression from toy experiments to method design is logically coherent.
- Value: ⭐⭐⭐⭐ Provides a practical and efficient solution for long-tailed learning in the era of foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling](rat_bridging_rnn_efficiency_and_attention_accuracy_via_chunk-based_sequence_mode.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](../../ICCV2025/model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](vision-centric_token_compression_in_large_language_model.md)
- [\[NeurIPS 2025\] A Granular Study of Safety Pretraining under Model Abliteration](a_granular_study_of_safety_pretraining_under_model_abliteration.md)

</div>

<!-- RELATED:END -->
