---
title: >-
  [Paper Note] A Unified Shape-Aware Foundation Model for Time Series Classification
description: >-
  [AAAI 2026][Time Series][Time Series Classification] This paper proposes UniShape — a foundation model for time series classification that adaptively aggregates multi-scale discriminative subsequences (shapelets) via a shape-aware adapter, and learns transferable shapelet representations at both instance and shape levels through prototype-based contrastive pretraining. With only 3.1M parameters, UniShape achieves state-of-the-art performance on 128 UCR datasets (average accuracy 87.08%) while providing strong classification interpretability.
tags:
  - AAAI 2026
  - Time Series
  - Time Series Classification
  - Foundation Model
  - Shapelet
  - Prototype Learning
  - Interpretability
date: 2026-05-08
content_hash: 2dd87f0e8c7fb442
---

# A Unified Shape-Aware Foundation Model for Time Series Classification

**Conference**: AAAI 2026
**arXiv**: [2601.06429v1](https://arxiv.org/abs/2601.06429v1)
**Code**: [https://github.com/qianlima-lab/UniShape](https://github.com/qianlima-lab/UniShape)
**Area**: Time Series Classification / Foundation Models
**Keywords**: Time Series Classification, Foundation Model, Shapelet, Prototype Learning, Interpretability

## TL;DR
This paper proposes UniShape — a foundation model for time series classification that adaptively aggregates multi-scale discriminative subsequences (shapelets) via a shape-aware adapter, and learns transferable shapelet representations at both instance and shape levels through prototype-based contrastive pretraining. With only 3.1M parameters, UniShape achieves state-of-the-art performance on 128 UCR datasets (average accuracy 87.08%) while providing strong classification interpretability.

## Background & Motivation
Time series foundation models (FMs) have advanced rapidly in recent years, but the vast majority of work focuses on **forecasting tasks** (e.g., Chronos, Moirai), while **classification tasks** have long been neglected. Forecasting targets temporal dynamics such as trends and periodicity and outputs continuous value sequences, whereas classification requires extracting discriminative local patterns from fixed-length samples (e.g., T-wave anomalies in ECG) and outputting discrete labels. Consequently, directly transferring forecasting-oriented FMs to classification tasks yields poor results — GPT4TS, MOMENT, and UniTS perform even worse than non-deep-learning methods (the Rocket family) on UCR classification benchmarks.

Moreover, **interpretability** is critical in time series classification, especially in medical domains, and shapelets (discriminative subsequences) are a classical interpretability tool. However, existing shapelet methods rely on labeled supervision and are incompatible with FM pretraining paradigms. Furthermore, shapelets are inherently **multi-scale** (discriminative patterns of varying lengths), and how to uniformly model multi-scale shapelets within an FM remains an open problem.

## Core Problem
1. How to design a **classification-oriented** time series foundation model rather than simply repurposing forecasting FMs?
2. How to **learn multi-scale shapelet representations in an unsupervised/weakly supervised manner** within the FM pretraining framework, enabling transfer across diverse domains?
3. How to provide **interpretability** (i.e., which temporal segments are most decisive for classification) while maintaining competitive classification performance?

## Method

The core mechanism of UniShape is as follows: multi-scale sliding windows decompose the time series into subsequences (shapes) at different granularities; a lightweight adapter adaptively selects the most discriminative subsequence scales and aggregates them into a class token; prototype contrastive learning then guides the pretraining phase to learn transferable shapelet patterns.

### Overall Architecture
- **Input**: Univariate time series $x \in \mathbb{R}^T$ (uniformly resized to $T=512$)
- **Shape-Aware Adapter**: Multi-scale sliding windows → normalization + linear projection to shape tokens → multi-resolution CNN encoding → attention pooling to aggregate into a class token → coarse-to-fine hierarchical fusion
- **Transformer Encoder**: Receives the final class token and shape tokens, outputs refined representations
- **Prototype-based Pretraining**: Two-level prototype contrastive learning at instance and shape levels
- **Output**: Class token passed through a classification head to produce category predictions

### Key Designs

1. **Shape-Aware Adapter**:

    - $Q=5$ sliding window scales (window lengths $W_q = 64, 32, 16, 8, 4$) segment the time series into subsequence sets at different granularities
    - Each subsequence is normalized (subtracting global mean/std), then concatenated with raw value encodings, first-order difference encodings, and local statistical embeddings, and linearly projected to a $d$-dimensional shape token
    - Three parallel 1D CNNs with different kernel sizes extract multi-resolution features within the adapter
    - Linear-complexity **attention-based MIL pooling** aggregates all shape tokens into a single class token via learned weights $\alpha$, which directly reflect the discriminative importance of each subsequence (source of interpretability)
    - **Coarse-to-fine hierarchical fusion** across scales: the class token from the previous (coarser) scale is prepended to the shape token sequence of the next (finer) scale, progressively propagating contextual information
    - All scales share the same adapter parameters, substantially reducing parameter count

2. **Instance-Prototype Contrastive Learning**:

    - A set of learnable class prototype vectors $\{p_c\}$ is maintained and dynamically updated via EMA
    - Labeled samples update their corresponding prototypes using ground-truth labels; unlabeled samples are assigned pseudo-labels based on cosine similarity to the nearest prototype
    - The instance-level contrastive loss $\mathcal{L}_\text{ins}$ pulls the class token toward its corresponding class prototype while pushing it away from others

3. **Shape-Prototype Contrastive Learning**:

    - The top-$\varepsilon$ (default 60%) shape tokens with the highest attention scores are selected as high-confidence shape tokens
    - The shape-level contrastive loss $\mathcal{L}_\text{shape}$ aligns these high-confidence shape tokens with their corresponding class prototypes
    - This design encourages the model to learn not only globally discriminative class features (instance level) but also local shapelet patterns (shape level)

### Loss & Training
- **Pretraining loss**: $\mathcal{L}_\text{pretrain} = \mathcal{L}_\text{proto} + \mathcal{L}_\text{self}$
    - $\mathcal{L}_\text{proto} = (1-\lambda)\cdot\mathcal{L}_\text{ins} + \lambda\cdot\mathcal{L}_\text{shape}$, with $\lambda=0.01$ balancing instance-level and shape-level objectives
    - $\mathcal{L}_\text{self}$: MoCo v3 self-supervised contrastive loss (consistency between two randomly cropped views), enabling weakly supervised pretraining
- **Finetuning loss**: $\mathcal{L}_\text{finetune} = \mathcal{L}_\text{ce} + \mu\cdot\mathcal{L}_\text{shape}$
    - Cross-entropy loss combined with an auxiliary shape contrastive loss ($\mu=0.01$), maintaining discriminative shapelet learning during finetuning
- Pretraining: 30 epochs, batch size 2048, defaulting to only 10% labeled data (experiments show negligible performance difference between 10% and 100% labels)
- Finetuning: 300 epochs; the checkpoint with the lowest training loss is selected for evaluation
- Momentum coefficient $\beta=0.9$; shape token selection ratio $\varepsilon=60\%$

## Key Experimental Results

| Dataset | Metric | UniShape | Prev. SOTA (MR-H/Mantis) | Gain |
|---------|--------|----------|--------------------------|------|
| 128 UCR (fully supervised) | Avg. Acc | **0.8708** | 0.8621 / 0.8441 | +0.87% / +2.67% |
| 128 UCR (fully supervised) | Avg. Rank | **2.71** | 3.97 / 5.21 | — |
| 30 additional datasets (zero-shot feature extraction) | Avg. Acc | **0.7262** | 0.7052 (Mantis) | +2.1% |
| 30 additional datasets (zero-shot) | Avg. Rank | **3.07** | 3.67 (Mantis) | — |

- UniShape has only **3.1M parameters**, far fewer than GPT4TS (84.1M) and MOMENT (341.2M)
- All $p$-values $< 0.05$ (Wilcoxon signed-rank test), indicating statistically significant superiority over all baselines

### Ablation Study
- **Without pretraining**: accuracy drops from 85.29% to 83.65% (−1.64%), confirming substantial contribution of pretraining
- **Without Adapter**: accuracy drops to 84.28% (−1.01%), validating the effectiveness of multi-scale shapelet modeling
- Replacing CNN in the adapter with Transformer or MLP consistently degrades performance; CNN is more suitable for multi-scale shapelet feature extraction
- **Without Instance-Prototype (w/o Ins)**: −0.85%; **Without Shape-Prototype (w/o Shape)**: −0.59%; removing both: −1.18% — indicating that instance-level prototypes are more important, and the two are complementary
- Transformer encoder outperforms CNN encoder in the FM setting (contrary to findings in domain-specific training where CNN is preferred); MLP encoder performs extremely poorly (−28.8%)
- Label fraction: performance difference between 10% and 100% labels is not statistically significant ($p > 0.05$), demonstrating robustness to label scarcity

## Highlights & Insights
- **Classification-oriented FM design**: The paper explicitly identifies the unsuitability of forecasting FMs for classification tasks and reframes the problem from a shapelet perspective, representing one of the few works specifically designing an FM for TSC
- **Attention pooling provides natural interpretability**: Attention weights on shape tokens directly indicate discriminative temporal segments without requiring post-hoc explanation methods. On ECGFiveDays, the model accurately localizes the known T-wave anomaly interval $[75, 95]$
- **Parameter efficiency**: 3.1M parameters substantially outperform comparable FMs (MOMENT 341M, GPT4TS 84M) in both size and accuracy
- **Shared-parameter adapter**: A single adapter is shared across all scales, enabling unified processing of variable-length sequences while controlling parameter count
- **Feasible weakly supervised pretraining**: Only 10% labeled data is sufficient to achieve near-fully-supervised performance, validating the pseudo-label and prototype learning mechanism

## Limitations & Future Work
- **Univariate only**: Multivariate time series must be decomposed into independent channels (channel-independent), discarding inter-channel dependencies. The authors explicitly acknowledge this as the primary limitation in the Conclusion
- **Fixed-length assumption**: All inputs must be resized to $T=512$; interpolation may discard or introduce spurious features, making the model less friendly to sequences whose original length is far shorter or longer than 512
- **Pretraining data domain coverage**: Although 1.89M samples are drawn from multiple domains, they primarily consist of "clean" benchmark data from UCR/UEA; robustness to industrial-grade noisy time series is not validated
- **Fixed sliding window scales**: The five window lengths $\{64, 32, 16, 8, 4\}$ are manually specified; adaptive scale selection has not been explored

## Related Work & Insights
- **vs. Mantis/NuTime** (classification-oriented FMs): UniShape achieves higher accuracy and better interpretability through explicit shapelet modeling; Mantis and NuTime focus on multi-scale normalization but lack the shapelet concept. UniShape (3.1M) outperforms Mantis (8.7M) with fewer parameters
- **vs. MOMENT/GPT4TS/UniTS** (general-purpose FMs): These FMs are primarily designed for forecasting; when transferred to classification, they perform far below non-deep-learning methods, validating the argument that classification requires task-specific design
- **vs. SoftShape** (shapelet method): SoftShape is an end-to-end shapelet learning method but is limited to domain-specific training; UniShape extends the shapelet concept to the FM pretraining paradigm with stronger generalization

The work validates the view that **task-specific FMs outperform general-purpose FMs** for classification — which requires discriminative local features rather than temporal dynamic modeling. This perspective may transfer to the design of other task-specific FMs. The combination of attention pooling and prototype learning constitutes a general weakly supervised representation learning framework, extensible to other interpretability-demanding sequence classification scenarios (e.g., text classification, biological sequence classification).

## Rating
- Novelty: ⭐⭐⭐⭐ Integrating shapelet concepts into FM pretraining is a novel angle, though each component (MIL pooling, prototype learning, MoCo) is a combination of established techniques
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 128 UCR + 30 additional datasets + 16 baselines + multi-dimensional ablations + interpretability analysis — highly comprehensive
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-motivated arguments, though some sections with dense notation are moderately difficult to read
- Value: ⭐⭐⭐⭐ Fills a gap in time series classification FMs, though the restriction to univariate settings limits practical applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition](probfm_probabilistic_time_series_foundation_model_with_uncertainty_decomposition.md)
- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)
- [\[AAAI 2026\] A Theoretical Analysis of Detecting Large Model-Generated Time Series](a_theoretical_analysis_of_detecting_large_model-generated_time_series.md)
- [\[AAAI 2026\] ReCast: Reliability-aware Codebook Assisted Lightweight Time Series Forecasting](recast_reliability-aware_codebook_assisted_lightweight_time_series_forecasting.md)
- [\[ICLR 2026\] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification](../../ICLR2026/time_series/timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)

</div>

<!-- RELATED:END -->
