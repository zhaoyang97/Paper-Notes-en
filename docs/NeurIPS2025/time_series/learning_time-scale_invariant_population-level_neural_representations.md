---
title: >-
  [Paper Note] Learning Time-Scale Invariant Population-Level Neural Representations
description: >-
  [NeurIPS 2025][Time Series][neural time series] This paper proposes Time-Scale Augmented Pretraining (TSAP), a strategy that introduces data augmentation over multiple temporal window lengths during pretraining…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "neural time series"
  - "foundation models"
  - "time-scale invariance"
  - "population-level representations"
  - "brain-computer interface"
date: 2026-05-08
content_hash: 6f56f743afd50dac
---

# Learning Time-Scale Invariant Population-Level Neural Representations

**Conference**: NeurIPS 2025
**arXiv**: [2511.13022](https://arxiv.org/abs/2511.13022)
**Code**: None
**Area**: Time Series / Neural Signal Foundation Models
**Keywords**: neural time series, foundation models, time-scale invariance, population-level representations, brain-computer interface

## TL;DR

This paper proposes Time-Scale Augmented Pretraining (TSAP), a strategy that introduces data augmentation over multiple temporal window lengths during pretraining, enabling population-level neural signal foundation models to achieve invariance to input time scales and substantially improving decoding performance at both matched and unseen time scales.

## Background & Motivation

**Background**: Building general-purpose representations of neural time series is a fundamental goal in neuroscience and brain-computer interface (BCI) research. High-fidelity neural recordings such as intracranial EEG (iEEG) capture complex activity patterns across multiple brain regions, yet modeling them remains highly challenging due to inter-subject and inter-session variability and limited dataset scale.

**Limitations of Prior Work**: Recent population-level pretraining methods (e.g., Population Transformer, PopT) learn spatially aggregated representations on top of frozen temporal encoders and achieve strong downstream decoding performance; however, these models are highly sensitive to preprocessing parameters—particularly time scale. Performance degrades substantially when the temporal window lengths used during pretraining differ from those used in downstream tasks.

**Key Challenge**: Neural recordings vary widely in duration across datasets and tasks (ranging from 1 to 5 seconds), yet existing models are pretrained on fixed temporal windows and cannot generalize to inputs of varying lengths.

**Goal**: To quantify the performance degradation caused by time-scale mismatch and to propose a strategy that enables models to achieve optimal performance across arbitrary input time scales.

**Key Insight**: A data augmentation perspective is adopted, exposing the model to data across multiple temporal window lengths during pretraining.

**Core Idea**: By mixing iEEG segments of multiple time scales during pretraining (TSAP), PopT is trained to learn time-scale invariant population-level representations.

## Method

### Overall Architecture

The method builds upon the Population Transformer (PopT) framework: for a given temporal interval of each electrode channel, a frozen temporal encoder (BrainBERT) first produces temporal embeddings; positional embeddings derived from 3D electrode coordinates are then added; finally, a Transformer encoder yields spatially contextualized channel representations and an aggregated [CLS] output for downstream decoding.

### Key Designs

1. **Time-Scale Augmented Pretraining (TSAP)**:

   - **Function**: Modifies the data generation pipeline so that the model is exposed to iEEG signals of multiple temporal window lengths during pretraining.
   - **Design Motivation**: Eliminates overfitting to any specific time scale and establishes time-scale invariance.
   - **Mechanism**: Recording segments of length $l \in \{1, 2, 4, 5\}$ seconds are sampled (3 seconds is held out); each channel is independently encoded into BrainBERT embeddings. Embeddings from different window lengths contain overlapping windows that the temporal encoder maps to distinct representations.
   - **Novelty**: The original PopT is pretrained solely on fixed 5-second windows, whereas TSAP encourages cross-scale generalization through multi-scale exposure.

2. **Embedding Space Analysis (PCA + K-Means)**:

   - **Function**: Visualizes the distribution of temporal embeddings and [CLS] token representations across different time scales.
   - **Design Motivation**: Verifies whether TSAP genuinely eliminates time-scale-related clustering.
   - **Mechanism**: 100 samples are drawn from a specific subject–session across 1–5 second time scales, followed by 2D PCA projection and K-Means clustering analysis.
   - **Key Findings**: PopT pretrained on 5 seconds produces strong time-scale clusters, whereas the TSAP model's clusters are substantially mixed, indicating stronger time-scale invariance.

### Loss & Training

- Pretraining steps are doubled from 500,000 to 1,000,000 to accommodate the larger augmented dataset.
- The learning rate is fixed at $1 \times 10^{-4}$ to improve training stability.
- The best checkpoint is selected based on validation loss.
- During downstream fine-tuning, 90 electrodes are randomly selected per subject, and each experiment is repeated over 5 random seeds.

## Key Experimental Results

### Main Results

Experiments are conducted on the public BrainTreeBank dataset (10 subjects, 1,688 electrodes) across two auditory-language classification tasks: Word Onset and Sentence Onset.

| Model | 1s | 2s | 3s (held-out) | 4s | 5s |
|---|---|---|---|---|---|
| Non-Pretrained | 0.645 | 0.665 | 0.663 | 0.671 | 0.678 |
| 1s Pretrained | **0.770** | 0.807 | 0.809 | 0.817 | 0.819 |
| 5s Pretrained | 0.717 | 0.801 | 0.846 | 0.879 | **0.901** |
| **TSAP** | **0.777** | **0.843** | **0.866** | **0.893** | **0.907** |

*Word Onset ROC-AUC (mean ± standard error across subjects and 5 seeds)*

TSAP matches or surpasses the "optimal" baseline (i.e., models where pretraining and fine-tuning use the same time scale) at all time scales, including the held-out 3-second scale.

### Ablation Study

| Comparison | Statistical Significance (p-value) |
|---|---|
| TSAP vs. 1s Optimal (1s) | p = 0.017* |
| TSAP vs. 4s Optimal (4s) | p = 0.00005* |
| TSAP vs. 5s Optimal (5s) | p = 0.004* |
| TSAP vs. 3s Optimal (3s, held-out) | p = 0.442 |

Paired t-tests show that TSAP significantly outperforms the optimal baseline at most time scales; the improvement at the held-out 3-second scale does not reach significance but is occasionally observed.

### Key Findings

- Time-scale mismatch leads to substantial performance degradation: for example, a model pretrained on 1-second windows performs considerably worse on 5-second inputs than a 5-second pretrained model.
- Even under mismatch, any pretrained model outperforms the non-pretrained baseline, indicating that pretraining still captures valuable information.
- TSAP not only recovers the performance lost due to mismatch but also surpasses the matched "optimal" baseline in most cases.
- PCA analysis confirms that TSAP substantially reduces time-scale clustering in the embedding space.

## Highlights & Insights

- **Simplicity and Effectiveness**: TSAP is a purely data-augmentation-based strategy requiring no architectural modifications; mixing multiple time scales only during pretraining suffices.
- **Clear Physical Intuition**: Although different temporal windows share overlapping information, they produce markedly different embeddings after the temporal encoder, which is the root cause of performance degradation.
- **Generalization to Held-Out Time Scales**: The 3-second scale is never seen during pretraining, yet the model generalizes well, demonstrating that the invariance learned by TSAP is genuinely transferable.
- **High Practical Value**: In real-world BCI applications, using neural recordings of varying lengths across tasks and experimental paradigms is the norm rather than the exception.

## Limitations & Future Work

- Validation is currently limited to iEEG data; other modalities such as EEG have not been tested.
- Only the data augmentation strategy is explored; integration with invariance methods at the temporal encoder level (e.g., frequency-domain approaches such as TF-C or BioFAME) remains unexplored.
- The range of time scales examined is limited (1–5 seconds); generalization over a wider range requires further investigation.
- Computational cost doubles (pretraining steps increase from 500K to 1M), though this overhead is acceptable given the performance gains.

## Related Work & Insights

- **PopT (chau2025population)**: Population-level Transformer; the base framework of this paper.
- **BrainBERT (wang2023brainbert)**: Channel-independent temporal encoder providing frozen temporal embeddings.
- **TS-Rep (somaiya2022ts)**: Encourages duration-agnostic representations via a triplet objective.
- **TF-C (zhang2022self)**: Promotes time-scale invariance through frequency-domain consistency.
- **Insight**: Data augmentation constitutes a lightweight yet effective solution to preprocessing diversity and is broadly applicable to other sensor data domains.

## Rating

- Novelty: ⭐⭐⭐ The method itself is straightforward multi-scale data augmentation, though the problem identification is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of multiple time scales, two tasks, statistical testing, and embedding analysis.
- Writing Quality: ⭐⭐⭐⭐ Concise workshop paper with clear structure and rigorous argumentation.
- Value: ⭐⭐⭐⭐ Directly beneficial for the engineering deployment of neural signal foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[NeurIPS 2025\] Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)

</div>

<!-- RELATED:END -->
