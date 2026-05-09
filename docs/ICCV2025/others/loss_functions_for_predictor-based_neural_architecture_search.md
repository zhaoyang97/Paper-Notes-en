---
title: >-
  [Paper Note] Loss Functions for Predictor-based Neural Architecture Search
description: >-
  [ICCV 2025][Neural Architecture Search] This paper presents the first comprehensive and systematic study of 8 loss functions for performance predictors, spanning regression, ranking, and weighting categories. Evaluated across 13 tasks on 5 search spaces, the study reveals the characteristics and complementarity of each loss type, and proposes PWLNAS—a piecewise loss (PW loss) combination method—that surpasses existing state-of-the-art on multiple benchmarks.
tags:
  - ICCV 2025
  - Neural Architecture Search
  - Performance Predictor
  - Loss Functions
  - Learning to Rank
  - Weighted Loss
date: 2026-05-08
content_hash: ee96d9217690ec7a
---

# Loss Functions for Predictor-based Neural Architecture Search

**Conference**: ICCV 2025
**arXiv**: [2506.05869](https://arxiv.org/abs/2506.05869)
**Code**: N/A
**Area**: Neural Architecture Search
**Keywords**: Neural Architecture Search, Performance Predictor, Loss Functions, Learning to Rank, Weighted Loss

## TL;DR

This paper presents the first comprehensive and systematic study of 8 loss functions for performance predictors, spanning regression, ranking, and weighting categories. Evaluated across 13 tasks on 5 search spaces, the study reveals the characteristics and complementarity of each loss type, and proposes PWLNAS—a piecewise loss (PW loss) combination method—that surpasses existing state-of-the-art on multiple benchmarks.

## Background & Motivation

Performance predictors are widely adopted in NAS as an evaluation acceleration strategy, and their effectiveness is critically influenced by the choice of loss function:

**Background**: Although multiple loss functions such as MSE and ranking losses exist, the performance differences among them across various search spaces and training data sizes have not been systematically studied.

**Limitations of Prior Work**: Each individual loss function has inherent limitations. Regression losses excel at predicting absolute accuracy but are weak at ranking; ranking losses achieve good global ordering but struggle to identify top-performing architectures; weighted losses focus on high-performing architectures but are prone to overfitting under small data regimes.

**Key Challenge**: Researchers lack empirical guidance for selecting appropriate loss functions in specific NAS tasks.

## Method

### Overall Architecture

This work combines a comprehensive empirical study with a novel method proposal. The core contributions are:
1. Categorizing loss functions into four major types—regression, pairwise ranking, listwise ranking, and weighted—comprising 8 loss functions in total.
2. Systematically evaluating these losses on 13 tasks across 5 search spaces using multiple metrics.
3. Proposing PWLNAS: a piecewise loss function designed based on the observed complementarity among loss types.

### Key Designs

1. **Loss Function Taxonomy and Selection**:

    - **Regression Loss**: MSE, minimizing the discrepancy between predicted scores and ground-truth accuracy.
    - **Pairwise Ranking Losses**: Hinge Ranking (HR), Logistic Ranking (LR), MSE+Sequence Ranking (MSE+SR), focusing on the relative ordering of architecture pairs.
    - **Listwise Ranking Loss**: ListMLE, optimizing the consistency between the predicted ranking list and the ground-truth ranking.
    - **Weighted Losses**: Exponential Weighted (EW), MAPE, and Weighted Approximate-Rank Pairwise (WARP), assigning higher weights to high-performing architectures.
    - **Design Motivation**: The selection covers mainstream paradigms of predictor loss functions; WARP is introduced to NAS for the first time.

2. **Multi-dimensional Evaluation Metrics**:

    - **Kendall's Tau ($\tau$)**: Overall rank correlation.
    - **Precision@T**: The proportion of architectures predicted as top-$T$% that actually belong to the top-$T$% (higher is better).
    - **N@K**: The actual rank of the best architecture among the predicted top-$K$ (lower is better).
    - **Design Motivation**: Since the core objective of NAS is to identify the best architecture, top-$K$ metrics are more informative than global ranking metrics.

3. **Piecewise Loss PWLNAS**:

    - During the early iterations of predictor training, ranking or regression losses are used for warm-up.
    - In later iterations, the loss is switched to a weighted loss to better identify high-performing architectures.
    - The specific combination is task-dependent: HR→MAPE for NAS-Bench-201, ListMLE→WARP for NAS-Bench-101, and HR→MAPE for DARTS.
    - **Design Motivation**: This design exploits the observed complementarity—ranking losses perform better under small data regimes, while weighted losses are superior when sufficient data is available.

### Loss & Training

All experiments employ a unified GCN-based performance predictor. Each loss function is fairly tuned with different levels of hyperparameters. Training data is sampled randomly from the search space, and predictors are evaluated over the entire search space. All results are averaged over 30 independent runs.

## Key Experimental Results

### Main Results — Search Results on Each Benchmark

| Method | Loss | NAS-Bench-201 C10 | NAS-Bench-201 C100 | NAS-Bench-201 IN-16 |
|--------|------|-------------------|--------------------|--------------------|
| NASBOT | MSE | 6.36 | 28.62 | 54.12 |
| ReNAS | LR | 6.01 | 27.88 | 54.03 |
| NPENAS | MSE | 5.69 | 26.54 | 53.52 |
| **PWLNAS** | **PW** | **5.63** | **26.51** | **52.88** |
| Global Best | - | 5.63 | 26.49 | 52.69 |

| Method | Loss | NAS-Bench-101 Test Err. |
|--------|------|------------------------|
| BANANAS | MAPE | 5.92 |
| FlowerFormer | HR | 5.86 |
| NPENAS | MSE | 5.85 |
| PWLNAS-HR | HR | 5.83 |
| **PWLNAS-PW** | **PW** | **5.80** |

| Method | Loss | DARTS Test Err. | Params |
|--------|------|----------------|--------|
| GMAENAS | BPR | 2.50±0.03 | 3.6M |
| DCLP | ListMLE | 2.48±0.02 | 3.3M |
| **PWLNAS** | **PW** | **2.47±0.05** | **3.6M** |

### Ablation Study — Performance of Loss Functions under Different Conditions

| Predictor Backbone | Loss | N@10↓ | Ptop@0.5↑ | $\tau$↑ |
|--------------------|------|-------|-----------|---------|
| AP (MLP) | MSE | 250.94 | 4.41 | 0.43 |
| AP (MLP) | HR | 23.58 | 22.15 | 0.65 |
| AP (MLP) | **ListMLE** | **22.74** | **24.15** | **0.66** |
| AP (MLP) | WARP | 113.20 | 9.36 | 0.43 |
| PINAT (Trans.) | MSE | 146.60 | 8.62 | 0.62 |
| PINAT (Trans.) | HR | 8.44 | 29.32 | 0.67 |
| PINAT (Trans.) | **WARP** | **3.78** | **38.71** | 0.65 |

### Key Findings

- **Weighted losses are best with sufficient data**: WARP and MAPE consistently lead on top-$K$ metrics (Precision@0.5, N@10).
- **Ranking losses outperform under extremely small data**: When training data is very limited, ranking losses such as HR outperform weighted losses, as the latter over-emphasize locally good architectures.
- **MSE is worst at identifying top architectures**: It achieves the lowest Precision@0.5 in most search spaces.
- **The hybrid loss MSE+SR underperforms single losses**: Jointly optimizing two objectives results in a compromised performance.
- **Simple backbones pair better with ranking losses; complex backbones with weighted losses**: MLP-based predictors achieve the best results with ListMLE, while Transformer-based predictors benefit most from WARP.
- **Ground-truth weights outperform rank-based weights**: Using the actual accuracy of architectures as weights in weighted losses is more effective than using rank-based weights.
- **More training data does not necessarily improve top-$K$ performance**: The top-$K$ metrics of regression and ranking losses may degrade as training data increases.
- **PW piecewise loss consistently wins**: PWLNAS achieves the lowest error rates on NAS-Bench-201, NAS-Bench-101, and DARTS.

## Highlights & Insights

1. **First systematic study**: This work provides the most comprehensive empirical investigation of loss functions for NAS performance predictors, offering significant reference value to the community.
2. **Valuable discovered patterns**: The complementarity between weighted and ranking losses, and the interaction effect between data size and loss type, offer practically actionable guidance.
3. **Introduction of WARP**: The WARP loss, originally from multi-label image annotation, is introduced to NAS with notable effectiveness.
4. **Simple yet effective PW method**: Switching loss functions in a piecewise manner suffices to surpass state-of-the-art, demonstrating consistent gains with minimal implementation complexity.

## Limitations & Future Work

- The switching threshold in PW loss is fixed and relies on manual selection based on empirical experience.
- More flexible loss combination strategies are unexplored, such as gradually increasing the weighting strength rather than using a hard switch.
- Loss function designs that directly optimize top-$K$ metrics have not been considered.
- The training data sampling strategy also plays an important role but is not studied in depth; random sampling may not adequately represent the search space distribution.
- Validation on larger-scale search spaces (e.g., very large NAS benchmarks or realistic open search spaces) is absent.
- Adaptive mechanisms for selecting loss functions based on the search progress remain to be explored.

## Related Work & Insights

- The study forms a systematic comparison with methods such as GATES (HR), DCLP (ListMLE), and NAR-Former (MSE+SR).
- The discovered patterns can directly guide improvements to existing predictor-based methods by simply replacing the loss function.
- The complementarity observations can inspire other learning-to-rank scenarios, such as recommender systems and information retrieval.
- A deeper information-theoretic understanding of why different losses behave so differently under varying data regimes represents a valuable direction for future work.

## Rating

- **Novelty**: ⭐⭐⭐ The individual loss functions already exist; the core contributions lie in the systematic study and the PW combination, representing moderate innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covering 5 search spaces, 13 tasks, multiple predictor architectures, varying training data sizes, and 30-run averages—extremely thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Well-organized and logically structured with accurate summaries of findings, though the density of figures and tables is high.
- **Value**: ⭐⭐⭐⭐ Provides the NAS community with a practical guide for loss function selection; the PW method is simple yet consistently effective.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](../../ICLR2026/others/on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[AAAI 2026\] Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details](../../AAAI2026/others/learning_compact_latent_space_for_representing_neural_signed_distance_functions_.md)
- [\[NeurIPS 2025\] Note 5: ReSearch — Learning to Reason with Search](../../NeurIPS2025/others/research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[ICCV 2025\] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)

<!-- RELATED:END -->
