---
title: >-
  [Paper Note] Credal Ensemble Distillation for Uncertainty Quantification
description: >-
  [AAAI 2026][Model Compression][Knowledge Distillation] This paper proposes the Credal Ensemble Distillation (CED) framework, which distills a deep ensemble (DE) teacher into a single-model student called CREDIT. Rather than predicting a single softmax distribution, CREDIT outputs class probability intervals that define a credal set, achieving superior or comparable uncertainty estimation on OOD detection tasks while substantially reducing inference overhead (from 5× to 1×).
tags:
  - AAAI 2026
  - Model Compression
  - Knowledge Distillation
  - Deep Ensembles
  - Uncertainty Quantification
  - Credal Sets
  - OOD Detection
date: 2026-05-08
content_hash: 74585913d24e35af
---

# Credal Ensemble Distillation for Uncertainty Quantification

**Conference**: AAAI 2026
**arXiv**: [2511.13766](https://arxiv.org/abs/2511.13766)
**Code**: Unavailable (experimental code provided in supplementary material)
**Area**: Model Compression / Uncertainty Quantification
**Keywords**: Knowledge Distillation, Deep Ensembles, Uncertainty Quantification, Credal Sets, OOD Detection

## TL;DR

This paper proposes the Credal Ensemble Distillation (CED) framework, which distills a deep ensemble (DE) teacher into a single-model student called CREDIT. Rather than predicting a single softmax distribution, CREDIT outputs class probability intervals that define a credal set, achieving superior or comparable uncertainty estimation on OOD detection tasks while substantially reducing inference overhead (from 5× to 1×).

## Background & Motivation

Uncertainty quantification (UQ) for deep neural networks is critical for model trustworthiness and robustness. Uncertainty is decomposed into two types:
- **Aleatoric Uncertainty (AU)**: inherent randomness in the data-generating process
- **Epistemic Uncertainty (EU)**: uncertainty due to insufficient model knowledge

**Deep Ensembles (DE)** have emerged as a strong UQ baseline by combining multiple independently trained networks, effectively distinguishing AU from EU. However, their key bottleneck is **high inference cost**: $M$ models incur $M$-fold computation and storage.

Limitations of existing distillation approaches:

**Ensemble Distillation (ED)**: distills DE into a single neural network (SNN) outputting one softmax distribution, but loses EU information (a single distribution cannot express uncertainty about uncertainty).

**Ensemble Distribution Distillation (EDD)**: distills into a model outputting a Dirichlet distribution, but suffers from the lack of ground-truth Dirichlet labels; in practice, accuracy drops severely (74.56% vs. SNN's 91.79% on VGG16), and recent theoretical work has criticized its EU interpretation.

**Key Challenge**: How can a single forward pass simultaneously preserve predictive capability and EU quantification?

**Key Insight**: This work replaces single distributions or parametric distributions with **credal sets** (convex sets of probability distributions defined by class probability intervals) as a second-order uncertainty representation. Credal sets are naturally derived from the multiple predictive distributions of a DE and require no distributional assumptions such as those underlying the Dirichlet.

## Method

### Overall Architecture

CED consists of three steps: (1) a credal wrapper extracts probability intervals and the cross-probability from the $M$ softmax outputs of the DE teacher; (2) a CREDIT student model is designed to output a $\mathbb{R}^{2C+1}$ vector encoding the cross-probability, interval lengths, and a weight factor; (3) a novel loss function trains the student to match the teacher's credal information. At inference, the cross-probability is used for classification, while the full output reconstructs the credal set for UQ.

### Key Designs

1. **Credal Wrapper (Teacher Side)**:

    - **Function**: Extracts class probability intervals from the $M$ predictive distributions of the DE.
    - **Mechanism**: For each class $k$, the upper bound is $\overline{p}_k = \max_m p_{m,k}$ and the lower bound is $\underline{p}_k = \min_m p_{m,k}$. These intervals define the credal set $\mathbb{Q}$. A normalized cross-probability is then computed as $p^*_k = \underline{p}_k + \beta(\overline{p}_k - \underline{p}_k)$, where $\beta = (1-\sum_k \underline{p}_k)/\sum_k \Delta p_k$.
    - **Design Motivation**: The cross-probability is the most representative single-point estimate of a probability interval system. The factor $\beta$ ensures the cross-probability is properly normalized.

2. **CREDIT Student Architecture**:

    - **Function**: Modifies the final layer of a standard SNN to output $2C+1$ values.
    - **Mechanism**: The first $C$ logits pass through a softmax to yield the cross-probability $\mathbf{p}_S^*$; the next $C$ pass through a sigmoid to yield interval lengths $\Delta\mathbf{p}_S$; the final scalar passes through a sigmoid to yield the weight factor $\beta_S$. The probability intervals are reconstructed as $\underline{p}_{S,k} = p^*_{S,k} - \beta_S \Delta p_{S,k}$ and $\overline{p}_{S,k} = p^*_{S,k} + (1-\beta_S)\Delta p_{S,k}$.
    - **Design Motivation**: The key constraint is to guarantee that the reconstructed probability intervals are valid ($\underline{p} \leq p^* \leq \overline{p}$, $\sum \underline{p} \leq 1 \leq \sum \overline{p}$). This is ensured by the combination of softmax and sigmoid activations, as verified mathematically.

3. **Distillation Loss**:

    - **Function**: Trains CREDIT to match the DE teacher's credal information.
    - **Mechanism**: $\mathcal{L}_{ced} = \text{CE}(\mathbf{p}^*, \mathbf{p}_S^*) + \text{MSE}(\Delta\mathbf{p}, \Delta\mathbf{p}_S) + \text{MSE}(\beta, \beta_S)$. The first term (cross-entropy) preserves predictive performance; the latter two (MSE) capture the imprecision of the credal set.
    - **Design Motivation**: The credal set distillation objective is decomposed into three independently optimizable targets: accurate classification, interval width, and interval position. Temperature scaling ($T=2.5$) is supported.

### Uncertainty Quantification

From the reconstructed credal set $\mathbb{Q}_S$ of CREDIT, the following quantities are computed via constrained optimization:
- **TU** (Total Uncertainty) = maximum Shannon entropy $\overline{H}(\mathbb{Q}_S)$ (the maximum-entropy distribution within the credal set)
- **AU** (Aleatoric Uncertainty) = minimum Shannon entropy $\underline{H}(\mathbb{Q}_S)$
- **EU** (Epistemic Uncertainty) = $\overline{H} - \underline{H}$ (interval width reflects insufficient model knowledge)

## Key Experimental Results

### Main Results (VGG16, CIFAR-10 vs. SVHN OOD Detection)

| Method | AUROC(EU) | AUROC(TU) | AUPRC(EU) | AUPRC(TU) | Inference Time |
|--------|-----------|-----------|-----------|-----------|----------------|
| DE (5×) | 89.99 | 91.53 | 93.78 | 95.09 | 5×2.22s |
| SNN | / | 89.44 | / | 93.71 | 2.22s |
| ED | / | 91.07 | / | 94.51 | 2.22s |
| EDD* | 90.94 | 90.96 | 93.66 | 93.78 | 2.22s |
| MCDO | 51.42 | 89.12 | 74.72 | 93.64 | 2.22s |
| **CED** | **93.56** | **92.51** | **96.09** | **95.21** | **2.26s** |

### Ablation Study (ResNet50 + CIFAR-10-C OOD)

| Method | AUROC(EU)↑ | AUROC(TU)↑ | Accuracy | Note |
|--------|-----------|-----------|----------|------|
| DE | 87.78 | 94.08 | 93.40 | 5-model ensemble, performance upper bound |
| CED | **96.80** | 95.23 | 91.77 | Single model; EU surpasses DE |
| ED | / | 94.09 | 92.02 | No EU estimation capability |
| EDD* | 89.48 | 91.04 | 80.38 | Severe accuracy degradation |

### Key Findings

- **CED's EU estimation significantly outperforms all baselines**: On VGG16/SVHN, CED achieves EU-AUROC of 93.56%, substantially surpassing DE (89.99%) and EDD* (90.94%), indicating that credal sets capture EU more faithfully than DE's discrete sampling or the Dirichlet distribution.
- **CED does not compromise accuracy**: CED (92.23%) is on par with ED (92.18%) and SNN (91.79%), whereas EDD accuracy collapses to 74.56% on VGG16.
- **EU vs. TU**: CED performs better at OOD detection using EU than TU, whereas other methods benefit more from TU, indicating a qualitative improvement in CED's EU estimation.
- **Inference efficiency**: CED inference time (2.26s) is nearly identical to SNN (2.22s), compared to 5×2.22s = 11.1s for DE.
- **Ensemble size ablation**: DE performance continues to improve with ensemble size, whereas CED nearly converges at $M=5$, demonstrating the effectiveness of distillation.
- **Temperature scaling**: $T=2.5$ yields the best performance; excessively high temperatures ($T=10$) degrade results.
- **Medical Imaging Case Study** (Camelyon17): CED achieves EU AUARC of 97.12% under OOD settings, outperforming DE (95.92%).

## Highlights & Insights

- Introducing credal sets into knowledge distillation is an elegant innovation: credal sets, as a second-order representation, are more flexible than Dirichlet (requiring no distributional assumptions) and more compact than DE (single model).
- The CREDIT architecture is minimally invasive, adding only $C+1$ output nodes with no modifications to the backbone.
- The paper mathematically proves that CREDIT's output probability intervals are always valid (satisfying credal set conditions), providing an important correctness guarantee from an engineering perspective.
- The loss function design is conceptually clear: CE preserves classification and MSE preserves imprecision, without requiring complex learning strategies as in EDD.

## Limitations & Future Work

- **Scalability to large label spaces**: When $C$ is large (100 or 1,000 classes), softmax produces very small probability values, which may destabilize the regression loss.
- **Calibration**: CED's ECE (6.71%) is considerably higher than DE's (1.46%), indicating that calibration requires further improvement.
- **Optimization overhead**: Computing $\overline{H}$ and $\underline{H}$ requires solving constrained optimization problems, which may incur non-negligible cost when $C > 10$.
- **Evaluation limited to classification**: Generalization to regression, detection, and other tasks remains unexplored.
- **Dependence on teacher quality**: CED's performance ceiling is bounded by the DE teacher.

## Related Work & Insights

- Compared to BNNs: CED does not require a posterior distribution over weights, making training considerably simpler.
- Compared to EDD: CED avoids both the missing ground-truth Dirichlet labels and the accuracy degradation associated with EDD.
- Credal sets have a rich theoretical foundation in classical machine learning (Levi 1980, imprecise probability theory); their introduction into deep learning distillation represents an effective bridge between theory and practice.
- **Insight**: Probability intervals are better suited than point estimates or parametric distributions for expressing "what the model does not know."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First proposal of credal sets combined with distillation; theoretically well-motivated)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple backbones, datasets, ablations, and a medical imaging case study)
- Writing Quality: ⭐⭐⭐⭐ (Content-dense but clearly structured)
- Value: ⭐⭐⭐⭐⭐ (Strong candidate for a new standard method in UQ; highly practical)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Ensemble++: Scalable Exploration via Ensemble](../../NeurIPS2025/model_compression/scalable_exploration_via_ensemble.md)
- [\[AAAI 2026\] CTPD: Cross Tokenizer Preference Distillation](ctpd_cross_tokenizer_preference_distillation.md)
- [\[AAAI 2026\] Condensed Data Expansion Using Model Inversion for Knowledge Distillation](condensed_data_expansion_using_model_inversion_for_knowledge_distillation.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)
- [\[AAAI 2026\] EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training](eeg-dlite_dataset_distillation_for_efficient_large_eeg_model_training.md)

</div>

<!-- RELATED:END -->
