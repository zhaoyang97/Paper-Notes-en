---
title: >-
  [Paper Note] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One
description: >-
  [NeurIPS 2025][LLM Evaluation][Model Recovery] Through large-scale model recovery experiments, this paper demonstrates that even with 4.5 million behavioral data points…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Model Recovery"
  - "Representational Alignment"
  - "Linear Probing"
  - "Identifiability"
  - "THINGS Dataset"
date: 2026-05-08
content_hash: b9cae38fe15f8e48
---

# Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One

**Conference**: NeurIPS 2025
**arXiv**: [2510.23321](https://arxiv.org/abs/2510.23321)  
**Code**: [GitHub](https://github.com/brainsandmachines/oddoneout_model_recovery)  
**Area**: Computational Neuroscience / Representational Alignment
**Keywords**: Model Recovery, Representational Alignment, Linear Probing, Identifiability, THINGS Dataset

## TL;DR

Through large-scale model recovery experiments, this paper demonstrates that even with 4.5 million behavioral data points, flexible evaluation methods based on linear probing achieve model recovery accuracy below 80% across 20 visual models. This reveals a fundamental trade-off between predictive accuracy and model identifiability, challenging the prevailing paradigm that the best-fitting model is the most appropriate one.

## Background & Motivation

Representations from deep neural networks (DNNs) are widely used as computational models of biological visual systems. The standard evaluation pipeline involves extracting ANN representations, aligning them to brain/behavioral data via some metric, and designating the model with the highest predictive accuracy as the best biological representational model.

Flexible, data-driven alignment methods (e.g., linear probing) substantially improve predictive accuracy—but this raises a critical question: **does predictive accuracy genuinely reflect representational similarity?**

Limitations of Prior Work:
- Kornblith et al. found that non-cross-validated flexible metrics fail to distinguish between layers (though this may be attributed to overfitting)
- Han et al. tested under idealized settings (noise-free ANN activations), which does not reflect real noisy data
- Schütt et al. validated the recovery capacity of non-flexible RSA but did not evaluate flexible RSA in noise-calibrated settings

Root Cause: Flexible evaluation improves predictive accuracy, potentially at the cost of model identifiability. The authors use the THINGS odd-one-out dataset (4.7 million behavioral judgments) to quantitatively investigate this trade-off.

## Method

### Overall Architecture

The paper adopts a model recovery experimental design: synthetic behavioral data are generated from model A → all models (including A) compete to fit the data → it is assessed whether model A can be correctly identified. If the best-fitting model is not the data-generating model, the evaluation method suffers from an identifiability problem.

### Key Designs

1. **Mapping from ANN representations to behavioral predictions**: For each pretrained ANN, the final representation layer $\mathbf{X} \in \mathbb{R}^{n \times p}$ ($n=1854$ images) is extracted, and a linear transformation $\mathbf{W} \in \mathbb{R}^{p \times p}$ is learned. The similarity matrix is $\mathbf{S} = (\mathbf{X}\mathbf{W})(\mathbf{X}\mathbf{W})^\top$. Odd-one-out predictions for triplets $\{a,b,c\}$ use a softmax:

    $p(\text{odd-one-out}=a \mid \{a,b,c\}) = \frac{\exp(S_{b,c}/T)}{\exp(S_{a,b}/T) + \exp(S_{a,c}/T) + \exp(S_{b,c}/T)}$

   Optimization minimizes negative log-likelihood plus regularization via L-BFGS.

2. **Improved regularization**: Standard Frobenius norm regularization is replaced by "shrinkage toward a scalar matrix":
    $\mathcal{R}(\mathbf{W}) = \min_\gamma \|\mathbf{W} - \gamma\mathbf{I}\|_F^2 = \|\mathbf{W}\|_F^2 - \frac{(\text{tr}(\mathbf{W}))^2}{p}$
   This avoids the degenerate behavior of Frobenius regularization, which can cause performance to drop below zero-shot under strong penalty.

3. **Noise calibration**: Rather than maximizing predictive likelihood, the temperature parameter $T$ is tuned so that the model's response variability matches the human noise ceiling (67.8% leave-one-subject-out consistency). This ensures that synthetic data have noise levels consistent with real experiments.

4. **Model recovery experimental pipeline**:

    - 20 diverse ANNs (varying architectures and training objectives)
    - Each model is first fit to the full human dataset to obtain $\mathbf{W}$ and a calibrated temperature
    - Synthetic behavioral data are sampled from the calibrated model
    - All models fit the synthetic data from scratch
    - 3-fold cross-validation (over different image subsets) is used to compare predictive accuracy
    - 30 random seeds × 20 generative models × 18 dataset sizes

### Identifiability Analysis

Regression analysis is used to identify causes of model misidentification:
- **Alignment-induced representational geometry shift** of candidate models (positive predictor of accuracy difference)
- **Magnitude of shift** of the data-generating model (negative predictor—models that shift more produce data more easily predicted by other models)
- **Effective dimensionality** (ED) of the data-generating model (negative predictor—higher-dimensional representations are harder to correctly recover)

## Key Experimental Results

### Main Results: Model Recovery Accuracy vs. Data Volume

| Training Triplets | Model Recovery Accuracy | Notes |
|-------------------|------------------------|-------|
| ~1,000 | <10% | Near chance (5%) |
| ~10,000 | ~15% | |
| ~100,000 | ~45% | Typical experimental scale |
| ~1,000,000 | ~70% | |
| 4,200,000 | <80% | Maximum data volume, still not saturated |

### Flexibility vs. Accuracy vs. Identifiability Trade-off

| Evaluation Method | Mean Predictive Accuracy | Model Recovery Accuracy (4.2M) |
|-------------------|--------------------------|-------------------------------|
| Zero-shot | ~34% | ~95% |
| Diagonal $\mathbf{W}$ | ~47% | ~85% |
| $p \times 10$ rectangular $\mathbf{W}$ | ~55% | ~75% |
| $p \times p$ full matrix | ~63% (near ceiling) | <80% |

### Ablation Study

| Control Variable | Change in Recovery Accuracy | Notes |
|------------------|-----------------------------|-------|
| Fixed PCA to 500 dimensions | No improvement | Parameter count is not the main factor |
| Expanded to 30 models | Drops to ~70% | More competitors make discrimination harder |
| Grouped by training objective | 73.7% | Difficult even within objective categories |
| Grouped by architecture | 70.3% | CNN vs. ViT also hard to distinguish |

### Key Findings

- **Systematic bias**: OpenAI CLIP ResNet-50 is systematically misidentified as the best model; 4 models have mean rank >2 (i.e., more than one competitor ranks higher on average)
- **Representational geometry shift**: After linear probing, all models converge toward VICE (a human embedding model); models initially farther from VICE exhibit larger shifts
- **Three significant regression predictors** (after Bonferroni correction): candidate model shift ($\beta=0.495$, $p=0.02$), generative model shift ($\beta=-0.251$, $p=0.01$), and generative model effective dimensionality ($\beta=-0.455$, $p=0.01$)

## Highlights & Insights

- **Rigorous quantitative proof that "best fit ≠ most correct"**: This is not a philosophical argument but a large-scale simulation-based validation
- **Noise calibration as a key innovation**: Prior model recovery studies used noise-free ANN activations, which are unrepresentative of real conditions. Temperature calibration matches synthetic data noise to human levels, yielding sobering results
- **Shrinkage-to-scalar-matrix regularization** is a small but practically important modification that avoids the degenerate behavior of the standard approach
- **The experimental design is analogous to knowledge distillation**—candidate models act as "students" attempting to imitate the behavior of the "teacher" (data-generating model)

## Limitations & Future Work

- Restricted to behavioral data (THINGS odd-one-out); neural data (fMRI/EEG) may exhibit different trade-off characteristics
- Quantitative model recovery results depend on the specific candidate model set (20 models)
- In practice, the "true model" (biological representation) is absent from the candidate set, making the problem even harder
- The paper proposes three improvement directions without implementing them: (1) active stimulus selection, (2) biologically constrained metrics, and (3) models with built-in alignment capacity

## Related Work & Insights

- Complements Kornblith et al.'s comparison of CKA vs. linear encoding: CKA is more conservative but potentially more reliable
- Muttenthaler et al. (2023)'s large-scale THINGS study serves as the direct foundation for this work
- Serves as a cautionary message for the broader representational alignment community: optimizing predictive accuracy may be a misleading objective
- Suggests that adaptive, model-discriminating stimulus design may be more effective than simply collecting more data

## Rating

- Novelty: ⭐⭐⭐⭐ The model recovery paradigm itself is not new, but noise calibration and large-scale application constitute important contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive design spanning 20 models × 18 data sizes × 30 random seeds
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, experimental design is rigorous, and discussion is thorough
- Value: ⭐⭐⭐⭐⭐ Has fundamental methodological implications for computational neuroscience; an exemplary instance of a negative result

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](bayesian_evaluation_of_large_language_model_behavior.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[NeurIPS 2025\] Reliably Detecting Model Failures in Deployment Without Labels](reliably_detecting_model_failures_in_deployment_without_labels.md)
- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[NeurIPS 2025\] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)

</div>

<!-- RELATED:END -->
