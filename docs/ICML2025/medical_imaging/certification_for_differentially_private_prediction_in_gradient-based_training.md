---
title: >-
  [Paper Note] Certification for Differentially Private Prediction in Gradient-Based Training
description: >-
  [ICML2025][Medical Imaging][Differential Privacy] An Abstract Gradient Training (AGT) framework is proposed to compute the upper bounds of the reachable set of model parameters during training using convex relaxation and bound propagation techniques. This leverages the smooth sensitivity mechanism to significantly tighten the privacy analysis of private prediction, achieving privacy bounds several orders of magnitude tighter than global sensitivity on medical imaging and NLP…
tags:
  - "ICML2025"
  - "Medical Imaging"
  - "Differential Privacy"
  - "Private Prediction"
  - "Smooth Sensitivity"
  - "Parameter-Space Bounds"
  - "Interval Propagation"
  - "Natural Language Processing"
date: 2026-05-08
content_hash: 010d8b4558b343dd
---

# Certification for Differentially Private Prediction in Gradient-Based Training

**Conference**: ICML2025  
**arXiv**: [2406.13433](https://arxiv.org/abs/2406.13433)  
**Code**: To be confirmed  
**Area**: Medical Images  
**Keywords**: Differential Privacy, Private Prediction, Smooth Sensitivity, Parameter-Space Bounds, Interval Propagation, Medical Imaging, Natural Language Processing

## TL;DR

An Abstract Gradient Training (AGT) framework is proposed to compute the upper bounds of the reachable set of model parameters during training using convex relaxation and bound propagation techniques. This leverages the smooth sensitivity mechanism to significantly tighten the privacy analysis of private prediction, achieving privacy bounds several orders of magnitude tighter than global sensitivity on medical imaging and NLP tasks.

## Background & Motivation

Differential privacy (DP) is a core tool for protecting user data privacy in machine learning models. Existing works follow two main paradigms:

**Private Training** (e.g., DP-SGD): Injecting noise during training to ensure the final model parameters satisfy DP. The disadvantages are that privacy parameters must be fixed beforehand, the training cost is high, and it may introduce bias against specific groups.

**Private Prediction**: Injecting noise to the outputs of non-private models. The advantages include dynamic adjustment of privacy budgets and easy integration with complex training paradigms like federated learning. However, existing methods rely on global sensitivity $\text{GS}(f)$, leading to a privacy-utility trade-off far inferior to private training.

Recent auditing studies (Chadha et al., 2024) indicate that the privacy analysis of private prediction has significant slackness. Targeting this slackness, this paper proposes a verification-centric framework to tighten the privacy guarantees of private prediction.

**Key Insight**: The primary source of slackness in private prediction lies in the use of global prediction sensitivity—defined as the maximum variation in prediction across all observable datasets. In practice, for a given dataset, local sensitivity can be far smaller than global sensitivity.

## Method

### 1. Prediction Stability Certification

Define **Prediction Stability**: For a model $f$ trained on dataset $D$, the prediction $f_x(D)$ at query point $x$ is stable at distance $k$ if and only if:

$$\|f_x(D) - f_x(D')\|_1 = 0, \quad \forall D': d(D, D') \leq k$$

meaning that adding or removing at most $k$ samples in the training set does not change the prediction result.

### 2. Valid Parameter-Space Bounds

The key idea is to transform the optimization problem in the dataset space into a verification problem in the parameter space. Define the valid parameter-space bound $T^k \subseteq \Theta$: for all datasets $D'$ with distance no more than $k$ from $D$, the trained parameters lie within $T^k$.

**Lemma 4.4** states that: proving the prediction of $f^\theta(x)$ remains unchanged for all $\theta \in T^k$ is sufficient to certify that the prediction is stable at distance $k$.

### 3. Abstract Gradient Training (AGT) Algorithm

The AGT algorithm (Algorithm 1) simultaneously maintains parameter interval bounds $[\theta_L, \theta_U]$ during standard SGD training:

- **Initialization**: $[\theta_L, \theta_U] \leftarrow [\theta', \theta']$ (a point interval of initial parameters)
- **For each batch**:
    - Compute the standard gradient update $\Delta\theta$
    - Compute the lower and upper bounds $[\Delta\theta_L, \Delta\theta_U]$ of the set of all possible descent directions $\Delta\Theta$
    - Update parameter bounds using interval arithmetic: $\theta_L \leftarrow \theta_L - \alpha \Delta\theta_U$, $\theta_U \leftarrow \theta_U - \alpha \Delta\theta_L$

**Theorem 4.6** provides the method for computing the bounds of descent directions, which fundamentally utilizes SEMax/SEMin operations (summing element-wise top/bottom-$a$ elements) and introduces gradient clipping $\text{Clip}_\gamma$ to bound single-sample gradient contributions:

$$\Delta\theta_L = \frac{1}{b}\left(\underset{b-k}{\text{SEMin}}\{\delta_L^{(i)}\}_{i=1}^b - k\gamma \mathbf{1}_d\right)$$

$$\Delta\theta_U = \frac{1}{b}\left(\underset{b-k}{\text{SEMax}}\{\delta_U^{(i)}\}_{i=1}^b + k\gamma \mathbf{1}_d\right)$$

where $\delta_L^{(i)}$ and $\delta_U^{(i)}$ are single-sample gradient bounds computed via Interval Bound Propagation (IBP).

### 4. Smooth Sensitivity Upper Bound

**Theorem 5.1**: If prediction $f_x(D)$ is stable at distance $k'$, then the $\beta$-smooth sensitivity satisfies:

$$\text{SS}^\beta(f_x, D) \leq e^{-\beta k'}$$

When $k' \gg 1$, compared to the global sensitivity $\text{GS}(f_x) = 1$, the smooth sensitivity bound can be exponentially smaller. The corresponding private prediction mechanism uses Cauchy noise:

$$z \sim \text{Cauchy}\left(\frac{6 \exp(-\epsilon k'/6)}{\epsilon}\right)$$

### 5. Private Aggregation for Ensemble Models

For an ensemble of $T$ sub-models, **Theorem 5.3** aggregates single-model stable distances into an ensemble stable distance $K = \sum_{i \in S_n} k^{(i)} + n - 1$, where $S_n$ is the index set of the $n$ models with the smallest stable distances, and $n = \lceil |n_1(x) - n_0(x)| / 2 \rceil$.

## Key Experimental Results

### Experimental Setup

| Dataset | Task | Model |
|--------|------|------|
| Blobs | Synthetic Binary Classification | Logistic Regression |
| OctMNIST | Retina OCT Medical Imaging | CNN (Fine-tuning the last FC layer) |
| IMDB | Sentiment Classification | Neural Network + GPT-2 Embeddings |

### Main Results: Performance Comparison of Teacher Mechanisms

Privacy budget $(ϵ,δ) = (10, 10^{-5})$, $Q=100$ queries:

| Teacher Mechanism | Blobs | OctMNIST | IMDB |
|-------------|-------|----------|------|
| Single Model + Global Sensitivity (GS) | 82.8 | 12.7 | 54.4 |
| Single Model + Smooth Sensitivity (SS) | **99.8** | 18.7 | **73.5** |
| Subsampled Aggregation + GS | 99.5 | 14.1 | 73.0 |
| Subsampled Aggregation + SS | 98.1 | **19.8** | 71.7 |
| DP-SGD | 1.0 | **81.2** | 70.5 |

### Key Findings

- Single Model + SS improves from 82.8% to **99.8%** on Blobs, and from 54.4% to **73.5%** on IMDB.
- The smooth sensitivity bound is **several orders of magnitude** tighter than global sensitivity.
- For small-query scenarios ($Q < 100$), Private Prediction + SS can **outperform** DP-SGD.
- Single model SS increases the maximum usable queries on IMDB by approximately **one order of magnitude**.
- When the ensemble size $T \leq 20$, SS is comparable to or better than GS; as $T$ increases, the reduced data per sub-model leads to looser bounds.

## Highlights & Insights

1. **Cross-disciplinary Application of Verification Techniques**: Translating interval propagation from neural network verification (adversarial robustness certification) to differential privacy analysis for the first time, opening up a brand-new research direction.
2. **High Practicality**: Private prediction allows dynamic adjustment of privacy budgets, making it suitable for users with different access permissions and sensitivity scenarios, ideal for actual deployment.
3. **Solid Theoretical Foundation**: Establishes a complete certification chain from parameter-space bounds to prediction stability and then to smooth sensitivity, with rigorous theoretical guarantees for each step.
4. **Computational Feasibility**: A single run of AGT is only 2-4x that of standard training, and 20-40x in total yields most of the privacy benefits.

## Limitations & Future Work

1. **Limited Multi-class Performance**: Only binary classification is currently considered. The interval relaxation for multi-class cross-entropy is particularly loose, and the bounds may significantly degrade when generalized to multi-class classification.
2. **Reliance on Large Batches / Few Epochs**: Bound propagation takes the worst-case scenario at each iteration. Obtaining meaningful guarantees requires using larger batch sizes or fewer epochs.
3. **Ensemble Scale Bottleneck**: As the ensemble size increases, the data for each sub-model decreases, causing the single-model stable distance bounds to degrade.
4. **Limited Improvement on OctMNIST**: In scenarios where data is not easily separable or the dataset size is small, the improvement of SS compared to GS is not significant.
5. **No Integration with Tighter Aggregation Mechanisms**: Advanced aggregation strategies like PATE's Confident GNMax could potentially yield further improvements.

## Related Work & Insights

- **DP-SGD** (Abadi et al., 2016): The benchmark method for private training and the primary baseline in this paper.
- **PATE** (Papernot et al., 2016): Private prediction based on ensemble voting, upon which this paper introduces smooth sensitivity.
- **Smooth Sensitivity** (Nissim et al., 2007): The core privacy tool utilized in this paper, which uses local sensitivity to provide tighter noise calibration than global sensitivity.
- **Interval Propagation** (Gowal et al., 2018; Wicker et al., 2022): Bound propagation techniques from adversarial robustness verification, creatively applied to the reachable set analysis of the training process.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Introduces a verification framework to private prediction for the first time, opening up a new direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three categories of tasks: synthetic, medical, and NLP, but lacks multi-class and large-scale experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — The theoretical proofs are clear and rigorous; while there are many notation symbols, it is rationally organized.
- **Value**: ⭐⭐⭐⭐ — Provides a practical and feasible improvement path for private prediction, showing real-world deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures](sgd_jittering_a_training_strategy_for_robust_and_accurate_model-based_architectu.md)
- [\[AAAI 2026\] Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection](../../AAAI2026/medical_imaging/personality-guided_public-private_domain_disentangled_hypergraph-former_network_.md)
- [\[NeurIPS 2025\] Interpretable Next-token Prediction via the Generalized Induction Head](../../NeurIPS2025/medical_imaging/interpretable_next-token_prediction_via_the_generalized_induction_head.md)
- [\[ICML 2026\] CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support](../../ICML2026/medical_imaging/cascade_conformal_prediction_uncertainty-adaptive_prediction_intervals_for_two-s.md)
- [\[CVPR 2025\] Distilled Prompt Learning for Incomplete Multimodal Survival Prediction](../../CVPR2025/medical_imaging/distilled_prompt_learning_for_incomplete_multimodal_survival_prediction.md)

</div>

<!-- RELATED:END -->
