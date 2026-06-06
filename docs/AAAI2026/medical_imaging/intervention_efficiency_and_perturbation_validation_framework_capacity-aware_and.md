---
title: >-
  [Paper Note] Intervention Efficiency and Perturbation Validation Framework: Capacity-Aware and Robust Clinical Model Selection under the Rashomon Effect
description: >-
  [AAAI 2026][Medical Imaging][Rashomon Effect] To address the challenge of model selection under the Rashomon Effect—where multiple models achieve similar performance on small…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Rashomon Effect"
  - "Model Selection"
  - "Intervention Efficiency"
  - "Perturbation Validation"
  - "Class Imbalance"
  - "Clinical Deployment"
date: 2026-05-08
content_hash: a084574a7b232db6
---

# Intervention Efficiency and Perturbation Validation Framework: Capacity-Aware and Robust Clinical Model Selection under the Rashomon Effect

**Conference**: AAAI 2026
**arXiv**: [2511.14317](https://arxiv.org/abs/2511.14317)  
**Code**: [GitHub](https://github.com/YuwenZhang-Peter/PVF-IE)  
**Area**: Medical Imaging / Clinical Machine Learning
**Keywords**: Rashomon Effect, Model Selection, Intervention Efficiency, Perturbation Validation, Class Imbalance, Clinical Deployment

## TL;DR

To address the challenge of model selection under the Rashomon Effect—where multiple models achieve similar performance on small, class-imbalanced clinical datasets—this paper proposes **Intervention Efficiency (IE)**, a capacity-aware evaluation metric, and the **Perturbation Validation Framework (PVF)**, a robustness validation framework, jointly enabling reliable model selection under resource constraints.

## Background & Motivation

### Core Challenges in Clinical Predictive Modeling

**Small samples + class imbalance**: High data acquisition costs and strict ethical constraints in clinical settings result in extremely rare positive events (e.g., adverse reactions), rendering traditional accuracy metrics misleading.

**Rashomon Effect (model multiplicity)**: On small datasets, different models (e.g., logistic regression, SVM, random forest) may achieve similar performance while relying on entirely different feature subsets, making the question of "which model to choose" non-trivial.

**Limitations of Prior Work**:
- F1 Score ignores true negatives and may favor suboptimal models
- AUC-ROC overestimates performance on imbalanced data
- AUC-PR, though more suitable for rare events, is sensitive to prevalence and offers poor clinical interpretability

**Validation instability**: Small datasets amplify variance, causing model rankings to fluctuate drastically across different data splits, making single-split validation results unreliable.

### Core Motivation

- **Resource constraints are neglected**: Clinical settings often allow intervention for only a limited number of patients, yet existing metrics do not account for "intervention capacity."
- **Robustness is overlooked**: Traditional validation provides only point estimates without assessing model stability under data perturbation.
- **A single deployable model is required**: Clinical practice prioritizes interpretable single-model predictions over ensemble methods.

## Method

### Overall Architecture

Two complementary tools are proposed:

```
┌─────────────────────────────────────────────────────┐
│        Candidate Model Set F = {f₁, f₂, ..., f_Q}   │
├───────────────┬─────────────────────────────────────┤
│  IE Evaluation │        PVF Robustness Screening     │
│ (Capacity-Aware)│   (Perturbation + Aggregation)     │
│               │  Original Val Set → M Perturbed Sets │
│  IE_γ(f,D)    │  → Evaluate all models on M sets     │
│               │  → Aggregate into a single score     │
├───────────────┴─────────────────────────────────────┤
│          Select f* = argmax A_f                      │
└─────────────────────────────────────────────────────┘
```

### Key Design 1: Intervention Efficiency (IE)

**Core Idea**: Quantify how many additional true positive cases a model-guided intervention captures over random intervention, given a finite intervention capacity $\gamma$ (the ratio of intervened individuals to the total population).

**Closed-form formula**:

$$IE_\gamma(f) = \frac{s \cdot p + (\gamma - s) \cdot \frac{\pi - s \cdot p}{1 - s}}{\gamma \cdot \pi}$$

where:
- $p$ = precision, $r$ = recall, $\pi$ = prevalence
- $s = \min(\gamma, \frac{\pi r}{p})$, representing the actual proportion of model-guided interventions that can be utilized
- $\gamma = c / \beta$, the ratio of intervention capacity to total population size

**Two operating regimes**:
- **Resource-scarce (Regime A)**: The number of model-predicted positives exceeds capacity $c$; only the top-$c$ predictions are intervened upon, and IE simplifies to $\beta p / \alpha$
- **Resource-abundant (Regime B)**: Intervention capacity covers all model-predicted positives; remaining capacity is allocated randomly

### Key Design 2: Perturbation Validation Framework (PVF)

**Procedure**:
1. Fix the original validation set $\mathcal{D}_{val}$
2. Independently perturb each sample's features to generate $M$ perturbed validation sets (each containing $k \cdot n$ samples)
3. Evaluate all candidate models on each perturbed validation set
4. Aggregate $M$ scores per model using an aggregation function $\mathcal{A}$ (e.g., 25th percentile) into a single robustness score
5. Select the model with the highest aggregated score: $f^* = \arg\max_{f \in \mathcal{F}} A_f$

**Perturbation mechanism** (by feature type):

| Feature Type | Perturbation Method | Control Parameter |
|---|---|---|
| Numerical | Add Gaussian noise $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ | Noise std $\sigma$ |
| Categorical (nominal) | Randomly flip to another category with probability $\xi$ (uniform sampling) | Flip probability $\xi$ |
| Ordered (ordinal) | Sample neighboring categories with distance-decay probability $\xi$ | Flip probability $\xi$, decay parameter $\lambda$ |

**Key hyperparameters**:
- $d$: number of features to perturb
- $k$: number of copies per original sample (preserving distribution and class imbalance)
- $M$: number of perturbed validation sets
- $\mathcal{A}$: aggregation function (Q1, i.e., 25th percentile, used in experiments)

**Computational complexity**: $\mathcal{O}(Q \cdot M \cdot k \cdot n \cdot d)$; no model retraining required.

### Why Not Perturb Labels in the Validation Set?

Label perturbation is explicitly excluded: flipping labels disproportionately penalizes stronger models (as their previously correct predictions are flipped) while minimally affecting weaker models, compressing or even reversing performance differences. On highly imbalanced datasets, even flipping a small number of labels can drastically alter precision and recall.

### Theoretical Guarantees

The paper provides a complete theoretical analysis of PVF in the appendix:
- **Proposition B.1**: Perturbed scores exhibit an i.i.d. structure
- **Propositions B.2–B.4**: PVF scores converge as $M$, $k$, $n \to \infty$, respectively
- **Proposition B.5**: Uniform convergence guarantee
- **Proposition B.6**: Selection consistency of PVF — asymptotically selects the optimal model
- **Proposition B.7**: PVF is essentially an optimized selection toward a user-specified property $\Phi_{A,K}$

## Key Experimental Results

### Main Results: Synthetic Data

**Setup**: 2 informative features + 3 noise features, 10 feature combinations → 10 candidate logistic regression models, 5,000 repetitions.

| Configuration | Sample size $n$ | Class separation $\mu$ | Perturbation noise $\sigma$ | Total combinations |
|---|---|---|---|---|
| Sample size | 50, 100 | — | — | 2 |
| Separation | — | 0.1–2.9 (step 0.2) | — | 15 |
| Perturbation noise | — | — | 1e-6 ~ 0.1 | 6 |
| **Total** | | | | **180 × 5,000** |

**Key findings** (Figure 3):
- At $\gamma=0.1$ and $\gamma=0.3$, PVF outperforms traditional methods in approximately **90%** of configurations
- The advantage diminishes as $\gamma$ increases but remains consistent
- PVF also consistently outperforms traditional methods under F1/accuracy metrics

### Main Results: Real Clinical Data

**Datasets**:
- **Cervical cancer dataset** (808 samples, 34 features): optimal $\sigma=0.01$ (IE) / $\sigma=1e{-6}$ (F1)
- **Breast cancer dataset** (569 samples, 30 features): optimal $\sigma=0.2$–$0.3$

| Dataset | $\gamma$ | PVF Win Rate | Traditional Win Rate | Tie |
|---|---|---|---|---|
| Cervical | 0.1 | **60.0%** | 26.7% | 13.3% |
| Cervical | 0.3–0.9 | 43.3–46.7% | 33.3–36.7% | ~20% |
| Cervical | F1 | **50.0%** | 33.3% | 16.7% |
| Breast | 0.1 | **52.0%** | 20.0% | 28.0% |
| Breast | 0.3 | **48.0%** | 20.0% | 32.0% |
| Breast | F1 | **52.0%** | 16.0% | 32.0% |

### Ablation Study: $\sigma$ Sensitivity Analysis

| Scenario | Optimal $\sigma$ Range | Key Pattern |
|---|---|---|
| Low separation ($\mu \leq 0.9$) | $\sigma \leq 1e{-3}$ | Small perturbations yield stable positive gains |
| Medium separation ($\mu$ 1.1–1.9) | Small $\sigma$ or $\sigma=0.1$ | Moderate perturbations reduce the advantage; largest perturbation recovers it |
| High separation ($\mu \geq 2.1$) | $\sigma=0.1$ | Large perturbations consistently amplify PVF advantage |
| Cervical cancer (real) | $\sigma \approx 0.01$ | Small $\sigma$ most effective |
| Breast cancer (real) | $\sigma \approx 0.2$–$0.3$ | Larger $\sigma$ required |

**Key Findings**:
- $\sigma$ is the most critical hyperparameter of PVF; no universally optimal value exists and dataset-specific tuning is necessary
- $\sigma=0.01$ is a reasonable empirical starting point
- PVF's advantage is most pronounced when $\gamma$ is small (i.e., intervention capacity is highly constrained)
- PVF requires no model retraining, keeping computational overhead manageable

## Highlights & Insights

1. **Novelty of IE**: For the first time, intervention capacity constraints are explicitly incorporated into an evaluation metric; the closed-form formula is elegant and interpretable, directly linking the precision–recall tradeoff to clinical resource limitations.
2. **Flexibility of PVF**: Compatible with arbitrary evaluation metrics (IE, F1, accuracy, etc.), composable with cross-validation, and requires no model retraining.
3. **Theoretical completeness**: A complete proof chain for PVF convergence and selection consistency is provided (6 propositions).
4. **Rigorous experimental design**: Synthetic experiments involve $180 \times 5{,}000 = 900{,}000$ repetitions, thoroughly exploring the hyperparameter space.
5. **Clinical orientation**: Emphasis on interpretable single-model deployment rather than black-box ensembles.

## Limitations & Future Work

1. **σ tuning requires prior knowledge**: The optimal perturbation noise scale varies by dataset, necessitating domain expert input or additional hyperparameter search.
2. **Validation limited to binary classification**: The closed-form IE formula is derived for binary classification; extension to multi-class settings remains incomplete.
3. **Limited scale of real-data experiments**: Validation is conducted on only 2 public datasets, insufficient to cover the diversity of clinical scenarios.
4. **Fairness constraints not considered**: IE does not incorporate fairness factors across different subgroups.
5. **Comparison with stronger baselines absent**: Methods such as nested CV and Bayesian model selection are not included.
6. **Area classification is debatable**: This work is more aligned with "clinical ML evaluation methodology" than traditional "medical imaging" research.

## Related Work & Insights

- **Rashomon Effect**: Breiman (2001) introduced the "two cultures" perspective; Rudin et al. (2024) advocated leveraging model diversity.
- **Perturbation robustness**: Mutation Validation (Zhang et al. 2023) injects noise into training labels; PVF differs by applying perturbations exclusively to validation set features.
- **Interpretable clinical models**: FIGS (Tan et al. 2022) and sparse logistic regression focus on model-level interpretability, whereas PVF targets the reliability of the selection process.
- **Cross-validation improvements**: Nested CV (Wainer & Cawley 2021) remains unstable on small data; PVF can complement such approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of IE and PVF offers a genuinely novel perspective, unifying resource constraints with robustness evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic experiments are exceptionally thorough (900K repetitions); real-data experiments are limited in scale.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear and rigorous, with complete theoretical proofs; however, the balance between main text and appendix is uneven, with substantial content relegated to the appendix.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to resource-constrained clinical deployment scenarios; the $\sigma$ tuning requirement limits plug-and-play usability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[ACL 2026\] Reliable Automated Triage in Spanish Clinical Notes: A Hybrid Framework for Risk-Aware HIV Suspicion Identification](../../ACL2026/medical_imaging/reliable_automated_triage_in_spanish_clinical_notes_a_hybrid_framework_for_risk-.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](../../ACL2026/medical_imaging/rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[AAAI 2026\] Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)

</div>

<!-- RELATED:END -->
