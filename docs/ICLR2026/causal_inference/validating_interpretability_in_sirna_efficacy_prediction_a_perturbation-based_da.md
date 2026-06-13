---
title: >-
  [Paper Note] Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol
description: >-
  [ICLR 2026][Causal Inference][siRNA] This paper proposes a standardized perturbation-based saliency faithfulness validation protocol for siRNA efficacy prediction…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "siRNA"
  - "saliency maps"
  - "faithfulness validation"
  - "perturbation testing"
  - "biological regularization"
date: 2026-05-08
content_hash: 16c25c941165b6c9
---

# Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol

**Conference**: ICLR 2026
**arXiv**: [2602.10152](https://arxiv.org/abs/2602.10152)  
**Code**: [https://github.com/shadi97kh/BioPrior](https://github.com/shadi97kh/BioPrior)  
**Area**: Medical/Biological Interpretable AI
**Keywords**: siRNA, saliency maps, faithfulness validation, perturbation testing, biological regularization

## TL;DR
This paper proposes a standardized perturbation-based saliency faithfulness validation protocol for siRNA efficacy prediction, serving as a "pre-synthesis checkpoint" to assess the reliability of saliency maps. The authors further introduce BioPrior, a biologically informed regularization method to improve saliency faithfulness. Results show that 19/20 fold-dataset instances pass the validation, while cross-dataset transfer reveals two distinct failure modes.

## Background & Motivation

**Background**: siRNA therapeutics (e.g., patisiran, givosiran) have received FDA approval. Deep learning models are widely used to predict siRNA knockdown efficacy, and researchers examine saliency maps to identify which nucleotide positions are "important" for guiding sequence editing.

**Limitations of Prior Work**: Saliency methods (gradients, integrated gradients, etc.) are broadly applied in the siRNA domain but rarely validated. Attribution methods do not guarantee they reflect true feature importance and may silently fail under protocol or distributional shifts.

**Key Challenge**: A model may achieve accurate predictions on a given dataset with seemingly reasonable saliency maps, yet when transferred to a different experimental protocol (e.g., a different assay), the saliency maps may become entirely unreliable—a failure that cannot be detected prior to deployment.

**Goal**: (a) Provide a standardized saliency faithfulness testing protocol; (b) identify and categorize failure modes in cross-dataset transfer; (c) improve saliency faithfulness via biological prior regularization.

**Key Insight**: The paper defines "counterfactual faithfulness"—whether mutating high-saliency positions causes larger prediction changes than controls—and uses this actionable test as a pre-synthesis deployment checkpoint.

**Core Idea**: An expected-effect perturbation operator (averaging prediction changes over three alternative bases at each position) + nucleotide composition-matched random baselines + paired Wilcoxon test → pass/fail determination.

## Method

### Overall Architecture

Training: Conv→BiLSTM→Transformer hybrid encoder + bidirectional cross-attention (siRNA↔mRNA) + MLP prediction head + BioPrior differentiable biological regularization. Validation: compute gradient saliency → select top-$k$ positions → compute expected-effect → compare against composition-matched random baselines → statistical testing.

### Key Designs

1. **Counterfactual Faithfulness Validation Protocol**:

    - **Function**: Tests whether model sensitivity at high-saliency positions exceeds that of matched controls.
    - **Mechanism**: $\Delta_i = \frac{1}{3}\sum_{b \neq x_i} |\hat{y}(\mathbf{X}) - \hat{y}(\mathbf{X}^{i \leftarrow b})|$ computes the expected-effect at each position. Top-$k$ positions are selected, and $\Delta(T)$ is compared against nucleotide composition-matched random baseline $\Delta_{match}$. Pass criteria: $p < 0.05$, $d_z > 0.2$, and win rate $> 50\%$.
    - **Design Motivation**: Distinctions from standard ISM: (1) expected-effect operator rather than single-mutation; (2) composition-matched baseline to control for nucleotide-specific biases; (3) explicit pass/fail criteria; (4) cross-dataset diagnostic taxonomy.

2. **BioPrior Biological Regularization**:

    - **Function**: Encodes established siRNA design rules as differentiable penalty terms.
    - **Core Rules**: Thermodynamic asymmetry, seed-region composition constraints, global GC heuristics, immunostimulatory motif avoidance, and duplex stability proxies. $\mathcal{L}_{bio} = \sum_c \bar{\alpha}_c \mathcal{L}_c$.
    - **Design Motivation**: Biological priors guide the model toward features consistent with known mechanisms, thereby improving saliency faithfulness.

3. **Transfer Failure Mode Taxonomy**:

    - **Faithful-but-wrong**: The faithfulness test passes but prediction fails (the model is internally consistent but has learned incorrect rules).
    - **Inverted saliency**: High-saliency positions exhibit lower sensitivity than random ($d_z < 0$)—a smokescreen failure.

### Loss & Training

$\mathcal{L}_{total} = \mathcal{L}_{pred} + \lambda(t) \mathcal{L}_{bio} + \lambda_{aux} \mathcal{L}_{aux}$, where $\lambda(t)$ follows a warmup+ramp schedule (linearly increasing from 0.10 to 0.30 after epoch 8), allowing the model to first learn predictive features before gradually incorporating biological regularization.

## Key Experimental Results

### Main Results

| Dataset | Model | AUC | Pearson $r$ | Faithfulness Win Rate | Cohen's $d_z$ |
|--------|------|-----|-----------|----------------|-------------|
| Huesken (2431 siRNAs) | +BioPrior | ~0.78 | ~0.65 | 85.2% | 0.86 |
| Huesken | Baseline | ~0.77 | ~0.64 | 82% | 0.77 |
| Katoh (702 siRNAs) | +BioPrior | ~0.76 | ~0.58 | 80%+ | 0.82 |
| Mix (581 siRNAs) | +BioPrior | ~0.77 | ~0.62 | 83%+ | 0.79 |

19/20 fold-dataset combinations pass the faithfulness test.

### Ablation Study

| Configuration | Faithfulness $d_z$ | AUC Change | Notes |
|------|------------|---------|------|
| +BioPrior (full) | 0.86 | +0.01 | Faithfulness improved |
| Baseline (no BioPrior) | 0.77 | Reference | |
| Random weights | -0.45~0.03 | N/A | Negative control, confirms failure |
| Shuffled labels | <0.03 | N/A | Negative control |
| Bottom-$k$ (low saliency) | Fail | N/A | Reverse control |

### Key Findings
- **Cross-dataset transfer exposes critical issues**: Transfer between the Katoh dataset (luciferase reporter assay) and the other three datasets (mRNA-level assays) fails—saliency learned under one experimental protocol may be entirely invalid under another.
- **Two failure modes**: faithful-but-wrong (prediction fails but faithfulness passes) and inverted saliency (Taka→Hu transfer yields $d_z = -1.25$).
- **BioPrior improves faithfulness with limited predictive gain**: AUC improves by +0.01, while faithfulness $d_z$ increases from 0.77 to 0.86.
- **High-saliency positions cluster in functional regions**: The seed region (5′ end) and 3′ end—consistent with biological priors.

## Highlights & Insights
- **Practical value of the "pre-synthesis checkpoint" concept**: In lab-AI loops, saliency validation should be a standard operating procedure, analogous to significance thresholds in statistical testing.
- **Predictive value of transfer failure warnings**: Protocol/assay shifts can silently invalidate deployment-stage saliency, even when in-domain performance appears strong.
- **Rigorous negative control design**: Four negative controls—random weights, shuffled labels, shuffled saliency, and bottom-$k$—all fail, confirming the protocol's discriminative power.

## Limitations & Future Work
- Only model sensitivity faithfulness is validated; this does not imply biological causality, which requires wet-lab confirmation.
- RNA-FM embeddings are held fixed during perturbation (for computational reasons), potentially introducing error.
- BioPrior rules are manually encoded; additional rules or data-driven priors may be more effective.
- The study is limited to four relatively small datasets.

## Related Work & Insights
- **vs. ISM (in vitro mutation scanning)**: ISM produces interpretive outputs; the proposed protocol is a statistical acceptance test—the objectives are distinct.
- **vs. OligoFormer**: Shares architectural foundations, but this work adds BioPrior and faithfulness validation.
- **Connection to physics-informed ML**: BioPrior is analogous to physical constraints in PINNs, though biological priors are inherently more uncertain.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combines known components, but the application to siRNA is novel and the failure mode taxonomy is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four datasets, 5-fold CV, cross-dataset transfer, multiple negative controls, and ablations—highly comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure; protocol descriptions are detailed and reproducible.
- **Value**: ⭐⭐⭐⭐ Provides practical guidance for interpretability validation in biological sequence models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)
- [\[ICML 2026\] Outcome-Aware Spectral Feature Learning for Instrumental Variable Regression](../../ICML2026/causal_inference/outcome-aware_spectral_feature_learning_for_instrumental_variable_regression.md)
- [\[NeurIPS 2025\] LLM Interpretability with Identifiable Temporal-Instantaneous Representation](../../NeurIPS2025/causal_inference/llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)
- [\[ICLR 2026\] An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes](an_orthogonal_learner_for_individualized_outcomes_in_markov_decision_processes.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)

</div>

<!-- RELATED:END -->
