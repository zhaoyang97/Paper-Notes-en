---
title: >-
  [Paper Note] MIB: A Mechanistic Interpretability Benchmark
description: >-
  [ICML 2025][Interpretability][Mechanistic Interpretability] This paper proposes MIB (Mechanistic Interpretability Benchmark), which includes two tracks (circuit discovery and causal variable localization), four tasks, and five models. Through standardized counterfactual intervention evaluation and new metrics (CPR/CMD), MI methods are systematically compared. The study finds that attribution + mask optimization methods perform best in circuit discovery…
tags:
  - "ICML 2025"
  - "Interpretability"
  - "Mechanistic Interpretability"
  - "benchmark"
  - "circuit discovery"
  - "causal variables"
  - "SAE"
  - "attribution patching"
date: 2026-05-08
content_hash: 28c76d6d8b2db4ac
---

# MIB: A Mechanistic Interpretability Benchmark

**Conference**: ICML 2025  
**arXiv**: [2504.13151](https://arxiv.org/abs/2504.13151)  
**Code**: [Available](https://github.com/)  
**Area**: Interpretability  
**Keywords**: Mechanistic Interpretability, benchmark, circuit discovery, causal variables, SAE, attribution patching

## TL;DR

This paper proposes MIB (Mechanistic Interpretability Benchmark), which includes two tracks (circuit discovery and causal variable localization), four tasks, and five models. Through standardized counterfactual intervention evaluation and new metrics (CPR/CMD), MI methods are systematically compared. The study finds that attribution + mask optimization methods perform best in circuit discovery, while SAE features do not outperform original neurons in causal variable localization.

## Background & Motivation

**Background**: Mechanistic interpretability (MI) methods are rapidly growing to understand the causal pathways and key concepts behind the internal behavior of language models. However, a standardized benchmark for comparing different methods is lacking.

**Limitations of Prior Work**: New methods usually compare using ad-hoc evaluation metrics and different tasks, making it impossible to determine if actual progress has been made. The faithfulness metric is used for two different goals: (i) finding components that positively contribute to the task vs. (ii) finding components that have any significant impact on the task.

**Key Challenge**: The MI field lacks a unified evaluation standard for cross-method comparison. Existing benchmarks either only compare specific categories of methods or target only specific tasks and models.

**Key Insight**: To build a standardized benchmark across methods, tasks, and models, providing fixed counterfactual inputs and standardized metrics.

**Core Idea**: To split faithfulness into two complementary metrics (CPR focusing on positive impact and CMD focusing on total impact) and integrate them over multiple circuit sizes (AUC concept) to eliminate the influence of threshold hyperparameters.

## Method

### Overall Architecture

MIB contains two tracks:
- **Circuit Discovery Track**: Evaluates the ability of methods to find subgraphs of model components (circuits) that are most important for a specific task.
- **Causal Variable Localization Track**: Evaluates the ability of methods to feature-ize hidden vectors (e.g., SAE/DAS) and align them to task-related causal variables.

Each track covers 4 tasks × multiple models, utilizing standardized counterfactual inputs for intervention evaluation.

### Key Designs

1. **Two New Metrics: CPR and CMD**:

    - **CPR (Integrated Circuit Performance Ratio)**: The AUC of the faithfulness curve with respect to circuit size, measuring the ability of a method to find components with positive contributions. Higher is better.
    - **CMD (Integrated Circuit-Model Distance)**: The area between the faithfulness curve and $f=1$, measuring the distance between circuit behavior and the full model. Smaller is better.
    - **Mechanism**: $f(\mathcal{C}, \mathcal{N}; m) = \frac{m(\mathcal{C}) - m(\emptyset)}{m(\mathcal{N}) - m(\emptyset)}$, computing faithfulness at 10 different circuit size ratios and integrating using the trapezoidal rule.
    - **Design Motivation**: Eliminates the impact of the threshold $\lambda$ on method comparison while capturing minimality (small circuit $\to$ high faithfulness).

2. **Weighted Edge Count**: Standardizes the size measurement of circuits of different granularities (edge-level vs. neuron-level). $|\mathcal{C}| = \sum_{(u,v) \in \mathcal{C}} \frac{|N_u \cap N_\mathcal{C}|}{|N_u|}$. **Design Motivation**: Including one neuron of a submodule is equivalent to including $1/d_{\text{model}}$ of the outgoing edges of that submodule.

3. **Four Task Designs**: IOI (Indirect Object Identification, a classic MI task), Arithmetic (addition and subtraction), MCQA (Multiple-Choice Question Answering, synthetic data), and ARC (real science questions). The first two are extensively studied tasks (to verify existing progress), while the latter two are unstudied (to prevent overfitting). Each task has a fixed mapping of counterfactual inputs.

4. **Causal Variable Localization Track**: Users submit featurization methods (e.g., SAE, DAS, PCA) to map hidden vectors into a new space, evaluating whether interchange interventions can precisely manipulate specific causal variables. The Interchange Intervention Accuracy (IIA) metric is used.

### Loss & Training

MIB itself is an evaluation framework and does not involve training. However, the InterpBench models (synthetic models with known ground-truth circuits) in the benchmark are trained using standard procedures. Masking methods like UGS are optimized jointly using KL divergence and L1 regularization.

## Key Experimental Results

### Main Results — Circuit Discovery CMD Score (Lower is Better)

| Method | IOI(GPT-2) | IOI(Qwen) | Arithmetic(Llama) | MCQA(Qwen) | ARC-E(Gemma) | ARC-C(Llama) |
|------|-----------|-----------|-------------------|-----------|-------------|-------------|
| Random | 0.75 | 0.72 | 0.74 | 0.73 | 0.68 | 0.74 |
| EAP (CF) | 0.03 | 0.15 | **0.01** | 0.07 | 0.04 | 0.18 |
| EAP-IG-inp (CF) | **0.03** | **0.02** | **0.01** | 0.08 | **0.04** | 0.22 |
| EAP-IG-act (CF) | **0.03** | **0.01** | **0.01** | **0.05** | **0.04** | 0.37 |
| NAP (CF) | 0.38 | 0.33 | 0.29 | 0.30 | 0.33 | 0.69 |
| UGS | **0.03** | **0.03** | - | 0.20 | - | - |
| IFR | 0.42 | 0.69 | 0.83 | 0.60 | 0.66 | 0.76 |

### Ablation Study — Causal Variable Localization (IIA Score)

| Method | Featurization | IOI | Arithmetic | Characteristics |
|------|--------|-----|-----------|------|
| DAS | Supervised rotation | Highest | Highest | Requires annotation |
| SAE | Unsupervised sparse | Medium | Medium | Does not outperform neurons |
| Neuron (Probe) | No featurization | Medium | Medium | Baseline |
| PCA | Unsupervised linear | Lower | Lower | Simple baseline |

### Causal Variable Localization — Key IIA Results

| Method | Featurization Type | ARC-E Gemma | ARC-E Llama | Characteristics |
|------|-----------|------------|------------|------|
| DAS | Supervised rotation direction | 88 (best:94) | 88 (best:99) | Requires causal model annotation |
| DBM+SAE | Unsupervised + mask | 82 (best:99) | Medium | SAE $\approx$ neurons |
| Full Vector | No featurization | Medium | Medium | Coarse-grained intervention |
| PCA | Unsupervised linear | Lower | Lower | Simple baseline |

### Key Findings

- **Edge-level attribution methods (EAP-IG) perform the best in circuit discovery**, especially when using counterfactual ablation.
- **Exact activation patching is not always optimal**—due to limitations of using small sample sizes or independent edge evaluations (e.g., EActP is worse than EAP-IG on Qwen).
- **SAE features do not outperform original neurons in causal variable localization**—the IIA of DBM selecting SAE features is close to selecting standard neurons.
- **Supervised methods (DAS) lead significantly in causal variable localization**—DAS achieves up to 94% IIA (best layer) on ARC-Easy Gemma.
- Node-level circuits perform poorly because each node "costs" too many edges.
- Counterfactual ablation > Mean ablation $\approx$ Optimal ablation.
- IFR (a non-causal method) outperforms random but is far inferior to causal methods, validating the necessity of causal analysis.
- The UGS mask method performs well on CMD because it directly optimizes KL divergence, but it does not excel in CPR.

## Highlights & Insights

- The design concept of CPR/CMD metrics is clever—AUC eliminates hyperparameters, and the two metrics target different analysis goals.
- The finding that SAE $\le$ neurons poses a serious challenge to the current enthusiasm of the interpretability community.
- Balancing existing and new tasks helps prevent overfitting on specific benchmarks.
- The public leaderboard accepts submissions, creating a mechanism for continuously tracking progress.

## Limitations & Future Work

- Only 4 models (GPT-2, Qwen-0.5B, Gemma-2B, Llama-8B) are evaluated, lacking larger models.
- Circuits in InterpBench synthetic models may have different properties compared to those in real models.
- The causal variable localization track only evaluates known causal variables and cannot evaluate the ability to discover new causal variables.
- Some methods (e.g., EActP, UGS) cannot be run on all models due to computational limits.
- Causal model assumptions for MCQA and ARC (e.g., order ID $\to$ answer token) may be oversimplified.
- Long-term maintenance of private test sets and preventing data leakage pose ongoing challenges.

## Related Work & Insights

- **Wang et al., 2023** (IOI Circuit): The most classic MI case study, which MIB standardizes.
- **Marks et al., 2025** (AP-IG-activations): One of the strongest current attribution methods.
- **Geiger et al., 2024** (DAS/Causal Abstraction): The theoretical framework for causal variable localization.
- **Karvonen et al., 2025** (SAE benchmark): MIB extends SAE evaluation to a broader framework.
- Insight: The MI field needs more standardized evaluations like MIB to determine real progress.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of CPR/CMD metrics and the cross-method comparison framework represent novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 tasks, 5 models, 10+ methods, public and private test sets.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, rigorous definitions, and abundant figures and tables.
- Value: ⭐⭐⭐⭐⭐ Of foundational importance to the MI field; the SAE findings have a significant impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](../../NeurIPS2025/interpretability/nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](../../ACL2025/interpretability/a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](../../ACL2025/interpretability/mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](../../ACL2025/interpretability/an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)

</div>

<!-- RELATED:END -->
