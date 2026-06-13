---
title: >-
  [Paper Note] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features
description: >-
  [ICML 2026][Interpretability][Sparse Autoencoders] Interpretable steering features are selected by computing the Pearson correlation between SAE activations on generation tokens and task outcomes. Using mean activations…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "Representation Steering"
  - "Feature Selection"
  - "Pearson Correlation"
  - "Side-Effect Rate"
date: 2026-05-08
content_hash: f9d2e45112489541
---

# CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features

**Conference**: ICML 2026  
**arXiv**: [2508.12535](https://arxiv.org/abs/2508.12535)  
**Code**: https://github.com/seonglae/CorrSteer  
**Area**: Mechanistic Interpretability / SAE / Behavior Steering  
**Keywords**: Sparse Autoencoders, Representation Steering, Feature Selection, Pearson Correlation, Side-Effect Rate

## TL;DR
Interpretable steering features are selected by computing the Pearson correlation between SAE activations on generation tokens and task outcomes. Using mean activations of positive samples as coefficients without contrastive datasets or backpropagation, this method improves MMLU by +3.3% and HarmBench by +27.1% on Gemma-2 2B / LLaMA-3.1 8B, achieving lower side-effect rates than fine-tuning.

## Background & Motivation
**Background**: Sparse Autoencoders (SAE) decompose LLM superposition representations into tens of thousands of sparse, interpretable features, becoming vital tools for mechanistic interpretability. SAE-based "behavior steering" modifies model behavior by adding a feature direction vector to the residual stream without fine-tuning weights, previously applied to bias mitigation, unlearning, and jailbreak defense.

**Limitations of Prior Work**: Existing SAE steering methods face three issues: (1) Methods like SPARE and DSG require constructing contrastive datasets or storing all sample activations, leading to memory and compute costs scaling linearly with sample size. (2) Methods like AlphaEdit and CAA select features using hidden states of context tokens, whereas steering truly affects "generation behavior," making the selection misaligned with the target. (3) Most methods are limited to specific axes like bias or refusal, lacking a general "task-related feature discovery" pipeline.

**Key Challenge**: Identifying features within an SAE dictionary of size $10^5$ that "change behavior without damaging general capabilities" requires scalability (limited memory), causal reliability (going beyond mere correlation), and interpretability (no black-box probes).

**Goal**: To construct a fully automated, $O(1)$ memory, backpropagation-free pipeline for feature selection, coefficient estimation, and steering that simultaneously achieves accuracy gains and low side-effect rates across multiple benchmarks.

**Key Insight**: The authors note that the SAE dictionary is linearly additive, aligning with the "Linear Representation Hypothesis." Thus, Pearson correlation faithfully captures linear dependencies between SAE activations and task outcomes as a natural, lightweight heuristic for SAE structures. Intervention tests then upgrade correlation to causal evidence.

**Core Idea**: Use streaming Pearson correlation on generation tokens to filter candidate features, use mean activations on positive samples for coefficients, and apply intervention validation to retain features that truly "improve performance when amplified" — treating "correlation as filtering, intervention as judgment."

## Method

### Overall Architecture
CorrSteer takes a base LLM, corresponding multi-layer SAEs, a small labeled evaluation set (~4k samples), and a benchmark category (multi-choice/safety/factuality, etc.) as input. The pipeline consists of three automated stages: (1) Calculate Pearson correlation $r_i$ between activations $z_i$ of each SAE feature and result $y_j \in \{0,1\}$ at generation tokens using a Welford-style streaming accumulator for $O(1)$ memory. (2) For each candidate feature $i$, use the mean activation on positive samples $c_i = \frac{1}{|\{j:y_j>0\}|}\sum_{j:y_j>0} z_{i,j}$ as the steering coefficient. (3) Add $v_{\text{steer}} = \sum_{i \in \mathcal{F}} c_i \cdot W_{\text{dec}}[:,i]$ to the residual stream at generation positions. The final output includes three variants, CorrSteer-S/A/P, corresponding to "global single feature / one per layer / intervention pruning."

### Key Designs
1. **Generation-time Pearson Correlation + Streaming Accumulation**:
    - **Function**: Ranks the linear association between each feature in a $10^5$ SAE dictionary and task success rate to select top-k candidates with memory independent of sample size.
    - **Mechanism**: Activations $z_i$ are extracted only at the "final generation token" position to compute $r_i = \text{Cov}(z_i, y) / \sqrt{\text{Var}(z_i) \cdot \text{Var}(y)}$. A Welford-like online algorithm accumulates mean, variance, and covariance scalars, requiring $O(1)$ memory per feature. Multi-token tasks use max-pooling, while long reasoning tasks (GSM8K) switch to mean-pooling to prevent coefficient explosion. Only positively correlated features are kept, as ablations show negative features consistently degrade performance.
    - **Design Motivation**: Compared to non-linear metrics like mutual information (SPARE) or Fisher matrix (DSG), Pearson best matches the linear additive structure of SAE decoders (Linear Representation Hypothesis). Unlike context-token selection (CAA, AlphaEdit), generation-token activations are closer to the "causal path" of output behavior, avoiding features that only reflect input processing.

2. **Positive Sample Mean Coefficients + Intervention Pruning (CorrSteer-P)**:
    - **Function**: Assigns physically meaningful, low-variance steering coefficients to each feature and filters candidates based on actual performance improvement, upgrading correlation to causal evidence.
    - **Mechanism**: The coefficient $c_i$ is defined as the average activation of the feature on correct samples; leveraging the non-negativity of SAE activations, it has lower variance than contrastive differences. CorrSteer-P then performs an intervention on top of CorrSteer-A (top correlated feature per layer), retaining only features where "adding them yields higher performance than non-steered." For LLaMA-3.1 8B, MMLU retains 24/31 layers, HarmBench retains 27/31, and MMLU-Pro retains only 5/31; more specialized tasks require stricter pruning.
    - **Design Motivation**: Correlation may capture spurious features "co-occurring" with success but not "driving" it. Intervention is the gold standard for causal inference. Automating this as pruning removes the need for human priors or task-specific hyperparameters.

3. **Side-Effect Rate (SER) as an Evaluation Axis**:
    - **Function**: Quantifies "how many originally correct answers are made incorrect while steering," revealing reward hacking risks.
    - **Mechanism**: $\text{SER} = \#\text{negatively changed} / \#\text{all changed}$. This decomposes "whether steering improves the model" into "how many changed + percentage of correct changes." CorrSteer-A achieves SER=0.21 on MMLU compared to 0.41 for fine-tuning, while accuracy remains comparable (55.48% vs 55.75%).
    - **Design Motivation**: Current steering work often reports only accuracy, hiding instances where target improvement comes at the cost of suppressing other correct behaviors, similar to reward hacking in RLHF. SER is a simple yet highly discriminative measure of side effects.

### Loss & Training
The method involves no gradient training — no weights fine-tuning, no SAE training, and no backpropagation. It only requires forward activation calculation, correlation accumulation, and intervention. A full pipeline for 4000 samples on Gemma-2 2B (including loading, streaming correlation, and evaluation) takes approximately 9 minutes on a single RTX 5090. At inference, pre-calculated steering vectors are added to the residual stream with < 0.1% overhead.

## Key Experimental Results

### Main Results
Evaluated on Gemma-2 2B + Gemma Scope (16K features × 26 layers) and LLaMA-3.1 8B + LLaMA Scope (32K × 32) across MMLU / MMLU-Pro / SimpleQA / BBQ / HarmBench / XSTest / GSM8K.

| Dataset | Non-steered | CorrSteer-A | Fine-tuning | Gain (vs base) |
|--------|-------------|-------------|-------------|----------------|
| MMLU | 52.21 | **55.48** | 55.75 | +3.3 (≈ FT) |
| HarmBench (refusal) | 46.61 | **73.75** | – | +27.1 |
| BBQ Disambig | 75.38 | 76.53 | – | +1.2 |
| XSTest | 86.35 | 86.98 | – | +0.6 |
| MMLU-Pro | 30.40 | 30.93 | 35.32 | +0.5 (FT is higher) |

| Method | MMLU | SER | Notes |
|------|------|--------------|------|
| CorrSteer-A | 55.48 | 0.21 | Accuracy matches FT |
| Fine-tuning | 55.75 | 0.41 | SER is 2× CorrSteer |
| SPARE (MI) | 54.97 | – | Requires heavy activation storage |
| DSG (Fisher) | 52.81 | – | Requires contrastive data + backprop |
| CAA | 55.13 | – | High SER on XSTest |

### Ablation Study

| Configuration | MMLU | HarmBench | Description |
|------|------|-----------|------|
| Full (max-pool + pos-corr) | 56.32 | 67.50 | Main configuration |
| Mean-pool | 56.32 | 0.00 | Safety tasks (multi-token); mean dilutes coefficients to 0 |
| All-token pool | 52.91 | 47.14 | Introduces context noise |
| Neg-correlated (Neg-A) | 49.45 | – | Negative features degrade performance |
| Sample size ≤ 100 | High Var | – | Requires ≥ 4000 for convergence |

### Key Findings
- On HarmBench (108 samples), CorrSteer-A variance is ±8.84, while on MMLU (4000 samples) it is only ±0.59; an empirical threshold of ~4000 samples is the lower bound for stable feature selection.
- Steering coefficient scale 1.0× is Pareto optimal: HarmBench 60.36% / XSTest over-refusal 21.89% / MMLU −0.21%. At 2.0×, the model collapses (HarmBench drops to 7.50%).
- Cross-task transfer: Features selected on MMLU still improve BBQ, indicating some features encode general QA capability rather than task specificity.
- In LLaMA-3.1 8B without any safety training, CorrSteer can turn answers on "how to steal enriched uranium" into actual refusals, proving it activates existing but suppressed capabilities rather than injecting new knowledge.

## Highlights & Insights
- **The minimal "Correlation + Intervention" approach best fits SAE linear structure**: This is the paper's most elegant contribution — using the SAE’s own linear assumption to justify the method theoretically rather than applying complex non-linear objectives. It easily transfers to any sparse additive representation system.
- **Generation-time tokens instead of context tokens**: A simple change that addresses a major blind spot in previous SAE steering methods — if the goal is output behavior, feature selection must occur at the output position. This insight is valuable for all steering-related research.
- **SER is a more honest metric than accuracy**: By separating "correcting errors" from "introducing errors," the study reveals that FT's SER on MMLU is twice that of CorrSteer. While FT shows high accuracy, it actually breaks previously correct answers. This metric is a strong candidate for the RLHF/DPO community.
- **Fully interpretable and reversible**: Selected features can be cross-referenced on Neuronpedia (e.g., refusal / multiple-choice format / nuclear physics); disabling them restores the model without re-training.

## Limitations & Future Work
- The authors acknowledge that CorrSteer-A has high variance (±24.43) on long reasoning tasks like GSM8K, suggesting "static behavior steering" is unsuitable for tasks requiring dynamic chain-of-thought.
- The 4000-sample lower bound is problematic for small benchmarks (HarmBench 108), leading to low stability in safety steering; future work could explore bootstrapping or fusion with contrastive signals.
- Validated only on Gemma Scope and LLaMA Scope; SAE training quality remains the performance ceiling. Implementing on closed models requires training custom SAEs, increasing deployment barriers.
- While "positive sample mean" is robust, it is sensitive to long-tail high activations; trimmed mean or robust estimators could be considered.

## Related Work & Insights
- **vs SPARE (Mutual Information selection)**: SPARE uses MI for non-linear dependency but requires massive activation storage that scales with sample size. CorrSteer's Pearson correlation in an SAE context is "just enough," outperforming SPARE on MMLU/BBQ with $O(1)$ memory.
- **vs DSG (Fisher Information selection)**: DSG requires contrastive datasets and Fisher matrix backpropagation, limiting task coverage. CorrSteer requires zero backpropagation and only a single forward pass per benchmark.
- **vs AlphaEdit / CAA (Context-token activation)**: Both select features from the input side, leading to high SER and severe over-refusal on XSTest. CorrSteer moves the selection point to generation tokens to align with objectives.
- **vs Fine-tuning / LoRA**: Fine-tuning achieves higher raw accuracy on GSM8K/MMLU-Pro but doubles the SER. CorrSteer can be stacked on FT models for complementary benefits.

## Rating
- Novelty: ⭐⭐⭐⭐ The "Pearson + Generation-token + Intervention" triad is systematically proposed for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two model scales, 8 benchmarks, extensive ablations on pooling/sample size/scale, and a new SER metric.
- Writing Quality: ⭐⭐⭐⭐ Clear formulas and pipelines; Figures and qualitative refusal examples are intuitive.
- Value: ⭐⭐⭐⭐⭐ Provides a "default baseline" for SAE steering — hyperparameter-free, $O(1)$ memory, 9-minute runtime, and lower SER than FT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](../../ICLR2026/interpretability/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[AAAI 2026\] SparseRM: A Lightweight Preference Modeling with Sparse Autoencoder](../../AAAI2026/interpretability/sparserm_a_lightweight_preference_modeling_with_sparse_autoencoder.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)

</div>

<!-- RELATED:END -->
