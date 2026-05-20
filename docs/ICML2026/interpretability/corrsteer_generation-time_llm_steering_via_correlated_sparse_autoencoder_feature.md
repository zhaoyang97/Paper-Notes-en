---
title: >-
  [Paper Note] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features
description: >-
  [ICML 2026][Interpretability][Sparse Autoencoder] By selecting interpretable steering features whose SAE activations on generated tokens are Pearson-correlated with task correctness…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Sparse Autoencoder"
  - "Representation Steering"
  - "Feature Selection"
  - "Pearson Correlation"
  - "Side Effect Rate"
date: 2026-05-08
content_hash: e0f211c0968c43ee
---

# CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features

**Conference**: ICML 2026  
**arXiv**: [2508.12535](https://arxiv.org/abs/2508.12535)  
**Code**: https://github.com/seonglae/CorrSteer  
**Area**: Mechanistic Interpretability / SAE / Behavioral Steering  
**Keywords**: Sparse Autoencoder, Representation Steering, Feature Selection, Pearson Correlation, Side Effect Rate

## TL;DR
By selecting interpretable steering features whose SAE activations on generated tokens are Pearson-correlated with task correctness, and directly using the mean activation on positive samples as the coefficient—without needing contrastive datasets or backpropagation—CorrSteer achieves +3.3% on MMLU and +27.1% on HarmBench for Gemma-2 2B / LLaMA-3.1 8B, with lower side effect rates than fine-tuning.

## Background & Motivation
**Background**: Sparse Autoencoders (SAE) decompose LLM superposed representations into tens of thousands of sparse, interpretable features, becoming a key tool for mechanistic interpretability. SAE-based "behavioral steering" modifies model behavior by adding a feature direction vector to the residual stream without weight fine-tuning, and has been used in narrow scenarios like bias mitigation, knowledge erasure, and jailbreak defense.

**Limitations of Prior Work**: Existing SAE steering methods face three main issues: (1) Methods like SPARE and DSG require constructing contrastive datasets or storing all activations, with memory and compute scaling linearly with sample size; (2) AlphaEdit and CAA select features using hidden states of context tokens, while the actual steering affects "generation behavior," leading to misaligned feature selection; (3) Most methods are limited to axes like bias/refusal, lacking a general "task-relevant feature discovery" pipeline.

**Key Challenge**: In a $10^5$-scale SAE dictionary, the challenge is to find a few features that "can change behavior without harming general capability"—requiring scalability (cannot store all activations), causal reliability (cannot rely solely on correlation), and interpretability (cannot depend on black-box probes).

**Goal**: To construct a fully automated, $O(1)$ memory, backpropagation-free pipeline for feature selection, coefficient estimation, and steering, achieving both accuracy gains and low side effect rates across multiple benchmarks.

**Key Insight**: The SAE dictionary is linearly compositional, consistent with the "Linear Representation Hypothesis," making Pearson correlation a natural, lightweight heuristic to capture the linear dependence between SAE activations and task outcomes. Intervention testing can further upgrade correlation to causal evidence.

**Core Idea**: Use streaming Pearson correlation on generated tokens to filter candidate features, use mean activation on correct samples as coefficients, and retain only those features that "improve performance when amplified" via intervention—"correlation for screening, intervention for decision."

## Method

### Overall Architecture
CorrSteer takes as input a base LLM, corresponding multi-layer SAE, a small labeled evaluation set (~4k samples), and a benchmark category (multi-choice/safety/factuality, etc.). The pipeline is fully automated in three stages: (1) At generation time, compute the Pearson correlation $r_i$ between each SAE feature $z_i$'s activation and sample correctness $y_j \in \{0,1\}$, using a Welford-style streaming accumulator for $O(1)$ memory; (2) For each candidate feature $i$, use the mean activation on positive samples $c_i = \frac{1}{|\{j:y_j>0\}|}\sum_{j:y_j>0} z_{i,j}$ as the steering coefficient; (3) Add $v_{\text{steer}} = \sum_{i \in \mathcal{F}} c_i \cdot W_{\text{dec}}[:,i]$ to the residual stream at the generation position. Three variants are output: CorrSteer-S/A/P, corresponding to "global single feature / one per layer / with intervention pruning," for different task granularities.

### Key Designs
1. **Generation-Time Pearson Correlation + Streaming Accumulation**:

    - **Function**: Ranks each feature in a $10^5$-scale SAE dictionary by linear association with task success, selecting top-k candidates, with memory independent of sample size.
    - **Mechanism**: Extract activations $z_i$ only at the "final generated token" position, compute $r_i = \text{Cov}(z_i, y) / \sqrt{\text{Var}(z_i) \cdot \text{Var}(y)}$ with sample correctness $y_j$, and use a Welford-like online algorithm to accumulate mean, variance, and covariance scalars per feature ($O(1)$ memory). For multi-token generation, use max-pooling; for long reasoning tasks (GSM8K), use mean-pooling to avoid coefficient explosion. Only retain positively correlated features—ablation shows negatively correlated features consistently degrade performance.
    - **Design Motivation**: Compared to nonlinear metrics like mutual information (SPARE) or Fisher matrix (DSG), Pearson correlation best matches the linear additive structure of SAE decoders (linear representation hypothesis). Compared to context-token feature selection (CAA, AlphaEdit), using activations at generation tokens better aligns with the causal path to output behavior, avoiding features that only reflect input processing.

2. **Positive Sample Mean Coefficient + Intervention Pruning (CorrSteer-P)**:

    - **Function**: Assigns each feature a physically meaningful, low-variance steering coefficient, and filters candidates by "actual performance improvement," upgrading correlation to causal evidence.
    - **Mechanism**: Coefficient $c_i$ is the mean activation on correct samples, leveraging the non-negativity of SAE activations for lower variance than contrastive differences. CorrSteer-P runs an intervention on CorrSteer-A (most correlated feature per layer), retaining only features that "improve over non-steered" when added. For example, on LLaMA-3.1 8B, MMLU retains 24/31 layers, HarmBench 27/31, MMLU-Pro only 5/31—the more specialized the task, the more aggressive the pruning.
    - **Design Motivation**: Correlation may select spurious features that co-occur with success but do not drive it; intervention is the gold standard for causality. Automated pruning requires no manual priors or task-specific hyperparameters.

3. **Side Effect Rate (SER) as an Evaluation Axis**:

    - **Function**: Quantifies "how many originally correct answers are made incorrect by steering while improving target answers," directly exposing reward hacking risk.
    - **Mechanism**: $\text{SER} = \#\text{negatively changed} / \#\text{all changed}$, decomposing "did steering actually improve the model" into "how many changed + proportion improved," providing a fairer metric than accuracy alone. CorrSteer-A achieves SER=0.21 on MMLU, while fine-tuning is 0.41, with nearly identical accuracy (55.48% vs 55.75%).
    - **Design Motivation**: Existing steering work reports only accuracy, masking cases where "other correct behaviors are suppressed to boost the target," akin to RLHF reward hacking; SER is a simple yet highly discriminative side effect metric.

### Loss & Training
The method involves no gradient training—no weight fine-tuning, no SAE training, no backpropagation. Only forward-pass activation computation, correlation accumulation, and intervention. On Gemma-2 2B, the full pipeline for 4000 samples (including model loading, streaming correlation, and evaluation) takes about 9 minutes on a single RTX 5090; at inference, only the precomputed steering vector is added to the residual stream, with overhead < 0.1%.

## Key Experimental Results

### Main Results
Evaluated on Gemma-2 2B + Gemma Scope (16K features × 26 layers) and LLaMA-3.1 8B + LLaMA Scope (32K × 32) across MMLU / MMLU-Pro / SimpleQA / BBQ / HarmBench / XSTest / GSM8K.

| Dataset | Non-steered | CorrSteer-A | Fine-tuning | Gain (vs base) |
|---------|-------------|-------------|-------------|---------------|
| MMLU | 52.21 | **55.48** | 55.75 | +3.3 (≈on par with FT) |
| HarmBench (refusal) | 46.61 | **73.75** | – | +27.1 |
| BBQ Disambig | 75.38 | 76.53 | – | +1.2 |
| XSTest | 86.35 | 86.98 | – | +0.6 |
| MMLU-Pro | 30.40 | 30.93 | 35.32 | +0.5 (FT higher) |

| Method | MMLU | Side Effect Rate (SER) | Notes |
|--------|------|-----------------------|-------|
| CorrSteer-A | 55.48 | 0.21 | Accuracy matches FT |
| Fine-tuning | 55.75 | 0.41 | SER is 2× CorrSteer |
| SPARE (MI) | 54.97 | – | Requires large activation storage |
| DSG (Fisher) | 52.81 | – | Needs contrastive data + backprop |
| CAA | 55.13 | – | Poor SER on XSTest |

### Ablation Study

| Configuration | MMLU | HarmBench | Notes |
|---------------|------|-----------|-------|
| Full (max-pool + positive correlation) | 56.32 | 67.50 | Main config |
| Mean-pool | 56.32 | 0.00 | For safety tasks with multi-token, mean dilutes coefficients to 0 |
| All-token pool | 52.91 | 47.14 | Introduces context noise |
| Neg-correlated (Neg-A) | 49.45 | – | Negatively correlated features degrade performance |
| Sample size ≤ 100 | High variance | – | ≥ 4000 needed for convergence |

### Key Findings
- On HarmBench (108 samples), CorrSteer-A has std ±8.84, while on MMLU (4000 samples) only ±0.59; empirically, ~4000 samples is the lower bound for stable feature selection.
- Steering coefficient scale at 1.0× is Pareto optimal: HarmBench 60.36% / XSTest over-refusal 21.89% / MMLU −0.21%; at 2.0×, model collapses (HarmBench drops to 7.50%).
- Cross-task transfer: Features selected on MMLU also improve BBQ, indicating some features encode general QA ability rather than task specificity.
- On LLaMA-3.1 8B without any safety training, CorrSteer can turn "teach you how to steal enriched uranium" from compliant to true refusal, proving it activates latent but suppressed capabilities, rather than injecting new knowledge.

## Highlights & Insights
- **The minimalist "correlation + intervention" two-step best matches SAE's linear structure**: This is the most elegant aspect—using SAE's own linear hypothesis as theoretical justification, rather than imposing a complex nonlinear objective. Directly transferable to any "dictionary sparse + linearly compositional" representation system (e.g., dictionary learning over MLP).
- **Generation token replaces context token**: A one-line change that addresses the blind spot of all prior SAE steering methods—the steering target is output behavior, so feature selection should focus on output positions. This insight is valuable for any steering work.
- **SER is much more honest than accuracy alone**: Reporting "improved" and "degraded" separately directly reveals that FT's SER on MMLU is 2× CorrSteer—FT may have higher accuracy but actually harms some originally correct answers. This metric is highly recommended for RLHF/DPO communities.
- **Fully interpretable and reversible**: Selected features can be semantically checked in Neuronpedia (refusal / multiple-choice format / nuclear physics, etc.); disabling them restores the model, with no retraining needed.

## Limitations & Future Work
- The authors acknowledge: On long reasoning tasks like GSM8K, CorrSteer-A has very high variance (±24.43), indicating that "static behavioral steering" is unsuitable for tasks requiring dynamic chain-of-thought.
- The 4000-sample lower bound is unfriendly to small benchmarks (HarmBench 108), leading to instability in safety steering; future work could try bootstrapping or fusing with contrastive signals for small-sample scenarios.
- Only validated on two public SAEs (Gemma Scope / LLaMA Scope); SAE training quality itself is a ceiling. For closed-source models, SAE must be trained independently, raising deployment barriers.
- While "positive sample mean" is robust as a coefficient, it is sensitive to long-tail high-activation samples; trimmed mean or robust estimators could be considered.

## Related Work & Insights
- **vs SPARE (mutual information feature selection)**: SPARE uses MI to capture nonlinear dependencies but requires large activation storage, scaling linearly with sample size during training; CorrSteer uses Pearson, which is "just right" for SAE's linear architecture, and empirically outperforms on MMLU/BBQ, with $O(1)$ memory.
- **vs DSG (Fisher information feature selection)**: DSG requires contrastive datasets and Fisher matrix backpropagation, with narrow task coverage; CorrSteer requires no backpropagation, and a single forward pass per benchmark suffices.
- **vs AlphaEdit / CAA (context-token activation)**: Both select features from the input side, leading to high SER and severe over-refusal on XSTest for behavioral tasks; CorrSteer moves the selection point to post-generation token, directly aligning with the target.
- **vs Fine-tuning / LoRA**: Fine-tuning achieves higher raw accuracy on GSM8K/MMLU-Pro, but SER doubles; CorrSteer can be stacked on FT models, making them complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The "Pearson + generation token + intervention pruning" trio is systematically proposed for the first time; each component alone is not new, but the combination is extremely concise.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two model scales, 8 benchmarks, ablations covering pooling/sample size/scale/positive-negative correlation, and a new SER metric.
- Writing Quality: ⭐⭐⭐⭐ Formulas and pipeline are clearly explained, Figure 1 and qualitative refusal examples are intuitive; some overlap between method and ablation, but overall very readable.
- Value: ⭐⭐⭐⭐⭐ Provides the "default baseline" for SAE steering—no hyperparameters, $O(1)$ memory, 9-minute runtime, lower SER than FT, extremely low barrier and plug-and-play for the mechanistic interpretability community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICLR 2026\] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](../../ICLR2026/interpretability/salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)
- [\[AAAI 2026\] Hypothesis Generation via LLM-Automated Language Bias for ILP](../../AAAI2026/interpretability/hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)

</div>

<!-- RELATED:END -->
