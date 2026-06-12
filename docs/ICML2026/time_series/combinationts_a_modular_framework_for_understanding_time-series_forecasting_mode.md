---
title: >-
  [Paper Note] CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models
description: >-
  [ICML 2026][Time Series][Modular Attribution] CombinationTS decouples time-series forecasting models into five orthogonal modules: Input Transformation, Embedding, Encoder, Decoder…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Modular Attribution"
  - "Probabilistic Evaluation"
  - "Identity Paradox"
  - "Data View"
  - "Evaluatology"
date: 2026-05-08
content_hash: 8f8916a22a6390c8
---

# CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models

**Conference**: ICML 2026  
**arXiv**: [2605.01231](https://arxiv.org/abs/2605.01231)  
**Code**: https://github.com/BenchCouncil/CombinationTS  
**Area**: Time-Series Forecasting  
**Keywords**: Modular Attribution, Probabilistic Evaluation, Identity Paradox, Data View, Evaluatology

## TL;DR
CombinationTS decouples time-series forecasting models into five orthogonal modules: Input Transformation, Embedding, Encoder, Decoder, and Output Transformation. By performing paired Monte Carlo sampling across a shared "Evaluation Condition Space," it replaces fragile single-point MSE with marginal performance $\mu$ and stability $\sigma$. The conclusion is: once the data view (Embedding) is well-designed, a parameter-free Identity Encoder can match or even exceed complex Transformers. The "SOTA gains" in the field of time-series forecasting largely stem from the way data is viewed rather than modeling capacity.

## Background & Motivation

**Background**: Long-term time-series forecasting has rapidly shifted from early sparse attention Transformers like Informer/Autoformer to architectures such as PatchTST, iTransformer, TimeMixer, and CycleNet, which tout "data view reshaping" as their core selling point. Papers have become increasingly complex, with MSE on leaderboards continuously decreasing.

**Limitations of Prior Work**: DLinear previously challenged the Transformer family with a single linear layer. Audit works such as Tan et al. 2024 and Brigato et al. 2025 further found that the so-called SOTA gaps are often on the same order of magnitude as seed noise and hyperparameter alignment errors—changing a seed or batch size can overturn the rankings. While existing benchmarks (TSLib, BasicTS, TFB, TAB) have unified implementation interfaces, they do not resolve the issue of "whether the model is winning or the evaluation protocol is winning."

**Key Challenge**: The authors attribute this predicament to two methodological flaws. The first is the **Attribution Gap**: models are treated as indivisible black boxes where the contributions of Embedding and Encoder are entangled; it remains unclear whether the improvements in PatchTST come from patch tokenization or the Transformer. The second is the **Benchmarking Crisis**: single-point estimation falls into both the "fairness trap" (constraining all models with the same suboptimal hyperparameters) and the "best trap" (cherry-picking the single best result), making it neither fair nor robust.

**Goal**: This study addresses three sub-problems: (1) how to decompose architectures into independently replaceable orthogonal modules; (2) how to upgrade the evaluation of a module from single-point MSE to statistics robust to hyperparameter noise; (3) whether this auditing tool can replicate or refute current perceptions of Transformers, frequency-domain modeling, and multi-scale decomposition.

**Key Insight**: Borrowing the perspective of Evaluatology (Zhan et al. 2025), the authors view "evaluation" as a signal-noise separation problem. The Evaluated Object (EO) is the component to be attributed, while the Evaluation Condition (EC) includes the other four modules, training hyperparameters, datasets, look-back length, and horizon, which are treated collectively as a stochastic process. A module is considered truly superior only if it wins across a large number of ECs.

**Core Idea**: By using "modular decoupling + paired EC Monte Carlo," the architecture problem is reformulated into a statistical attribution problem—evaluating $(\mu(\theta), \sigma(\theta))$ instead of $\mathrm{MSE}(\theta, c^\ast)$.

## Method

### Overall Architecture

CombinationTS reformulates any time-series forecasting model $f$ as a five-stage composition:

$$f = \mathcal{T}^{-1}_{out}\circ \mathcal{D}\circ \varPhi\circ \mathcal{E}\circ \mathcal{T}_{in}$$

The input is historical observations $\mathbf{X}\in\mathbb{R}^{T\times N}$, and the output is the prediction $\mathbf{Y}\in\mathbb{R}^{P\times N}$. The five stages are: Input Transformation $\mathcal{T}_{in}$ (injecting structural priors like RevIN, trend-seasonal decomposition, multi-scale downsampling, or cycle embedding into the raw signal); Embedding $\mathcal{E}$ (determining the tokenization view and mapping to a unified tensor interface $\mathbb{R}^{B\times C\times L\times D}$); Encoder $\varPhi$ (performing interaction between tokens on the latent tensor, e.g., Self-Attention, MLP, or Identity); Decoder $\mathcal{D}$ (projecting to the prediction horizon); and Output Transformation $\mathcal{T}^{-1}_{out}$ (inverse normalization, adding back trends, etc.).

A key constraint is that the Embedding is solely responsible for "view + intra-token encoding," while all cross-token dependency modeling is delegated to the Encoder. This interface constraint allows the "Embedding contribution vs. Encoder contribution" to be evaluated independently for the first time.

On this unified interface, each module can be replaced and combined freely. The paper comprises a search space of over 100 architectural variants from 4 Embeddings × 3 Encoders × 1 Decoder + 4 Input Transformations. Paired EC Monte Carlo is then used across 6 datasets × 4 horizons to derive the $(\mu, \sigma)$ for each module.

### Key Designs

1.  **Modular Decoupling + Tensor Interface Standardization**:
    - **Function**: Forces heterogeneous architectures into a "five-stage" pipeline, allowing components from any paper to be extracted and plugged in.
    - **Mechanism**: Establishes that the Embedding output must be a four-dimensional tensor $\mathcal{Z}\in\mathbb{R}^{B\times C\times L\times D}$ (batch / variable / temporal token / hidden dimension). Point-wise projects each step ($L=T$); Patch-wise projects after patching ($L=\lceil T/S\rceil$); Variate-wise compresses the entire history into 1 token ($L=1$); Identity performs no projection; and Time-as-Feature reshapes $T$ directly into the feature dimension (zero parameters). The Encoder strictly operates across tokens in $\mathcal{Z}$, ensuring that replacing it does not contaminate the Embedding attribution.
    - **Design Motivation**: Previously, evaluating PatchTST could not distinguish whether the patch view or the Transformer reasoner was responsible for gains. With a standardized interface, running a Patch-wise Embedding with an Identity Encoder directly reveals the "independent contribution of the view." This is the methodological prerequisite for the Identity Paradox.

2.  **Probabilistic Evaluation Protocol (EC Space + $\mu / \sigma$ Dual Statistics)**:
    - **Function**: Replaces single-point MSE reporting with the expected performance and stability of each module within a "sea of hyperparameters."
    - **Mechanism**: Defines an evaluation condition space $\Omega$, where each $\mathbf{c}\in\Omega$ is a configuration tuple (dataset, look-back $T\in\{96,192,336,512\}$, horizon $P\in\{96,192,336,720\}$, encoder layers, latent dimension $D\in\{64,128,256,512\}$, learning rate, batch size, dropout, seed, etc.). Module performance is treated as a random variable $L(\theta, \mathbf{c})$, and two statistics are estimated: $\mu(\theta)=\mathbb{E}_{\mathbf{c}\sim\Omega}[L(\theta,\mathbf{c})]$ (marginal performance) and $\sigma(\theta)=\sqrt{\mathrm{Var}_{\mathbf{c}\sim\Omega}[L(\theta,\mathbf{c})]}$ (stability). Simultaneously, $L_{best}=\min_k L(\theta,\mathbf{c}_k)$ is reported as the "peak reachable via hyperparameter search" to explicitly contrast with the "best trap."
    - **Design Motivation**: The "fairness trap" and "best trap" are dual failure modes of single-point protocols. $\mu$ directly corresponds to the average level a user achieves by picking hyperparameters randomly, and $\sigma$ corresponds to hyperparameter sensitivity—these are the metrics truly relevant for deployment.

3.  **Stratified Paired EC Sampling**:
    - **Function**: Reduces the cost of estimating $(\mu, \sigma)$ while eliminating confounding noise from using different ECs for different modules.
    - **Mechanism**: A fixed set of conditions $\{\mathbf{c}_k\}_{k=1}^K$ ($K=600$ in main experiments) is sampled from $\Omega$, stratified by dataset and horizon. All EOs are evaluated strictly on the same $\{\mathbf{c}_k\}$, using a paired experimental design. This significantly reduces the variance of the difference between two modules, making statistical significance easier to obtain (verified using a one-tailed Mann–Whitney U test at $\alpha=0.05$).
    - **Design Motivation**: If Identity and Transformer are evaluated on different ECs, much of the difference serves as noise from the different EC distributions. Stratification ensures uniform coverage, and pairing cancels systematic bias from varying task difficulty.

### Loss & Training

MSE is used uniformly as the loss $L$. The main experiments fix the seed and dropout=0.1, attributing all observed variance to architecture and hyperparameter selection; multi-seed verification in Appendix A.7 excludes seed sensitivity. Training is standardized to batch=32, 30 epochs, and early-stopping patience=3. The EC space expanded across three experiments: Exp.1 used $D \in \{64,128,256,512\}$, $\eta \in \{10^{-3}, 10^{-4}\}$, and layers $\in \{1,2,3\}$; Exp.2 extended tests to $D=16$ and wider $\eta$ to test if input priors allow for "thinner" models.

## Key Experimental Results

Datasets: Weather, Electricity, ETTh1/h2/m1/m2 (6 standard benchmarks); horizon $\in\{96,192,336,720\}$.

### Main Results

Exp.1 deconstructs backbones: 4 Embeddings × 3 Encoders, with each EO evaluated on $K=600$ paired ECs.

**Embedding Level (Table 2 excerpt, $\mu/\sigma$, lower is better)**:

| Dataset | Horizon | Point-wise | Patch-wise | Variate-wise | Time→Dim |
|---------|---------|------------|------------|--------------|----------|
| ETTh1   | 96      | 0.3898 / 0.0143 | **0.3772 / 0.0058** | 0.3834 / 0.0080 | 0.3937 / 0.0156 |
| ETTh1   | 192     | 0.4324 / 0.0166 | **0.4240 / 0.0133** | 0.4247 / 0.0128 | 0.4343 / 0.0161 |
| ETTm2   | 720     | 0.3981 / 0.0216 | **0.3813 / 0.0140** | 0.3863 / 0.0170 | 0.3892 / 0.0174 |
| Electricity | 96  | 0.1609 / 0.0228 | **0.1512 / 0.0176** | 0.1527 / 0.0135 | 0.1573 / 0.0190 |

Structured views (Patch-wise / Variate-wise) significantly outperform Point-wise. Zero-parameter Time→Dim reshape also matches learnable projections in many cases—structure itself is the signal.

**Encoder Level (Table 3 excerpt)**:

| Dataset | Horizon | MLP | Transformer | Identity |
|---------|---------|-----|-------------|----------|
| ETTh1   | 96      | 0.4371 / 0.0816 | 0.4518 / 0.0930 | **0.3907 / 0.0231** |
| ETTh1   | 720     | 0.5538 / 0.1039 | 0.5679 / 0.1014 | **0.4722 / 0.0406** |
| ETTh1   | avg     | 0.4960 / 0.0938 | 0.5095 / 0.1004 | **0.4392 / 0.0424** |

The parameter-free Identity Encoder outperforms both MLP and Transformer in $\mu$ and $\sigma$ simultaneously—this is the **Identity Paradox**. Statistically, the Mann–Whitney U test gives $p<0.0001$ for Identity vs. Transformer and $p=0.0115$ for Identity vs. MLP.

Exp.2 Auditing Input Transformation (Table 4 excerpts):

| Dataset | Baseline (RevIN) | Cycle | MultiScale | TrendSeasonal |
|---------|------------------|-------|------------|---------------|
| ETTh1   | 0.3975 / 0.0315  | **0.3891 / 0.0283** | 0.3921 / 0.0183 | 0.3947 / 0.0248 |
| ETTh2   | 0.3013 / 0.0116  | **0.2969 / 0.0115** | 0.3071 / 0.0343 | 0.3024 / 0.0135 |
| Electricity | 0.2061 / 0.0238 | 0.1999 / 0.0225 | **0.1952 / 0.0202** | 0.1976 / 0.0197 |

Cycle (learnable cycle embedding) is overall the most stable. Naive TrendSeasonal and MultiScale often fail to beat the RevIN baseline unless paired with deep cross-branch interactions like those in TimeMixer.

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Patch-wise + Identity Encoder | ETTh1 96 $\mu/\sigma$ ≈ 0.38 / Ultra-low | Structural view + param-free Encoder is the new baseline. |
| Patch-wise + Transformer | Same as above but $\sigma$ increases several-fold | Encoder primarily introduces hyperparameter noise. |
| Point-wise + Transformer | $\sigma$ significantly higher | Self-Attention depends heavily on "semantically sound tokens." |
| Cycle Prior + Variate-wise | ETTh2 $\mu/\sigma$ = 0.2969 / 0.0115 | Periodic prior is the most robust. |
| Naive MultiScale (No interaction) | Fails to beat RevIN | "Divide and conquer" without cross-branch interaction loses information. |

### Key Findings

- **Identity Paradox**: Given a well-designed Embedding, the marginal performance $\mu$ and stability $\sigma$ of the Identity Encoder are superior to those of Transformer and MLP. Most complex Encoders act as "over-parameterized noise sources" rather than true feature extractors on standard benchmarks.
- **Transformer Reliance on Views**: Transformer is only stable under structured views like Patch-wise or Variate-wise. Under a "bare time-step" Point-wise view, $\sigma$ expands significantly.
- **Cycle Prior is a Panacea; Decomposition Needs Interaction**: Cycle improves both $\mu$ and $\sigma$ almost universally; TrendSeasonal/MultiScale alone often fail.
- **Frequency Domain Improves Representation, Not Robustness**: SimpleTM improves $\mu$ over iTransformer, but $\sigma$ remains unchanged.
- **Statistical Significance is Reproducible**: The Identity series still yields the lowest MSE on non-stationary data (Exchange-Rate), excluding "lucky seeds" as an explanation.

## Highlights & Insights

- **Translating Architecture Problems into Statistical Attribution**: The EO/EC framework cleanly separates "signal" from "noise." This methodology is generalizable to CV/NLP auditing.
- **Tensor Interface Constraints are Underestimated**: Forcing Embedding to output $\mathbb{R}^{B\times C\times L\times D}$ without cross-token mixing allows Identity Encoder to be meaningful; otherwise, "secret" cross-token mixing in the Embedding would lead to incorrect attribution.
- **Variance Control via Paired Monte Carlo**: Compared to independent sampling, the paired design significantly lowers $\mathrm{Var}(L_A - L_B)$, providing stronger statistical significance within a fixed budget.
- **"Burden of Proof" Manifesto**: The authors propose that future works adding architectural complexity must first prove superiority over an Identity baseline in $(\mu, \sigma)$ space.

## Limitations & Future Work

- Evaluation is limited to MSE; whether the Identity Paradox holds for MAE, CRPS, or probabilistic forecasting requires verification.
- The EC space, while $K=600$, remains a pre-defined discrete grid; representative coverage of all hyperparameters (warmup, weight decay, etc.) is not theoretically guaranteed.
- Datasets are primarily periodic/stationary; long-tail, breakpoint, or missing data scenarios are not systematically covered.
- The conclusions hold for small-to-medium scale models ($D \leq 512$). Whether it holds for Foundation Models (Chronos, Moirai, etc.) is an open question.

## Related Work & Insights

- **vs. DLinear (Zeng et al. 2023)**: DLinear used a single linear layer to challenge Transformers; CombinationTS generalizes this observation to the module level.
- **vs. PatchTST / iTransformer / TimeMixer**: Instead of treating them as monolithic models, this paper extracts their core innovations (Patch, Variate token) as modules, revealing that the Embedding/Input designs are the true sources of contribution.
- **vs. Evaluatology (Zhan et al. 2025)**: This work serves as the first complete case study of the Evaluatology EO/EC concepts in the time-series domain.
- **Insight**: Future architecture design should prioritize iterating on "data views" rather than stacking Encoders. Providing $(\mu, \sigma)$ dual metrics should become the default reporting standard.

## Rating
- Novelty: ⭐⭐⭐⭐ (A new evaluation paradigm rather than just a model; the Identity Paradox is counter-intuitive and statistically backed.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (6 datasets × 4 horizons × $K=600$ paired ECs, 100+ combinations, and Mann-Whitney U tests.)
- Writing Quality: ⭐⭐⭐⭐ (RQ structure is clear, and the narrative on methodological flaws is compelling.)
- Value: ⭐⭐⭐⭐⭐ (Directly challenges current SOTA culture and provides an actionable open-source framework.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TimeOmni-VL: Unified Models for Time Series Understanding and Generation](timeomni-vl_unified_models_for_time_series_understanding_and_generation.md)
- [\[ICML 2026\] Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)
- [\[ICML 2026\] Interpretability in Deep Time Series Models Demands Semantic Alignment](interpretability_in_deep_time_series_models_demands_semantic_alignment.md)

</div>

<!-- RELATED:END -->
