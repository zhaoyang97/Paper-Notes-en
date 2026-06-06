---
title: >-
  [Paper Note] CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models
description: >-
  [ICML 2026][Time Series][Modular Attribution] CombinationTS decomposes time-series forecasting models into five orthogonal modules: Input Transformation / Embedding / Encoder / Decoder / Output Transformation. It perform…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Modular Attribution"
  - "Probabilistic Evaluation"
  - "Identity Paradox"
  - "Data View"
  - "Evaluatology"
date: 2026-05-08
content_hash: f537d0a7349f04a9
---

# CombinationTS: A Modular Framework for Understanding Time-Series Forecasting Models

**Conference**: ICML 2026  
**arXiv**: [2605.01231](https://arxiv.org/abs/2605.01231)  
**Code**: https://github.com/BenchCouncil/CombinationTS  
**Area**: Time-Series Forecasting  
**Keywords**: Modular Attribution, Probabilistic Evaluation, Identity Paradox, Data View, Evaluatology

## TL;DR
CombinationTS decomposes time-series forecasting models into five orthogonal modules: Input Transformation / Embedding / Encoder / Decoder / Output Transformation. It performs paired Monte Carlo sampling over a shared "evaluation condition space," replacing fragile single-point MSE with marginal performance $\mu$ and stability $\sigma$. The main conclusion: with well-designed data views (Embedding), a parameter-free Identity Encoder can match or even outperform complex Transformers. Much of the "SOTA gain" in time-series forecasting stems from how data is viewed, not from modeling capacity.

## Background & Motivation

**Background**: Long-term time-series forecasting has rapidly shifted from early sparse-attention Transformers (Informer, Autoformer) to architectures like PatchTST, iTransformer, TimeMixer, and CycleNet, which focus on "data view reshaping." Papers are increasingly complex, and leaderboard MSEs keep dropping.

**Limitations of Prior Work**: DLinear has already shown that a single linear layer can outperform Transformer-based models. Further auditing works (Tan et al. 2024, Brigato et al. 2025) reveal that the so-called SOTA gap is often on par with seed noise or hyperparameter misalignment—changing the seed or batch size can flip the leaderboard. Existing benchmarks (TSLib, BasicTS, TFB, TAB) have unified interfaces but have not resolved whether "the model is winning, or the evaluation protocol is winning."

**Key Challenge**: The authors attribute this dilemma to two methodological flaws. First, the **Attribution Gap**: models are treated as inseparable black boxes, with Embedding and Encoder contributions entangled—no one can clearly say whether PatchTST's gains come from patch tokenization or the Transformer. Second, the **Benchmarking Crisis**: single-point estimates fall into the "fairness trap" (using the same suboptimal hyperparameters for all models) and the "best-case trap" (cherry-picking the best single run), making results neither fair nor robust.

**Goal**: Decompose into three subproblems—(1) How to break architectures into independently replaceable orthogonal modules; (2) How to upgrade "module evaluation" from single-point MSE to statistics robust to hyperparameter noise; (3) Whether this auditing tool can reproduce or overturn current beliefs about Transformers, frequency-domain modeling, and multi-scale decomposition.

**Key Insight**: Drawing on Evaluatology (Zhan et al. 2025), the authors treat "evaluation" as a signal–noise separation problem: the Evaluated Object (EO) is the component to be attributed, and the Evaluation Condition (EC) includes the other four modules, training hyperparameters, dataset, look-back length, horizon, etc., all treated as a random process. Only if a module wins across many ECs is it truly superior.

**Core Idea**: By combining "modular decoupling + paired EC Monte Carlo," the architecture problem is reframed as a statistical attribution problem—evaluating $(\mu(\theta), \sigma(\theta))$ instead of $\mathrm{MSE}(\theta, c^\ast)$.

## Method

### Overall Architecture

CombinationTS rewrites any time-series forecasting model $f$ as a five-stage composition:

$$f = \mathcal{T}^{-1}_{out}\circ \mathcal{D}\circ \varPhi\circ \mathcal{E}\circ \mathcal{T}_{in}$$

Input is historical observation $\mathbf{X}\in\mathbb{R}^{T\times N}$, output is prediction $\mathbf{Y}\in\mathbb{R}^{P\times N}$. The five stages are: Input Transformation $\mathcal{T}_{in}$ (injecting structural priors into the raw signal, e.g., RevIN, trend-seasonal decomposition, multi-scale downsampling, Cycle embedding), Embedding $\mathcal{E}$ (determines tokenization view, unified mapping to $\mathbb{R}^{B\times C\times L\times D}$ tensor interface), Encoder $\varPhi$ (token interactions in latent tensor, e.g., Self-Attention / MLP / Identity), Decoder $\mathcal{D}$ (projects to prediction horizon), Output Transformation $\mathcal{T}^{-1}_{out}$ (inverse normalization, adding back trend, etc.).

A key constraint: Embedding is responsible only for "view + intra-token encoding," while all cross-token dependency modeling is delegated to the Encoder—this interface constraint enables, for the first time, decoupled evaluation of "Embedding contribution vs Encoder contribution."

With this unified interface, each module can be independently replaced and freely combined. The paper constructs a search space of 4 Embeddings × 3 Encoders × 1 Decoder + 4 Input Transformations, yielding 100+ architecture variants. Paired EC Monte Carlo is then used to run each module's $(\mu, \sigma)$ over 6 datasets × 4 horizons.

### Key Designs

1. **Modular Decoupling + Tensor Interface Standardization**:

    - **Function**: Forces heterogeneous architectures into a "five-stage" pipeline, allowing any paper's component to be extracted and reinserted.
    - **Mechanism**: Embedding output must be $\mathcal{Z}\in\mathbb{R}^{B\times C\times L\times D}$ (batch / variable / time token / hidden dim). Point-wise projects each time step ($L=T$); Patch-wise projects after patching ($L=\lceil T/S\rceil$); Variate-wise compresses the entire variable history into one token ($L=1$); Identity does not project; Time-as-Feature reshapes $T$ directly to feature dim (zero parameters). Encoder strictly operates only on $\mathcal{Z}$, so replacing Encoder does not contaminate Embedding attribution.
    - **Design Motivation**: Previously, when evaluating PatchTST, it was impossible to distinguish whether the patch view or the Transformer reasoner was responsible for gains. With a unified tensor interface, running Patch-wise Embedding with Identity Encoder directly reveals the "view's independent contribution." This is the methodological prerequisite for the Identity Paradox.

2. **Probabilistic Evaluation Protocol (EC Space + $\mu / \sigma$ Statistics)**:

    - **Function**: Replaces single-point MSE reporting, computing each module's "expected performance and stability across a sea of hyperparameters."
    - **Mechanism**: Defines evaluation condition space $\Omega$, each $\mathbf{c}\in\Omega$ is a configuration tuple (dataset, look-back $T\in\{96,192,336,512\}$, horizon $P\in\{96,192,336,720\}$, encoder layers, latent dim $D\in\{64,128,256,512\}$, learning rate, batch, dropout, seed, etc.). Module performance is treated as a random variable $L(\theta, \mathbf{c})$, estimating two statistics: $\mu(\theta)=\mathbb{E}_{\mathbf{c}\sim\Omega}[L(\theta,\mathbf{c})]$ (marginal performance) and $\sigma(\theta)=\sqrt{\mathrm{Var}_{\mathbf{c}\sim\Omega}[L(\theta,\mathbf{c})]}$ (stability). $L_{best}=\min_k L(\theta,\mathbf{c}_k)$ is also reported as the "peak achievable by hyperparameter search," explicitly contrasting the "best-case trap."
    - **Design Motivation**: The "fairness trap" and "best-case trap" are dual failure modes of single-point protocols. $\mu$ reflects "average performance a user can expect with arbitrary hyperparameters," $\sigma$ reflects "hyperparameter sensitivity"—the two numbers that matter in practice. $L_{best}$ is reported separately to show whether "SOTA claims" are due to general capability or outlier luck.

3. **Stratified Paired EC Sampling**:

    - **Function**: Reduces the cost of estimating $(\mu, \sigma)$ and eliminates confusion from "comparing modules on different ECs."
    - **Mechanism**: Samples a fixed, stratified-by-dataset/horizon set $\{\mathbf{c}_k\}_{k=1}^K$ from $\Omega$ (main experiment $K=600$, 100 per dataset). All EOs are evaluated strictly on the same $\{\mathbf{c}_k\}$, i.e., paired experimental design; irrelevant dimensions are ignored for a given EO. Then $\hat{\mu}(\theta)=\frac{1}{K}\sum_k L(\theta,\mathbf{c}_k)$, $\hat{\sigma}(\theta)$ is the sample standard deviation. Pairing greatly reduces the variance of module differences, making statistical significance easier to achieve (one-tailed Mann–Whitney U test at $\alpha=0.05$ is used).
    - **Design Motivation**: If Identity and Transformer are evaluated on different ECs, much of the difference is due to "different EC distributions"—the core problem of current leaderboards. Stratification ensures even coverage of datasets/horizons, pairing cancels out systematic bias from "hard ECs," and Monte Carlo balances bias–variance under a fixed sample budget.

### Loss & Training

MSE is uniformly used as $L$. The main experiment fixes seed and dropout=0.1, attributing all observable variance to architecture and hyperparameter choices; Appendix A.7 performs multi-seed validation to rule out seed sensitivity. Training uses batch=32, 30 epochs, early-stopping patience=3. The EC space is expanded in three experiments: Exp.1 uses $D\in\{64,128,256,512\}$, $\eta\in\{10^{-3},10^{-4}\}$, encoder layers $\in\{1,2,3\}$; Exp.2 extends $D$ down to 16 and expands $\eta\in\{10^{-2},10^{-3},10^{-4}\}$ to test "whether input priors allow thinner models"; Exp.3 shares the same EC space as Exp.1 for paired comparison.

## Key Experimental Results

Datasets: Weather, Electricity, ETTh1/h2/m1/m2—6 standard long-term forecasting benchmarks; horizon $\in\{96,192,336,720\}$.

### Main Results

Exp.1 deconstructs the backbone: 4 Embeddings × 3 Encoders, each EO evaluated on $K=600$ paired ECs.

**Embedding Layer (Table 2 excerpt, $\mu/\sigma$, lower is better):**

| Dataset | Horizon | Point-wise | Patch-wise | Variate-wise | Time→Dim |
|---------|---------|------------|------------|--------------|----------|
| ETTh1 | 96 | 0.3898 / 0.0143 | **0.3772 / 0.0058** | 0.3834 / 0.0080 | 0.3937 / 0.0156 |
| ETTh1 | 192 | 0.4324 / 0.0166 | **0.4240 / 0.0133** | 0.4247 / 0.0128 | 0.4343 / 0.0161 |
| ETTm2 | 720 | 0.3981 / 0.0216 | **0.3813 / 0.0140** | 0.3863 / 0.0170 | 0.3892 / 0.0174 |
| Electricity | 96 | 0.1609 / 0.0228 | **0.1512 / 0.0176** | 0.1527 / 0.0135 | 0.1573 / 0.0190 |

Structured views (Patch-wise / Variate-wise) consistently outperform Point-wise; the zero-parameter Time→Dim reshape can match learnable projections in many cases—structure itself is signal.

**Encoder Layer (Table 3 excerpt):**

| Dataset | Horizon | MLP | Transformer | Identity |
|---------|---------|-----|-------------|----------|
| ETTh1 | 96 | 0.4371 / 0.0816 | 0.4518 / 0.0930 | **0.3907 / 0.0231** |
| ETTh1 | 720 | 0.5538 / 0.1039 | 0.5679 / 0.1014 | **0.4722 / 0.0406** |
| ETTh1 | avg | 0.4960 / 0.0938 | 0.5095 / 0.1004 | **0.4392 / 0.0424** |

The parameter-free Identity Encoder outperforms both MLP and Transformer in $\mu$ and $\sigma$—the **Identity Paradox**. Statistically, Mann–Whitney U test yields Identity vs Transformer $p<0.0001$, Identity vs MLP $p=0.0115$, significant across all datasets.

Exp.2 audits Input Transformation (Table 4, ETTh1 / ETTh2 / Electricity excerpt):

| Dataset | Baseline (RevIN) | Cycle | MultiScale | TrendSeasonal |
|---------|------------------|-------|------------|---------------|
| ETTh1 | 0.3975 / 0.0315 | **0.3891 / 0.0283** | 0.3921 / 0.0183 | 0.3947 / 0.0248 |
| ETTh2 | 0.3013 / 0.0116 | **0.2969 / 0.0115** | 0.3071 / 0.0343 | 0.3024 / 0.0135 |
| Electricity | 0.2061 / 0.0238 | 0.1999 / 0.0225 | **0.1952 / 0.0202** | 0.1976 / 0.0197 |

Cycle (CycleNet's learnable periodic embedding) is the most stable in both $\mu$ and $\sigma$; naive TrendSeasonal and MultiScale often underperform the RevIN baseline, only surpassing it when paired with deep cross-branch interaction like TimeMixer.

Exp.3 Frequency vs Time Domain (Table 5 ETTh1):

| Dataset | Horizon | iTransformer | SimpleTM (Spectral) | Variate + Identity |
|---------|---------|--------------|---------------------|--------------------|
| ETTh1 | 96 | 0.3934 / 0.0133 | 0.3812 / 0.0076 | **0.3805 / 0.0098** |
| ETTh1 | 720 | 0.4711 / 0.0155 | 0.4646 / 0.0151 | **0.4520 / 0.0191** |
| ETTh1 | avg | 0.4401 / 0.0335 | 0.4324 / 0.0354 | **0.4271 / 0.0355** |

Spectral SimpleTM improves $\mu$ over time-domain iTransformer, but $\sigma$ is nearly unchanged; under the same Variate-wise Embedding, **Identity Encoder remains superior**—frequency-domain advantage comes from representation, not robustness.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Patch-wise + Identity Encoder | ETTh1 96 $\mu/\sigma$ ≈ 0.38 / very low | Structured view + parameter-free Encoder is the new baseline |
| Patch-wise + Transformer | Same but $\sigma$ increases several times | Encoder mainly introduces hyperparameter noise |
| Point-wise + Transformer | $\sigma$ significantly higher than other combos | Self-Attention heavily depends on "semantically meaningful tokens" |
| Cycle prior + Variate-wise | ETTh2 $\mu/\sigma$ = 0.2969 / 0.0115 | Periodic prior is most robust |
| Naive MultiScale without interaction | Underperforms RevIN baseline on multiple datasets | "Divide and conquer" without cross-branch interaction loses information |
| Multi-seed validation (Appendix A.7) | Conclusion unchanged | Rules out seed sensitivity |

### Key Findings

- **Identity Paradox**: On well-designed Embeddings, the parameter-free Identity Encoder outperforms Transformer and MLP in both $\mu$ and $\sigma$. Most complex Encoders act as "overparameterized noise sources" rather than true feature extractors on standard benchmarks.
- **Transformer's dependence on view**: Transformer is stable only under structured views like Patch-wise/Variate-wise; under Point-wise ("raw time step") view, $\sigma$ inflates—self-attention is not a universal feature extractor.
- **Universal periodic prior, decomposition prior needs deep interaction**: Cycle improves both $\mu$ and $\sigma$ on almost all datasets; TrendSeasonal/MultiScale often fail alone, TimeMixer excels because multi-scale is followed by cross-component mixing.
- **Frequency domain improves representation, not robustness**: SimpleTM outperforms iTransformer in $\mu$, but not in $\sigma$—wavelets/Fourier change how signals are represented, not how optimization becomes robust.
- **Statistical significance is reproducible**: The authors verify that Identity models still achieve the lowest MSE on non-stationary data (Exchange-Rate), and multi-seed runs rule out "lucky periodic benchmark" and "lucky seed" explanations.

## Highlights & Insights

- **Translating architecture problems into statistical attribution**: The EO/EC concept cleanly separates "signal vs noise." This idea can be directly applied to architecture auditing in CV/NLP (e.g., decomposing LLaMA into tokenizer + position encoding + attention + FFN + lm-head for EC-paired scans), making it a general methodology.
- **Tensor interface constraint is an underrated methodological requirement**: Forcing Embedding to output $\mathbb{R}^{B\times C\times L\times D}$ without cross-token operations is essential for meaningful Identity Encoder evaluation; otherwise, Embedding could "secretly do cross-token mixing," leading to misattribution. This interface is valuable in engineering.
- **Variance control via paired Monte Carlo + stratified sampling**: Compared to independent sampling, pairing significantly reduces $\mathrm{Var}(L_A - L_B)$, achieving stronger statistical significance under a fixed $K=600$ budget; Mann–Whitney U significance in the appendix can be directly reused.
- **Zero-parameter Time-as-Feature reshape is competitive**: Simply reshaping tensor structure can approach learnable projections, indicating that much of the "learnable" part in current benchmarks is redundant.
- **"Burden of proof reversal" declaration**: The authors explicitly propose that future work adding architectural complexity must first prove superiority over the Identity baseline in $(\mu, \sigma)$ space, imposing a paradigm shift on the time-series forecasting community.

## Limitations & Future Work

- Only MSE is used as the evaluation metric; whether the Identity Paradox holds for MAE, CRPS, probabilistic forecasting, etc., remains to be tested.
- The EC space, though $K=600$, is still a predefined discrete grid ($D, T, P, \eta$, etc., are limited choices), not covering wilder hyperparameters (warmup, weight decay, activation functions, etc.), and the "representativeness" of $\Omega$ is not theoretically guaranteed.
- Datasets are mainly ETT/Weather/Electricity, which are strongly periodic/stationary; only Exchange-Rate is used for non-stationary validation. Long-tail, breakpoint, and missing-data scenarios (e.g., high-frequency finance, ICU signals) are not systematically covered.
- Conclusions hold only for small to medium model scales ($D\leq 512$, layers $\leq 3$); whether the Identity Paradox holds for large/foundation models (Chronos, TimeGPT, Moirai, etc.) is unanswered.
- The conclusion that "Encoder is a noise source" strongly depends on "Embedding is already good"; with very few tokens (Variate-wise, $L=1$), the Encoder has little to do, possibly overestimating Embedding's contribution.
- Directions for improvement: (i) Expand EC space to continuous distributions and use importance sampling; (ii) Apply the method to other time-series tasks (classification, anomaly detection, imputation); (iii) Explicitly connect with causal inference (Treatment Effect) frameworks for stronger causal semantics in "module contribution."

## Related Work & Insights

- **vs DLinear (Zeng et al. 2023)**: DLinear used a single linear layer to outperform Transformers, representing the first wave of "complexity skepticism." CombinationTS generalizes this to the module level—not only can the model be simple, but with good Embedding, the Encoder can be omitted entirely.
- **vs PatchTST / iTransformer / TimeMixer**: Previously compared as "whole models," this work decomposes their core innovations (Patch, Variate token, multi-scale downsampling) as replaceable modules. Recombination reveals that Embedding/Input designs are the true contributors, while attention/mixer are redundant.
- **vs TFB / TAB / BasicTS / TSLib (Wang et al. 2024c, Qiu et al. 2024/2025a, Shao et al. 2024)**: These benchmarks unify implementation but remain leaderboard-driven; this work rewrites the evaluation protocol layer, effectively adding a "paired Monte Carlo" plugin to existing benchmarks.
- **vs Brigato et al. 2025 ("There are no Champions")**: Brigato's audit of 3500+ models showed SOTA is fragile but only "broke" the paradigm; this work provides a **constructive** alternative evaluation protocol ($\mu, \sigma, L_{best}$ triple reporting).
- **vs Evaluatology (Zhan et al. 2025)**: Directly implements Evaluatology's EO/EC concepts as an executable engineering framework, providing the first complete case study in time-series forecasting.
- **Insights**: (i) Recommender systems, CV detection/segmentation, and other leaderboard-driven subfields can adopt this "module decoupling + paired EC" protocol for attribution; (ii) The Identity Paradox suggests that future architecture design should prioritize "data view" iteration over stacking Encoders; (iii) The $(\mu, \sigma)$ dual-metric should become the default reporting standard in papers.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new model but a new evaluation paradigm, proposing the counterintuitive Identity Paradox and supporting it with statistical tests; high value at the framework level.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 datasets × 4 horizons × $K=600$ paired ECs, 100+ combinations, Mann–Whitney U significance, non-stationary and multi-seed robustness checks—comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Three RQs are clearly connected, the "two methodological flaws" narrative is clean, formulas and tables are dense but appropriate; some appendix references are frequent, and figure explanations in the main text are brief.
- Value: ⭐⭐⭐⭐⭐ Directly challenges the current SOTA culture, provides an immediately usable open-source framework, and has structural impact on time-series forecasting and broader benchmarking practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](../../ICLR2026/time_series/scits_scientific_time_series_llm.md)
- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[ICLR 2026\] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](../../ICLR2026/time_series/swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)
- [\[ICML 2026\] Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
