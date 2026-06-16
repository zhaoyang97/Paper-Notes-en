---
title: >-
  [Paper Note] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders
description: >-
  [ICML 2026][Interpretability][feature death] This paper identifies that the true root cause of the "dead feature" problem in SAEs is not training dynamics but the geometric properties of the activation distribution. By quantifying "dimension-level outliers" using $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$, the authors analytically predict death rates from initializati
tags:
  - ICML 2026
  - Interpretability
  - feature death
  - mean-centering
  - TopK SAE
date: 2026-05-08
content_hash: ee00461571502714
---
# On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders

**Conference**: ICML 2026  
**arXiv**: [2605.31518](https://arxiv.org/abs/2605.31518)  
**Code**: None  
**Area**: Interpretability
**Keywords**: Sparse Autoencoders, feature death, activation outliers, mean-centering, TopK SAE

## TL;DR
This paper identifies that the true root cause of the "dead feature" problem in SAEs is not training dynamics but the geometric properties of the activation distribution. By quantifying "dimension-level outliers" using $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$, the authors analytically predict death rates from initialization (Spearman $\rho=0.82\sim0.89$ across 454 model-layer combinations) and demonstrate that mean-centering alone can reduce death rates in high-$\gamma$ models like AlphaFold3/ESM3 from 70%+ to near zero.

## Background & Motivation

**Background**: Sparse Autoencoders (SAEs) are a primary tool for mechanistic interpretability, mapping neural network activations to a high-dimensional sparse dictionary space ($n>d$), where each dictionary direction represents an interpretable concept. Architectural variants include ReLU-SAE, TopK-SAE, and JumpReLU-SAE, with this paper focusing on TopK-SAE.

**Limitations of Prior Work**: The same SAE configuration (identical architecture, dictionary size, sparsity, and AuxK) yields a dead feature rate of <5% on GPT-2 but as high as 72% on AlphaFold3. Even within a single model like ESM3, death rates fluctuate violently between 20% and 80% across different layers. Dead features mean dictionary capacity is severely wasted, forcing surviving features to "crowd in" more concepts, which reintroduces the superposition that SAEs aimed to eliminate.

**Key Challenge**: Previous revival techniques (AuxK, Ghost Gradient, Resampling) treat dead features as a "training dynamics problem"—assuming the features are stuck and need a "push" via various tricks. However, these tricks fail entirely on the most problematic models, suggesting the issue does not stem from training.

**Goal**: (1) Find an interpretable diagnostic capable of predicting death rates across different modalities and models; (2) Explain why revival methods like AuxK fail on high-death-rate models; (3) Provide a principled preprocessing solution and clarify when its use is mandatory.

**Key Insight**: The authors discovered that high-death layers share a common activation pattern: the mean of a few dimensions is significantly larger than the per-token standard deviation ("high mean, low variance" dimension-level outliers). This is distinct from the token-level outliers (spikes in individual tokens) studied in the quantization field. This geometric property determines the fate of most features at initialization.

**Core Idea**: Dead features are a geometric issue rather than a training issue. A single scalar $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ is sufficient to predict the death rate, and mean-centering (initializing the bias with the activation mean) can fundamentally eliminate deaths caused by outliers.

## Method

### Overall Architecture

The paper addresses why the same SAE results in 70%+ dead features in certain models. The answer lies in the geometry of the activation distribution rather than the training process. The paper performs three tasks: analytically predicting death rates using a pre-training scalar $\gamma$, decomposing the revival mechanism into fast and slow paths to explain why AuxK fails on difficult models, and providing mean-centering as a preprocessing step that adds zero inference overhead. The input is the activation distribution of a layer from any pre-trained network (GPT-2, Pythia, DINOv3, ESM3, AlphaFold3, Evo2, etc.), and the output is the $\gamma$ diagnostic, an initial death rate formula derived from $\gamma$, and a one-line code modification for SAE bias initialization.

The analysis is built on the standard TopK-SAE structure: $\mathbf{z}_{\text{pre}}=\mathbf{W}_{\text{enc}}(\mathbf{x}-\mathbf{b})+\mathbf{b}_{\text{enc}}$, $\mathbf{z}=\text{TopK}(\text{ReLU}(\mathbf{z}_{\text{pre}}))$, $\hat{\mathbf{x}}=\mathbf{W}_{\text{dec}}^{\top}\mathbf{z}+\mathbf{b}$. A feature "dies" through two distinct paths: **dead-by-ReLU** (pre-activation is negative for all inputs, perpetually truncated by ReLU) and **dead-by-TopK** (pre-activation is positive but never enters the top-$k$). The diagnostic, revival analysis, and preprocessing are developed along these two paths.

### Key Designs

**1. $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ Diagnostic and Analytical Death Formula: Predicting Death Rates with One Scalar Before Training**

The pain point was that prior work either used token-level outlier metrics (like kurtosis) or lacked a diagnostic altogether, failing to explain why death rates fluctuate so much across models. This paper decomposes single-token activations into $\mathbf{x}=\bm{\mu}+(\mathbf{x}-\bm{\mu})$, prompting the pre-activation to split into a constant shift term $\mathbf{w}_i\cdot\bm{\mu}$ and an input-dependent signal term $\mathbf{w}_i\cdot(\mathbf{x}-\bm{\mu})$. $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ quantifies the ratio of "mean" to "per-token standard deviation" (after per-token LayerNorm to remove scale differences). When $\gamma$ is large, the shift dominates: features aligned opposite to $\bm{\mu}$ have a permanently negative pre-activation and are "dead-by-ReLU"; features strongly aligned with $\bm{\mu}$ activate on every input; only features nearly orthogonal to $\bm{\mu}$ truly respond to input. Treating both shift and signal as projections of random unit vectors onto fixed directions, high-dimensional probability approximations (see Appendix B) yield $P(\text{dead-by-ReLU})\approx \Phi(-C/\gamma)$, where $C=\Phi^{-1}(1-1/N)\approx 4.26$ ($N=10^5$ samples). The TopK case raises the survival threshold to the $(1-k/n)$ quantile of the shift distribution $t_k=\Phi^{-1}(1-k/n)$, resulting in $P(\text{dead-by-TopK})\approx \Phi(t_k-C/\gamma)$. This is effective because $\gamma$ captures a truly "dimension-level" geometric quantity, and the formula requires no fitted parameters—achieving a Spearman $\rho$ as high as 0.89 (dead-by-TopK) / 0.82 (dead-by-ReLU) across 454 cross-modal model-layer combinations. Practitioners can thus predict severe feature death before committing compute to training.

**2. Two Revival Paths and the "Bias Learning $\bm{\mu}$" Bottleneck: Explaining AuxK Failure under High $\gamma$**

Prior work implicitly treated dead features as a "failure of training dynamics" and attempted to inject more gradients. This paper ablates bias freezing vs. non-freezing and the presence of AuxK on synthetic data, decoupling the revival mechanism by death path for the first time. **dead-by-TopK** revival depends on alive features lowering their activation magnitudes after convergence to let the $(k+1)$-th ranked feature in—a path that takes roughly 200K steps and is unaffected by bias freezing. **dead-by-ReLU** revival, however, relies entirely on the bias slowly absorbing $\bm{\mu}$, as only the bias can lift a permanently negative pre-activation above zero. The problem is that the speed at which the bias learns $\bm{\mu}$ depends heavily on $\gamma$: at $\gamma\le 5$, it reaches 99% in 200K steps; at $\gamma\approx 20$, it takes 2M steps to reach 90%; and at $\gamma\ge 30$, it only reaches 50–70% after 2M steps. Intuitively, weights act on inputs and their effects scale with input magnitude, whereas the bias is added directly; thus, the larger $\|\bm{\mu}\|$ is, the more the bias lags behind. Furthermore, once alive features learn $\bm{\mu}$, they suppress the bias gradient further. This logic clarifies the true role of AuxK: during TopK revival, some alive features are pushed below zero and become new "dead-by-ReLU" instances; AuxK provides gradients to stabilize "dead-by-TopK" features, preventing them from falling into "dead-by-ReLU." It suppresses "collateral death" rather than performing true revival. Since it does not accelerate bias learning, it is powerless against features that are "dead-by-ReLU" from initialization, explaining why AuxK works for moderate $\gamma$ but fails for high $\gamma$. The conclusion is that high $\gamma$ doesn't need better revival techniques, but rather a bias initialized at $\bm{\mu}$.

**3. Mean-centering: Initializing Bias with Activation Mean to Eliminate the Shift Term**

Since the bottleneck is the bias failing to catch up with $\bm{\mu}$, the most direct solution is to initialize the bias to the activation mean. By setting $\mathbf{b}=\bm{\mu}$, the pre-activation becomes $z_i=\mathbf{w}_i\cdot(\mathbf{x}-\bm{\mu})+b_{\text{enc}}$, the shift term $\mathbf{w}_i\cdot\bm{\mu}$ vanishes, and all feature pre-activations center around zero, responding only to input changes. This eliminates deaths caused by outliers from initialization. The geometric median is used by default instead of the arithmetic mean because some activation distributions are heavily skewed (see Appendix D.5). This is equivalent to mean subtraction at runtime but carries no extra inference overhead when folded into bias initialization. Notably, it only eliminates "outlier-induced death"; layers where variance is concentrated in small-dimensional subspaces (common in some protein/gene model layers) may still have residual deaths, requiring PCA whitening (Appendix E). Its true value lies in formalizing a trick that appeared sporadically (Bricken 2023b, Gao 2024) but lacked consistency and criteria. $\gamma$ provides the criterion: mandatory for high $\gamma$, optional for low $\gamma$, upgrading an empirical practice to a theoretically grounded standard preprocessing.

### Loss & Training

The training objective remains standard TopK-SAE (reconstruction MSE + TopK sparsification), with $k$, dictionary size, and learning rate kept consistent across models. Mean-centering does not change the loss; it only modifies the bias initialization. Synthetic experiments average 10 seeds, while real-world data uses mid-network layers across 454 model-layer combinations.

## Key Experimental Results

### Main Results: $\gamma$ Predicting Death Rates on Synthetic and Real Data

| Data | Metric | dead-by-ReLU | dead-by-TopK | Remarks |
|------|------|--------------|--------------|------|
| Synthetic Activations (controlled $\gamma$) | Spearman $\rho$ | 1.0 | 1.0 | Almost perfect alignment with $\Phi(-C/\gamma)$ curve |
| 454 Real Model-Layers (Language/Vision/Protein/Gene) | Spearman $\rho$ | 0.82 | 0.89 | No fitted parameters |
| AlphaFold3 mid layer | Dead Feature Rate | — | 98% → <5% | After mean-centering |
| ESM3 mid layer | Dead Feature Rate | — | 83% → ≈0 | After mean-centering |

### Ablation Study: mean-centering vs baseline vs AuxK (ESM3 L24, $\gamma\approx 8$)

| Configuration | Final Dead Rate | Interpretable Bio-concepts |
|------|----------------|------------------|
| baseline | ≈75% | 73 (dict=8192) |
| baseline + AuxK | ≈25% (plateau) | — |
| LayerNorm + $\sqrt{d}$ rescale | ≈20% | Fewer than baseline |
| mean-centering (dict=2048) | ≈0 | **100** |
| mean-centering (dict=8192) | ≈0 | Higher |

### Synthetic Ground-truth Feature Recovery ($\gamma=40$)

| Configuration | MMCS (Mean Max Cosine Similarity) |
|------|----------------------------|
| baseline | 0.38 |
| mean-centering | 0.97 |

### Key Findings

- **$\gamma$ is a genuine "pre-calculable" diagnostic**: Achieving $\rho\approx 0.89$ on 454 real model-layers without fitting means one can check $\gamma$ before deciding to invest compute in training an SAE.
- **Bias learning is the bottleneck under high $\gamma$**: At $\gamma\ge 30$, the bias only learns 50–70% of $\bm{\mu}$ within 2M steps, leaving "dead-by-ReLU" rates stuck at 75–90%.
- **AuxK's true role is suppressing collateral death**: It provides gradients to "dead-by-TopK" features to keep them stable, rather than reviving features that were "dead-by-ReLU" from the start.
- **Mean-centering outperforms baseline with a 4× smaller dictionary**: On ESM3, a mean-centered SAE with dict=2048 (100 concepts) outperforms a baseline with dict=8192 (73 concepts), significantly reducing training compute.
- **Mean-centering stabilizes learning rate sensitivity**: While the baseline shows high variance in death rates during an LR sweep, mean-centered models maintain consistently low death rates.
- **Theory slightly overestimates dead-by-ReLU**: When activation distributions are heavy-tailed (diagnosable via per-dim kurtosis), the actual maximum signal can exceed the $C\approx 4.26$ assumed under Gaussianity, reviving some features predicted to be dead.

## Highlights & Insights

- **Reframing a "Training Problem" as a "Geometric Problem"**: The SAE community has long focused on tuning AuxK / Ghost Gradient / Resampling. This paper proves these tricks fail in the hardest cases because most features are dead from initialization—this is a shift in the research paradigm, not just a new method.
- **A 0-parameter analytical formula outperforms empirical diagnostics**: Previous community metrics like kurtosis were token-level. $\gamma$ captures dimension-level outliers, the true source of the problem. Deriving a formula from high-dimensional geometry that predicts death across four modalities is a powerful application of "prior + geometry."
- **Asymmetry in bias vs. weight learning speed**: Weights interact with inputs and scale accordingly, while the bias is a direct addition. This observation explains why many normalization/centering techniques work and serves as a reusable intuition.
- **Reinterpreting the true function of AuxK**: Previously thought to "revive" features, AuxK was found through detailed subdivision to "prevent alive features from dying." This "phenomenon → mechanism re-evaluation" is a model for rigorous analysis.
- **MMCS from 0.38 → 0.97**: The near-perfect alignment with ground-truth on synthetic data shows mean-centering doesn't just lower death rates; it ensures the SAE learns the correct directions, directly benefiting interpretability.

## Limitations & Future Work

- **Inadequacy for a tiny minority of layers**: In protein and gene models, some layers retain dead features even after centering because variance is concentrated in a few directions, requiring PCA whitening (Appendix E). Mean-centering is "necessary but not sufficient."
- **Gaussian signal assumption**: In the presence of heavy-tailed activations, $\Phi(-C/\gamma)$ overestimates the death rate. While the authors use per-dim kurtosis as a backup, a unified formula for heavy-tailed activations is missing.
- **Systematic comparison limited to mid-network layers**: While Appendix data exists, the transferability across all layers and tasks is not yet fully comprehensive, particularly regarding unified preprocessing selection when $\gamma$ varies wildly.
- **No direct comparison with the "low-rank attention" hypothesis (Wang 2025)**: Wang et al. attribute death rates to the low-rank structure of attention activations. The interaction between low-rank properties and dimension-level outliers has not been systematically ablated.
- **Application extensions**: $\gamma$ could be used as a target for "activation normalization" or "architectural regularization"—reducing $\gamma$ during model training might lower downstream SAE costs or even aid quantization.

## Related Work & Insights

- **vs AuxK / Ghost Grad / Resampling (Gao 2024; Bricken 2023b)**: These methods attempt to "revive" dead features with gradients. This paper proves that under high $\gamma$, it's a bias-distance issue rather than a gradient issue. AuxK's true role is preventing collateral death.
- **vs token-level outlier studies (Sun 2024; Dettmers 2022)**: Those studies focus on spikes in individual tokens for quantization; this paper focuses on dimension-level outliers—dimensions that deviate from zero across every token, which is a different geometric property.
- **vs Lu et al. 2025 (ESMFold dimension outliers) / Wang et al. 2025 (Low-rank dead features)**: Lu et al. observed similar outliers in ESMFold but offered no diagnostic/solution. Wang et al. attributed death to low-rank attention. This paper provides the first cross-modal predictive diagnostic, analytical formula, and minimal solution.
- **vs early SAE work (Bricken 2023b; Gao 2024 used mean-centering)**: They used it inconsistently without a criterion. This paper uses $\gamma$ to formalize "when to center," upgrading it from an empirical trick to a theoretically supported standard procedure.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframing feature death from a "training dynamics problem" to an "activation geometry problem" and deriving a zero-parameter death formula is a paradigm-shifting perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive evidence from 454 real model-layers, controlled synthetic experiments, multi-modal data, and bias-freezing ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent coordination between derivations and figures. Key insights are clear, and the appendix addresses potential concerns (heavy tails, geometric medians, cross-layer transfer).
- Value: ⭐⭐⭐⭐⭐ Provides an immediately actionable mean-centering solution and a pre-training diagnostic $\gamma$. Highly recommended for SAE interpretability researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](sparse_autoencoders_are_topic_models.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](../../ICLR2026/interpretability/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)

</div>

<!-- RELATED:END -->
