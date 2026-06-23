---
title: >-
  [Paper Note] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders
description: >-
  [ICML 2026][Interpretability][feature death] This paper identifies the true root cause of the "dead feature" problem in SAEs as the geometric properties of the activation distribution rather than training dynamics. It quantifies the severity of "dimension-level outliers" using $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$, analytically predicts the death rate from initia
tags:
  - ICML 2026
  - Interpretability
  - feature death
  - mean-centering
  - TopK SAE
date: 2026-05-08
content_hash: 5cf3d180490a0bb3
---
# On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders

**Conference**: ICML 2026  
**arXiv**: [2605.31518](https://arxiv.org/abs/2605.31518)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Sparse Autoencoders, feature death, activation outliers, mean-centering, TopK SAE

## TL;DR
This paper identifies the true root cause of the "dead feature" problem in SAEs as the geometric properties of the activation distribution rather than training dynamics. It quantifies the severity of "dimension-level outliers" using $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$, analytically predicts the death rate from initialization (Spearman $\rho=0.82\sim0.89$ across 454 model-layer combinations), and demonstrates that mean-centering alone can reduce the death rate of high-$\gamma$ models like AlphaFold3/ESM3 from 70%+ to near zero.

## Background & Motivation

**Background**: Sparse Autoencoders (SAEs) are primary tools for mechanistic interpretability, mapping neural network activations into a high-dimensional sparse dictionary space ($n>d$), where each dictionary direction represents an interpretable concept. Architecturally, variants like ReLU-SAE, TopK-SAE, and JumpReLU-SAE exist; this paper focuses on TopK-SAE.

**Limitations of Prior Work**: The same SAE configuration (architecture, dictionary size, sparsity, and AuxK) yields a dead feature rate of <5% on GPT-2 but as high as 72% on AlphaFold3. Even within a single model like ESM3, death rates fluctuate drastically between 20%–80% across different layers. Dead features result in wasted dictionary capacity, forcing surviving features to "crowd" into more concepts, reintroduced the superposition that SAEs intended to resolve.

**Key Challenge**: Previous revival techniques (AuxK, Ghost Gradient, Resampling) treat dead features as a "training dynamics problem"—if they get stuck, use tricks to push them. However, these tricks fail entirely on the most problematic models, suggesting the issue does not originate in training.

**Goal**: (1) Find an interpretable diagnostic metric that predicts death rates across modalities and models; (2) Explain why revival methods like AuxK fail on high-death-rate models; (3) Provide a principled preprocessing scheme and specify when it must be used.

**Key Insight**: High-death-rate layers share a common activation pattern: the mean of a few dimensions is significantly larger than the per-token standard deviation ("dimension-level outliers" with high mean and low variance). This is distinct from the token-level outliers (spikes in specific tokens) studied in quantization. This geometric property determines the fate of most features at initialization.

**Core Idea**: Dead features are a geometric problem rather than a training problem—a single scalar $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ can predict the death rate, and mean-centering (initializing the bias with the activation mean) can eliminate deaths caused by outliers at their source.

## Method

### Overall Architecture

The paper investigates why the same SAE results in 70%+ dead features in certain models, attributing the cause to the geometry of the activation distribution itself. Around this theme, the paper accomplishes three things: it analytically predicts the death rate using a scalar $\gamma$ calculable before training, decomposes the revival mechanism into fast and slow paths to explain why AuxK fails on difficult models, and provides mean-centering as a zero-inference-overhead preprocessing step. The input is the activation distribution of any layer in a pre-trained network (e.g., GPT-2, Pythia, DINOv3, ESM3, AlphaFold3, Evo2), and the output is the diagnostic value $\gamma$, an initial death rate formula, and a modified code for SAE bias initialization.

The analysis is based on the standard TopK-SAE structure: $\mathbf{z}_{\text{pre}}=\mathbf{W}_{\text{enc}}(\mathbf{x}-\mathbf{b})+\mathbf{b}_{\text{enc}}$, $\mathbf{z}=\text{TopK}(\text{ReLU}(\mathbf{z}_{\text{pre}}))$, and $\hat{\mathbf{x}}=\mathbf{W}_{\text{dec}}^{\top}\mathbf{z}+\mathbf{b}$. A feature "dies" via two distinct paths: **dead-by-ReLU** (pre-activation is negative for all inputs, permanently truncated by ReLU) and **dead-by-TopK** (pre-activation is positive but never enters the top-$k$). The diagnostic metric, revival analysis, and preprocessing follow these two paths.

### Key Designs

**1. $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ Diagnostic and Analytic formula: Predicting death rates with a scalar before training**

Prior work either used token-level outlier metrics (like kurtosis) or lacked diagnostic metrics entirely, failing to explain why death rates fluctuate across models. This paper decomposes single-token activations as $\mathbf{x}=\bm{\mu}+(\mathbf{x}-\bm{\mu})$, leading the pre-activation to split into a constant shift term $\mathbf{w}_i\cdot\bm{\mu}$ and an input-varying signal term $\mathbf{w}_i\cdot(\mathbf{x}-\bm{\mu})$. The scalar $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ quantifies the ratio of the "mean" to the "per-token standard deviation" (after applying per-token LayerNorm to remove scale differences). When $\gamma$ is large, the shift term dominates: features anti-aligned with $\bm{\mu}$ are dead-by-ReLU, while those strongly aligned with $\bm{\mu}$ activate on every input; only features approximately orthogonal to $\bm{\mu}$ truly respond to input. Treating both shift and signal as projections of random unit vectors and using high-dimensional probability approximations, the paper derives $P(\text{dead-by-ReLU})=\Phi(-C/\gamma)$, where $C=\Phi^{-1}(1-1/N)\approx 4.26$ for $N=10^5$ samples. For TopK, the threshold is raised to the $(1-k/n)$ quantile of the shift distribution, $t_k=\Phi^{-1}(1-k/n)$, yielding $P(\text{dead-by-TopK})\approx \Phi(t_k-C/\gamma)$. The key to its effectiveness is that $\gamma$ captures "dimension-level" geometry, and the formula requires no fitted parameters—achieving a Spearman $\rho$ of 0.89 (dead-by-TopK) and 0.82 (dead-by-ReLU) across 454 model-layer combinations.

**2. Two Revival Paths and the "Bias Learning $\bm{\mu}$ Bottleneck": explaining AuxK failure at high $\gamma$**

Prior work implicitly treated dead features as a "failure of training dynamics" and focused on injecting gradients. This paper uses ablations on synthetic data (bias frozen/unfrozen, with/without AuxK) to decouple revival mechanisms. **Dead-by-TopK** revival depends on alive features actively reducing their activation magnitude after convergence to let the $(k+1)$-th feature in—this path takes about 200K steps and is unaffected by frozen biases. **Dead-by-ReLU** revival relies on the bias slowly absorbing $\bm{\mu}$ to lift permanently negative pre-activations above zero. However, the speed at which the bias learns $\bm{\mu}$ depends heavily on $\gamma$: it takes 200K steps to reach 99% for $\gamma\le 5$, but 2M steps only reach 90% for $\gamma\approx 20$, and only 50–70% for $\gamma\ge 30$. Intuitively, feature weights work on inputs (scaled by magnitude), while the bias is strictly additive and not amplified, making it harder to catch up as $\|\bm{\mu}\|$ increases. Furthermore, alive features learning $\bm{\mu}$ suppress the bias gradient. This clarifies the role of AuxK: during TopK revival, some alive features are pushed below zero and become new dead-by-ReLU; AuxK provides gradients to stabilize them and prevent this "collateral death," but it does not accelerate bias learning. Thus, AuxK is ineffective for features that are dead-by-ReLU from initialization. The conclusion is that high $\gamma$ requires no better revival technique, only that the bias starts at $\bm{\mu}$.

**3. Mean-centering: Initializing bias with activation mean to eliminate the shift term**

Since the bottleneck is the bias failing to catch up with $\bm{\mu}$, the direct solution is initializing the bias to the activation mean. By setting $\mathbf{b}=\bm{\mu}$, the pre-activation becomes $z_i=\mathbf{w}_i\cdot(\mathbf{x}-\bm{\mu})+b_{\text{enc}}$, removing the shift term $\mathbf{w}_i\cdot\bm{\mu}$. All pre-activations then center around zero and vary only with the input, preventing outlier-induced deaths from the start. By default, the geometric median is used instead of the arithmetic mean because some activation distributions are heavily skewed; this is equivalent to runtime mean subtraction but involves no additional inference cost when folded into bias initialization. Notably, this only eliminates "outlier-induced death." Layers where variance is concentrated in a small dimensional subspace (common in some protein/gene models) may still have residual deaths, requiring PCA whitening. The value lies in turning a previously scattered trick into a principled standard: $\gamma$ provides the criterion—mandatory for high $\gamma$, optional for low $\gamma$.

### Loss & Training

The training objective remains the standard TopK-SAE (Reconstruction MSE + TopK Sparsification). Hyperparameters like $k$, dictionary size, and learning rate are consistent across comparisons. Mean-centering does not change the loss; it only modifies the bias initialization. Synthetic experiments use an average of 10 seeds, while 454 real model-layer combinations are evaluated using mid-network layers.

## Key Experimental Results

### Main Results: $\gamma$ Predicts Death Rate on Synthetic and Real Data

| Data | Metric | dead-by-ReLU | dead-by-TopK | Note |
|------|------|--------------|--------------|------|
| Synthetic activations (controlled $\gamma$) | Spearman $\rho$ | 1.0 | 1.0 | $\Phi(-C/\gamma)$ curve aligns perfectly |
| 454 Real model-layers (Language/Vision/Protein/Gene) | Spearman $\rho$ | 0.82 | 0.89 | No fitted parameters |
| AlphaFold3 mid layer | Death feature rate | — | 98% → <5% | After mean-centering |
| ESM3 mid layer | Death feature rate | — | 83% → ≈0 | After mean-centering |

### Ablation Study: mean-centering vs baseline vs AuxK (ESM3 L24, $\gamma\approx 8$)

| Configuration | Final Death Rate | Interpretable Bio-concepts |
|------|----------------|------------------|
| baseline | ≈75% | 73 (dict=8192) |
| baseline + AuxK | ≈25% (plateau) | — |
| LayerNorm + $\sqrt{d}$ rescale | ≈20% | Fewer than baseline |
| mean-centering (dict=2048) | ≈0 | **100** |
| mean-centering (dict=8192) | ≈0 | Higher |

### Ground-truth feature recovery (synthetic, $\gamma=40$)

| Configuration | MMCS (Mean Max Cosine Similarity) |
|------|----------------------------|
| baseline | 0.38 |
| mean-centering | 0.97 |

### Key Findings

- **$\gamma$ is a true "calculable pre-training" diagnostic**: Achieving $\rho\approx 0.89$ across 454 real layers without fitting means $\gamma$ can be calculated to decide whether to invest compute in SAE training.
- **Bias learning is the bottleneck under high $\gamma$**: At $\gamma\ge 30$, the bias only learns 50–70% of $\bm{\mu}$ within 2M steps, leaving the dead-by-ReLU rate stuck at 75–90%.
- **AuxK primarily inhibits collateral death**: It provides gradients to dead-by-TopK features to stabilize them, rather than reviving features that were dead-by-ReLU at initialization.
- **Mean-centering outperforms baseline with a 4× smaller dictionary**: On ESM3, a mean-centered SAE with dict=2048 (100 concepts) outperforms the baseline with dict=8192 (73 concepts), significantly reducing training compute.
- **Mean-centering stabilizes learning rate sensitivity**: While the baseline shows high death rate variance in LR sweeps, mean-centered SAEs maintain a consistently low death rate.
- **Theory slightly overestimates dead-by-ReLU**: When the activation distribution is heavy-tailed (diagnosed via per-dim kurtosis), the maximum signal can exceed the Gaussian assumption's $C\approx 4.26$, reviving some features predicted to be dead.

## Highlights & Insights

- **Reframing "Training Problems" as "Geometric Problems"**: While the SAE community focused on tuning AuxK / Ghost Gradient / Resampling, this paper proves these tricks fail on difficult models because features are dead at initialization due to geometry. This shifts the research paradigm.
- **A zero-parameter analytic formula beats empirical diagnostics**: Previous kurtosis-based metrics were token-level; $\gamma$ captures dimension-level outliers. The formula is derived from high-dimensional geometry without fitting, yet predicts death rates across four modalities.
- **Bias vs. Weight Learning Asymmetry**: Weights scale with input magnitude, while bias is additive. This observation explains why normalization/centering techniques are effective and serves as a reusable intuition.
- **Reinterpretation of AuxK**: Previously thought to "revive" features, this paper shows AuxK actually "prevents alive features from dying." Such mechanistic re-evaluations of phenomena are highly insightful.
- **MMCS improvement from 0.38 → 0.97**: Perfect alignment with ground-truth in synthetic data suggests mean-centering does more than reduce death rates; it helps SAEs find the correct directions, benefiting interpretability.

## Limitations & Future Work

- **Mean-centering is insufficient for a few layers**: Some protein and gene model layers maintain residual deaths after centering due to variance concentrated in few directions, requiring PCA whitening. Mean-centering is "necessary but not sufficient."
- **Gaussian signal assumption**: Heavy-tailed distributions cause $\Phi(-C/\gamma)$ to overestimate the death rate. While per-dim kurtosis is used as a fallback, a unified formula for heavy-tailed activations is missing.
- **Systematic comparison limited to mid-network layers**: Although Appendix data exists, the transferability across all layers and tasks is not fully comprehensive, especially regarding how to unify preprocessing when $\gamma$ varies.
- **No direct comparison with "low-rank attention" hypotheses**: While existing work attributes death rates to low-rank structures in attention activations, this paper attributes them to dimension-level outliers. The interaction between these geometric factors has not been systematically ablated.
- **Future applications**: $\gamma$ could be used as a target for "activation normalization" or "architecture regularization" during model training—reducing $\gamma$ at the source could lower the cost of training downstream SAEs and potentially aid quantization.

## Related Work & Insights

- **vs. AuxK / Ghost Grad / Resampling (Gao 2024; Bricken 2023b)**: These attempt to inject gradients to revive features. This paper proves that under high $\gamma$, the issue is bias distance, not gradients. AuxK inhibits collateral death rather than reviving initially dead features.
- **vs. Token-level outlier studies (Sun 2024; Dettmers 2022)**: These focus on individual token spikes for quantization; this paper focuses on dimension-level outliers—dimensions that deviate from zero across all tokens.
- **vs. Lu et al. 2025 (ESMFold outliers) / Wang et al. 2025 (Low-rank dead features)**: Lu et al. observed similar outliers in ESMFold but lacked diagnostics/solutions. Wang et al. attributed death to low-rank attention. This paper is the first to provide a cross-modal diagnostic, analytic formula, and simple solution.
- **vs. Early SAE work (Bricken 2023b; Gao 2024 used mean-centering)**: Previous use was inconsistent and lacked criteria. This paper uses $\gamma$ to formalize "when to center," upgrading an empirical trick to a theoretically supported standard.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframes feature death from training dynamics to activation geometry and provides a zero-parameter formula; a paradigm-shifting perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 454 real model-layers, synthetic experiments, multiple modalities, and bias-freezing ablations with consistent evidence.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent coordination between derivations and figures; key insights are concise; Appendix addresses heavy-tailed activations and geometric medians.
- Value: ⭐⭐⭐⭐⭐ Provides an immediately applicable mean-centering solution and pre-training $\gamma$ diagnostic; essential reading for SAE researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](sparse_autoencoders_are_topic_models.md)
- [\[ICML 2026\] Ensembling Sparse Autoencoders](ensembling_sparse_autoencoders.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](../../ICLR2026/interpretability/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)

</div>

<!-- RELATED:END -->
