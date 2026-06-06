---
title: >-
  [Paper Note] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders
description: >-
  [ICML 2026][Interpretability][Sparse Autoencoders] This paper identifies that the root cause of "dead features" in SAEs is not training dynamics but the geometric properties of activation distributions. By quantifying "d…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Sparse Autoencoders"
  - "feature death"
  - "activation outliers"
  - "mean-centering"
  - "TopK SAE"
date: 2026-05-08
content_hash: ed5abce54a054f7b
---

# On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders

**Conference**: ICML 2026  
**arXiv**: [2605.31518](https://arxiv.org/abs/2605.31518)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Sparse Autoencoders, feature death, activation outliers, mean-centering, TopK SAE

## TL;DR
This paper identifies that the root cause of "dead features" in SAEs is not training dynamics but the geometric properties of activation distributions. By quantifying "dimension-level outliers" using $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$, the authors analytically predict dead rates from initialization (Spearman $\rho=0.82\sim0.89$ across 454 model-layer pairs) and demonstrate that mean-centering alone can reduce dead rates in high-$\gamma$ models (like AlphaFold3/ESM3) from over 70% to near zero.

## Background & Motivation

**Background**: Sparse Autoencoders (SAEs) are primary tools for mechanistic interpretability, mapping neural network activations into a high-dimensional sparse dictionary space ($n>d$), where each dictionary direction represents an interpretable concept. Architecture variants include ReLU-SAE, TopK-SAE, and JumpReLU-SAE; this paper focuses on TopK-SAE.

**Limitations of Prior Work**: The same SAE configuration (architecture, dictionary size, sparsity, Co-occurrence AuxK) results in a dead feature rate of <5% on GPT-2 but as high as 72% on AlphaFold3. Even within a single model like ESM3, dead rates fluctuate sharply between 20%–80% across different layers. Dead features imply a severe waste of dictionary capacity, forcing surviving features to "crowd" more concepts, leading to the return of the very superposition SAEs aim to eliminate.

**Key Challenge**: Previous revival techniques (AuxK, Ghost Gradient, Resampling) treat dead features as a "training dynamics issue"—assuming that since they are stuck, they need a "push" from various tricks. However, these tricks fail on the models with the most severe death rates, suggesting the problem is not fundamentally about training.

**Goal**: (1) Find an interpretable diagnostic metric that can predict dead rates across modalities and models; (2) explain why revival methods like AuxK fail on high-death models; (3) provide a principled preprocessing solution and clarify when it must be used.

**Key Insight**: The authors found that high-death layers share a common activation pattern: the mean of a few dimensions is significantly larger than the per-token standard deviation ("dimension-level outliers" with high mean and low variance), which is distinct from the token-level outliers (spikes in individual tokens) studied in the quantization field. This geometric property determines the fate of most features at initialization.

**Core Idea**: Feature death is an activation geometry problem rather than a training problem. A single scalar $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ can predict the dead rate, while mean-centering (initializing the bias with the activation mean) can eliminate outliers-induced death from the root.

## Method

### Overall Architecture

The paper revolves around three components: analytically linking the single scalar $\gamma$ to the dead rate, decomposing the revival mechanism during training into fast and slow paths, and providing a "zero-extra-computation" preprocessing step (mean-centering).

Input: Activation distributions from a specific layer of any pretrained neural network (GPT-2, Pythia, DINOv3, ESM3, AlphaFold3, Evo2, etc.);  
Output: (1) A diagnostic $\gamma$ value calculated before training; (2) a closed-form formula for predicting initial dead rate from $\gamma$; (3) a one-line modification to SAE bias initialization.

Formally, the standard structure of a TopK-SAE is:

$\mathbf{z}_{\text{pre}}=\mathbf{W}_{\text{enc}}(\mathbf{x}-\mathbf{b})+\mathbf{b}_{\text{enc}}$,  
$\mathbf{z}=\text{TopK}(\text{ReLU}(\mathbf{z}_{\text{pre}}))$,  
$\hat{\mathbf{x}}=\mathbf{W}_{\text{dec}}^{\top}\mathbf{z}+\mathbf{b}$.

Feature death occurs through two paths: **dead-by-ReLU** (pre-activation is negative for all inputs) and **dead-by-TopK** (pre-activation is positive but never enters the top-$k$).

### Key Designs

1.  **$\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ Diagnostic and Analytical Dead Rate Formula**:
    - **Function**: To quantify the ratio of "activation mean" to "per-token variance" using a single scalar, thereby providing a closed-form prediction of the dead feature rate before training starts.
    - **Mechanism**: Decomposing a single token activation as $\mathbf{x}=\bm{\mu}+(\mathbf{x}-\bm{\mu})$, the pre-activation splits into a constant shift term $\mathbf{w}_i\cdot\bm{\mu}$ and an input-dependent signal term $\mathbf{w}_i\cdot(\mathbf{x}-\bm{\mu})$. When $\gamma$ is large, the shift dominates the signal: features anti-aligned with $\bm{\mu}$ have permanently negative pre-activations (dead-by-ReLU), while features strongly aligned with $\bm{\mu}$ activate for every input; only features approximately orthogonal to $\bm{\mu}$ truly respond to input variations. Treating both shift and signal as projections of random unit vectors onto a fixed direction, high-dimensional probability approximations (detailed in Appendix B) yield $P(\text{dead-by-ReLU})=\Phi(-C/\gamma)$, where $C=\Phi^{-1}(1-1/N)\approx 4.26$ ($N=10^5$ evaluation samples). In the TopK case, the survival threshold is raised to the $(1-k/n)$ quantile of the shift distribution $t_k=\Phi^{-1}(1-k/n)$, yielding $P(\text{dead-by-TopK})\approx \Phi(t_k-C/\gamma)$. Per-token LayerNorm is applied to activations before calculating $\gamma$ to decouple scale differences across tokens.
    - **Design Motivation**: Prior work used either token-level outliers (kurtosis) or had no diagnostic at all. $\gamma$ is a truly "dimension-level" geometric quantity. Across 454 model-layer combinations, its Spearman $\rho$ reaches 0.89 (dead-by-TopK) and 0.82 (dead-by-ReLU) without any fitting parameters. Practitioners can predict severe feature death before committing compute to training an SAE.

2.  **Two Revival Paths and the "Bias learning $\bm{\mu}$ as Bottleneck"**:
    - **Function**: To explain how dead features revive (and why they fail to do so) during training, revealing the functional limits of existing methods like AuxK.
    - **Mechanism**: Ablations were conducted on synthetic data by freezing/unfreezing SAE bias and enabling/disabling AuxK. **Dead-by-TopK** revival depends on alive features lowering their activation magnitudes after convergence to allow features ranked $k+1$ to enter the top-$k$; this path completes within $\sim$200K steps regardless of bias freezing. **Dead-by-ReLU** revival relies solely on the bias gradually absorbing $\bm{\mu}$, as only the bias can lift permanently negative pre-activations above zero. The bottleneck is that the speed of learning $\bm{\mu}$ via the bias is heavily dependent on $\gamma$: at $\gamma\le 5$, it reaches 99% in 200K steps; at $\gamma\approx 20$, it only reaches 90% in 2M steps; at $\gamma\ge 30$, it only reaches 50–70% in 2M steps. Intuitively, weights multiply the input and their effect scales with input magnitude, whereas the bias is purely additive and not amplified; thus, the larger $\|\bm{\mu}\|$, the slower the bias catches up. Once alive features learn $\bm{\mu}$, they further suppress the bias gradient. The hidden role of AuxK is actually suppressing "collateral death"—during the TopK revival process, some alive features are pushed below zero as they shrink, becoming new dead-by-ReLU features. AuxK provides gradients to dead-by-TopK features to stabilize them, but it does not accelerate bias learning, making it helpless against features that are dead-by-ReLU from initialization.
    - **Design Motivation**: Previous work implicitly assumed "dead features = broken training dynamics" and focused on "how to inject more gradients." This study decouples revival by death path, proving that under high $\gamma$, no better revival technique is needed—only an initialization where the bias starts at $\bm{\mu}$.

3.  **Mean-centering: Initializing Bias with Activation Mean**:
    - **Function**: A one-line code preprocessing step that initializes the SAE bias to the geometric median (default) or arithmetic mean of the activations, directly canceling the shift term in pre-activation.
    - **Mechanism**: By setting $\mathbf{b}=\bm{\mu}$, the pre-activation reduces to $z_i=\mathbf{w}_i\cdot(\mathbf{x}-\bm{\mu})+b_{\text{enc}}$. The shift term $\mathbf{w}_i\cdot\bm{\mu}$ vanishes, and all feature pre-activations center around zero, varying only with input. This eliminates outliers-induced death from initialization. The geometric median is preferred over the arithmetic mean due to heavy skew in some models (comparison in Appendix D.5). This is equivalent to performing mean subtraction at runtime but involves zero additional inference overhead when folded into bias initialization. Note that it only eliminates "outlier-induced death"; for the few layers where variance is concentrated in a tiny subspace (common in some protein/gene models), residual death persists and requires PCA whitening (Appendix E).
    - **Design Motivation**: Mean-centering appeared sporadically in Bricken 2023b and Gao 2024 but was used inconsistently and lacked a clear criterion. $\gamma$ provides a principled rule: mandatory for high $\gamma$, optional for low $\gamma$.

### Loss & Training
The standard TopK-SAE training objective (Reconstruction MSE + TopK sparsification) is maintained. Hyperparameters such as $k$, dictionary size, and learning rate are kept consistent across model comparisons. Mean-centering does not modify the loss function; it only changes the bias initialization. Synthetic experiments are averaged over 10 seeds; for real-world data, mid-network layers are used consistently for training across 454 model-layer combinations.

## Key Experimental Results

### Main Results: $\gamma$ Predicts Dead Rates on Synthetic and Real Data

| Data | Metric | dead-by-ReLU | dead-by-TopK | Remarks |
|------|------|--------------|--------------|------|
| Synthetic (Controlled $\gamma$) | Spearman $\rho$ | 1.0 | 1.0 | Near-perfect alignment with $\Phi(-C/\gamma)$ |
| 454 Real Layers | Spearman $\rho$ | 0.82 | 0.89 | No fitting parameters used |
| AlphaFold3 mid layer | Dead Rate | — | 98% → <5% | After mean-centering |
| ESM3 mid layer | Dead Rate | — | 83% → ≈0 | After mean-centering |

### Ablation Study: mean-centering vs Baseline vs AuxK (ESM3 L24, $\gamma\approx 8$)

| Configuration | Final Dead Rate | Interpretable Bio-concepts |
|------|----------------|------------------|
| Baseline | ≈75% | 73 (dict=8192) |
| Baseline + AuxK | ≈25% (plateau) | — |
| LayerNorm + $\sqrt{d}$ rescale | ≈20% | Fewer than baseline |
| Mean-centering (dict=2048) | ≈0 | **100** |
| Mean-centering (dict=8192) | ≈0 | Higher |

### Ground-truth Feature Recovery (Synthetic, $\gamma=40$)

| Configuration | MMCS (Mean Max Cosine Similarity) |
|------|----------------------------|
| Baseline | 0.38 |
| Mean-centering | 0.97 |

### Key Findings

- **$\gamma$ is a robust pre-training diagnostic**: Achieving $\rho\approx 0.89$ on 454 real layers without fitting means practitioners can calculate $\gamma$ before deciding to invest compute in SAE training.
- **Bias learning is the bottleneck under high $\gamma$**: At $\gamma\ge 30$, the bias only learns 50–70% of $\bm{\mu}$ within 2M steps, keeping the dead-by-ReLU rate at 75–90%.
- **AuxK primarily suppresses collateral death**: It stabilizes features that are dead-by-TopK rather than reviving those that were dead-by-ReLU at initialization.
- **Ours (Mean-centering) outperforms baseline with 4× smaller dictionary**: On ESM3, a mean-centered SAE with dict=2048 (100 concepts) outperformed a baseline with dict=8192 (73 concepts), significantly reducing training compute.
- **Mean-centering stabilizes sensitivity to learning rate**: While the baseline shows high variance in dead rates during LR sweeps, mean-centered SAEs maintain consistently low dead rates.
- **Theory slightly overestimates dead-by-ReLU**: When activation distributions are heavy-tailed (diagnosable via per-dim kurtosis), the actual maximum signal can exceed the Gaussian-derived $C\approx 4.26$, reviving some features predicted to be dead.

## Highlights & Insights

- **Reframing "Training Problem" as "Geometric Problem"**: For a long time, the SAE community focused on tuning AuxK / Ghost Gradient / Resampling. This paper proves these tricks fail on the hardest models because features are dead at initialization—a paradigm shift in research perspective.
- **Zero-parameter Analytical Formula Beats Empirical Diagnostics**: Previous metrics like kurtosis were token-level. $\gamma$ captures the true source—dimension-level outliers. The formula derived from high-dimensional geometry predicts dead rates across four modalities without parameter fitting.
- **Learning Rate Asymmetry between Bias and Weight**: Weights are amplified by inputs while the bias is additive. This observation provides a clear intuition for why normalization/centering techniques work.
- **Redefining AuxK's Role**: Previously thought to "revive" features, detailed decomposition shows AuxK actually "prevents alive features from sliding into death."
- **MMCS Improvement (0.38 → 0.97)**: Near-perfect recovery of ground-truth features on synthetic data suggests mean-centering doesn't just lower dead rates; it ensures SAEs learn the correct directions.

## Limitations & Future Work

- **Mean-centering is insufficient for a few layers**: In some protein and gene models, residual feature death remains even after centering due to variance being concentrated in a few directions, necessitating PCA whitening (Appendix E).
- **Gaussian Signal Assumption**: In the presence of heavy tails, $\Phi(-C/\gamma)$ overestimates the dead rate. While per-dim kurtosis is used as a fallback, a unified formula for heavy-tailed activations is missing.
- **Systematic Comparison Limited to Mid-network**: While Appendix data shows cross-layer transferability, a comprehensive study on how to handle layers with vastly different $\gamma$ values is still open.
- **Comparison with "Low-rank Attention" Hypothesis (Wang 2025)**: Wang et al. attribute dead rates to the low-rank structure of attention activations. The interaction between low-rank and outlier geometric factors remains to be explored.
- **Applications**: $\gamma$ could be used as a target for normalization or architecture regularization during LLM training to reduce the cost of downstream SAEs.

## Related Work & Insights

- **vs AuxK / Ghost Grad / Resampling (Gao 2024; Bricken 2023b)**: These methods try to "push" dead features with gradients. This paper proves that at high $\gamma$, the bottleneck is bias distance, not gradient deficiency.
- **vs Token-level Outlier Studies (Sun 2024; Dettmers 2022)**: Those studies focus on spikes in individual tokens for quantization; this paper focuses on "dimension-level outliers"—fixed dimensions that deviate from zero for every token.
- **vs Lu et al. 2025 / Wang et al. 2025**: Lu et al. observed similar outliers in ESMFold but offered no diagnostic/solution. Wang et al. linked death to low-rank attention. This paper is the first to provide a cross-modal diagnostic, analytical formula, and a minimalist solution.
- **vs Early SAE works**: While mean-centering was mentioned in Bricken 2023b, it was used inconsistently. This paper upgrades it from a trick to a theoretical requirement based on $\gamma$.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Reframes feature death as a geometric problem; provides a zero-fitting analytical formula.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 454 real layers + synthetic controlled experiments + multiple modalities + bias freezing ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Excellent coordination between derivations and figures; complex insights are made clear.
- **Value**: ⭐⭐⭐⭐⭐ Provides an immediately actionable mean-centering solution and a pre-training diagnostic $\gamma$ for all SAE researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](sparse_autoencoders_are_topic_models.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)
- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](../../ICLR2026/interpretability/temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](../../CVPR2026/interpretability/beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)

</div>

<!-- RELATED:END -->
