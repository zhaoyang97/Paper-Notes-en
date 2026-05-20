---
title: >-
  [Paper Note] From Observations to States: Latent Time Series Forecasting
description: >-
  [ICML 2026][Time Series][time series forecasting] The authors observe that even state-of-the-art TSF models with high prediction accuracy often exhibit "temporal disorder" (Latent Chaos) in their latent spaces. They prop…
tags:
  - "ICML 2026"
  - "Time Series"
  - "time series forecasting"
  - "latent state space"
  - "Latent Chaos"
  - "representation alignment"
  - "mutual information"
date: 2026-05-08
content_hash: cb2d7b22eb60778e
---

# From Observations to States: Latent Time Series Forecasting

**Conference**: ICML 2026  
**arXiv**: [2602.00297](https://arxiv.org/abs/2602.00297)  
**Code**: https://github.com/Muyiiiii/LatentTSF (available)  
**Area**: Time Series Forecasting / Representation Learning  
**Keywords**: time series forecasting, latent state space, Latent Chaos, representation alignment, mutual information

## TL;DR
The authors observe that even state-of-the-art TSF models with high prediction accuracy often exhibit "temporal disorder" (Latent Chaos) in their latent spaces. They propose LatentTSF: first, an AutoEncoder compresses observations into a high-dimensional latent state space; then, any mainstream backbone predicts future states in this space (using a Pred + Align dual loss); finally, the predictions are decoded back to the observation space. On six standard benchmarks, this approach consistently reduces MSE/MAE and restores temporal locality and spectral structure in the latent representations.

## Background & Motivation

**Background**: Modern TSF almost exclusively adopts the "observation space regression" paradigm: given a historical window $\mathbf{X} \in \mathbb{R}^{C \times L}$, RNN/CNN/MLP/Transformer models learn a mapping $\mathcal{F}_\theta: \mathbb{R}^{C \times L} \to \mathbb{R}^{C \times T}$ to directly predict future observations $\mathbf{Y}$, minimizing MSE/MAE.

**Limitations of Prior Work**: Through multi-perspective representation diagnostics on strong backbones like iTransformer, the authors uncover a paradox: a model may achieve low MAE in observation space, yet its internal latent representations are "temporally disordered"—adjacent time-step embeddings do not cluster, t-SNE plots lose continuity, and spectral structure is destroyed. On Electricity, the average Euclidean distance between adjacent latent states jumps from 12.94 to 94.03, and dominant periodic signals nearly vanish.

**Key Challenge**: The authors attribute this to two fundamental issues. **(i) Systems Theory**: The true observation $\mathbf{X}$ is a "noisy, partial projection" of an underlying high-dimensional dynamical system; key latent variables are unobservable in the observation space, and minimizing observation MSE encourages shortcut learning—focusing on mean regression, periodicity, and autocorrelation rather than true generative dynamics. **(ii) Optimization**: Pointwise MAE/MSE loss imposes no inductive bias for "temporal continuity," so models have no incentive to learn temporally coherent latent spaces.

**Goal**: To construct a new training paradigm that explicitly learns temporal dynamics in a "structured latent state space," rather than optimizing only for observation space accuracy; the paradigm should (a) be compatible with any existing backbone, and (b) be more robust than standard approaches on noisy, partially observed real-world data.

**Key Insight**: Rather than modifying backbone architectures, change the training paradigm itself—replace the "observation → observation" objective with a four-step pipeline: "observation → latent state → latent state prediction → decode back to observation," with all supervision applied in the latent space.

**Core Idea**: Use a pretrained and frozen AutoEncoder to independently encode each time step into a high-dimensional latent state $\mathbf{Z}$; let the backbone predict future latent states $\widehat{\mathbf{Z}}_Y$ in this space, supervised by "latent space prediction loss $\mathcal{L}_\text{Pred}$ + latent space alignment loss $\mathcal{L}_\text{Align}$," and finally use the frozen decoder to map back to the observation space to obtain $\widehat{\mathbf{Y}}$.

## Method

### Overall Architecture
A two-stage pipeline: (1) **Latent State Space Construction**: A point-wise AutoEncoder $\mathcal{E}, \mathcal{D}$ encodes each $\mathbf{x}_t \in \mathbb{R}^C$ to $\mathbf{z}_t \in \mathbb{R}^D$ ($D$ can be larger or smaller than $C$, with the focus on "better suited for dynamics modeling"), pretrained with MAE reconstruction loss and then **frozen**. (2) **Latent State Prediction**: Any TSF backbone $\mathcal{F}^\mathbf{Z}_\theta$ takes $\mathbf{Z}_X = \mathcal{E}(\mathbf{X})$ as input, outputs $\widehat{\mathbf{Z}}_Y$, and uses the frozen $\mathcal{D}$ to decode $\widehat{\mathbf{Y}} = \mathcal{D}(\widehat{\mathbf{Z}}_Y)$. During training, loss is computed not on $\widehat{\mathbf{Y}}$, but by bringing $\widehat{\mathbf{Z}}_Y$ closer to the ground-truth latent state $\mathbf{Z}_Y = \mathcal{E}(\mathbf{Y})$.

### Key Designs

1. **Point-wise AutoEncoder + Frozen Target Encoder**:

    - **Function**: Constructs a latent state space that is smoother and better suited for learning dynamics, and provides a stable regression target.
    - **Mechanism**: The AutoEncoder encodes each time step **independently** (no temporal convolution/attention), leaving all temporal structure learning to the backbone; pretrained with $\mathcal{L}_\text{Rec} = \frac{1}{L}\sum_t \|\mathbf{x}_t - \mathcal{D}(\mathcal{E}(\mathbf{x}_t))\|_1$ and then frozen. After freezing, $\mathbf{Z}_Y = \mathcal{E}(\mathbf{Y})$ serves as a **stationary target** for the backbone to regress towards.
    - **Design Motivation**: Freezing and point-wise encoding offer two benefits. First, freezing structurally prevents representation collapse—as long as the AutoEncoder maps different inputs to different latent points, constant solutions cannot be optimal (formally proven in Remark 3.1 + App. C.3), eliminating the need for SimSiam/BYOL-style stop-gradient or EMA. Second, point-wise encoding ensures the backbone receives "pure" latent states rather than temporally smoothed sequences, preserving the challenge of dynamics modeling.

2. **Joint Latent Space Loss $\mathcal{L}_\text{Pred} + \mathcal{L}_\text{Align}$**:

    - **Function**: Ensures predicted latent states $\widehat{\mathbf{Z}}_Y$ match both in "magnitude" and "direction."
    - **Mechanism**: Total loss $\mathcal{L}_\text{Total} = \alpha \cdot \|\mathbf{Z}_Y - \widehat{\mathbf{Z}}_Y\|_F^2 + \beta \cdot (1 - \cos(\mathbf{Z}_Y, \widehat{\mathbf{Z}}_Y))$. The first term is a Frobenius norm prediction loss, enforcing numerical similarity; the second is a cosine alignment loss, enforcing directional consistency. The authors provide an information-theoretic interpretation (§4): $\mathcal{L}_\text{Pred}$ can be viewed as a variational lower bound on $I(\mathbf{Z}_Y; \widehat{\mathbf{Z}}_Y)$ (reducing to squared error under Gaussian assumptions), and $\mathcal{L}_\text{Align}$ as a practical proxy for maximizing $I(\mathbf{Y}; \widehat{\mathbf{Z}}_Y)$ via a simplified InfoNCE.
    - **Design Motivation**: Ablations show that using only one term is clearly inferior to both, with consistent ranking: "full > w/o Align > w/o Pred ≈ baseline." Pred alone lacks directional constraint; Align alone lacks magnitude constraint. Default weights $\alpha=10, \beta=15$ lie in a broad "plateau" in the Pred-Align 2D heatmap, requiring no fine-tuning.

3. **Strictly No Observation Space Loss (Perceptual Loss)**:

    - **Function**: Ensures all supervision is confined to the latent space, with no MSE after the decoder.
    - **Mechanism**: The authors experimented with adding $\mathcal{L}_\text{Perc} = \|\widehat{\mathbf{Y}} - \mathbf{Y}\|^2$, but found this "double insurance" actually destabilizes the latent space—because the frozen decoder is nonlinear, small latent deviations are amplified into large reconstruction errors, introducing high gradient noise to the backbone. Thus, the final training recipe **disables** $\mathcal{L}_\text{Perc}$ by default.
    - **Design Motivation**: This overturns the conventional intuition that "supervising both latent and observation spaces is more stable," providing strong empirical support for the central claim that latent space prediction alone suffices for TSF.

### Loss & Training

Two stages. **Stage 1**: Pretrain the AutoEncoder with $\mathcal{L}_\text{Rec}$ (MAE reconstruction, per time step), then freeze all parameters. **Stage 2**: Train the backbone with $\mathcal{L}_\text{Total} = 10 \cdot \mathcal{L}_\text{Pred} + 15 \cdot \mathcal{L}_\text{Align}$, inputting $\mathbf{Z}_X$ and outputting $\widehat{\mathbf{Z}}_Y$, finally decoding with the frozen $\mathcal{D}$. AdamW + cosine scheduling + early stopping (patience=5) are used.

## Key Experimental Results

### Main Results
Comprehensive comparisons on six standard benchmarks (ETTh1/h2/m1/m2, Traffic, Electricity) × six backbones (CMoS, DLinear, PatchTST, TimeBase, TimeXer, iTransformer), evaluating "Original" vs. "with LatentTSF" training.

| Dataset | Metric | Prev. SOTA | +LatentTSF | Gain |
|---------|--------|------------|------------|------|
| Electricity | MSE (PatchTST) | 0.389 | 0.207 | -0.182 (-47%) |
| Electricity | MSE (iTransformer) | 0.268 | 0.194 | -0.074 (-28%) |
| Traffic | MSE (TimeXer) | 1.270 | 0.636 | -0.634 (-50%) |
| Traffic | MSE (PatchTST) | 0.982 | 0.719 | -0.263 (-27%) |
| ETTh1 | MSE (TimeXer) | 0.485 | 0.432 | -0.053 (-11%) |
| ETTm2 | MSE (PatchTST) | 0.261 | 0.247 | -0.014 (-5%) |

LatentTSF reduces error across almost all backbone × dataset combinations, with **greater benefits for higher variable dimensions and longer horizons**. On Electricity (321 variables), PatchTST's MSE is halved; on low-dimensional data like ETTm2 (7 variables), improvements are milder but still positive.

### Ablation Study

| Config | ETTh1 CKA ↓ | Eff. Rank ↑ | TTC ↑ | Description |
|--------|-------------|-------------|-------|-------------|
| Observation space | – | 2.86 | 0.913 | Standard paradigm |
| LatentTSF latent space | 0.015 | 3.36 | 0.983 | Nontrivial mapping + ~7% temporal consistency gain |
| Electricity observation space | – | 7.89 | 0.894 | – |
| Electricity LatentTSF | 0.023 | 34.90 | 0.967 | Effective Rank 4.4×, TTC +7% |

| Config | Electricity MSE | Description |
|--------|-----------------|-------------|
| DLinear baseline | 0.201 | Original observation space |
| LatentTSF (full) | 0.182 | Full version |
| w/o $\mathcal{L}_\text{Align}$ | 0.183 | Pred is main driver (-8.8% vs baseline) |
| DLinear + Align on observation | ≈baseline | Align alone in observation space is ineffective or harmful |
| LatentTSF + Perceptual | Worse than full | Observation space supervision disrupts latent space |

### Key Findings
- $\mathcal{L}_\text{Pred}$ is the **main driver** of gains (removing Align retains 90% of improvement), but $\mathcal{L}_\text{Align}$ is only effective in latent space—moving it to observation space nullifies its effect, strongly supporting the claim that "latent space supervision is key."
- On ETTh1, adding noise $\sigma \in \{0, 0.1, 0.2, 0.5\}$ or missing rates 0%-30%, LatentTSF consistently achieves lower MSE than observation space training at every perturbation level, indicating that structured latent spaces enhance noise robustness.
- AE learning rate sweeps show that even with perceptual loss and joint fine-tuning of encoder/decoder, performance is inferior to **frozen AE + latent space loss only**—repeatedly confirming that freezing the target encoder is the source of stability.
- For long horizons ($T=720$), LatentTSF's advantage is further amplified, as it transforms the "error accumulation" problem into "drift on a stable manifold," fundamentally avoiding the chain amplification of first-order errors in observation space.

## Highlights & Insights
- **"Latent Chaos" is a phenomenon worth naming**: Using t-SNE, spectral analysis, and adjacent Euclidean distance, the authors jointly verify the counterintuitive fact that "accurate prediction ≠ learning temporal structure," sounding an alarm for the TSF community—future evaluations should consider not just MSE/MAE, but also the geometry/dynamics of latent representations.
- **Frozen target encoder structurally prevents collapse**: Unlike SimSiam/BYOL, which rely on stop-gradient or EMA engineering tricks, this work proves that as long as $\mathcal{E}$ is frozen and discriminative, cosine alignment loss cannot be minimized by a constant solution. This theoretical insight is valuable for self-supervised representation learning.
- **Striking contrast between training paradigm and architectural innovation**: Without changing a single line of backbone code, simply "training in a different space" pushes six different backbones to SOTA, clearly illustrating "paradigm > architecture"—a reflection for TSF papers that endlessly tweak Transformer architectures.

## Limitations & Future Work
- The default weights ($\alpha=10, \beta=15$) are "universal values" chosen after extensive sweeps; while robust, they may not be optimal for every dataset, and extreme long-horizon or ultra-high-dimensional scenarios may still require tuning.
- The AE encodes each time step independently, intentionally ignoring temporal information—while deliberate, this limits the "richness" of the latent space; adding lightweight temporal structures (e.g., short-range conv) may further improve latent state quality.
- Experiments are limited to multivariate time series regression, not covering probabilistic forecasting, long-tail distributions, or irregular sampling.
- Comparisons with some of the latest strong backbones (e.g., TimeMixer++, latest TimeXer) and large-scale TSF foundation models are lacking.

## Related Work & Insights
- **vs Representation Regularization Methods (Glocal-IB / TimeAlign)**: These methods still train the backbone in observation space, using latent space terms as regularizers; LatentTSF **moves the backbone entirely to latent space**, a more thorough approach.
- **vs Patch-wise loss**: The latter refines local supervision in observation space but does not address the "noisy observation space" issue; LatentTSF changes the playing field entirely.
- **vs SimSiam / BYOL**: Both use cosine alignment + non-contrastive learning, but this work replaces the learnable target with a **pretrained + frozen AE target**, structurally avoiding collapse—a clean transfer of this idea to supervised learning.
- **vs InfoNCE**: The authors show InfoNCE is a strict MI lower bound, but when negative samples are omitted, it reduces to cosine alignment, losing strictness but retaining practicality—a trade-off relevant for similar settings (small batch, frozen target).

## Rating
- Novelty: ⭐⭐⭐⭐ "Moving TSF to latent space" is conceptually clear despite its simplicity; the naming of Latent Chaos and theoretical guarantee of frozen encoder have notable research value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 backbones × 6 datasets × multiple horizons × ablations + noise robustness tests, with very comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ From phenomenon diagnosis → mechanism analysis → theoretical framework → empirical validation, the logical chain is very clear, with well-explained formulas and intuition.
- Value: ⭐⭐⭐⭐ Paradigm-level work, can be directly applied as a plug-in to almost any TSF backbone, with significant potential community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation](helix_hybrid_encoding_with_learnable_identity_and_cross-dimensional_synthesis_fo.md)
- [\[ICML 2026\] Time-series Forecasting Through the Lens of Dynamics](time-series_forecasting_through_the_lens_of_dynamics.md)
- [\[ICML 2026\] PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)
- [\[ICML 2026\] Doubly Outlier-Robust Online Infinite Hidden Markov Model](doubly_outlier-robust_online_infinite_hidden_markov_model.md)
- [\[ICML 2026\] Fern: 椭球时间序列预测——用 Brenier 定理直接参数化 Jacobian 谱结构](ellipsoidal_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
