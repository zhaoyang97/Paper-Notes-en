---
title: >-
  [Paper Note] From Observations to States: Latent Time Series Forecasting
description: >-
  [ICML 2026][Time Series][Time Series Forecasting] The authors observe that existing TSF models often exhibit "Latent Chaos" in their latent spaces despite high prediction accuracy. They propose LatentTSF: an AutoEncoder…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time Series Forecasting"
  - "Latent State Space"
  - "Latent Chaos"
  - "Representation Alignment"
  - "Mutual Information"
date: 2026-05-08
content_hash: d1ccfc03b9a7511c
---

# From Observations to States: Latent Time Series Forecasting

**Conference**: ICML 2026  
**arXiv**: [2602.00297](https://arxiv.org/abs/2602.00297)  
**Code**: https://github.com/Muyiiiii/LatentTSF (Available)  
**Area**: Time Series Forecasting / Representation Learning  
**Keywords**: Time Series Forecasting, Latent State Space, Latent Chaos, Representation Alignment, Mutual Information

## TL;DR
The authors observe that existing TSF models often exhibit "Latent Chaos" in their latent spaces despite high prediction accuracy. They propose LatentTSF: an AutoEncoder first compresses observations into a high-dimensional latent state space, allowing any mainstream backbone to perform future prediction within this space (using a dual Pred + Align loss) before decoding back to the observation space. This approach consistently reduces MSE/MAE across 6 standard benchmarks and restores the temporal locality and spectral structure of latent representations.

## Background & Motivation

**Background**: Modern TSF almost exclusively adopts the "observation space regression" paradigm: given a historical window $\mathbf{X} \in \mathbb{R}^{C \times L}$, an RNN / CNN / MLP / Transformer learns a mapping $\mathcal{F}_\theta: \mathbb{R}^{C \times L} \to \mathbb{R}^{C \times T}$ to directly predict future observations $\mathbf{Y}$ by minimizing MSE / MAE.

**Limitations of Prior Work**: Through multi-perspective representation-level diagnostics on strong backbones like iTransformer, the authors identify a surprising paradox: a model can achieve low MAE in the observation space while its internal latent representations remain "temporally chaotic." Neighboring time-step embeddings fail to cluster, continuous trajectories vanish in t-SNE, and spectral structures are destroyed. On the Electricity dataset, the average Euclidean distance between adjacent latent states surges from an original 12.94 to 94.03, and dominant periodic signals almost disappear.

**Key Challenge**: This phenomenon is attributed to two fundamental issues. **(i) System Theory**: Real-world observations $\mathbf{X}$ are "noisy + partial projections" of underlying high-dimensional dynamical systems. Key latent variables are invisible in the observation space. Minimizing observation MSE encourages models to learn shortcuts—capturing shallow statistics like mean reversion, periodicity, and autocorrelation rather than true generative dynamics. **(ii) Optimization**: Point-wise MAE/MSE losses provide no inductive bias for "temporal continuity," so models do not actively learn temporally coherent latent spaces.

**Goal**: Construct a new training paradigm that enables models to explicitly learn temporal dynamics within a "structured latent state space" rather than strictly optimizing for observation space accuracy. The paradigm must (a) be compatible with any existing backbone and (b) be more robust than the standard paradigm on noisy, partially observable real-world data.

**Key Insight**: Instead of modifying backbone architectures, modify the training paradigm itself. The "observation $\to$ observation" objective is transformed into a four-step pipeline: "observation $\to$ latent state $\to$ latent state prediction $\to$ decode to observation," where all supervision is applied in the latent space.

**Core Idea**: A pre-trained and frozen AutoEncoder encodes each time step independently into a high-dimensional latent state $\mathbf{Z}$. A backbone learns to predict future latent states $\widehat{\mathbf{Z}}_Y$ within the $\mathbf{Z}$ space. The supervision signals consist of a "latent space prediction loss $\mathcal{L}_\text{Pred}$ + latent space alignment loss $\mathcal{L}_\text{Align}$." Finally, a frozen decoder maps the results back to the observation space to obtain $\widehat{\mathbf{Y}}$.

## Method

### Overall Architecture
A two-stage pipeline: (1) **Latent State Space Construction**: A point-wise AutoEncoder $\mathcal{E}, \mathcal{D}$ maps $\mathbf{x}_t \in \mathbb{R}^C$ to $\mathbf{z}_t \in \mathbb{R}^D$ ($D$ can be larger or smaller than $C$; the focus is "suitability for dynamics modeling"). It is pre-trained with MAE reconstruction loss and then **frozen**. (2) **Latent State Prediction**: Any TSF backbone $\mathcal{F}^\mathbf{Z}_\theta$ takes $\mathbf{Z}_X = \mathcal{E}(\mathbf{X})$ as input and outputs $\widehat{\mathbf{Z}}_Y$. The frozen $\mathcal{D}$ generates $\widehat{\mathbf{Y}} = \mathcal{D}(\widehat{\mathbf{Z}}_Y)$. During training, losses are no longer calculated for $\widehat{\mathbf{Y}}$; instead, $\widehat{\mathbf{Z}}_Y$ is pulled closer to the ground-truth latent state $\mathbf{Z}_Y = \mathcal{E}(\mathbf{Y})$ in the latent space.

### Key Designs

1. **Point-wise AutoEncoder + Frozen Target Encoder**:
    - **Function**: Construct a latent state space that is smoother and more suitable for learning dynamics than the observation space, while providing a stable regression target.
    - **Mechanism**: The AutoEncoder encodes **independently per time step** (no convolution/attention along the time dimension), leaving all temporal structure for the backbone to learn. It is frozen after pre-training with $\mathcal{L}_\text{Rec} = \frac{1}{L}\sum_t \|\mathbf{x}_t - \mathcal{D}(\mathcal{E}(\mathbf{x}_t))\|_1$. Once frozen, $\mathbf{Z}_Y = \mathcal{E}(\mathbf{Y})$ serves as a **stationary target** for backbone regression.
    - **Design Motivation**: Freezing + point-wise processing offers two benefits. First, freezing provides a natural guarantee against representation collapse—as long as the AutoEncoder maps different inputs to different latent points, a constant solution cannot be optimal (formalized in Remark 3.1 + App. C.3). This eliminates the need for stop-gradient or EMA hacks like SimSiam/BYOL. Second, point-wise encoding ensures the backbone receives "pure" latent states rather than time series already smoothed by an AE, which would make the dynamics modeling task trivial.

2. **Latent Space Joint Loss $\mathcal{L}_\text{Pred} + \mathcal{L}_\text{Align}$**:
    - **Function**: Ensure predicted latent states $\widehat{\mathbf{Z}}_Y$ are correct in both "magnitude" and "direction."
    - **Mechanism**: Total loss $\mathcal{L}_\text{Total} = \alpha \cdot \|\mathbf{Z}_Y - \widehat{\mathbf{Z}}_Y\|_F^2 + \beta \cdot (1 - \cos(\mathbf{Z}_Y, \widehat{\mathbf{Z}}_Y))$. The former is a Frobenius norm prediction loss to constrain numerical values; the latter is a cosine alignment loss to constrain directional consistency. Section §4 provides an information-theoretic interpretation: $\mathcal{L}_\text{Pred}$ maximizes the variational lower bound of $I(\mathbf{Z}_Y; \widehat{\mathbf{Z}}_Y)$ (degenerating to squared error under Gaussian assumptions), while $\mathcal{L}_\text{Align}$ acts as a practical proxy for maximizing $I(\mathbf{Y}; \widehat{\mathbf{Z}}_Y)$ via a simplified InfoNCE form.
    - **Design Motivation**: Ablations show that using either loss alone is significantly weaker than using both, with a consistent ranking of "full > w/o Align > w/o Pred ≈ baseline." Pred lacks directional constraints, while Align lacks magnitude. Default weights $\alpha=10, \beta=15$ sit on a stable "plateau" in the Pred-Align heatmap, requiring minimal tuning.

3. **Complete Rejection of Observation Space Loss (Perceptual Loss)**:
    - **Function**: Explicitly lock the entire supervision signal within the latent space, preventing post-decoder MSE from interfering with training.
    - **Mechanism**: The authors attempted to add $\mathcal{L}_\text{Perc} = \|\widehat{\mathbf{Y}} - \mathbf{Y}\|^2$ but found this seemingly natural "double insurance" actually destabilizes the latent space. Since the frozen decoder is non-linear, small deviations in latent space are amplified into large reconstruction errors, introducing significant gradient noise back to the backbone. Thus, $\mathcal{L}_\text{Perc}$ is **disabled** by default.
    - **Design Motivation**: This contradicts the intuition that supervising both spaces is more stable, serving as strong empirical support for the central thesis that latent space prediction alone is sufficient for TSF.

### Loss & Training
Two stages. **Stage 1**: Pre-train the AutoEncoder using $\mathcal{L}_\text{Rec}$ (MAE reconstruction, point-wise). All parameters are frozen upon completion. **Stage 2**: Train the backbone using $\mathcal{L}_\text{Total} = 10 \cdot \mathcal{L}_\text{Pred} + 15 \cdot \mathcal{L}_\text{Align}$, taking $\mathbf{Z}_X$ as input to produce $\widehat{\mathbf{Z}}_Y$, which is then decoded by the frozen $\mathcal{D}$. Optimization via AdamW + cosine scheduler + early stopping (patience=5).

## Key Experimental Results

### Main Results
Complete comparisons were conducted across 6 standard benchmarks (ETTh1/h2/m1/m2, Traffic, Electricity) × 6 backbones (CMoS, DLinear, PatchTST, TimeBase, TimeXer, iTransformer), comparing the "Original" training paradigm with "LatentTSF."

| Dataset | Metric | Best Original | +LatentTSF | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Electricity | MSE (PatchTST) | 0.389 | 0.207 | -0.182 (-47%) |
| Electricity | MSE (iTransformer) | 0.268 | 0.194 | -0.074 (-28%) |
| Traffic | MSE (TimeXer) | 1.270 | 0.636 | -0.634 (-50%) |
| Traffic | MSE (PatchTST) | 0.982 | 0.719 | -0.263 (-27%) |
| ETTh1 | MSE (TimeXer) | 0.485 | 0.432 | -0.053 (-11%) |
| ETTm2 | MSE (PatchTST) | 0.261 | 0.247 | -0.014 (-5%) |

LatentTSF reduces error across nearly all backbone × dataset combinations. **Gains are larger for higher variable dimensions and longer horizons.** On Electricity (321 variables), PatchTST's MSE is nearly halved; improvements on low-dimensional data like ETTm2 (7 variables) are more modest but still positive.

### Ablation Study

| Config | ETTh1 CKA ↓ | Eff. Rank ↑ | TTC ↑ | Description |
| :--- | :--- | :--- | :--- | :--- |
| Observation Space | – | 2.86 | 0.913 | Standard paradigm |
| LatentTSF Latent Space | 0.015 | 3.36 | 0.983 | Non-trivial mapping + ~7% temporal consistency gain |
| Electricity Observation | – | 7.89 | 0.894 | – |
| Electricity LatentTSF | 0.023 | 34.90 | 0.967 | Effective Rank 4.4×, TTC +7% |

| Config | Electricity MSE | Description |
| :--- | :--- | :--- |
| DLinear baseline | 0.201 | Original observation space |
| LatentTSF (full) | 0.182 | Full version |
| w/o $\mathcal{L}_\text{Align}$ | 0.183 | Pred is the main driver (-8.8% vs. baseline) |
| DLinear + Align on obs | ≈baseline | Align alone on observations is ineffective/harmful |
| LatentTSF + Perceptual | Weaker than full | Observation supervision disturbs latent space |

### Key Findings
- $\mathcal{L}_\text{Pred}$ is the **main driver** of gains (retaining 90% of the improvement without Align), but $\mathcal{L}_\text{Align}$ is only effective in latent space—it fails when applied to observations. This strongly supports the argument that latent space supervision is key.
- Under noise $\sigma \in \{0, 0.1, 0.2, 0.5\}$ or missing rates 0%-30% on ETTh1, LatentTSF consistently maintains lower MSE than observation-space training, demonstrating that a structured latent space enhances noise robustness.
- AE learning rate scans show that even with perceptual loss for fine-tuning the encoder/decoder, performance is inferior to **frozen AE + latent space loss only**. This confirms that a frozen target encoder is the source of stability.
- The advantages of LatentTSF are amplified for long horizons ($T=720$), as it transforms the "error accumulation" problem into a "drift on a stable manifold" problem, effectively avoiding the chained amplification of first-order errors in the observation space.

## Highlights & Insights
- **"Latent Chaos" is a phenomenon worth naming**: Use of t-SNE + spectral analysis + adjacent Euclidean distance validates the counter-intuitive fact that "accurate prediction ≠ learned temporal structure." This serves as a warning to the TSF community: evaluating models should look beyond MSE/MAE to the geometric/dynamical properties of latent representations.
- **Frozen target encoder structurally prevents collapse**: Unlike SimSiam/BYOL which rely on engineering hacks like stop-gradients or EMA, this paper proves that as long as $\mathcal{E}$ is frozen and distinguishes inputs, the cosine alignment loss cannot reach an optimum at a constant solution. This theoretical observation is valuable for self-supervised representation learning.
- **Paradigm vs. Architecture Innovation**: The paper achieves SOTA across 6 different backbones without modifying a single line of backbone code. By demonstrating "Paradigm > Architecture," it provides a reflective critique of the trend of incremental Transformer modifications in TSF research.

## Limitations & Future Work
- Default weights ($\alpha=10, \beta=15$) were selected from extensive scanning as "universal values." While robust, they might not be optimal for every dataset, particularly in extremely long-horizon or high-dimensional scenarios.
- The AE encodes time steps independently, meaning it ignores temporal information by design. This limits latent space "richness"; adding lightweight temporal structures (e.g., short-range convolutions) could improve latent state quality.
- Experiments are restricted to multivariate numerical forecasting, without addressing probabilistic forecasting, long-tail distributions, or irregular sampling.
- Comparisons with some highly competitive recent backbones (e.g., TimeMixer++, latest TimeXer) and large-scale TSF foundation models are currently missing.

## Related Work & Insights
- **vs. Representation Regularization (Glocal-IB / TimeAlign)**: These methods still train backbones in the observation space using latent terms as regularizers; LatentTSF is more radical by **moving the backbone entirely into the latent space**.
- **vs. Patch-wise loss**: The latter refines local supervision in the observation space without solving the "noisy observation space" issue; LatentTSF changes the battlefield entirely.
- **vs. SimSiam / BYOL**: Also uses cosine alignment + non-contrastive learning, but replaces the learnable target with a **pre-trained + frozen AE target**. This is a clean migration of those ideas to a supervised setting while structurally avoiding collapse.
- **vs. InfoNCE**: The authors derive InfoNCE as a strict MI lower bound. Simplifying it by removing negative samples results in cosine alignment, sacrificing strictness for utility—a trade-off useful for similar settings (small batch, frozen target).

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of "moving TSF to latent space" is clear, and the naming of "Latent Chaos" plus the theoretical guarantees of frozen encoders provide significant research value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 6 backbones × 6 datasets × multiple horizons × wide-ranging ablations and robustness tests.
- Writing Quality: ⭐⭐⭐⭐⭐ Very clear logical chain from phenomenon diagnosis $\to$ mechanism analysis $\to$ theoretical framework $\to$ empirical validation.
- Value: ⭐⭐⭐⭐ A paradigm-level work that can be used as a plug-in for almost any TSF backbone, with high potential for community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Latent Laplace Diffusion for Irregular Multivariate Time Series](latent_laplace_diffusion_for_irregular_multivariate_time_series.md)
- [\[ICML 2026\] It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks](its_time_towards_the_next_generation_of_time_series_forecasting_benchmarks.md)
- [\[ICML 2026\] Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting](beyond_extrapolation_knowledge_utilization_paradigm_with_bidirectional_inspirati.md)
- [\[ICML 2026\] Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)
- [\[ICML 2026\] Time-series Forecasting Through the Lens of Dynamics](time-series_forecasting_through_the_lens_of_dynamics.md)

</div>

<!-- RELATED:END -->
