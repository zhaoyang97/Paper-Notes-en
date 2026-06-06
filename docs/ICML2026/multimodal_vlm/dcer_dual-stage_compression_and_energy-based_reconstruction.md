---
title: >-
  [Paper Note] DCER: Robust Multimodal Fusion via Dual-Stage Compression and Energy-Based Reconstruction
description: >-
  [ICML 2026][Multimodal VLM][Dual-stage compression] DCER utilizes "intra-modal frequency domain compression + cross-modal bottleneck tokens" as a unified robust fusion pipeline. It employs a learned energy function for g…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Dual-stage compression"
  - "frequency domain transform"
  - "bottleneck token"
  - "energy-based model"
  - "missing modality"
date: 2026-05-08
content_hash: c860cb6aee1586e4
---

# DCER: Robust Multimodal Fusion via Dual-Stage Compression and Energy-Based Reconstruction

**Conference**: ICML 2026  
**arXiv**: [2602.04904](https://arxiv.org/abs/2602.04904)  
**Code**: To be open-sourced on GitHub (promised in paper)  
**Area**: Multimodal Fusion / Multimodal Sentiment Analysis  
**Keywords**: Dual-stage compression, frequency domain transform, bottleneck token, energy-based model, missing modality

## TL;DR
DCER utilizes "intra-modal frequency domain compression + cross-modal bottleneck tokens" as a unified robust fusion pipeline. It employs a learned energy function for gradient-descent-based reconstruction of missing modalities, while treating the final energy value as intrinsic uncertainty, achieving new SOTAs on MOSI, MOSEI, and SIMS.

## Background & Motivation

**Background**: Currently, the mainstream approaches for Multimodal Sentiment Analysis (MSA) involve schemes like cross-modal attention (MulT), tensor fusion, or MAG-BERT, which "encode modalities independently then concatenate/cross-attend." These achieve high scores under complete modality settings.

**Limitations of Prior Work**: (a) These schemes allow models to learn "independent bypass channels for each modality"—the entire prediction collapses when a modality is absent; (b) the majority of dimensions in raw audio/video are noise, and without an explicit compression mechanism, models struggle to capture the frequency bands where sentiment actually resides; (c) existing missing-modality evaluations use zero-padding, which models can exploit as a hidden token, leading to an **overestimation** of robustness by 15–51% (as empirically tested by the authors).

**Key Challenge**: The trade-off between "high performance on complete data" and "robustness during modality absence"—the former encourages learning independent modality pathways, while the latter requires forcing all modalities to share a single bottleneck.

**Goal**: (1) Simultaneously address noise robustness and missing modality robustness; (2) provide intrinsic uncertainty; (3) propose a more rigorous evaluation protocol (noise-masking instead of zero-masking).

**Key Insight**: Information Bottleneck principle + frequency domain priors. Sentiment signals are sparse in the frequency domain (low-frequency prosody in audio, low spatial frequency in facial video), whereas noise is typically uniformly distributed across a wide spectrum—frequency domain transforms naturally perform lossy compression. By stacking "a small number of learnable query tokens as a cross-modal fixed-capacity bottleneck," all modality information is forced through it, prohibiting modality shortcuts.

**Core Idea**: Treat "compression" as a universal language—integrated across intra-modal (frequency domain), cross-modal (query bottleneck), and missing reconstruction (energy function), with the final energy value providing "free" uncertainty.

## Method

### Overall Architecture
The pipeline consists of three stages: **Stage 1 (Intra-modal Frequency Compression)**: Audio uses learnable wavelets (DWT, 3 levels) for multi-scale time-frequency compression; video uses 2D DCT to segment 4 frequency bands with perception-aware attention; text uses RoBERTa + projection (text is already a discrete symbolized compression). **Stage 2 (Cross-modal Bottleneck)**: Three modalities are concatenated as $H=[h_a;h_v;h_t]$. $K=4$ learnable queries $Q$ extract $Z=\mathrm{softmax}(QH^\top/\sqrt D)H \in \mathbb R^{K\times D}$ via cross-attention. Since $K\ll T_a+T_v+T_t$, a strong capacity constraint is formed. **Stage 3 (Energy Reconstruction)**: When modalities are missing, the energy function $E_\theta(h_m;Z)$ is used to recover $h_m^*$ through $T=3$ steps of gradient descent with momentum; the final energy $E_\theta(h_m^*)$ serves as uncertainty. Finally, an MLP prediction head outputs the sentiment score.

### Key Designs

1.  **Learnable Frequency Compression (Stage 1)**:
    - **Function**: Filters out wide-band noise components and preserves sparse sentiment frequency bands before cross-modal fusion.
    - **Mechanism**: Audio uses 3-level DWT with learnable wavelet bases initialized by Daubechies-4 to obtain multi-scale coefficients $\{c_L,d_L,\ldots,d_1\}$, fused via cross-scale attention: $h_a=\mathrm{Proj}(\mathrm{CrossScaleAttn}(W(x_a)))$. Video undergoes 2D-DCT split into 4 bands with **learnable** boundaries, followed by frequency-aware attention: $h_v=\mathrm{Proj}(\mathrm{FreqAttn}(D(x_v)))$.
    - **Design Motivation**: Fixed BatchNorm/LayerNorm cannot "filter by frequency," but wavelets/DCT provide sparse orthogonal bases in the frequency domain. Making decomposition parameters learnable preserves priors while remaining task-adaptive. Table 4 shows MAE increases by 3.7% without frequency compression, and 12.3% in extreme missing scenarios.

2.  **K=4 Query Tokens as Cross-modal Fixed Capacity Bottleneck (Stage 2)**:
    - **Function**: Block modality-specific shortcuts—prediction cannot be based on independent channel summation $\hat y = f_a(h_a)+f_v(h_v)+f_t(h_t)$ because it only stems from $Z$.
    - **Mechanism**: A 6-layer fusion transformer where each layer performs (bottleneck $\leftarrow$ modalities) cross-attention, bottleneck self-attention, and FFN with residual connections. $K=4$ is identified as the optimal point between compression and expressiveness.
    - **Design Motivation**: Inspired by Perceiver, but DCER treats latent queries as modality-agnostic information aggregation points; this ensures that under missing modality conditions, $K$ tokens still participate in downstream inference.

3.  **Energy Function + Gradient Reconstruction + Intrinsic Uncertainty (Stage 3)**:
    - **Function**: Estimates $h_m^*$ when modality $m$ is missing; simultaneously outputs a scalar energy value as confidence.
    - **Mechanism**: $E_\theta(h_m;Z)=f_\theta(h_m - \mathrm{CrossAttn}(h_m, Z)) + \lambda_E g_\theta(h_m)$, where $f_\theta, g_\theta$ are learned MLPs. Initialized from $\mu_\theta(Z,h_\mathrm{obs})+\epsilon$, it performs $T=3$ steps of momentum gradient descent: $h_m^{(t)}\leftarrow h_m^{(t-1)} - (\rho v^{(t-1)}+\eta\nabla E_\theta)$. For complete modalities, $T=0$; for missing ones, $T=3$ significantly improves performance. The correlation coefficient $\rho$ between $E_\theta(h_m^*)$ and prediction error is $>0.72$.
    - **Design Motivation**: Traditional VAE/GAN reconstructions are unstable in high dimensions. The energy function converts "reconstruction" into "moving toward learned low-energy basins via gradients," which is stable and naturally provides uncertainty—a core differentiator from VAE/GAN schemes.

### Loss & Training
The total loss is $\mathcal L = \mathcal L_\mathrm{pred} + \alpha\mathcal L_\mathrm{recon} + \beta\mathcal L_\mathrm{energy} + \gamma\mathcal L_\mathrm{joint}$, with $\alpha=0.1, \beta=0.01, \gamma=0.05$. $\mathcal L_\mathrm{joint}=\|Z_\mathrm{full}-Z_\mathrm{recon}\|^2$ ensures consistency between bottlenecks calculated from real and reconstructed modalities. Optimization uses AdamW with lr=1e-5, batch size 32, 40 epochs, averaged over 5 seeds.

## Key Experimental Results

### Main Results

| Dataset | Metric | DCER | Suboptimal | Gain |
|---|---|---|---|---|
| CMU-MOSI | MAE↓ | **0.669** | 0.710 (MMA) | -5.8% |
| CMU-MOSI | Corr↑ | **0.823** | 0.796 (EMT) | +3.4% |
| CMU-MOSI | Acc-7↑ | **51.6%** | 48.1% (MSAmba) | +7.3% |
| CMU-MOSEI | MAE↓ | **0.498** | 0.514 (MSAmba) | -3.1% |
| CMU-MOSEI | Acc-7↑ | **55.0%** | 53.6% (EMT) | +2.6% |
| CMU-MOSEI | F1↑ | **84.9** | 83.9 (MSAmba) | +1.2% |
| CH-SIMS | Acc-5↑ | **55.5%** | 45.2% (MTFN) | +22.8% |
| CH-SIMS | F1↑ | **81.7** | 79.7 (MMA) | +2.5% |

### Ablation Study

| Configuration | MOSI MAE↓ | MOSI Acc-7↑ |
|---|---|---|
| Full DCER ($K=4$, freq on, $T=3$ for missing) | 0.669 | 51.6 |
| w/o Freq Compression (Linear instead of Wavelet/DCT) | 0.694 | -3.7% (Full) / -12.3% (Extreme Missing) |
| $K=2$ tokens | Significant info loss | Decrease |
| $K=8$ tokens | Near zero bottleneck, overfitting | Decrease |
| $T=0$ (No energy iteration) | Significant drop in missing scenarios | $T=3$ is the sweet spot |
| zero-mask evaluation vs noise-mask | Former overestimates by 15–51% | Suggests protocol flaws |

### Key Findings
- **Robustness is U-shaped**: Multimodal fusion is strongest with full modalities and also strongest under heavy missing rates (>50%)—the intermediate range almost ties with unimodal performance because losing one or two modalities still allows modality shortcuts to "deceive" the model, whereas heavy missingness forces the use of truly learned cross-modal compressed representations.
- The correlation between energy values and prediction error ($\rho > 0.72$) allows $E_\theta$ to be used as a rejection threshold for risk-sensitive predictions.
- The +22.8% Acc-5 gain on CH-SIMS is a remarkable fine-grained improvement, indicating that the cross-modal bottleneck maximizes the discriminative power for "ambiguous emotions."

## Highlights & Insights
- **Compression as a unified language**: Intra-modal (frequency), cross-modal (query bottleneck), and missing reconstruction (energy function) are all interpreted under the "capacity-limited channel" principle—a rare and cohesive narrative.
- **Intrinsic Uncertainty**: Provides uncertainty for free without the complexity of MC dropout or ensembles, and $\rho > 0.72$ is sufficient for engineering applications.
- **Correcting Zero-mask Evaluation**: Reveals that 15–51% of performance in prior missing-rate benchmarks may stem from the model treating "zero" as a signal for missingness. Advocating for noise-mask evaluation is a valuable contribution to the community.

## Limitations & Future Work
- Frequency selection relies on manual priors (Audio $\rightarrow$ Wavelet, Video $\rightarrow$ DCT, Text $\rightarrow$ RoBERTa); it lacks an automated mechanism for new modalities like IMU.
- The energy function is relatively simple (two MLPs) and is not aligned with mainstream EBM training like SGLD/Contrastive Divergence, potentially falling into local minima with sparse data.
- Validation is limited to sentiment analysis; tasks requiring multi-step reasoning or long-context (e.g., VideoQA) remain untested, and whether $K=4$ remains optimal for long sequences is unverified.
- The $\rho > 0.72$ correlation is not yet calibrated uncertainty; comparison with conformal prediction or Platt scaling is missing.

## Related Work & Insights
- **vs MulT (Tsai et al. 2019)**: MulT uses direct cross-modal attention without compression; DCER adds bottlenecks around cross-attn to prevent collapse during modality absence.
- **vs Perceiver (Jaegle et al. 2021)**: Uses learnable queries as bottlenecks but lacks intra-modal frequency compression and missing modality handling. DCER completes these components.
- **vs EMT (Sun et al. 2023)**: EMT performs dual-level feature reconstruction using a deterministic decoder; DCER uses energy gradient descent and outputs uncertainty.
- **vs VAE/GAN-based Reconstruction**: These are unstable in high dimensions; EBM bypasses this by "only learning energy, not an explicit generator," while providing uncertainty naturally.

## Rating
- **Novelty**: ⭐⭐⭐⭐ While frequency transforms and bottlenecks are not new, integrating EBM into multimodal missing scenarios for uncertainty is a unique combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets, 8 baselines, systematic ablations, and a discussion on evaluation protocols, though lacks complex VideoQA verification.
- **Writing Quality**: ⭐⭐⭐⭐ Figure 1 explains the entire story very clearly; formulas are well-balanced.
- **Value**: ⭐⭐⭐⭐ High practical value for industrial-grade multimodal systems—providing both SOTA performance and free uncertainty while advocating for better evaluation standards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](../../CVPR2026/multimodal_vlm/duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)
- [\[ICML 2026\] Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_.md)
- [\[ICML 2026\] VLANeXt: A Recipe for Building Robust VLA Models](vlanext_recipes_for_building_strong_vla_models.md)
- [\[ICML 2026\] Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)
- [\[CVPR 2026\] Unbiased Dynamic Multimodal Fusion](../../CVPR2026/multimodal_vlm/unbiased_dynamic_multimodal_fusion.md)

</div>

<!-- RELATED:END -->
