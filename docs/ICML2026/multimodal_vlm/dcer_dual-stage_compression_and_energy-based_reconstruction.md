---
title: >-
  [Paper Note] DCER: Robust Multimodal Fusion via Dual-Stage Compression and Energy-Based Reconstruction
description: >-
  [ICML 2026][Multimodal VLM][Dual-stage compression] DCER unifies "intra-modal frequency domain compression + cross-modal bottleneck token" as a robust fusion pipeline…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Dual-stage compression"
  - "frequency domain transform"
  - "bottleneck token"
  - "energy model"
  - "missing modality"
date: 2026-05-08
content_hash: e2f50e33fde9e8a2
---

# DCER: Robust Multimodal Fusion via Dual-Stage Compression and Energy-Based Reconstruction

**Conference**: ICML 2026  
**arXiv**: [2602.04904](https://arxiv.org/abs/2602.04904)  
**Code**: To be open-sourced on GitHub (as promised in the paper)  
**Area**: Multimodal Fusion / Multimodal Sentiment Analysis  
**Keywords**: Dual-stage compression, frequency domain transform, bottleneck token, energy model, missing modality

## TL;DR
DCER unifies "intra-modal frequency domain compression + cross-modal bottleneck token" as a robust fusion pipeline, employs a learned energy function for gradient-based reconstruction of missing modalities, and uses the final energy value as intrinsic uncertainty, achieving new SOTA on MOSI/MOSEI/SIMS.

## Background & Motivation

**Background**: In multimodal sentiment analysis (MSA), mainstream approaches such as cross-modal attention (MulT), tensor fusion, and MAG-BERT typically "encode each modality separately, then concat/cross-attend," achieving high scores when all modalities are present.

**Limitations of Prior Work**: (a) These methods allow the model to learn "independent bypass channels" for each modality—if one modality is missing, the entire prediction collapses; (b) Most dimensions in raw audio/video are noise, and without explicit compression, models struggle to capture the true emotional frequency bands; (c) Existing missing-modality evaluations use zero-masking, allowing the model to treat "zero" as a hidden token, **overestimating** robustness by 15–51% (as measured by the authors).

**Key Challenge**: The trade-off between "high performance with complete data" and "robustness to missing modalities"—the former encourages independent modality pathways, while the latter requires forcing all modalities to share a bottleneck.

**Goal**: (1) Achieve both noise and missing-modality robustness; (2) Provide intrinsic uncertainty; (3) Propose a stricter evaluation protocol (noise-masking instead of zero-masking).

**Key Insight**: Information bottleneck principle + frequency domain prior. Emotional signals are sparse in the frequency domain (audio: prosodic low-frequency, video: facial low spatial frequency), while noise is broadband and uniformly distributed—frequency domain transforms naturally perform lossy compression. Adding a "small number of learnable query tokens as a fixed-capacity cross-modal bottleneck" forces all modality information to be squeezed in, prohibiting modality shortcuts.

**Core Idea**: Treat "compression" as the unifying language—applied intra-modally (frequency domain), cross-modally (query bottleneck), and for missing modality reconstruction (energy function), with the final energy value "for free" as uncertainty.

## Method

### Overall Architecture
The pipeline has three stages: **Stage 1 (Intra-modal Frequency Compression)**: Audio uses learnable wavelets (DWT, 3 levels) for multi-scale time-frequency compression; video uses 2D DCT to split into 4 frequency bands with frequency-aware attention; text uses RoBERTa + projection (text is already a discrete compressed representation). **Stage 2 (Cross-modal Bottleneck)**: Concatenate the three modalities $H=[h_a;h_v;h_t]$, and $K=4$ learnable query $Q$ extract $Z=\mathrm{softmax}(QH^\top/\sqrt D)H \in \mathbb R^{K\times D}$ via cross-attn, forming a strong capacity constraint since $K\ll T_a+T_v+T_t$. **Stage 3 (Energy-based Reconstruction)**: When a modality is missing, the energy function $E_\theta(h_m;Z)$ is used for $T=3$ steps of momentum-based gradient descent to recover $h_m^*$, and the final energy $E_\theta(h_m^*)$ serves as uncertainty. An MLP prediction head outputs the sentiment score.

### Key Designs

1. **Learnable Frequency Compression (Stage 1)**:

    - **Function**: Remove broadband noise and retain sparse emotional frequency bands before cross-modal fusion.
    - **Mechanism**: Audio uses Daubechies-4 initialized learnable wavelet bases for 3-level DWT, yielding $\{c_L,d_L,\ldots,d_1\}$ multi-scale coefficients, fused via cross-scale attention: $h_a=\mathrm{Proj}(\mathrm{CrossScaleAttn}(W(x_a)))$; video uses 2D-DCT to split into 4 frequency bands with **learnable** boundaries, then applies frequency-aware attention: $h_v=\mathrm{Proj}(\mathrm{FreqAttn}(D(x_v)))$.
    - **Design Motivation**: Fixed BatchNorm/LayerNorm cannot "filter by frequency band," but wavelet/DCT are sparse orthogonal bases in the frequency domain; making their decomposition parameters learnable preserves priors while being task-adaptive. Table 4 shows removing frequency compression increases MAE by 3.7%, and by 12.3% in extreme missing scenarios.

2. **K=4 Query Tokens as Cross-modal Fixed-capacity Bottleneck (Stage 2)**:

    - **Function**: Block modality-specific shortcuts—disallow $\hat y = f_a(h_a)+f_v(h_v)+f_t(h_t)$ style independent channel summation, as prediction must be based on $Z$.
    - **Mechanism**: Six-layer fusion transformer, each layer performs (bottleneck ← modalities) cross-attn, bottleneck self-attn, FFN, with residual connections; $K=4$ is optimal between compression and expressiveness ($K=2$ loses information, $K=8$ approaches no bottleneck).
    - **Design Motivation**: Inspired by Perceiver, but DCER directly uses Perceiver's latent queries as modality-agnostic aggregation points; this ensures that "even with missing modalities, there are still $K$ tokens for downstream reasoning."

3. **Energy Function + Gradient-based Reconstruction + Built-in Uncertainty (Stage 3)**:

    - **Function**: Estimate $h_m^*$ when modality $m$ is missing; output a scalar energy value as confidence.
    - **Mechanism**: $E_\theta(h_m;Z)=f_\theta(h_m - \mathrm{CrossAttn}(h_m, Z)) + \lambda_E g_\theta(h_m)$, where $f_\theta,g_\theta$ are learned MLPs; initialized from $\mu_\theta(Z,h_\mathrm{obs})+\epsilon$, perform $T=3$ steps of momentum-based gradient descent $h_m^{(t)}\leftarrow h_m^{(t-1)} - (\rho v^{(t-1)}+\eta\nabla E_\theta)$. For complete modalities, $T=0$ suffices; for missing, $T=3$ yields significant improvement. Final $E_\theta(h_m^*)$ correlates with prediction error ($\rho>0.72$).
    - **Design Motivation**: Traditional VAE/GAN reconstruction is unstable in high dimensions, and explicit IB optimization is hard to train; the energy function reframes "reconstruction" as "a few gradient steps towards a learned low-energy basin," which is simple, stable, and naturally provides uncertainty—this is the core difference from VAE/GAN-based reconstruction.

### Loss & Training
Total loss $\mathcal L = \mathcal L_\mathrm{pred} + \alpha\mathcal L_\mathrm{recon} + \beta\mathcal L_\mathrm{energy} + \gamma\mathcal L_\mathrm{joint}$, with $\alpha=0.1,\beta=0.01,\gamma=0.05$; $\mathcal L_\mathrm{joint}=\|Z_\mathrm{full}-Z_\mathrm{recon}\|^2$ ensures the bottleneck computed from real and reconstructed modalities are consistent. Optimizer: AdamW lr=1e-5, batch 32, 40 epochs, averaged over 5 seeds.

## Key Experimental Results

### Main Results

| Dataset | Metric | DCER | Prev. SOTA | Gain |
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
| w/o Frequency Compression (linear projection instead of wavelet/DCT) | 0.694 | -3.7% (complete) / -12.3% (extreme missing) |
| $K=2$ tokens | Significant information loss | Decrease |
| $K=8$ tokens | Near no bottleneck, overfitting | Decrease |
| $T=0$ (no energy iteration) | Significant drop in missing scenarios | $T=3$ is the sweet spot |
| zero-mask eval vs noise-mask | Former overestimates by 15–51% | Indicates protocol issue |

### Key Findings
- Robustness shows a **U-shape**: Multimodal fusion is strongest with all modalities present and also under severe missing (>50%); in the middle, performance is close to unimodal, as losing one or two modalities still allows modality shortcuts to "fool" the model, while severe missing forces true cross-modal compressed representations.
- Energy value correlates with prediction error ($\rho>0.72$), providing free uncertainty—downstream tasks can directly use $E_\theta$ as a risk-sensitive rejection threshold.
- Acc-5 on CH-SIMS improves by +22.8%, indicating the cross-modal bottleneck maximizes discriminative power for "ambiguous emotions."

## Highlights & Insights
- Elevates "compression" as a unifying language: intra-modal (frequency domain), cross-modal (query bottleneck), and missing reconstruction (energy function) are all explained by the same "capacity-limited channel" principle—a rare and coherent narrative.
- Energy value naturally serves as uncertainty—eliminates the need for MC dropout or ensemble, and $\rho>0.72$ is sufficient for engineering use.
- Exposes and corrects the zero-mask evaluation flaw: prior SOTA missing-rate performance may be inflated by 15–51% due to models treating zeros as "I know nothing here" signals; advocating noise-mask evaluation benefits the entire community.

## Limitations & Future Work
- Frequency domain choices are manual priors (audio→wavelet, video→DCT, text→RoBERTa): for other modalities like IMU/physiological signals, manual selection is still required, lacking automation.
- The energy function itself is simple (two MLPs), not aligned with mainstream EBM training like SGLD/contrastive divergence; with many missing modalities, it may fall into local minima.
- Only validated on sentiment analysis, not on multimodal tasks requiring multi-step reasoning or long context (e.g., VideoQA); whether $K=4$ remains optimal for long sequences needs verification.
- "$\rho>0.72$ correlation" is not calibrated uncertainty, and has not been compared with conformal prediction/Platt scaling.

## Related Work & Insights
- **vs MulT (Tsai et al. 2019)**: MulT uses direct cross-modal attention without compression; DCER adds bottlenecks before and after cross-attn, preventing collapse when modalities are missing.
- **vs Perceiver (Jaegle et al. 2021)**: Also uses learnable queries as bottleneck, but Perceiver lacks intra-modal frequency compression and missing modality handling. DCER complements both.
- **vs EMT (Sun et al. 2023)**: EMT uses dual-level feature reconstruction for missing modalities with a deterministic decoder; DCER uses energy-based gradient descent and outputs uncertainty.
- **vs VAE/GAN-based missing modality reconstruction**: The latter is unstable in high dimensions; EBM sidesteps this by "learning only energy, not explicit generators," and the energy value itself is uncertainty.

## Rating
- Novelty: ⭐⭐⭐⭐ Frequency domain + bottleneck are not new, but integrating EBM for missing modalities and uncertainty is a rare combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three major datasets + 8 baselines + systematic ablation + missing protocol discussion, but no complex VideoQA.
- Writing Quality: ⭐⭐⭐⭐ Figure 1 clearly illustrates the entire story; formula count is reasonable.
- Value: ⭐⭐⭐⭐ Directly applicable to industrial multimodal systems—provides both SOTA and free uncertainty, and exposes zero-mask evaluation flaws, promoting evaluation reform.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](../../CVPR2026/multimodal_vlm/duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)
- [\[ICML 2026\] Breaking Dual Bottlenecks: Evolving Unified Multimodal Models into Self-Adaptive Interleaved Visual Reasoners](breaking_dual_bottlenecks_evolving_unified_multimodal_models_into_self-adaptive_.md)
- [\[ICML 2026\] Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)
- [\[CVPR 2026\] Unbiased Dynamic Multimodal Fusion](../../CVPR2026/multimodal_vlm/unbiased_dynamic_multimodal_fusion.md)
- [\[ICML 2026\] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)

</div>

<!-- RELATED:END -->
