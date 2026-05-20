---
title: >-
  [Paper Note] STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting
description: >-
  [CVPR 2026][Time Series][weather forecasting] This paper proposes STCast, a framework that replaces static boundary cropping with learnable global-regional distributions via Spatial-Aligned Attention (SAA) to adaptively…
tags:
  - "CVPR 2026"
  - "Time Series"
  - "weather forecasting"
  - "spatial-aligned attention"
  - "temporal MoE"
  - "global-regional coupling"
  - "adaptive boundary"
date: 2026-05-08
content_hash: 2ca65219bb3e1b58
---

# STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting

**Conference**: CVPR 2026
**arXiv**: [2509.25210](https://arxiv.org/abs/2509.25210)  
**Code**: N/A  
**Area**: Spatiotemporal Forecasting / Weather Prediction
**Keywords**: weather forecasting, spatial-aligned attention, temporal MoE, global-regional coupling, adaptive boundary

## TL;DR

This paper proposes STCast, a framework that replaces static boundary cropping with learnable global-regional distributions via Spatial-Aligned Attention (SAA) to adaptively integrate global atmospheric information into regional forecasting, and employs Temporal Mixture-of-Experts (TMoE) with month-conditioned dynamic routing to enhance temporal modeling. STCast achieves state-of-the-art performance across four tasks: global forecasting, high-resolution regional forecasting, typhoon track prediction, and ensemble forecasting.

## Background & Motivation

**Accurate regional weather forecasting** requires global atmospheric dynamics as context — Siberian cold surges can trigger East Asian cold waves, and Tibetan Plateau surface heating simultaneously influences the East Asian monsoon and the North American jet stream. The true "boundary" for regional forecasting is therefore not the neighboring region but the entire globe.

**Limitations of Prior Work**: Traditional NWP methods solve PDEs on fine grids at prohibitive computational cost. Data-driven methods (e.g., Pangu-Weather, GraphCast) substantially reduce this cost but face two core challenges: (1) directly training high-resolution (~1 km, 0.01°) global models is computationally infeasible (requiring a 19980×39960 grid); (2) existing global-regional coupling methods rely solely on **static neighboring regions** as boundaries — OneForecast, for instance, directly concatenates neighborhood crops, which contradicts the theory of atmosphere-ocean-land-biosphere coupling.

**Core Innovations of STCast**: (1) replacing static neighborhood boundaries with learnable global-regional distributions, initialized via great-circle distance and adaptively optimized during training; (2) employing month-specific Gaussian priors to guide MoE expert routing, explicitly modeling atmospheric variability across different months.

## Method

### Overall Architecture

STCast unifies four tasks: low-resolution global forecasting → high-resolution regional forecasting (via SAA for global information fusion) → typhoon track prediction (from regional sea-level pressure predictions) → ensemble forecasting (via Perlin noise injection over multiple simulations). The backbone follows an Encoder-Processor-Decoder architecture, with the Processor alternating between window attention and self-attention.

### Key Designs

1. **Spatial-Aligned Attention (SAA)**:

    - **Function**: Adaptively aggregates global atmospheric information into regional forecasting, replacing static boundary concatenation.
    - **Mechanism**: Employs linear cross-attention with global features as Query and Key, and regional features as Value. The key innovation lies in prior initialization — the great-circle distance $d(\phi,\lambda)$ from each global grid point to the target region is computed, and an exponential decay function $f(\phi,\lambda) = \exp(-\alpha \cdot d^2)$ is used to initialize the global-regional distribution. This prior modulates attention weights via Hadamard product and is continuously optimized during training. $O(n)$ linear attention reduces computational complexity.
    - **Design Motivation**: Atmospheric influence decays with distance but does not vanish. The exponential decay prior encodes this physical intuition while allowing the model to learn long-range teleconnections beyond geographic proximity.

2. **Temporal Mixture-of-Experts (TMoE)**:

    - **Function**: Dynamically routes forecasting tasks across different months to specialized expert models.
    - **Mechanism**: A discrete Gaussian distribution is learned for each month, with its peak rotationally aligned to the month of the input variables. The month embedding $M \in \mathbb{R}^{12\times 1}$ is concatenated with the routing weights derived from input features, and softmax selects the Top-K experts: $I = \text{Softmax}(\text{Conv}(X^t) + M)$. The Gaussian distribution ensures that activation probability decreases monotonically with temporal distance.
    - **Design Motivation**: Atmospheric variables exhibit significant month-to-month variability (e.g., summer convection vs. winter radiation). Implicit MoE assignment struggles to capture such temporal specificity and is prone to expert homogenization. TMoE provides an explicit expert specialization signal via month embeddings.

3. **Unified Four-Task Framework**:

    - **Function**: Handles global forecasting, regional forecasting, typhoon track prediction, and ensemble forecasting within a single framework.
    - **Mechanism**: Global forecasting: $X_g^{t+1} = \Phi_g(X_g^t)$; regional forecasting via SAA coupling: $X_r^{t+1} = \Phi_r(X_r^t, X_g^t)$; typhoon tracks are inferred from regional sea-level pressure predictions; ensemble forecasting injects Perlin noise into the global initial state and averages $n$ simulations.
    - **Design Motivation**: All four tasks share underlying atmospheric physical representations; a unified framework better exploits global-regional-temporal coupling.

### Loss & Training

AdamW optimizer, learning rate 0.0002, 100 epochs, batch size 16. Training is conducted on the ERA5 dataset (1979–2019, 0.25° resolution, 721×1440, 70 variables). Global and regional models are trained separately. Hardware: 16× NVIDIA A100 GPUs.

## Key Experimental Results

### Main Results

**Global Forecasting (ERA5, Normalized RMSE↓ / ACC↑)**:

| Method | 1-day RMSE↓ | 4-day RMSE↓ | 7-day RMSE↓ | 10-day RMSE↓ |
|--------|------------|------------|------------|-------------|
| Pangu-Weather | 0.1571 | 0.3380 | 0.5092 | 0.6215 |
| GraphCast | 0.1304 | 0.2861 | 0.4597 | 0.6009 |
| OneForecast | 0.1231 | 0.2732 | 0.4468 | 0.5918 |
| **STCast** | **0.1197** | **0.2578** | **0.4348** | **0.5763** |

### Ablation Study

| Configuration | Key Impact | Remarks |
|---------------|-----------|---------|
| w/o SAA (direct neighborhood concat) | Significant degradation in regional forecasting | Static boundaries are insufficient |
| w/o TMoE (standard MoE) | Weakened temporal generalization | Month-specificity is lost |
| Distance decay prior fixed (non-learnable) | Moderate performance | Learnable prior is superior |
| Euclidean distance instead of great-circle | Marginally worse | Spherical distance is more accurate |

### Key Findings

- SAA's adaptive boundaries outperform OneForecast's neighborhood concatenation and direct training strategies across all regional forecasting variables.
- TMoE's month embeddings effectively prevent expert homogenization, with different experts learning distinct atmospheric patterns for different months.
- STCast shows greater advantages in 10-day long-range forecasting (RMSE 0.5763 vs. 0.5918), indicating that global-regional coupling is especially beneficial for extended-range prediction.
- Typhoon track prediction and ensemble forecasting further demonstrate the generalizability of the unified framework.

## Highlights & Insights

- The prior design combining great-circle distance with exponential decay initialization is physically well-motivated — it encodes the fundamental principle that "atmospheric influence decays with distance" while leaving room for the model to learn teleconnection patterns that deviate from pure distance decay. This paradigm of "physics-informed prior initialization + data-driven optimization" holds strong reuse potential for other Earth science applications.
- TMoE uses discrete Gaussian distributions for month-conditioned routing — providing explicit month information is a low-cost, high-return design choice compared to expecting the model to implicitly learn temporal patterns.

## Limitations & Future Work

- Regional forecasting is currently validated only over East Asia; generalization to other geographic regions (e.g., equatorial or polar regions) remains to be examined.
- Uncertainty quantification relies solely on Perlin noise-based ensemble forecasting, with no probabilistic calibration analysis provided.
- Computational cost remains high (16× A100s, 100 epochs), posing challenges for resource-constrained research groups.

## Related Work & Insights

- **vs. OneForecast**: The most direct competitor. OneForecast performs global-regional coupling via neighborhood concatenation, whereas STCast learns adaptive distributions via SAA, outperforming OneForecast across all four tasks.
- **vs. GraphCast/FuXi**: These methods focus on global forecasting. STCast extends to high-resolution regional forecasting via SAA, bridging a critical gap in the global → regional pipeline of AI-based weather prediction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — SAA's distance-decay prior and TMoE's month embedding are both well grounded in physical motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four tasks × multiple baselines with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐ — Framework description is clear, but some implementation details are relegated to the appendix.
- **Value**: ⭐⭐⭐⭐ — Achieves comprehensive state-of-the-art across four unified tasks; highly valuable to the AI weather forecasting community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting](../../ICCV2025/time_series/va-moe_variables-adaptive_mixture_of_experts_for_incremental_weather_forecasting.md)
- [\[CVPR 2026\] L2GTX: From Local to Global Time Series Explanations](l2gtx_from_local_to_global_time_series_explanations.md)
- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[ICLR 2026\] VoT: Event-Driven Reasoning and Multi-Level Alignment Unlock the Value of Text for Time Series Forecasting](../../ICLR2026/time_series/unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)

</div>

<!-- RELATED:END -->
