---
title: >-
  [Paper Note] Scaling Laws of Global Weather Models
description: >-
  [ICML 2026][Earth Science][Weather Forecasting Models] This paper presents the first cross-model scaling law analysis of five mainstream data-driven weather models (Aurora, AIFS, Pangu, GraphCast, SFNO) under a unified training/evaluation protocol. It finds that weather models favor "width over depth," compute budgets should prioritize more training data over larger models, and scaling behaviors vary significantly across meteorological variables—distinct patterns from NLP/Vis…
tags:
  - "ICML 2026"
  - "Earth Science"
  - "Weather Forecasting Models"
  - "Scaling Laws"
  - "Compute-Optimal"
  - "Width-First"
  - "ERA5"
date: 2026-05-08
content_hash: 72b44e0896f5d2f5
---

# Scaling Laws of Global Weather Models

**Conference**: ICML 2026  
**arXiv**: [2602.22962](https://arxiv.org/abs/2602.22962)  
**Code**: https://github.com/spcl/scaling-laws-weather-model  
**Area**: Earth Science / Weather Forecasting / Scaling Laws  
**Keywords**: Weather Forecasting Models, Scaling Laws, Compute-Optimal, Width-First, ERA5

## TL;DR
This paper presents the first cross-model scaling law analysis of five mainstream data-driven weather models (Aurora, AIFS, Pangu, GraphCast, SFNO) under a unified training/evaluation protocol. It finds that weather models favor "width over depth," compute budgets should prioritize more training data over larger models, and scaling behaviors vary significantly across meteorological variables—distinct patterns from NLP/Vision scaling laws.

## Background & Motivation
**Background**: Data-driven neural weather forecasting (GraphCast, Pangu, Aurora, etc.) is rapidly approaching or even surpassing traditional Numerical Weather Prediction (NWP). With the availability of high-resolution reanalysis data (e.g., ERA5) and large-scale training infrastructure, these models are entering a stage of performance gains driven by scaling.

**Limitations of Prior Work**: While mature scaling laws (Kaplan, Chinchilla) exist in NLP and Vision to guide the allocation of compute between parameters and data, systematic cross-model scaling research in weather forecasting is nearly non-existent. Existing works focus only on single models or configurations and fail to answer which architectures scale more efficiently or whether laws are universal.

**Key Challenge**: Directly applying NLP/Vision scaling laws to weather models is unsafe. The atmosphere is a chaotic system with physical limits on predictability; weather models must simultaneously predict hundreds of correlated variables (temperature, wind speed, pressure, etc.) with diverse physical properties and difficulties. Moreover, how GNNs, Transformers, and Fourier operators depend on spatial resolution differs fundamentally, suggesting potentially divergent scaling behaviors.

**Goal**: To characterize how the validation loss $\mathcal{L}$ varies with model size $N$, data volume $D$, compute budget $C$, and model shape (width vs. depth) under strictly unified experimental conditions, and to answer how to allocate resources under a fixed compute budget.

**Key Insight**: Instead of inventing new models, the authors place five representative models into a "wind tunnel"—using the same ERA5 data, identical resolution, and a unified validation loss function. By systematically varying $N/D/C$/shape and fitting power-law coefficients using log-log linear regression, architectural differences are isolated from external noise.

**Core Idea**: Use "unified, standardized cross-model scaling experiments" to reveal scaling laws unique to weather forecasting, elevating scaling laws from "single-model empirical findings" to "domain-level design guidelines."

## Method

### Overall Architecture
This work is essentially a large-scale empirical study rather than a new method. It can be understood as an analysis pipeline: "Standardized Wind Tunnel → Multi-dimensional Scanning → Power-law Fitting → Extracting Principles." The inputs are ERA5 reanalysis data and five models; outputs are scaling law coefficients and four actionable design conclusions.

Specifically: All models perform 6-hour forecasts on ERA5 (1979–2020 training, 2021 validation, $0.25^\circ \times 0.25^\circ$ global grid, 6-hour intervals). "Non-architectural factors" are minimized by taking the intersection of input/target variables, unifying learning rate schedules, and using consistent weight initialization. Scanning is performed along $N$ (number of parameters), $D$ (cumulative training samples, defined by the total samples processed), $C$ (compute), and model shape. For each configuration, the power law for validation loss is fitted:

$$\mathcal{L}(D)=\alpha D^{-\beta},\quad \mathcal{L}(N)=\gamma N^{-\delta},\quad \mathcal{L}(C)=\lambda C^{-\epsilon}$$

Where larger $\beta$ and $\delta$ indicate faster loss reduction as data/parameters grow, suggesting better long-term potential. The experiments consumed over 430,000 GPU hours. The following "Key Designs" represent the core analysis dimensions and findings.

### Key Designs

**1. Unified Standardized Evaluation Protocol: Isolating Architecture from Noise**

The biggest trap in cross-model comparison is the inconsistency in default training loss, variable weighting, and data loading. Comparing losses directly would be "apples to oranges." The methodological cornerstone of this paper is forced alignment: all models use the same validation loss $\mathcal{L}$—a weighted average of the squared error $(\hat{x}-x)^2$ over spatial grids and atmospheric variables. Each variable is normalized by its standard deviation, each grid is weighted by its normalized area (handling the Earth's spherical geometry), and upper-air variables are additionally weighted by pressure levels (consistent with GraphCast, where higher pressure levels are prioritized). With unified resolution, learning rate schedules, and initialization, any observed performance differences can be credibly attributed to the architecture itself rather than optimization tricks. The authors also tested Maximal Update Parameterization ($\mu$P) to stabilize training across scales, finding it ineffective for AIFS, which underscores that the alignment protocol was tuned per model rather than blindly applied.

**2. Data/Parameter Dual Power-law Fitting: Using $\beta$ instead of $\alpha$ to Judge Potential**

For $D$, the paper fits $\mathcal{L}(D)=\alpha D^{-\beta}$ while fixing parameters. A key insight is that the intercept $\alpha$ is misleading because model rankings change as $D$ increases—for instance, GraphCast has the lowest loss at $D=30$ TB but is overtaken by Aurora at $D=100$ TB. The robust long-term indicator is the slope $\beta$. Results show Aurora’s $\beta \approx 0.51$ is significantly higher than other models ($0.30$–$0.46$), meaning it extracts information from more data most efficiently (a 10x increase in data reduces validation loss by up to 3.2x). For $N$, fitting $\mathcal{L}(N)=\gamma N^{-\delta}$ with fixed data shows stable power laws for all five models, with GraphCast exhibiting the best parameter efficiency (lowest loss at equivalent $N$). An interesting "threshold effect" was found in Pangu: at 15 TB, $\delta$ is low (adding parameters is ineffective), but at 30 TB, $\delta$ increases, suggesting parameter scaling is bottlenecked by data volume.

**3. Width Over Depth: Weather Forecasting Relies on Capacity Rather Than Deep Non-linearity**

This is the sharpest divergence from NLP scaling laws. While Kaplan et al. found that language model performance is nearly independent of shape at fixed $N$, this study compared "wide" vs. "narrow" configurations for each model under a fixed parameter budget. **All models consistently preferred the wide configuration.** Most extremely, GraphCast and SFNO performed well even with a depth of 1. The authors explain this through Geometric Deep Learning (linear graph models can rival deep ones) and Neural Operator theory (single-layer spectral/attention can approximate complex global optimization steps). Conclusion: 6-hour short-range weather dynamics can be approximated by near-linear models; stacking deep non-linear layers is redundant. Designing weather models should prioritize width over depth for a fixed parameter budget.

**4. Compute-Optimal Allocation: Prioritize Data Over Parameters Given Fixed Compute**

When compute $C$ is fixed, fitting validation loss against $D$ results in a parabola (Chinchilla-style approach). The minimum of the parabola indicates the optimal ratio of $N$ to $D$, formalized as $N_{\text{opt}} \propto C^{a}$ and $D_{\text{opt}} \propto C^{b}$. A nuance is that the NLP constraint $a+b=1$ does not universally hold for weather models: GraphCast/AIFS backbones run on latent graphs independent of input resolution, violating the standard $C \approx 6ND$ assumption (resolution is fixed here, so $C \sim N \cdot B$). Conversely, the patch mechanism in Transformer-like models scales the effective $D$ by $1/p^2$, maintaining $a+b=1$ only when $C \approx 6ND/p^2$. Despite different ratios, the parabolic behavior is universal, and the trend is consistent: as compute increases, **prioritizing longer training (more data) reduces forecast error more effectively than increasing model size.** It is better to train a smaller model for longer than to under-train a larger one.

## Key Experimental Results

### Main Results
Evaluation of five models on ERA5, comparing data scaling coefficient $\beta$ and hardware efficiency:

| Model | Parameters | Backbone | $\beta \uparrow$ (Data) | Note |
| :--- | :--- | :--- | :--- | :--- |
| Aurora | 1.3B | Swin Transformer | 0.51 | Best data scaling efficiency |
| AIFS | 255M | Graph Transformer | 0.46 | Spherical graph, reduced distortion |
| Pangu | 276M | Swin Transformer | 0.43 | Threshold effect in parameter scaling |
| GraphCast | 36.7M | GNN | 0.36 | Best parameter efficiency, low hardware use |
| SFNO | 433M | Spherical Fourier Op | 0.34 | Good performance even at Depth=1 |

### Hardware Efficiency Analysis
Scaling laws often treat compute as a static resource, ignoring wall-clock time. This study found that "good loss scaling" does not equate to "good compute utilization":

| Model | Tflop/s | GPU Peak (Tflop/s) | GPU Utilization (%) |
| :--- | :--- | :--- | :--- |
| Aurora | 368 | 989 (32-bit) | 37.2 |
| AIFS | 33.7 | 1979 (16-bit) | 1.70 |
| GraphCast | 10.15 | 989 (32-bit) | 1.03 |
| Pangu | 3.25 | 989 (32-bit) | 0.33 |
| SFNO | 0.215 | 989 (32-bit) | 0.022 |

### Key Findings
- **Data Scaling Champion $\neq$ Parameter Scaling Champion**: Aurora has the strongest data scaling ($\beta \approx 0.51$), but GraphCast wins in parameter efficiency (lowest loss at same $N$); these strengths are complementary and mutually exclusive.
- **Theoretical Efficiency $\neq$ Real-world Efficiency**: GraphCast is parameter-efficient, but its GNN message passing is memory-bound, reaching only 1.03% utilization on H100 (single precision), while Aurora reaches 37.2%—a $\sim$36x difference. Engineering implementation is critical for deployment.
- **Variable Heterogeneity**: Scaling behaviors differ significantly across variables. Aurora consistently has the lowest RMSE for 10m u-wind (10U), but GraphCast outperforms all models for 2m temperature (2T). Aggregate weighted loss is only a rough proxy; one must examine variable-level scaling.

## Highlights & Insights
- **The "Wind Tunnel" protocol is the major methodological contribution**: Cross-model comparisons are usually untrustworthy due to inconsistent losses. By isolating architectural factors through a unified protocol, this work provides a template for multi-model benchmarking in any field.
- **Using slope $\beta$ rather than intercept $\alpha$ to judge long-term potential**: This insight clarifies that performance rankings at specific data volumes can flip. Scaling speed is what matters, which is particularly valuable for teams that must extrapolate from small-scale experiments.
- **"Width > Depth" overturns NLP intuition**: Given a fixed parameter budget, widening is more effective than deepening (even Depth=1 is viable). This conclusion directly affects the architectural design of next-generation weather models.
- **Integrating hardware utilization into scaling discussions**: The paper highlights that ignoring wall-clock time and utilization is a blind spot. A 36x gap in utilization proves that "beautiful scaling curves" and "fast training" are distinct objectives.

## Limitations & Future Work
- **Compute/Memory constraints limited the search space**: The maximum explored width was 512, and Aurora's minimum width was constrained by its attention head size (64). This means only one side of the compute-optimal parabola was observable for some models; extrapolation remains uncertain.
- **Focus on 6-hour short-term forecasts**: The "Depth=1 is sufficient" conclusion likely only applies to near-linear 6-hour dynamics. Whether long-lead times or auto-regressive rollouts still favor wide-and-shallow structures is unverified.
- **$\mu$P and cross-scale stabilization are not universal**: The failure of $\mu$P on AIFS suggests these scaling conclusions partly depend on model-specific hyperparameter tuning, requiring caution for reproducibility.
- **Future Directions**: Incorporating variable-level hybrid modeling, adaptive tuning, and formally integrating wall-clock/utilization into the compute-optimal framework to make scaling laws more practical for deployment.

## Related Work & Insights
- **vs. Kaplan / Chinchilla (NLP Scaling Laws)**: They established compute-optimal laws like $N \propto C^{0.73}$ and $D \propto C^{0.27}$ with shape-independent performance. This paper finds $a+b=1$ is not universal for weather models and shows a strong preference for width, proving scaling laws are domain-specific.
- **vs. Single-model Weather Scaling (e.g., Nguyen et al., Couairon et al.)**: Prior works focused on single models/configs. This work is the first to perform a unified evaluation across five models, elevating scaling from "case-by-case experience" to "architectural selection guidelines."
- **vs. Compute Assumptions in Classical Scaling Laws**: Classical works treat compute as static. This paper uses the 36x utilization gap between Aurora and GraphCast to highlight the blind spot of ignoring engineering efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ First cross-model weather scaling analysis; "width over depth" is a counter-intuitive domain-level discovery.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × multi-dimensional scans, 430,000 GPU hours, unified protocol.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and rich visualizations, though some parabolic curves are incomplete.
- Value: ⭐⭐⭐⭐⭐ Provides actionable guidelines for next-generation weather model architecture and resource allocation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TianQuan-S2S: Constructing Subseasonal-to-Seasonal Global Weather Forecasting Models by Incorporating Climatology](../../ICLR2026/earth_science/tianquan-s2s_a_subseasonal-to-seasonal_global_weather_model_via_incorporate_clim.md)
- [\[ICML 2026\] (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models](sparse_attention_to_the_details_preserving_spectral_fidelity_in_ml-based_weather.md)
- [\[ICLR 2026\] Task-Adaptive Parameter-Efficient Fine-Tuning for Weather Foundation Models](../../ICLR2026/earth_science/task-adaptive_parameter-efficient_fine-tuning_for_weather_foundation_models.md)
- [\[CVPR 2026\] PhyOceanCast: Global Ocean Forecasting with Physics-Informed Diffusion](../../CVPR2026/earth_science/phyoceancast_global_ocean_forecasting_with_physics-informed_diffusion.md)
- [\[ICLR 2026\] RainPro-8: An Efficient Deep Learning Model to Estimate Rainfall Probabilities Over 8 Hours](../../ICLR2026/earth_science/rainpro-8_an_efficient_deep_learning_model_to_estimate_rainfall_probabilities_ov.md)

</div>

<!-- RELATED:END -->
