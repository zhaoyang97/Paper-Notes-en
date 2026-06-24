---
title: >-
  [Paper Note] Detection and Simulation of Urban Heat Islands Using a Fine-Tuned Geospatial Foundation Model
description: >-
  [NeurIPS 2025][Image Generation][Urban Heat Island] This paper proposes a unified three-stage workflow based on a fine-tuned geospatial foundation model (Granite-GFM): first establishing an empirical baseline via green space cooling effects to verify physical plausibility; then extrapolating urban temperatures under future climate scenarios; and finally simulating the cooling impact of greening interventions via inpainting. This elevates the foundation model from an evaluatio…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Urban Heat Island"
  - "Geospatial Foundation Model"
  - "LST Prediction"
  - "Climate Extrapolation"
  - "Inpainting Simulation"
date: 2026-05-08
content_hash: 1e2693bc945d0fdc
---

# Detection and Simulation of Urban Heat Islands Using a Fine-Tuned Geospatial Foundation Model

**Conference**: NeurIPS 2025
**arXiv**: [2510.18773](https://arxiv.org/abs/2510.18773)  
**Code**: Based on the Granite-Geospatial-LST model (available on HuggingFace)  
**Area**: Geospatial / Climate Science
**Keywords**: Urban Heat Island, Geospatial Foundation Model, LST Prediction, Climate Extrapolation, Inpainting Simulation

## TL;DR

This paper proposes a unified three-stage workflow based on a fine-tuned geospatial foundation model (Granite-GFM): first establishing an empirical baseline via green space cooling effects to verify physical plausibility; then extrapolating urban temperatures under future climate scenarios; and finally simulating the cooling impact of greening interventions via inpainting. This elevates the foundation model from an evaluation tool to an interactive simulation platform for urban planning.

## Background & Motivation

**Background**: The urban heat island (UHI) effect intensifies with urbanization and climate change, with urban cores reaching temperatures more than 5°C above surrounding areas, leading to increased energy consumption, heat-related illness, and degraded air quality. With over 70% of the global population projected to live in cities by 2050, UHI mitigation is increasingly urgent.

**Limitations of Prior Work**: Effective UHI mitigation requires high-resolution, high-frequency temperature data, yet current approaches face multiple bottlenecks: (1) sparse meteorological station networks and long satellite revisit intervals result in insufficient data coverage; (2) physics-based models (e.g., WRF, UrbClim) demand extensive input data and computational resources; (3) conventional ML models rely on large annotated datasets and generalize poorly across regions—particularly problematic for data-scarce developing areas.

**Key Challenge**: Urban planners require four types of information—current heat island distribution, the spatial extent of green space cooling, the intensity of future extreme heat events, and the expected impact of interventions—yet these are typically provided by separate toolchains with no unified framework. Moreover, existing work rarely verifies whether models genuinely capture physical mechanisms (e.g., green space cooling), reporting only error metrics.

**Goal**: How can a unified geospatial foundation model workflow simultaneously accomplish UHI detection and quantification, future prediction, and intervention simulation, while verifying the physical plausibility of the model?

**Key Insight**: Geospatial foundation models (GFMs) pre-trained on globally unstructured remote sensing data exhibit strong generalization and require only minimal fine-tuning to adapt to new cities. IBM's Granite-GFM has already demonstrated cross-regional LST prediction capability; this paper builds a complete three-stage workflow upon that foundation.

**Core Idea**: A single fine-tuned geospatial foundation model is used to chain together three stages—empirical validation → climate extrapolation → intervention simulation—upgrading UHI analysis from an assessment tool to a planning simulation platform.

## Method

### Overall Architecture

The three-stage workflow proceeds sequentially: (1) overlaying LULC data with high-resolution LST imagery to analyze both internal and spillover cooling effects of green spaces, establishing a physical baseline and validating the model; (2) sorting training data by temperature for a held-out city, fine-tuning on the cooler 90% and testing extrapolation on the hottest 10%, then substituting climate projection inputs for future forecasting; (3) replacing built-up pixel values with green space equivalents via inpainting to simulate intervention effects.

### Key Designs

1. **Stage 1: Empirical Cooling Baseline Construction and Model Validation**

    - **Function**: Quantify the actual cooling effects of green spaces and verify whether the model correctly captures these physical mechanisms.
    - **Mechanism**: Impact Observatory 10 m land-use classifications are overlaid with HLS L30 30 m multispectral imagery to extract park areas. The cooling anomaly is defined as $\Delta T$ = temperature at target point − baseline temperature of surrounding built-up area. Two effects are analyzed as a function of distance from park boundaries: internal cooling (the temperature gradient from park edge to center) and spillover cooling (the diffusion of cooling from parks into surrounding built-up areas). Data from 12 European cities are aggregated over summer daytime periods from 2017 to 2025.
    - **Design Motivation**: Conventional evaluation reports only MAE/RMSE metrics, which cannot determine whether a model truly understands physical processes. By comparing model predictions against empirical cooling curves, this work assesses "physical plausibility" rather than merely "numerical accuracy."

2. **Stage 2: Extreme Climate Extrapolation and Future Prediction**

    - **Function**: Assess the model's ability to extrapolate to unseen extreme heat conditions and predict future climate scenarios.
    - **Mechanism**: Brașov, Romania—a city outside the training set—is selected. Data are sorted by temperature; the cooler 90% are used for fine-tuning and the hottest 10% are held out as the test set. Extrapolation accuracy is evaluated on this 10% to establish the model's reliable extrapolation range. EURO-CORDEX climate projection data then replace ERA5 inputs to forecast UHI patterns in 2030, 2050, and 2100 under RCP 2.6, 4.5, and 8.5 emission scenarios.
    - **Design Motivation**: The core challenge in climate prediction is extrapolation—models must forecast extreme conditions outside the training distribution. By first quantifying extrapolation limits on known data (MAE = 1.74°C), the approach provides a principled basis for confidence when applying the model to future scenarios.

3. **Stage 3: Urban Greening Intervention Simulation (Inpainting)**

    - **Function**: Simulate temperature changes resulting from the addition of green space in specific urban areas.
    - **Mechanism**: Built-up pixel values in satellite imagery are replaced with pixels representative of green space, with corresponding adjustments to input spectral indices (e.g., increased NDVI), and the model predicts LST for the modified scene. Cooling effects are quantified by comparing pre- and post-intervention predictions.
    - **Design Motivation**: Inpainting is transferred from image generation to climate simulation—not to produce visually realistic images, but to predict changes in physical quantities. This transforms the model from a passive assessment tool into an active "what-if" simulation platform.

### Model Architecture and Training

Granite-GFM is based on the SWIN Transformer architecture, built upon the Prithvi-SWIN-L Earth observation foundation model. Two versions exist: V1 (fine-tuned on 28 cities) and V2 (52 cities, covering a broader range of hydroclimatic zones). Inputs consist of HLS multispectral bands and ERA5 2 m air temperature statistics. Outputs are land surface temperature (LST) at 30 m spatial resolution and hourly temporal resolution.

## Key Experimental Results

### Main Results

Evaluation is conducted on 13 European cities outside the training set. The empirical baseline shows average urban core UHI intensity of +3.3°C, internal park cooling of up to −2.6°C, and spillover cooling reaching −3.5°C near park boundaries and decaying to −1°C at 150 m.

| Experiment | Metric | V1 | V2 | Gain |
|---|---|---|---|---|
| Internal Cooling | MAE | 0.240°C | 0.231°C | −3.8% |
| Spillover Cooling | MAE | 0.302°C | **0.199°C** | **−34%** |
| Spillover Cooling | RMSE | 0.339°C | 0.243°C | −28% |
| Extreme Heat Extrapolation | MAE | 1.89°C | **1.74°C** | −7.9% |

V2 reduces MAE for spillover cooling by 34%—this more complex spatial interaction effect benefits substantially from more diverse training data.

### Ablation Study

| Configuration | Spillover Cooling MAE | Extrapolation MAE | Notes |
|---|---|---|---|
| V1 (28 cities) | 0.302°C | 1.89°C | Baseline |
| V2 (52 cities) | **0.199°C** | **1.74°C** | Data diversity is key |
| V1 high-temperature subset | — | 1.89°C | Limited extrapolation capacity |
| V2 high-temperature subset | — | 1.74°C | Broader coverage improves extrapolation |

### Key Findings

- **Data diversity > model scale**: The key difference between V1 and V2 is the expansion of training cities from 28 to 52, with no architectural change. The 34% improvement in spillover cooling indicates that the bottleneck for foundation models lies in data coverage rather than model capacity.
- **Internal vs. spillover cooling**: The two versions show little difference on internal cooling (a relatively simple temperature gradient), whereas spillover cooling involves complex interactions with urban morphology and requires more diverse training data to model accurately.
- **1.74°C extrapolation accuracy**: For planning-level decisions (e.g., determining whether a given area requires greening intervention), an error of approximately 2°C is within an acceptable range, though it remains insufficient for precise engineering design.
- **Physical consistency of inpainting simulation**: The model correctly predicts internal and spillover cooling for greened areas; the resulting temperature profiles are qualitatively consistent with the empirical baseline, confirming that the model has learned the physical mechanisms underlying green space cooling.

## Highlights & Insights

- **Unified three-in-one workflow**: Detection, prediction, and simulation are accomplished within a single model and framework, eliminating data conversion and inconsistencies across multiple toolchains. This paradigm of upgrading an "assessment tool" to a "simulation platform" is generalizable to other Earth science problems.
- **Physical plausibility validation methodology**: Rather than reporting only MAE/RMSE, the work validates whether the model understands physical processes through the matching of cooling effect curves. This validation paradigm is more convincing than conventional numerical error assessment and merits adoption in other AI for Earth science work.
- **Cross-domain transfer of inpainting**: Inpainting techniques from image generation are repurposed for climate "what-if" analysis—not to generate realistic images, but to predict changes in physical quantities. This idea of transferring generative model capabilities to scientific simulation is broadly inspiring.
- **Value of foundation models in data-scarce regions**: GFMs generalize to out-of-training cities with minimal fine-tuning, making them particularly suitable for developing regions with limited meteorological infrastructure—a capability beyond the reach of conventional ML approaches.

## Limitations & Future Work

- **Spatial resolution constraints**: 30 m resolution remains coarse for fine-grained urban microclimates (e.g., street canyon effects, building shadows). Higher resolution requires finer satellite data and greater model capacity.
- **LST vs. perceived temperature**: The model predicts land surface temperature (LST) rather than the air temperature experienced by humans. The two can differ substantially in areas with shade or wind.
- **Idealized inpainting assumptions**: Directly replacing built-up pixels with green space represents an extreme assumption that does not account for gradual greening processes, varying vegetation types, or maintenance costs.
- **Geographic scope**: Validation is limited to European cities. Results may differ for tropical cities (higher humidity, different vegetation types) and arid cities (water limitations on greening effectiveness).
- **Summer daytime data only**: UHI effects may be more severe at night (due to building thermal mass release), and UHI also occurs in winter—neither case is addressed by the current workflow.

## Related Work & Insights

- **vs. physics-based models (WRF, UrbClim)**: GFMs require neither extensive input data nor specialized expertise and computational resources. The trade-off is the absence of explicit physical process modeling—partially compensated by the three-stage validation procedure. Best suited for rapid deployment to new cities in planning assessment contexts.
- **vs. single-region ML methods**: Conventional ML models are trained separately for each city and require local annotated data. The transfer learning advantage of GFMs enables deployment in any city, particularly in data-scarce regions, though global models may sacrifice accuracy relative to locally trained models in specific cities.
- **vs. pure remote sensing LST retrieval**: This work extends LST retrieval with extrapolation and intervention simulation capabilities, advancing from perception to prediction and planning—a qualitative leap from "observing the past" to "forecasting the future and simulating interventions."

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified three-stage workflow and the application of inpainting to UHI analysis are novel, though the technical methods primarily involve fine-tuning existing models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validation across 13 cities, V1 vs. V2 comparison, and physical plausibility verification are thorough, though geographic coverage is limited to Europe.
- **Writing Quality**: ⭐⭐⭐⭐ The workflow is clearly presented, experiments are intuitively visualized, and motivation is compellingly argued.
- **Value**: ⭐⭐⭐⭐ Offers practical applicability to climate research and urban planning, and demonstrates the potential of foundation models in scientific applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](../../CVPR2025/image_generation/fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[ECCV 2024\] Beta-Tuned Timestep Diffusion Model](../../ECCV2024/image_generation/beta-tuned_timestep_diffusion_model.md)
- [\[CVPR 2025\] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](../../CVPR2025/image_generation/fade_fine_grained_erasure_diffusion.md)
- [\[NeurIPS 2025\] Epistemic Uncertainty for Generated Image Detection](epistemic_uncertainty_for_generated_image_detection.md)
- [\[ICML 2025\] GRAM: A Generative Foundation Reward Model for Reward Generalization](../../ICML2025/image_generation/gram_a_generative_foundation_reward_model_for_reward_generalization.md)

</div>

<!-- RELATED:END -->
