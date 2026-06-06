---
title: >-
  [Paper Note] Any2Any: Unified Arbitrary Modality Translation for Remote Sensing
description: >-
  [ICML2026][Remote Sensing][Any-to-any translation] Any2Any transforms remote sensing (RS) translation between sensors like RGB, SAR, NIR, MS…
tags:
  - "ICML2026"
  - "Remote Sensing"
  - "Any-to-any translation"
  - "Remote sensing multi-modality"
  - "Latent diffusion"
  - "RST-1M"
  - "Residual adapter"
date: 2026-05-08
content_hash: 16c103017b602c8d
---

# Any2Any: Unified Arbitrary Modality Translation for Remote Sensing

**Conference**: ICML2026  
**arXiv**: [2603.04114](https://arxiv.org/abs/2603.04114)  
**Code**: https://github.com/MiliLab/Any2Any (Available)  
**Area**: Remote Sensing / Multimodal Generation  
**Keywords**: Any-to-any translation, Remote sensing multi-modality, Latent diffusion, RST-1M, Residual adapter  

## TL;DR
Any2Any transforms remote sensing (RS) translation between sensors like RGB, SAR, NIR, MS, and PAN from a collection of paired models into a unified latent diffusion model within a shared latent space. Leveraging the million-scale RST-1M dataset and target-modality residual adapters, the model achieves superior fidelity and generalization across 14 seen translation directions and multiple unseen modality combinations.

## Background & Motivation
**Background**: Remote sensing scenarios increasingly rely on multi-source sensor synergy. A single geographic area can be observed simultaneously via optical RGB, Synthetic Aperture Radar (SAR), Near-Infrared (NIR), Multi-Spectral (MS), and Pansharpening (PAN) modalities, providing texture, structure, spectral, and all-weather information, respectively. In practice, certain modalities are often unavailable, making the synthesis of missing modalities from existing ones a critical problem for continuous earth observation and downstream analysis.

**Limitations of Prior Work**: Existing cross-modal translation methods in RS are typically trained for specific directions (e.g., SAR-to-RGB, NIR-to-RGB). This approach requires $O(N^2)$ models to support all inter-modality translations as the number of sensors increases. Furthermore, models trained on specific directions only learn from local data, failing to reuse semantic information from other modality pairs or generalize to combinations without paired training samples.

**Key Challenge**: RS modalities share the same geographic scene but are constrained by diverse physical imaging mechanisms. Differences in resolution, channel counts, sampling geometry, and noise characteristics across RGB, SAR, NIR, MS, and PAN mean that a fully shared model leads to distribution misalignment, while fully separated models lose cross-modal semantic sharing and scalability. The paper seeks an extensible compromise between "unified semantic mapping" and "preserving sensor differences."

**Goal**: The authors define the task as Any-to-Any RS modality translation, where the model performs translation for any given source and target modality within a single framework. This requires solving three sub-problems: constructing a large-scale supervised cross-modal graph, projecting diverse sensors into a comparable latent space, and learning semantic mapping with a shared backbone while correcting target-specific systematic biases.

**Key Insight**: Despite different observational forms, sensors point to the same geographic semantic scene. By using spatially registered samples as "latent space anchors," cross-modal generation can be transformed from unstable marginal distribution matching into supervised regression on target latents. This is particularly suitable for RS, which emphasizes geographic alignment and physical scale consistency.

**Core Idea**: Establish a unified latent space using modality-specific VAEs, use a shared DiT backbone conditioned on source/target modalities to predict target latents, and apply target-modality residual adapters for fine calibration.

## Method
Any2Any utilizes a three-stage framework comprising representation alignment, shared translation, and target-specific refinement. Instead of learning generators in pixel space, each modality has its own encoder/decoder to project images of varying resolutions and channels into a unified latent representation. A shared Diffusion Transformer (DiT) learns the semantic mapping in this latent space, followed by target-indexed residual adapters to refine the output.

### Overall Architecture
The input consists of a source RS image $x_i$ and a target modality identifier $j$. The source encoder $E_i$ produces source latent $z_i$. During training, the target image $x_j$ is passed through $E_j$ to obtain $z_j$, which serves as a supervised anchor. The diffusion backbone receives the concatenated noisy target latent $z_t$ and source latent $z_i$. The DiT AdaLN layers are modulated by timesteps, source modality, and target modality embeddings to predict the clean target latent $\hat{z}_j$. Finally, the target adapter $A_j$ performs residual calibration on $\hat{z}_j$ before the target decoder $D_j$ reconstructs the image.

### Key Designs
1. **RST-1M Supported Latent Anchors**:
	- **Function**: Converts cross-modal translation from ambiguous generation into spatially-registered target latent regression.
	- **Mechanism**: RST-1M aggregates public datasets like SEN1-2 and SpaceNet-5, covering five modalities and seven cross-modal pairings. Target images $x_j$ are treated as deterministic anchors $z_j = E_j(x_j)$ in the target distribution. The objective is to force the predicted results to match the latent representation of the specific geographic scene.
	- **Design Motivation**: RS translation requires maintaining consistency in geographic structures (boundaries, roads, water). Anchor supervision reduces mapping stochasticity and allows the backbone to learn stable geographic semantics.

2. **Modality-Specific VAE + Shared DiT Semantic Mapping**:
	- **Function**: Standardizes diverse sensor inputs into unified latent dimensions and processes all translation directions with one backbone.
	- **Mechanism**: Independent VAEs handle specific channel counts and imaging statistics, encoding images into $4 \times 64 \times 64$ latents. After freezing the VAEs, a DiT serves as the shared semantic backbone. Input is $[z_t, z_i]$, and the condition vector is a fusion of timestep, source, and target embeddings.
	- **Design Motivation**: Modality-specific VAEs absorb low-level physical differences, while the shared DiT learns high-level geographic semantic transformations.

3. **$x_0$ Prediction and Target Modality Residual Adapter**:
	- **Function**: Enhances structural stability and corrects residual misalignment between the shared backbone output and the target decoder's latent space.
	- **Mechanism**: Any2Any directly predicts the clean target latent $\hat{z}_j = f_\theta([z_t, z_i], c)$. The target adapter then executes $z'_j = \hat{z}_j + A_j(\hat{z}_j)$. The adapter is a compact convolutional network with zero-initialization on the final layer; training uses stop-gradient to ensure calibration loss does not perturb the shared backbone.
	- **Design Motivation**: The shared backbone maintains cross-direction generalization, while the adapter handles small, target-specific latent space corrections to prevent detail averaging.

### Loss & Training
Training involves two stages. Stage 1 trains modality-specific VAEs using $L_{VAE} = L_{rec} + \gamma L_{lpips} + \beta L_{KL}$. Stage 2 freezes VAEs and trains the shared DiT and adapters. The diffusion backbone uses an $x_0$ prediction loss $L_{z0}$ relative to the target anchor $z_j$, while the adapter uses $L_{calib} = \|\hat{z}_j + A_j(sg(\hat{z}_j)) - z_j\|_2^2$. The total objective is $L_{total} = L_{z0} + \lambda L_{calib}$ with $\lambda = 1.0$.

## Key Experimental Results

### Main Results
Any2Any is evaluated on 14 seen translation directions in the RST-1M test set against Pix2PixHD, BBDM, and others. While baselines require 14 separate models, a single Any2Any-L model achieves leading performance.

| Translation Direction | Metric | Any2Any-L | Best Baseline | Gain / Difference |
|----------|------|-----------|--------------|-------------|
| SAR → RGB | PSNR / RMSE | 25.20 / 16.85 | BBDM: 19.50 / 31.02 | PSNR +5.70, RMSE -14.17 |
| NIR → RGB | PSNR / RMSE | 27.03 / 13.70 | BBDM: 20.39 / 29.59 | PSNR +6.64, RMSE -15.89 |
| MS → RGB | PSNR / RMSE | 33.22 / 6.45 | BBDM: 26.39 / 12.76 | PSNR +6.83, RMSE -6.31 |
| RGB → PAN | PSNR / RMSE | 33.45 / 9.47 | LBM: 27.02 / 13.30 | PSNR +6.43, RMSE -3.83 |
| MS → NIR | PSNR / RMSE | 29.14 / 10.26 | LBM: 19.00 / 34.33 | PSNR +10.14, RMSE -24.07 |

Scalability comparison:
| Model Scale | SAR→RGB PSNR / RMSE | NIR→RGB PSNR / RMSE | MS→RGB PSNR / RMSE | RGB→PAN PSNR / RMSE | Observation |
|----------|----------------------|----------------------|---------------------|----------------------|------|
| Any2Any-S | 22.25 / 23.45 | 23.01 / 21.25 | 29.81 / 9.23 | 31.30 / 11.27 | Small model exceeds most paired baselines |
| Any2Any-B | 24.35 / 18.42 | 26.02 / 15.27 | 32.35 / 7.07 | 33.03 / 9.87 | Multi-direction improvement with scale |
| Any2Any-L | 25.20 / 16.85 | 27.03 / 13.70 | 33.22 / 6.45 | 33.45 / 9.47 | Strongest overall performance |

### Ablation Study
Ablations focus on the residual adapter and training strategies. Results are reported for SAR → RGB.

| Configuration | Training Direction / Strategy | PSNR | RMSE | Description |
|------|----------------|------|------|------|
| Setting 1 | SAR→RGB, w/o adapter | 20.68 | 28.51 | Lacks target latent calibration |
| Setting 2 | SAR→RGB, w/ adapter | 20.88 | 27.89 | Adapter provides +0.20 PSNR |
| Setting 3 | SAR↔RGB, from scratch | 19.63 | 32.83 | Bi-directional scratch training is unstable |
| Setting 4 | SAR↔RGB, incremental | 21.44 | 25.87 | Outperforms scratch by +1.81 PSNR |
| Setting 5 | SAR→All connected | 22.06 | 24.00 | Multi-direction improves specific tasks |
| Setting 6 | All connected→RGB | 21.36 | 26.32 | Alternative multi-direction is also superior |
| Setting 7 | Any→Any, 14 directions | 22.25 | 23.45 | Unified training performs best |

### Key Findings
- **Residual Adapters**: Offer consistent gains by correcting latent space misalignment with minimal parameters.
- **Incremental Training**: Outperforms training from scratch, suggesting that geographic structure representations learned in simpler directions transfer to new ones.
- **Unified Training Synergy**: Multi-direction training does not dilute performance but strengthens it, likely because additional directions provide more supervised anchors.
- **Zero-shot Capability**: Any2Any generates semantically reasonable results for pairs like SAR-PAN and NIR-PAN despite no paired training data.

## Highlights & Insights
- **Unified Inference**: Redefines RS translation as a unified inference task on a connected modality graph rather than isolated pairs.
- **RST-1M Connectivity**: Pragmatically links datasets via shared modalities (e.g., RGB), enabling the model to generalize across the graph.
- **Functional Partitioning**: VAEs manage physical sensing, DiT manages semantic mapping, and adapters manage distribution calibration.
- **Regularization through Diversity**: Multiple directions serve as a form of regularization and extra supervision, a key finding for RS foundation models.

## Limitations & Future Work
- **Dataset Bias**: RST-1M relies on public datasets; robustness against extreme weather or unconventional sensors remains untested.
- **Graph-based Generalization**: Any-to-Any generalization functions via transfer learning on a connected graph; isolated or vastly different new modalities may still require specific anchors.
- **Inference Latency**: 250-step DDIM sampling is computationally expensive for large-scale production.
- **Downstream Validation**: Lacks systematic verification of whether synthesized data improves tasks like segmentation or change detection.

## Related Work & Insights
- **vs. Pix2Pix/Pix2PixHD**: Traditional GANs suffer from boundary misalignment in cross-sensor scenarios and lack scalability for $N$ modalities.
- **vs. Diffusion/Bridge Models**: While high quality, they are typically limited to fixed domain pairs. Any2Any shifts the focus to anchor regression within a unified latent space.
- **vs. ControlNet**: While sharing the "lightweight adjustment" philosophy, Any2Any's adapters focus on correcting output distribution bias for specific target sensors rather than providing external control.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (First systematic formalization of Any-to-Any RS translation).
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ (Comprehensive baselines and seen directions; unseen directions are qualitative).
- **Writing Quality**: ⭐⭐⭐⭐☆ (Clear framework breakdown; rich tabular data).
- **Value**: ⭐⭐⭐⭐⭐ (Directly applicable to multi-sensor completion and RS foundation models).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](../../ICCV2025/remote_sensing/skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[CVPR 2026\] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](../../CVPR2026/remote_sensing/geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](../../NeurIPS2025/remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[ICCV 2025\] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images](../../ICCV2025/remote_sensing/smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](../../ICCV2025/remote_sensing/skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[CVPR 2026\] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](../../CVPR2026/remote_sensing/geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)
- [\[ECCV 2024\] Masked Angle-Aware Autoencoder for Remote Sensing Images](../../ECCV2024/remote_sensing/masked_angle-aware_autoencoder_for_remote_sensing_images.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[ICML 2025\] Resampling Augmentation for Time Series Contrastive Learning: Application to Remote Sensing](../../ICML2025/remote_sensing/resampling_augmentation_for_time_series_contrastive_learning_application_to_remo.md)

</div>

<!-- RELATED:END -->
