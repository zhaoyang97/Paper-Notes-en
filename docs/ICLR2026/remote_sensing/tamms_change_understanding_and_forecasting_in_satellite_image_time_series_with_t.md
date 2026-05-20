---
title: >-
  [Paper Note] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models
description: >-
  [ICLR 2026][Remote Sensing][satellite image time series] TAMMs is proposed as the first unified framework that jointly performs Temporal Change Description (TCD) and Future Satellite Image Forecasting (FSIF) within a sin…
tags:
  - "ICLR 2026"
  - "Remote Sensing"
  - "satellite image time series"
  - "temporal change description"
  - "future prediction"
  - "multimodal large language model"
  - "diffusion model"
date: 2026-05-08
content_hash: c11abd680b2ae952
---

# TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models

**Conference**: ICLR 2026
**arXiv**: [2506.18862](https://arxiv.org/abs/2506.18862)  
**Code**: None  
**Area**: Remote Sensing
**Keywords**: satellite image time series, temporal change description, future prediction, multimodal large language model, diffusion model

## TL;DR

TAMMs is proposed as the first unified framework that jointly performs Temporal Change Description (TCD) and Future Satellite Image Forecasting (FSIF) within a single MLLM-diffusion architecture. A Temporal Adaptation Module (TAM) awakens the temporal reasoning capability of a frozen MLLM, while a Semantic Fusion Control Injection (SFCI) mechanism converts change understanding into generative control signals.

## Background & Motivation

1. TCD and FSIF over Satellite Image Time Series (SITS) are two core yet historically disconnected tasks, both constrained by insufficient long-range temporal dynamic modeling.
2. Existing TCD methods (e.g., SITSCC) fuse multi-temporal information through simple interaction, limiting long-range temporal reasoning; existing FSIF methods (e.g., DiffusionSat) primarily rely on metadata-based conditioning and lack semantic change understanding.
3. MLLMs excel at vision-language tasks, but their video understanding capabilities are optimized for densely sampled short-interval sequences and cannot directly accommodate the sparse, long-term intervals spanning multiple years in SITS.
4. Control signals for diffusion models (e.g., edge maps or metadata) are typically low-level, lacking high-level semantic guidance over temporal evolution narratives.
5. Conventional evaluation metrics (PSNR, SSIM) suffer from an "evaluation gap"—they cannot penalize temporally inconsistent predictions; a perceptually realistic prediction that contradicts historical trends still receives a high score.
6. The paper's core hypothesis is that enabling MLLMs to deeply understand historical dynamics yields more consistent future predictions—a synergistic gain from joint understanding and generation.

## Method

### Overall Architecture

TAMMs comprises two collaborative stages: (1) a temporal change understanding stage, in which the TAM module enhances the frozen MLLM's temporal reasoning to produce textual descriptions and a semantic feature vector $\mathbf{M}_t$; and (2) a future prediction stage, in which the SFCI mechanism translates the MLLM's semantic features into multi-scale control signals that guide the denoising process of a frozen diffusion U-Net. The MLLM backbone is the frozen DeepSeek-VL2; the generation component is based on DiffusionSat (Stable Diffusion 2-1).

### Key Designs

#### 1. Temporal Adaptation Module (TAM)

- **Physical Time Encoder (PTE)**: Introduces a learnable temporal token `[TIME_DIFF]`, dynamically conditioned on the concrete time interval $\Delta t_i$ via an MLP, and inserted between adjacent image visual features. This allows the MLLM's attention mechanism to directly associate visual changes with corresponding time intervals.
- **Contextual Temporal Prompting (CTP)**: Provides structured textual prompts with detailed scene descriptions to guide the MLLM in performing specific temporal reasoning tasks (describing changes in the observed sequence), focusing the MLLM's general reasoning capability on temporal dynamic recognition.

#### 2. Semantic Fusion Control Injection (SFCI)

An Enhanced Control Module (ECM) operates in parallel with the frozen diffusion U-Net, consisting of a four-step pipeline:

- **Structural path**: A frozen 3D Control Block processes U-Net encoder features to yield structural control signals $\mathbf{h}_l^{(ctrl)}$ encoding visual dynamics.
- **Semantic path**: MLLM semantic features $\mathbf{M}_t$ are projected by layer-specific processors and tiled into spatially-aware guidance signals $\mathbf{s}_l$.
- **Adaptive gated fusion**: A dynamic gate $\mathbf{g}_l$ adaptively interpolates structural and semantic signals: $\mathbf{f}_l = (1-\mathbf{g}_l) \odot \mathbf{h}_l^{(ctrl)} + \mathbf{g}_l \odot \mathbf{s}_l$.
- **Temporal refinement**: A temporal Transformer models long-range dependencies, with outputs integrated via a weighted residual connection.

#### 3. Temporal Consistency Score (TCS)

A newly proposed evaluation metric that quantifies the consistency between predicted changes and historical dynamics:

$$\text{TCS} = \text{SPS} \cdot \text{ACS}$$

- SPS (Spatial Proximity Score): quantifies location consistency of change centroids.
- ACS (Area Consistency Score): assesses the agreement of change magnitudes.

### Loss & Training

- Two-stage training: the structural path is trained first to learn basic spatio-temporal priors, then frozen while the semantic components are trained.
- Understanding stage: composite loss $\mathcal{L} = \lambda_{\text{text}} \mathcal{L}_{\text{text}} + \lambda_{\text{temp}} \mathcal{L}_{\text{temp}}$, balancing text accuracy and temporal regularization.
- Generation stage: standard diffusion loss.
- Only lightweight adapter components are trained; both the MLLM and U-Net remain frozen.

## Key Experimental Results

### Main Results

**Temporal Change Description (TCD)**:

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr-D |
|-------|--------|--------|---------|---------|
| RSICC-Former | 0.1285 | 0.1930 | 0.3489 | 0.5344 |
| SITSCC | 0.2122 | 0.2961 | 0.4701 | 0.6244 |
| TEOChat | 0.2398 | 0.3102 | 0.4735 | 0.8267 |
| **TAMMs** | **0.2669** | **0.3312** | **0.4690** | **0.9030** |

**Future Satellite Image Forecasting (FSIF)**:

| Model | PSNR↑ | SSIM↑ | LPIPS↓ | TCS↑ |
|-------|-------|-------|--------|------|
| DiffusionSat | 11.89 | 0.1520 | 0.5225 | 0.7624 |
| MCVD | 9.22 | 0.2098 | 0.4970 | 0.1930 |
| **TAMMs** | **12.07** | 0.1831 | **0.4931** | **0.9690** |

### Ablation Study

| Configuration | BLEU-4 | CIDEr-D | TCS |
|---------------|--------|---------|-----|
| SFT only | 0.2134 | 0.7523 | 0.6842 |
| w/o PTE | 0.2387 | 0.8234 | 0.7456 |
| w/o CTP | 0.2445 | 0.8567 | 0.8234 |
| w/o Semantic Fusion | - | - | 0.7911 |
| w/o Text Guidance | - | - | 0.9410 |
| Base Control Block | - | - | 0.7624 |
| **TAMMs (Full)** | **0.2669** | **0.9030** | **0.9690** |

### Key Findings

1. **TCS demonstrates a decisive advantage**: TAMMs achieves TCS of 0.9690, far exceeding DiffusionSat (0.7624) and GeoSynth-Canny (0.2170), confirming that the generated future images are more consistent with historical evolution trajectories.
2. **Semantic fusion is critical**: Removing semantic feature fusion causes an 18% drop in TCS (0.9690→0.7911), validating the core hypothesis of injecting deep semantic reasoning from the MLLM directly into the generative control pathway.
3. **PTE contributes most to temporal understanding**: Removing PTE leads to a 23% drop in TCS, indicating that explicit time-interval encoding is the key to awakening temporal reasoning in the MLLM.

## Highlights & Insights

1. This is the first work to unify temporal change understanding and future prediction in a single framework; the bidirectional co-design—where understanding guides generation and generation validates understanding—is a novel contribution.
2. The TCS metric fills a gap in temporal prediction evaluation: standard quality metrics cannot measure temporal consistency, whereas TCS quantifies this through spatial proximity and area consistency.
3. The TAM module design is elegant—it "awakens" the latent temporal reasoning capacity of a frozen MLLM in a parameter-efficient manner, avoiding costly full-model fine-tuning.
4. SFCI bypasses the bottleneck of coarse-grained text control by directly converting the MLLM's multi-image temporal understanding features into patch-level fine-grained control signals.

## Limitations & Future Work

1. Training set annotations are automatically generated by Qwen2.5-VL (37K sequences), which may introduce systematic bias; the test set contains only 150 sequences.
2. MCVD achieves a higher SSIM (0.2098) than TAMMs (0.1831), revealing a trade-off between standard quality metrics and temporal consistency metrics.
3. The TCS metric relies on binary change detection and may have limited capacity to capture gradual changes (e.g., slow vegetation degradation).
4. Very long-term (>10 years) prediction scenarios and sudden-event forecasting remain unexplored.

## Related Work & Insights

- **DiffusionSat**: A foundational model for satellite image generation conditioned on metadata, lacking semantic understanding guidance.
- **SITSCC**: A multi-temporal change description method, lacking the deep semantic reasoning of MLLMs.
- **TEOChat**: A temporal MLLM for Earth observation, limited to description tasks with no generative capability.
- **ControlNet**: Controls diffusion model generation primarily via text, providing insufficient patch-level signal guidance for temporal prediction.
- Insight: Unifying understanding and generation is an important direction for remote sensing spatio-temporal analysis; the "awakening" paradigm of TAM is generalizable to other foundation models with insufficient temporal awareness.

## Rating

- ⭐ Novelty: 4.5/5 — First unified TCD+FSIF framework; TCS metric, TAM, and SFCI each represent distinct innovations
- ⭐ Experimental Thoroughness: 4/5 — Ablations are thorough and qualitative analysis is rich, but the test set is small (150 sequences)
- ⭐ Writing Quality: 4/5 — Problem-driven narrative is clear; two "How" questions effectively guide the method design
- ⭐ Value: 4/5 — Establishes a new paradigm of understanding-driven generation for remote sensing spatio-temporal analysis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Are Pretrained Image Matchers Good Enough for SAR-Optical Satellite Registration?](../../CVPR2026/remote_sensing/pretrained_image_matchers_for_sar_optical_satellite_registration.md)
- [\[NeurIPS 2025\] EcoCast: A Spatio-Temporal Model for Continual Biodiversity and Climate Risk Forecasting](../../NeurIPS2025/remote_sensing/ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models](../../NeurIPS2025/remote_sensing/connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](../../ICCV2025/remote_sensing/wildsat_learning_satellite_image_representations_from_wildlife_observations.md)
- [\[AAAI 2026\] TDCNet: Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving IRSTD](../../AAAI2026/remote_sensing/spatio-temporal_context_learning_with_temporal_difference_convolution_for_moving.md)

</div>

<!-- RELATED:END -->
