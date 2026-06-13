---
title: >-
  [Paper Note] CHARM: Using Multimodal JEPA + Channel Descriptions for Time Series Foundation Embedding
description: >-
  [ICML 2026][Multimodal VLM][Time Series Foundation Models] CHARM injects channel text descriptions (e.g., "temperature sensor °C") as inductive biases into a Time Series Transformer and trains it using the JEPA objective…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Time Series Foundation Models"
  - "JEPA"
  - "Channel Descriptions"
  - "equivariant attention"
  - "sensor embedding"
date: 2026-05-08
content_hash: 111b3ebb15c5a37d
---

# CHARM: Using Multimodal JEPA + Channel Descriptions for Time Series Foundation Embedding

**Conference**: ICML 2026  
**arXiv**: [2605.31580](https://arxiv.org/abs/2605.31580)  
**Code**: Not provided in the paper  
**Area**: Time Series / Self-Supervised / Multimodal  
**Keywords**: Time Series Foundation Models, JEPA, Channel Descriptions, equivariant attention, sensor embedding

## TL;DR
CHARM injects channel text descriptions (e.g., "temperature sensor °C") as inductive biases into a Time Series Transformer and trains it using the JEPA objective (latent prediction instead of raw signal reconstruction). The resulting embeddings achieve performance comparable to specialized models like PatchTST, MOMENT, and Moirai on anomaly detection, classification, and forecasting using a simple linear probe, while maintaining strict channel-permutation equivariance.

## Background & Motivation

**Background**: Time series models are critical for applications in manufacturing, energy, healthcare, and finance, but most remain narrow in scope and task-specific. While foundation models have achieved significant success in NLP, CV, and audio, and forecasting foundation models (TimeFM, Moirai, Chronos) are emerging in time series, their representations remain brittle and unsuitable for diverse downstream tasks.

**Limitations of Prior Work**: (1) Most SSL time series models use masked reconstruction or next-step prediction, requiring the encoder to impute raw signals—which are often noisy, low-resolution, and contain domain artifacts—causing representations to overfit sensor noise; (2) Almost all time series models treat channels as uncategorized streams, discarding crucial context like sensor identity; (3) Models such as UniTS extend reconstruction-based approaches but remain grounded at the raw signal level.

**Key Challenge**: General-purpose embeddings require SSL, yet existing SSL objectives (reconstruction) mismatch the goal of "semantic representation." There is a need for sensor-aware models that are also invariant to channel ordering.

**Goal**: Build a semantically grounded, channel-aware (using text descriptions), and channel-order equivariant time series foundation embedding model.

**Key Insight**: (1) Utilize JEPA for prediction in the latent space rather than raw signal space to avoid overfitting noise; (2) Inject text descriptions of each channel as inductive bias through contextual TCN and attention gating; (3) Design inter-channel time-offset attention and description-aware gating to ensure channel permutation equivariance.

**Core Idea**: CHARM gives "voices to sensors"—sensor text descriptions participate in convolution kernel generation, attention gating, and time-offset embedding, allowing the model to adapt to heterogeneous sensor configurations. The combination of JEPA latent prediction and multi-resolution L1 loss ensures embeddings are both fine-grained and abstractly robust.

## Method

### Overall Architecture

The input is a tuple $\mathbf{t} = (\mathbf{T}, \mathbf{D}, \mathbf{pos})$ where $\mathbf{T} \in \mathbb{R}^{T \times C}$ represents the time series, and $\mathbf{D}$ contains text descriptions for $C$ channels (encoded as $\mathbf{E}_d$ by a frozen text encoder). A Contextual TCN provides the initial embedding $\mathbf{T}_c \in \mathbb{R}^{T \times C \times H}$; contextual attention layers fuse spatio-temporal channel information using description-aware gating and inter-channel time-offset attention; three JEPA encoders (context, target, and predictor) perform latent prediction training.

### Key Designs

1. **Contextual TCN: Text Descriptions Participate in Convolution Kernel Generation**:

    - **Function**: Allows the TCN to adapt across domains without manual tuning of patch sizes.
    - **Mechanism**: Uses channel description embeddings $\mathbf{E}_d$ to generate: (a) Contextual Kernel Gating $\mathbf{G}_c = \mathrm{sigmoid}(\mathbf{E}_d \mathbf{W}_g)$, which applies soft gating to each convolution layer to control the effective field of view; (b) Contextual Kernels $\mathbf{G}_k = \mathbf{E}_d \mathbf{W}_k$, which generates convolution filters directly from descriptions. This ensures different sensor types are processed by distinct kernels.
    - **Design Motivation**: Previous multivariate TS models used fixed patches or simple convolutions requiring manual tuning across domains. Generating kernels from descriptions aligns kernels with sensor types, enabling cross-domain transfer without retuning.

2. **Description-aware Inter-channel Attention + Time-offset Attention**:

    - **Function**: Explicitly models selective interactions and temporal lags between channels within attention layers while maintaining channel-permutation equivariance.
    - **Mechanism**: (a) Gating — computes pairwise channel similarity $\mathbf{S} = \mathbf{E}_d \mathbf{E}_d^\top$ and thresholds $\mathbf{Z}[i,j] = \mathrm{sigmoid}(\mathbf{E}_d[i,:] \mathbf{W}_b \mathbf{E}_d[j,:]^\top)$, where the gate $\mathbf{G}_d = \mathrm{ReLU}(\mathbf{Z} - \mathbf{S})$ controls channel attention; (b) Time-offset Attention — utilizes a learnable tensor $\boldsymbol{\Delta} \in \mathbb{R}^{C \times C \times 2T_{\max}}$, constructed symmetrically as $\boldsymbol{\Delta}_{i,j,t} = \boldsymbol{\Delta}_{j,i,-t}$; (c) Custom attention $\mathbf{A}_{[(i,p),(j,q)]} = \mathrm{Softmax}(\frac{\hat{\mathbf{Q}}\hat{\mathbf{K}}^\top}{\sqrt{D_e}} + \boldsymbol{\Delta}[i,j,q-p] - \lambda_G \mathbf{G}_d[i,j])$ integrates both gating and lag.
    - **Design Motivation**: Channel permutation equivariance (validated on 8 datasets with max output diff $<10^{-4}$) makes the model robust to sensor configuration changes; time-offset attention expresses lag dependencies more explicitly than vanilla RoPE.

3. **JEPA Training + Multi-resolution L1 Loss**:

    - **Function**: Performs prediction in the latent space to prevent raw signal reconstruction from overfitting noise; multi-scale loss encourages embeddings to learn both detail and abstraction.
    - **Mechanism**: Three encoders—context (input perturbed data), target (input clean data, updated via EMA), and predictor (a narrow contextual attention layer predicting masked positions). Two SSL tasks: causal prediction and smoothing. The self-supervised loss uses multi-resolution L1: per (channel, time) fine-grained + per time cross-channel average + global average across channel and time. Regularization: $R_1$ biases channel similarity thresholds toward sparsity, and $R_2$ regularizes the time-offset tensor via L2.
    - **Design Motivation**: JEPA latent prediction filters sensor noise compared to raw signal reconstruction, encouraging the encoder to learn high-level temporal structures while avoiding the complexity of negative sampling in contrastive learning. Multi-resolution L1 aligns embeddings at multiple granularities.

### Inference

Once trained, CHARM yields $\mathbf{Y} = \mathbf{E}_\theta(\mathbf{T}, \mathbf{D}, \mathbf{pos}) \in \mathbb{R}^{T \times C \times H}$ (L2-normalized). Downstream tasks can use frozen $\mathbf{Y}$ with a linear probe or a non-linear head.

## Key Experimental Results

### Forecasting: LSF benchmark (5 datasets × 4 horizons)

| Dataset | CHARM+LP | CHARM+NLH FT | Moirai-Large | Toto | TimeMixer++ | VisionTS |
|---------|----------|--------------|--------------|------|-------------|----------|
| Weather (MSE) | 0.230 | 0.222 | 0.226 | 0.242 | 0.269 | 0.228 |
| ETTm1 | 0.416 | 0.411 | 0.368 | 0.448 | 0.373 | 0.344 |
| ETTm2 | 0.220 | **0.208** | 0.269 | 0.322 | 0.281 | 0.259 |
| ETTh1 | 0.592 | 0.557 | 0.395 | 0.400 | 0.392 | 0.418 |
| ETTh2 | 0.323 | **0.316** | 0.339 | 0.341 | 0.333 | 0.352 |

CHARM using frozen embedding linear probes outperforms billion-parameter Moirai-Large and Toto on ETTm2 and ETTh2; after fine-tuning, it achieves the lowest MSE on 3/5 ETT/Weather datasets.

### Classification

| Method | Wins ↑ | Avg. Acc ↑ |
|--------|--------|------------|
| TS2Vec | 2 | 78.1 |
| T-Rep | 3 | 78.5 |
| MOMENT | 3 | 72.5 |
| MiniROCKET | 4 | 77.6 |
| **CHARM frozen+SVM** | **4** | **79.6** |

CHARM frozen + SVM achieves the highest average accuracy at 79.6%.

### Key Findings

- **Frozen Embeddings + Linear Probe are Competitive**: Beating Moirai-Large on ETTm2/ETTh2 demonstrates that channel-aware semantic representations can outperform billion-parameter forecasting-specific models.
- **Cross-task Scalability**: Performance is strong across anomaly detection, classification, and forecasting using the same embedding, indicating a truly general-purpose representation.
- **Strict Validation of Channel Permutation Equivariance**: Max output difference < $10^{-4}$ in random permutation tests across 8 datasets.
- **JEPA > Reconstruction**: Compared to MOMENT's reconstruction-based pretraining (Acc 72.5), CHARM's JEPA latent prediction (79.6) provides a 7% Gain, verifying that latent prediction is superior for time series representation.
- **Descriptions as Channel Identifiers**: The authors clarify that "text descriptions serve as channel identifiers for cross-dataset generalization"—descriptions are used for sensor type inductive bias rather than language understanding.

## Highlights & Insights

- **Adapting JEPA from NLP/CV to Time Series**: This is the first work to systematically port JEPA + latent prediction to multivariate time series, proving the transferability of cross-modal SSL.
- **Text Descriptions as Inductive Bias, Not Captions**: Unlike previous multimodal TS models that treat text as captions, this work uses text as "sensor type identifiers" to control convolution kernels and attention mechanisms.
- **Achieving Channel-permutation Equivariance via Attention Design**: The symmetric time-offset tensor and description-aware gating ensure invariance to sensor order, allowing the model to handle heterogeneous fleets.
- **Multi-resolution L1 Loss Prevents Representation Collapse**: Aligning at both fine and coarse levels ensures that embeddings remain meaningful across multiple granularities.
- **Frozen + Linear Probe Surpassing Billion-param Moirai**: A striking demonstration of parameter efficiency with significant implications for industrial deployment.
- **Synergy of JEPA, Descriptions, and Equivariance**: JEPA alone lacks sensor awareness; descriptions alone lack noise robustness; equivariance alone lacks semantic representation. Together, they form the missing piece for foundation embeddings.

## Limitations & Future Work

- **Dependency on Channel Description Quality**: Poorly written sensor descriptions (missing units, ambiguous semantics) may degrade Contextual TCN and attention gating performance; real-world deployment requires description engineering.
- **Cross-domain Transferability**: While verified on LSF benchmarks, representation reuse in cross-industry transfers (e.g., Medical → Manufacturing) has not been systematically tested.
- **Training Stability of JEPA EMA Targets**: JEPA is known for collapse risks in vision; more evidence is needed regarding its stability over long trajectory training in time series.
- **Memory Overhead of Time-offset Tensor $\boldsymbol{\Delta}$**: The $\mathbb{R}^{C \times C \times 2T_{\max}}$ tensor may cause memory issues when $C$ is large (e.g., 500 sensors), necessitating sparse or low-rank approximations.
- **Lack of Large-scale Anomaly Detection Comparison**: Reconstruction is a traditional strength for anomaly detection; CHARM's advantage here has not been fully quantified.
- **Gap in Long-horizon Forecasting (720)**: On ETTh1, CHARM (0.592) still trails Moirai (0.395); because JEPA learns semantic rather than fine-grained signals, long-term precision may be limited.

## Related Work & Insights

- **vs MOMENT / UniTS**: These use reconstruction-based pretraining at the raw signal level; CHARM uses JEPA latent prediction for better noise robustness, leading to +7% in classification.
- **vs Moirai / TimesFM / Chronos / TimeFM**: These billion-parameter forecasting foundation models are specific to forecasting; CHARM provides general-purpose representations that match them even when frozen.
- **vs PatchTST / iTransformer**: These tokenize patches or channels; CHARM generates convolution kernels and attention gates from descriptions for finer sensor awareness.
- **vs I-JEPA (Assran et al. 2023)**: The vision prototype for JEPA; this work is an adaptation and enhancement for multivariate TS (via multi-resolution loss and description-aware attention).
- **vs CLIP for TS**: CLIP-style contrastive learning treats text as captions; this work treats text as a mechanistic inductive bias.
- **Insight**: (1) The "text description as inductive bias" approach can be extended to other heterogeneous data (point clouds, medical imaging); (2) JEPA latent prediction is better suited than reconstruction-based SSL for noisy/low-resolution data; (3) Channel permutation equivariance via symmetric tensors and description gating is a reusable architectural pattern.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Combines JEPA, description-as-inductive-bias, and channel-permutation equivariance in a comprehensive methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results across Forecasting, Classification, and Anomaly Detection; lacks cross-industry transfer tests and direct comparisons with multi-billion-parameter models under identical conditions.
- Writing Quality: ⭐⭐⭐⭐ Clear architecture figures and detailed loss mathematics; however, some design choices (symmetric tensor construction, multi-resolution L1) are relegated to the appendix.
- Value: ⭐⭐⭐⭐ The fact that Frozen + Linear Probe can rival billion-param models enables smaller research groups to use powerful TS representations; the channel description approach has direct practical value for industrial sensor fleet deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Text-Conditional JEPA for Learning Semantically Rich Visual Representations](text-conditional_jepa_for_learning_semantically_rich_visual_representations.md)
- [\[NeurIPS 2025\] GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images](../../NeurIPS2025/multimodal_vlm/gem_empowering_mllm_for_grounded_ecg_understanding_with_time_series_and_images.md)
- [\[CVPR 2026\] Scaling Spatial Intelligence with Multimodal Foundation Models](../../CVPR2026/multimodal_vlm/scaling_spatial_intelligence_with_multimodal_foundation_models.md)
- [\[ICML 2026\] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models](circle-rope_cone-like_decoupled_rotary_positional_embedding_for_large_vision-lan.md)
- [\[ICML 2026\] LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations](limssr_llm-driven_sequence-to-score_reasoning_under_training-time_incomplete_mul.md)

</div>

<!-- RELATED:END -->
