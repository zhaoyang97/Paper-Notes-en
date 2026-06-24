---
title: >-
  [Paper Note] RFDM: Residual Flow Diffusion Models for Video Editing
description: >-
  [CVPR 2026][Video Generation][Instructive Video Editing] RFDM transforms a 2D image-to-image (I2I) diffusion model into a frame-by-frame autoregressive video editor. By "shifting" the noise mean of the current frame relative to the previous frame's prediction, the model learns inter-frame residuals rather than full frames. This achieves temporal consistency and editing fidelity comparable to 3D spatio-temporal models with **zero additional computational cost** and support for…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Instructive Video Editing"
  - "Autoregressive Diffusion"
  - "Residual Flow"
  - "Exposure Bias"
  - "Temporal Consistency"
date: 2026-05-08
content_hash: 83f6466da850b854
---

# RFDM: Residual Flow Diffusion Models for Video Editing

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Salehi_RFDM_Residual_Flow_Diffusion_Models_for_Video_Editing_CVPR_2026_paper.html)  
**Code**: None  
**Area**: Video Editing / Diffusion Models  
**Keywords**: Instructive Video Editing, Autoregressive Diffusion, Residual Flow, Exposure Bias, Temporal Consistency

## TL;DR
RFDM transforms a 2D image-to-image (I2I) diffusion model into a frame-by-frame autoregressive video editor. By "shifting" the noise mean of the current frame relative to the previous frame's prediction, the model learns inter-frame residuals rather than full frames. This achieves temporal consistency and editing fidelity comparable to 3D spatio-temporal models with **zero additional computational cost** and support for arbitrary video lengths.

## Background & Motivation
**Background**: Instruction-based video editing (e.g., "remove the person" via natural language) typically follows two paths: 1) Large 3D spatio-temporal models (e.g., EVE), which offer high quality but require massive data and compute; 2) Adapting 2D T2I/I2I models (e.g., Fairy, VidToMe, TokenFlow) using cross-frame feature alignment or spatio-temporal attention to maintain consistency.

**Limitations of Prior Work**: Most existing methods rely on **non-causal** temporal mechanisms, requiring fixed-length video segments and causing computational costs to explode with frame counts due to global attention. Autoregressive video generation offers solutions for variable-length input and efficiency but remains largely unexplored and non-real-time for video editing.

**Key Challenge**: While I2I models are computationally efficient, independent frame-by-frame editing suffers from **temporal inconsistency** (e.g., flickering colors) due to diffusion stochasticity. Achieving consistency typically requires heavy spatio-temporal attention, sacrificing the efficiency of 2D models.

**Goal**: Achieve temporal consistency while maintaining I2I level computational efficiency and supporting arbitrary video lengths.

**Key Insight**: Successive video frames are highly redundant; most pixels merely need to be "carried over" from the previous frame. Reformulating the editing task as "predicting residuals between adjacent frames" avoids reconstructing the entire image from pure noise for every frame.

**Core Idea**: Shift the **noise mean** of the I2I forward diffusion process from 0 to the previous prediction $\hat{y}_{t-1}$. This focuses denoising on temporal changes (residuals) and uses the previous prediction as a condition for causal autoregression—defining the Residual Flow Diffusion Model (RFDM).

## Method

### Overall Architecture
RFDM converts a single-frame I2I model into a causal, variable-length, zero-overhead video editor. Given a video sequence $X=\{x_t\}$ and instruction $p$, the model autoregressively generates edited frames $Y=\{y_t^0\}$, conditioning the $t$-th frame on its own prediction from the $(t-1)$-th frame $\hat{y}_{t-1}$. The pipeline modifies standard I2I diffusion in two ways: 1) Concatenating the previous prediction as a condition (no extra compute); 2) Reformulating the forward process into a residual form where the noise mean shifts towards $\hat{y}_{t-1}$. During training, Diffusion Forcing is used instead of Teacher Forcing to sample $\hat{y}_{t-1}$ and mitigate exposure bias.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input frame x_t + Instruction p"] --> B["Conditioning:<br/>Concatenate ŷ_(t-1) as input"]
    B --> C["Residual Flow Forward Process:<br/>Shift noise mean to ŷ_(t-1)"]
    C --> D["Denoising Network ŷ_θ<br/>DDIM + CFG, predict y_t^0"]
    D -->|t=0 start from pure noise| D
    D --> E["Output ŷ_t"]
    E -->|Feedback as ŷ_(t-1) for next frame| B
    F["Diffusion Forcing<br/>Mitigate Exposure Bias"] -.Sampling ŷ_(t-1) during training.-> C
```

### Key Designs

**1. Previous Prediction Conditioning: Causal I2I with Zero Overhead**

To ensure consistency, the $t$-th frame must "know" the $t-1$ edit. RFDM maintains the standard forward process $y_t^s = \alpha^s y_t^0 + \sigma^s \epsilon$ (where $\alpha^s, \sigma^s$ follow a schedule) while feeding $\hat{y}_{t-1}$ to the denoising network via **channel-wise concatenation**. The training objective is:

$$\arg\min_\theta \big\| \hat{y}_\theta(y_t^s, \hat{y}_{t-1}, x_t, p, \lambda^s) - y_t^0 \big\|,\quad \hat{y}_{-1}=0 .$$

The first frame defaults to standard I2I. Subsequent frames use their own outputs as conditions, making the sequence **causally autoregressive**. Since conditioning only adds a tensor concatenation, it introduces **no cross-frame attention**, keeping compute identical to single-frame I2I.

**2. Residual Flow Forward Process: Learning Inter-frame Changes**

Simple conditioning is insufficient as the model still reconstructs frames from pure noise. RFDM shifts the generation target from the "entire frame $y_t^0$" to the "inter-frame residual" $m_t^0 = \hat{y}_{t-1} - y_t^0$ by shifting the **Gaussian mean** from 0 to $\hat{y}_{t-1}$:

$$q(y_t^s \mid y_t^0, \hat{y}_{t-1}) = \mathcal{N}\big(\alpha^s y_t^0 + \sigma^s \hat{y}_{t-1},\ (\sigma^s)^2 I\big) = \mathcal{N}\big(\gamma^s y_t^0 + \sigma^s m_t^0,\ (\sigma^s)^2 I\big),$$

where $\gamma^s = \sqrt{1-(\sigma^s)^2} + \sigma^s$. Equivalently, the sampling becomes:

$$y_t^s = \alpha^s y_t^0 + \sigma^s \hat{\epsilon},\quad \hat{\epsilon}\sim\mathcal{N}(\hat{y}_{t-1}, I).$$

The noise is centered on the previous prediction, transporting the target frame to a noisy version of the previous one. This **explicitly embeds the residual** into the noisy input. The network learns to shift pixels for already-edited regions in $\hat{y}_{t-1}$ (e.g., background) and only perform denoising for new areas described by $x_t$ and $p$.

**3. Diffusion Forcing: Eliminating Exposure Bias**

Autoregressive models suffer from exposure bias: the distribution of $\hat{y}_{t-1}$ differs between training and inference, leading to quality drift. Unlike Teacher Forcing, which uses ground-truth $y_{t-1}^0$, RFDM uses **Diffusion Forcing**: during training, frames are sampled with different noise levels, and $\hat{y}_{t-1}$ is sampled using the denoising network's own prediction at $t-1$. This keeps the condition frame "clean" but ensures it originates from the model's own training distribution, narrowing the train-inference gap.

### Loss & Training
Training utilizes a standard MSE reconstruction loss: for a video clip, $K$ ordered frames are sampled. Starting from $\hat{y}_{-1}=0$, each frame constructs a noisy input $y_t^s$ using the mean-shifted process. The model accumulates $\text{MSE}(\hat{y}_t, y_t^0)$ across all frames. Backbones include SD1.5 (RFDM1.5) and SD3.5-M (RFDM3.5). Training data is Señorita-2M, a large-scale real-world video editing dataset with 2 million pairs. Training involves 45k steps on 8×A100 GPUs with FusedAdam and a learning rate of 1e-4.

## Key Experimental Results

### Main Results
On the Señorita test set (16-frame generation across style transfer and object removal), RFDM outperforms existing I2I methods in fidelity (ViDreamSim↓, DVS↑, MLLM-Judge↑) and efficiency, approaching the quality of closed-source 3D models (EVE) with ~13× less VRAM:

| Method | Causal | DVS↑ | MLLM-Judge↑ | ViDreamSim↓ | TempCon↓ | Latency(s)↓ | Memory(GB)↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fairy | ✗ | 0.40 | 3.98 | 0.29 | 0.042 | 13 | 77 |
| VidToMe | ✗ | 0.37 | 1.77 | 0.59 | 0.014 | 86 | 9 |
| TokenFlow | ✗ | 0.29 | 3.23 | 0.48 | 0.010 | 128 | 11 |
| RAVE | ✗ | 0.34 | 3.54 | 0.42 | 0.017 | 92 | 9 |
| RFDM1.5 | ✓ | 0.43 | 6.60 | 0.23 | 0.010 | **8** | **2** |
| RFDM3.5 | ✓ | **0.48** | **7.37** | **0.20** | 0.009 | 13 | 6 |

RFDM variants achieved the highest CLIPFrame scores (temporal consistency) on TGVE benchmarks, with 1/13th the VRAM of Fairy and 4× faster latency than other baselines.

### Ablation Study
Ablations on SD1.5/Señorita (Default: 3 autoregressive frames, Residual Flow):

| Config | TempCon↓ | ErrAccu↓ | Notes |
|------|:---:|:---:|------|
| AR Frames = 0 | 0.068 | 0.21 | Degrades to frame-wise independent |
| AR Frames = 1 | 0.013 | 0.12 | Major improvement with previous condition |
| AR Frames = 3 (Def) | 0.009 | 0.07 | Best balance of consistency/error |
| Condition = $x$ only | 0.027 | 0.14 | Significant degradation without $\hat{y}_{t-1}$ |
| Type = Full Frame | 0.009 | 0.09 | Higher error accumulation |
| Type = Residual Flow| 0.009 | 0.07 | Lower error accumulation |

DAVIS tracking experiments showed that switching from "Full Frame" to "Residual Flow" improved J&F from 29.1 to **43.6**, confirming that residual modeling stabilizes edited objects across frames.

### Key Findings
- **Residual flow primarily reduces "Error Accumulation"**: While TempCon is similar between full-frame and residual modes (0.009), ErrAccu drops from 0.09 to 0.07, mitigating long-range autoregressive drift.
- **Previous frame conditioning is the "Master Switch"**: Increasing AR frames from 0 to 1 slashes TempCon from 0.068 to 0.013, proving that seeing the previous frame is the most critical factor for consistency.
- **Improved Benchmarks**: Standard CLIP text similarity fails to measure fidelity (avoiding unwanted changes). RFDM introduces ViDreamSim (per-frame fidelity to ground truth) and Error Accumulation (drift relative to the first frame) to better differentiate "smoothness" from "faithfulness."

## Highlights & Insights
- **Noise Mean Shifting is extremely lightweight**: Without changing network architecture or adding attention, reformulating the process centers the diffusion on residuals, matching pure I2I efficiency.
- **Implicit Pixel Partitioning**: The network naturally learns to shift pixels for previously edited areas and only generate for new content, elegantly encoding temporal redundancy into the diffusion process.
- **Diffusion Forcing for Video Editing**: Applying this technique to the train-inference gap in autoregressive editing effectively prevents long-range degradation.
- **Benchmark Contribution**: The identification of ViDreamSim/Error Accumulation provides a more robust framework for evaluating video editing fidelity versus mere smoothing.

## Limitations & Future Work
- RFDM's PickScore is slightly lower than protected 3D models like EVE (which uses 4.4B params vs RFDM's 2.5B and 17× more training data).
- While residual flow reduces drift, pure autoregressive models may still drift over extremely long sequences; experiments were mostly conducted on 16-frame clips.
- Dependence on the Señorita "paired ground truth" dataset; performance on edits requiring massive geometric structural changes (e.g., complex object replacement) is less explored.
- The first frame still relies on standard I2I; any artifacts in the first frame will propagate through the sequence.

## Related Work & Insights
- **vs. Fairy/VidToMe (I2I backbones, non-causal)**: These methods require the entire video at once and have high memory costs (Fairy at 77GB). RFDM is causal, uses 2–6GB VRAM, and provides better fidelity and consistency.
- **vs. EVE (3D Spatio-temporal)**: EVE provides high quality but massive compute; RFDM uses 2D backbones + residual flow to approach this quality with orders-of-magnitude lower latency.
- **vs. Teacher Forcing**: Traditional AR training uses ground truth; RFDM uses Diffusion Forcing to train on the model's own distribution, specifically suppressing the drift typical in autoregressive editing.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlowPortal: Residual-Corrected Flow for Training-Free Video Relighting and Background Replacement](flowportal_residual-corrected_flow_for_training-free_video_relighting_and_backgr.md)
- [\[CVPR 2026\] FlowDirector: Training-Free Flow Steering for Precise Text-to-Video Editing](flowdirector_training-free_flow_steering_for_precise_text-to-video_editing.md)
- [\[CVPR 2026\] Accelerating Autoregressive Video Diffusion via History-Guided Cache and Residual Correction](accelerating_autoregressive_video_diffusion_via_history-guided_cache_and_residua.md)
- [\[CVPR 2026\] BAgger: Backwards Aggregation for Mitigating Drift in Autoregressive Video Diffusion Models](bagger_backwards_aggregation_for_mitigating_drift_in_autoregressive_video_diffus.md)
- [\[CVPR 2026\] Flowception: Temporally Expansive Flow Matching for Video Generation](flowception_temporally_expansive_flow_matching_for_video_generation.md)

</div>

<!-- RELATED:END -->
