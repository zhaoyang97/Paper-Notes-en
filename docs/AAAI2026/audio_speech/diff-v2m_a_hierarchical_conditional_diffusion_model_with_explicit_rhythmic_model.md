---
title: >-
  [Paper Note] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation
description: >-
  [AAAI 2026][Audio & Speech][Video-to-music generation] Diff-V2M is proposed, a hierarchical conditional diffusion Transformer-based video-to-music generation framework. By integrating emotional, semantic, and rhythmic features through explicit rhythmic modeling (low-resolution ODF) and hierarchical cross-attention mechanisms, it achieves SOTA performance on both in-domain and out-of-domain datasets.
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Video-to-music generation"
  - "diffusion models"
  - "rhythmic modeling"
  - "hierarchical cross-attention"
  - "feature fusion"
date: 2026-05-08
content_hash: 0ae6d99224b4ee37
---

# Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation

**Conference**: AAAI 2026  
**arXiv**: [2511.09090](https://arxiv.org/abs/2511.09090)  
**Code**: [Demo](https://Tayjsl97.github.io/Diff-V2M-Demo/)  
**Area**: Image Generation  
**Keywords**: Video-to-music generation, diffusion models, rhythmic modeling, hierarchical cross-attention, feature fusion

## TL;DR

Diff-V2M is proposed, a hierarchical conditional diffusion Transformer-based video-to-music generation framework. By integrating emotional, semantic, and rhythmic features through explicit rhythmic modeling (low-resolution ODF) and hierarchical cross-attention mechanisms, it achieves SOTA performance on both in-domain and out-of-domain datasets.

## Background & Motivation

Video-to-music generation (V2M) aims to construct matching soundtracks for visual content in video creation. With the rise of video platforms like YouTube and TikTok, alongside video generation models like Sora and Veo, the demand for personalized audiovisual content has surged. Existing methods primarily face two core challenges:

**Lack of explicit rhythmic modeling**: Existing approaches implicitly map visual dynamics to musical rhythm via optical flow, frame differences, or frame-level visual features, failing to achieve precise audiovisual temporal alignment. LLM-based methods transcribing videos into text prompts also struggle to preserve fine-grained temporal dynamics.

**Difficulty in multi-view feature fusion**: Videos contain diverse visual characteristics, including emotional, semantic, and rhythmic features. Existing progressive fusion schemes increase computational overhead, while simple concatenation fails to capture complex inter-feature dependencies.

## Method

### Overall Architecture

Diff-V2M consists of two core components:

- **Visual Feature Extraction Module**: Extracts semantic features (CLIP), emotional features (Color Histogram), and rhythmic features (low-resolution ODF + rhythm predictor).
- **Conditional Music Generation Module**: A DiT-based latent diffusion model equipped with hierarchical cross-attention and a timestep-aware fusion strategy.

The audio sampling rate is 44.1 kHz, and the video frame rate is 1 FPS. A frozen VAE from Stable Audio Open is used to encode audio waveforms into latent representations, while the DiT model is trained from scratch.

### Key Designs

#### 1. Generalizable Rhythmic Representation

Three rhythmic representation schemes are systematically compared:

- **Low-resolution Mel-spectrogram** ($\text{Mel}_{LR}$): Obtained by normalizing and downsampling the original Mel-spectrogram to a target resolution of $[M, C]$ ($C=16$).
- **Low-resolution Tempo Map** ($\text{Tem}_{LR}$): A time-tempo representation capturing the local evolution of tempo over time ($B=16$).
- **Low-resolution ODF** ($\text{ODF}_{LR}$): Onset detection function, a 1D time series reflecting note onset strength. A second-level vector ($d=1$) is constructed via peak detection on the raw ODF.

Experiments demonstrate that **ODF is the most effective rhythmic representation**—more compact than Mel-spectrograms and tempo maps, directly highlighting key rhythmic events.

#### 2. Rhythm Predictor

Since audio is unavailable during inference, a decoder-only Transformer is trained to predict the rhythmic representation from video. The input comprises three components:

- **CLIP semantic features** $C_\mathbf{s}$
- **Scene transition embedding** $\mathbf{e} \in \{0,1\}^M$: Scene boundaries detected via PySceneDetect.
- **Visual beat vector** $\mathbf{v} \in \mathbb{R}^M$: Peak detection results from aggregated inter-frame differences.

The three inputs are summed and fed into the predictor: $\mathbf{X} = C_\mathbf{s} + \text{Embed}(\mathbf{e}) + \text{Linear}(\mathbf{v})$

#### 3. Hierarchical Cross-Attention Conditioning Module

A two-layer hierarchical structure is utilized to consolidate multi-view features:

- **First Layer**: Emotional features guide the overall emotional tone through standard cross-attention.
- **Second Layer**: Semantic and rhythmic features are processed independently using **parallel cross-attention** to prevent information entanglement.

#### 4. Timestep-Aware Fusion Strategy

Two fusion strategies are proposed to adaptively balance semantic and rhythmic branches:

**Weighted Fusion**: A gating network outputs a scalar weight $\alpha \in [0,1]$ depending on the diffusion timestep $t$:

$$\alpha = \sigma(f_{\text{gate}}(t)), \quad \mathbf{h}_{\text{fuse}} = \alpha \cdot \mathbf{h}_{\text{sem}} + (1-\alpha) \cdot \mathbf{h}_{\text{rhy}}$$

**FiLM Fusion**: Based on Feature-wise Linear Modulation, timestep-aware scaling and shifting parameters are generated for dimension-level modulation:

$$\text{FiLM}_{\text{sem}}(\mathbf{h}_{\text{sem}}) = \gamma_{\text{sem}}^t \cdot \mathbf{h}_{\text{sem}} + \beta_{\text{sem}}^t$$

The optimal setup is a combination of **Post-Attention FiLM + Feature Selection**.

### Loss & Training

**Diffusion Training Objective** (v-objective):

$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{t, \mathbf{z}_a^0, \epsilon}\left[\|\epsilon - G(\mathbf{z}_a^t, \mathbf{C}, t)\|_2^2\right]$$

**Scheduled Conditioning Training Strategy**: Addresses the training-inference gap (where ground-truth rhythm is used during training, and predicted rhythm is used during inference). A probability schedule is defined to progressively replace the ground truth with predicted rhythm:

$$p_{\text{pred}}(e) = \begin{cases} 0, & e < 10 \\ (e-10)/20, & 10 \leq e < 30 \\ 1, & e \geq 30 \end{cases}$$

The rhythm predictor and generator are **jointly trained** to ensure mutual adaptation. The optimizer is AdamW (lr = $1\times10^{-4}$), trained for 50 epochs on 2 A100 GPUs. Inference uses DDIM with 250 steps and a classifier-free guidance scale of 3.0.

## Key Experimental Results

### Main Results

**Table 4: Quantitative comparison with existing methods (Mixed Test Set)**

| Method | FAD↓ | FD↓ | KL↓ | Den.↑ | Cov.↑ | IB↑ |
|------|------|-----|-----|-------|-------|-----|
| CMT | 8.93 | 47.76 | 1.10 | 0.042 | 0.008 | 0.082 |
| MuMu-LLaMA | 2.84 | 27.12 | 1.25 | 0.107 | 0.090 | 0.145 |
| VidMuse | 3.44 | 21.04 | 0.94 | 0.150 | 0.130 | 0.180 |
| **Diff-V2M** | **1.52** | **10.96** | **0.86** | **0.376** | **0.399** | **0.181** |

**V2M-Bench (Out-of-Domain)**

| Method | FAD↓ | FD↓ | IB↑ |
|------|------|-----|-----|
| GVMGen | 2.15 | 21.55 | 0.203 |
| VidMuse | 2.59 | 22.03 | 0.196 |
| **Diff-V2M** | **1.76** | **22.02** | **0.197** |

Diff-V2M achieves leading performance across all metrics on the in-domain test set, while also showing the best generalizability out-of-domain. Subjective A/B testing indicates it outperforms all baselines.

### Ablation Study

**Table 5: Ablation study on training strategies**

| Configuration | FAD↓ | FD↓ | IB↑ |
|------|------|-----|-----|
| **Diff-V2M (Full)** | **1.52** | **10.96** | 0.181 |
| w/o Rhythmic Feature $C_r$ | 1.83 | 11.95 | 0.189 |
| w/o Emotional Feature $C_e$ | 1.68 | 12.89 | 0.181 |
| w/o Visual Rhythm | 2.22 | 13.61 | 0.180 |
| w/o Joint Training | 1.88 | 13.39 | 0.181 |
| w/o Scheduling Strategy | 1.62 | 10.67 | 0.186 |

### Key Findings

1. **ODF is the optimal rhythmic representation**: Compared to Mel-spectrograms and tempo maps, ODF is more compact and computationally effective.
2. **Post-Attention FiLM + Feature Selection is the optimal fusion strategy**: This configuration reduces FAD from 2.02 to 1.52.
3. **Removing visual rhythm (scene cuts + visual beats) has the largest impact**: This validates that fine-grained visual dynamics are critical for rhythm prediction.
4. **Scheduled Conditioning effectively mitigates the exposure bias (training-inference gap)**.
5. Removing rhythmic features paradoxically improves the IB score, indicating that ImageBind exhibits a preference for semantic alignment over rhythm.

## Highlights & Insights

1. **Explicit Rhythmic Modeling**: This work is the first to systematically compare multiple rhythm representations and demonstrate the superiority of ODF, establishing a standardized modeling solution for the V2M domain.
2. **Hierarchical Conditioning Design**: The two-layer architecture (Emotion $\rightarrow$ Semantics + Rhythm) avoids information entanglement while enabling flexible parameter manipulation for individual feature contributions.
3. **Timestep-Aware Fusion**: The FiLM mechanism allows the model to adaptively balance semantic and rhythmic features across different diffusion stages—focusing on semantics during the early steps and refining the rhythm in later stages.
4. **Joint Training + Scheduling**: Training the predictor and generator conjointly under Scheduled Conditioning offers an elegant, stable alternative to teacher forcing.

## Limitations & Future Work

1. **Limitations in Human Motion Scenes**: Relying on scene cuts and frame differences can neglect subtle human motion cues, resulting in sub-optimal rhythm alignment for human-centric videos.
2. **Lack of Explicit Style Control**: The framework does not currently support explicit style/genre or mood selection prompts.
3. **Limited Training Data Scale**: BGM909 contains only 909 piano tracks, and SymMV has approximately 79 hours—larger scale and higher diversity data can further unleash the model's potential.
4. Future plans include scaling to longer videos (> 30s) and multi-instrument arrangements.

## Related Work & Insights

- **VidMuse**: Also generates audio-level music but employs discrete token prediction, whereas this work utilizes continuous diffusion for a more natural generation process.
- **TiVA**: A pioneer in utilizing low-resolution Mel-spectrograms as audio layouts, which inspired this work to explore diverse rhythmic representations.
- **Stable Audio Open**: This work's generative backbone is based on its VAE+DiT architecture, proving the feasibility of adapting text-to-music pre-trained models to V2M tasks.
- Insight: Hierarchical conditional attention combined with timestep-aware fusion forms a robust and generic multi-conditional diffusion generation paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ (Explicit rhythmic modeling + hierarchical fusion)
- Technical Depth: ⭐⭐⭐⭐⭐ (Elaborate fusion strategies and training schedules)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive subjective/objective evaluations, out-of-domain testing, and ablation studies)
- Utility Value: ⭐⭐⭐⭐ (High demand for general video soundtrack generation)
- Overall Rating: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hierarchical Codec Diffusion for Video-to-Speech Generation](../../CVPR2026/audio_speech/hierarchical_codec_diffusion_for_video-to-speech_generation.md)
- [\[ICLR 2026\] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation](../../ICLR2026/audio_speech/synctrack_rhythmic_stability_and_synchronization_in_multi-track_music_generation.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)
- [\[CVPR 2026\] EchoFoley: Event-Centric Hierarchical Control for Video Grounded Creative Sound Generation](../../CVPR2026/audio_speech/echofoley_event-centric_hierarchical_control_for_video_grounded_creative_sound_g.md)

</div>

<!-- RELATED:END -->
