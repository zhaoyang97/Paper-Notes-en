---
title: >-
  [Paper Note] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation
description: >-
  [AAAI 2026][Audio & Speech][Video-to-music generation] This paper proposes Diff-V2M, a hierarchical conditional diffusion Transformer framework for video-to-music generation that integrates affective, semantic…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Video-to-music generation"
  - "diffusion model"
  - "rhythmic modeling"
  - "hierarchical cross-attention"
  - "feature fusion"
date: 2026-05-08
content_hash: 33bba60f58755df3
---

# Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation

**Conference**: AAAI 2026  
**arXiv**: [2511.09090](https://arxiv.org/abs/2511.09090)  
**Code**: [Demo](https://Tayjsl97.github.io/Diff-V2M-Demo/)  
**Area**: Image Generation  
**Keywords**: Video-to-music generation, diffusion model, rhythmic modeling, hierarchical cross-attention, feature fusion

## TL;DR

This paper proposes Diff-V2M, a hierarchical conditional diffusion Transformer framework for video-to-music generation that integrates affective, semantic, and rhythmic features via explicit rhythmic modeling (low-resolution ODF) and a hierarchical cross-attention mechanism, achieving state-of-the-art performance on both in-domain and out-of-domain datasets.

## Background & Motivation

Video-to-music generation (V2M) aims to compose background music that matches the visual content of a given video. With the rapid growth of video platforms such as YouTube and TikTok, and the emergence of video generation models such as Sora and Veo, the demand for personalized audio-visual content has surged dramatically. Existing methods face two core challenges:

**Lack of explicit rhythmic modeling**: Prior methods implicitly learn the mapping from visual dynamics to musical rhythm via optical flow, frame differences, or frame-level visual features, failing to achieve precise audio-visual temporal alignment. LLM-based approaches that translate video into text prompts also struggle to preserve fine-grained temporal dynamics.

**Difficulty in multi-perspective feature fusion**: Videos contain diverse visual features including affective, semantic, and rhythmic cues. Progressive fusion strategies incur additional computational overhead, while simple concatenation fails to capture inter-feature dependencies.

## Method

### Overall Architecture

Diff-V2M consists of two core modules:

- **Visual feature extraction module**: Extracts semantic features (CLIP), affective features (color histograms), and rhythmic features (low-resolution ODF + rhythm predictor).
- **Conditional music generation module**: A latent diffusion model based on DiT, equipped with hierarchical cross-attention and a timestep-aware fusion strategy.

The audio sampling rate is 44.1 kHz and the video frame rate is 1 FPS. A frozen VAE from Stable Audio Open encodes waveforms into latent representations, while the DiT is trained from scratch.

### Key Designs

#### 1. Generalizable Rhythmic Representation

Three rhythmic representation schemes are systematically compared:

- **Low-resolution mel spectrogram** ($\text{Mel}_{LR}$): The original mel spectrogram is normalized and downsampled to a target resolution of $[M, C]$ ($C=16$).
- **Low-resolution tempogram** ($\text{Tem}_{LR}$): A time-tempo representation capturing the evolution of local beat patterns over time ($B=16$).
- **Low-resolution ODF** ($\text{ODF}_{LR}$): An onset detection function, a one-dimensional temporal sequence encoding the intensity of note onsets. Peak detection is applied to the original ODF to construct a second-level vector ($d=1$).

Experiments demonstrate that **ODF is the most effective rhythmic representation**—more compact than the mel spectrogram and tempogram, and directly emphasizing key rhythmic events.

#### 2. Rhythm Predictor

Since audio is unavailable at inference time, a decoder-only Transformer is trained to predict the rhythmic representation from video. The input comprises three components:

- **CLIP semantic features** $C_\mathbf{s}$
- **Scene transition embeddings** $\mathbf{e} \in \{0,1\}^M$: Scene boundaries detected via PySceneDetect.
- **Visual beat vector** $\mathbf{v} \in \mathbb{R}^M$: Peak detection results aggregated from inter-frame differences.

These are summed and fed into the predictor: $\mathbf{X} = C_\mathbf{s} + \text{Embed}(\mathbf{e}) + \text{Linear}(\mathbf{v})$

#### 3. Hierarchical Cross-Attention Conditioning Module

A two-level hierarchical structure is adopted to integrate multi-perspective features:

- **First level**: Affective features shape the overall emotional tone via cross-attention.
- **Second level**: Semantic and rhythmic features are processed independently through **parallel cross-attention**, preventing feature entanglement.

#### 4. Timestep-Aware Fusion Strategy

Two fusion methods are proposed to adaptively balance the semantic and rhythmic branches:

**Weighted fusion**: A gating network outputs a scalar weight $\alpha \in [0,1]$ conditioned on the diffusion timestep $t$:

$$\alpha = \sigma(f_{\text{gate}}(t)), \quad \mathbf{h}_{\text{fuse}} = \alpha \cdot \mathbf{h}_{\text{sem}} + (1-\alpha) \cdot \mathbf{h}_{\text{rhy}}$$

**FiLM fusion**: Based on Feature-wise Linear Modulation, this generates timestep-aware scaling and shift parameters for dimension-wise modulation:

$$\text{FiLM}_{\text{sem}}(\mathbf{h}_{\text{sem}}) = \gamma_{\text{sem}}^t \cdot \mathbf{h}_{\text{sem}} + \beta_{\text{sem}}^t$$

The optimal configuration is the combination of **Post-Attention FiLM + Feature Selection**.

### Loss & Training

**Diffusion training objective** (v-objective):

$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{t, \mathbf{z}_a^0, \epsilon}\left[\|\epsilon - G(\mathbf{z}_a^t, \mathbf{C}, t)\|_2^2\right]$$

**Scheduled Conditioning training strategy**: Addresses the train-inference discrepancy (GT rhythm used during training vs. predicted rhythm at inference). A probability schedule progressively replaces GT rhythm with predicted rhythm:

$$p_{\text{pred}}(e) = \begin{cases} 0, & e < 10 \\ (e-10)/20, & 10 \leq e < 30 \\ 1, & e \geq 30 \end{cases}$$

The rhythm predictor and the generator are **jointly trained** to ensure co-adaptation. The optimizer is AdamW (lr=$1\times10^{-4}$), trained for 50 epochs on 2 A100 GPUs. Inference uses 250-step DDIM with a classifier-free guidance scale of 3.0.

## Key Experimental Results

### Main Results

**Table 4: Quantitative comparison with existing methods (Mixed Test Set)**

| Method | FAD↓ | FD↓ | KL↓ | Den.↑ | Cov.↑ | IB↑ |
|--------|------|-----|-----|-------|-------|-----|
| CMT | 8.93 | 47.76 | 1.10 | 0.042 | 0.008 | 0.082 |
| MuMu-LLaMA | 2.84 | 27.12 | 1.25 | 0.107 | 0.090 | 0.145 |
| VidMuse | 3.44 | 21.04 | 0.94 | 0.150 | 0.130 | 0.180 |
| **Diff-V2M** | **1.52** | **10.96** | **0.86** | **0.376** | **0.399** | **0.181** |

**V2M-Bench (out-of-domain)**

| Method | FAD↓ | FD↓ | IB↑ |
|--------|------|-----|-----|
| GVMGen | 2.15 | 21.55 | 0.203 |
| VidMuse | 2.59 | 22.03 | 0.196 |
| **Diff-V2M** | **1.76** | **22.02** | **0.197** |

Diff-V2M achieves comprehensive superiority on the in-domain test set and attains the best overall performance on the out-of-domain benchmark. It also outperforms all baselines in subjective A/B testing.

### Ablation Study

**Table 5: Ablation of training strategies**

| Configuration | FAD↓ | FD↓ | IB↑ |
|---------------|------|-----|-----|
| **Diff-V2M (full)** | **1.52** | **10.96** | 0.181 |
| w/o rhythmic feature $C_r$ | 1.83 | 11.95 | 0.189 |
| w/o affective feature $C_e$ | 1.68 | 12.89 | 0.181 |
| w/o visual rhythm | 2.22 | 13.61 | 0.180 |
| w/o joint training | 1.88 | 13.39 | 0.181 |
| w/o scheduling strategy | 1.62 | 10.67 | 0.186 |

### Key Findings

1. **ODF is the best rhythmic representation**: It is more compact and effective than the mel spectrogram and tempogram.
2. **Post-Attention FiLM + Feature Selection is the optimal fusion strategy**: FAD decreases from 2.02 to 1.52.
3. **Removing visual rhythm (scene transitions + visual beats) causes the largest performance drop**: This highlights the critical importance of fine-grained visual dynamics for rhythm prediction.
4. **Scheduled Conditioning effectively mitigates the train-inference gap**.
5. Removing rhythmic features yields a higher IB score, as ImageBind tends to favor semantic alignment.

## Highlights & Insights

1. **Explicit rhythmic modeling**: This work is the first to systematically compare multiple rhythmic representations and demonstrate the superiority of ODF, providing a standardized rhythmic modeling solution for V2M research.
2. **Hierarchical conditioning design**: The two-level architecture (affective → semantic + rhythmic) avoids feature entanglement while allowing flexible adjustment of each feature's contribution.
3. **Timestep-aware fusion**: The FiLM mechanism enables the model to adaptively shift the balance between semantic and rhythmic features across diffusion stages—prioritizing semantics in early steps and rhythm in later steps.
4. **Joint training + scheduling**: Simultaneously training the predictor and generator with Scheduled Conditioning constitutes an elegant solution to the teacher-forcing problem.

## Limitations & Future Work

1. **Limited performance on human-centric scenes**: Reliance on scene cuts and inter-frame differences may miss subtle motion cues, leading to imprecise rhythmic alignment in videos centered on human motion.
2. **Lack of explicit style control**: The framework does not support control over music genre, mood, or other stylistic attributes.
3. **Limited training data scale**: BGM909 contains only 909 piano pieces and SymMV covers approximately 79 hours—larger-scale data could further improve performance.
4. The approach is extensible to longer videos (>30s) and multi-instrument arrangement scenarios.

## Related Work & Insights

- **VidMuse**: Also generates audio-level music but employs discrete token prediction, whereas the continuous diffusion approach in this work yields more natural outputs.
- **TiVA**: Pioneered the use of low-resolution mel spectrograms as audio layout priors, inspiring this work's exploration of more effective rhythmic representations.
- **Stable Audio Open**: This work builds upon its VAE+DiT architecture, demonstrating the feasibility of adapting text-to-music models for V2M.
- Insight: Hierarchical conditional attention combined with timestep-aware fusion constitutes a general paradigm for multi-conditional diffusion generation.

## Rating

- Novelty: ⭐⭐⭐⭐ (explicit rhythmic modeling + hierarchical fusion)
- Technical Depth: ⭐⭐⭐⭐⭐ (sophisticated fusion strategies and training schedule design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (comprehensive objective/subjective evaluation, out-of-domain testing, and ablations)
- Practical Value: ⭐⭐⭐⭐ (broad demand for general-purpose video background music generation)
- Overall Score: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation](../../ICLR2026/audio_speech/synctrack_rhythmic_stability_and_synchronization_in_multi-track_music_generation.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)
- [\[ACL 2026\] ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling](../../ACL2026/audio_speech/controlaudio_tackling_text-guided_timing-indicated_and_intelligible_audio_genera.md)
- [\[ACL 2026\] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions](../../ACL2026/audio_speech/unisonate_a_unified_model_for_speech_music_and_sound_effect_generation_with_text.md)

</div>

<!-- RELATED:END -->
