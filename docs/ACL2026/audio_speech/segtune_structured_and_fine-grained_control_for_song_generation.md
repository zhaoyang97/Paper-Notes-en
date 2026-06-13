---
title: >-
  [Paper Note] SegTune: Structured and Fine-Grained Control for Song Generation
description: >-
  [ACL 2026][Audio & Speech][Song generation] Proposes SegTune, a song generation framework based on Diffusion Transformer, which achieves fine-grained temporal control over song structures and musical attributes through h…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Song generation"
  - "segmented control"
  - "Diffusion Transformer"
  - "duration prediction"
  - "hierarchical conditioning"
date: 2026-05-08
content_hash: 4a00592716abd291
---

# SegTune: Structured and Fine-Grained Control for Song Generation

**Conference**: ACL 2026 Best Paper Oral  
**arXiv**: [2606.02638](https://arxiv.org/abs/2606.02638)  
**Code**: To be confirmed  
**Area**: audio_speech  
**Keywords**: Song generation, segmented control, Diffusion Transformer, duration prediction, hierarchical conditioning

## TL;DR
Proposes SegTune, a song generation framework based on Diffusion Transformer, which achieves fine-grained temporal control over song structures and musical attributes through hierarchical text conditions (global + segment-level prompts) and an LLM duration predictor.

## Background & Motivation
**Background**: Neural song generation has achieved high-quality audio synthesis from lyrics and global text prompts. However, existing systems (AR such as YuE/LeVo and NAR such as DiffRhythm/ACE-Step) primarily rely on global control signals.

**Limitations of Prior Work**: (1) Global prompts fail to capture the temporal dynamics of a song (e.g., evolution of instrumentation, emotion, and energy across sections), leading to homogenized outputs; (2) Simultanously generating vocals and accompaniments under global conditions imposes a significant coordination burden on the model; (3) the lack of fine-grained control limits the expressive flexibility for creators.

**Key Challenge**: NAR models compress composition and rendering into a single diffusion process, making it difficult to simultaneously optimize music structure, temporal coherence, and vocal-instrument balance; furthermore, existing methods depend on low-quality lyric duration annotations (manual or zero-shot LLM generation).

**Goal**: To introduce segment-level fine-grained control capabilities into NAR song generation while eliminating the dependence on manual lyric duration annotations.

**Key Insight**: Textual prompts are divided into global and segment levels. Segment prompts are temporally broadcast to corresponding time windows, and a fine-tuned LLM automatically predicts sentence-level timestamps.

**Core Idea**: Hierarchical segment conditioning injection + LLM-based duration predictor = structured fine-grained controllable song generation.

## Method

### Overall Architecture
SegTune adopts the DiT (Diffusion Transformer) architecture based on Conditional Flow Matching (CFM). A 1D VAE compresses 44kHz raw audio into a latent sequence at 21.5Hz. The model receives conditions from three complementary sources: global text prompts, segment-level text prompts, and time-aligned lyrics. An LLM duration predictor generates sentence-level timestamps for segment prompt broadcasting and lyric alignment.

### Key Designs
1. **Hierarchical Segment Text Conditioning**:

    - Function: Decouples global style consistency from local musical variations.
    - Mechanism: Global prompts are encoded by Qwen3-Embedding-0.6B and broadcast to all frames; segment prompts are encoded into vectors $\mathbf{e}_s^i \in \mathbb{R}^{1 \times d_s}$ and broadcast only to frames within the corresponding time window; both are concatenated along the channel dimension and mapped to the final condition $E_{\text{text}} \in \mathbb{R}^{T \times 1024}$ via a 3-layer MLP.
    - Design Motivation: Musical elements like instrumentation, emotion, and rhythm naturally evolve across different sections, which global prompts cannot express.

2. **LLM-based Duration Predictor**:

    - Function: Automatically generates sentence-level lyric timestamps (LRC format), eliminating the need for manual annotation.
    - Mechanism: A fine-tuned Qwen3-4B-Base model takes lyrics and hierarchical prompts as input to autoregressively output LRC-formatted timestamps; it is efficiently trained using LoRA (rank=32) over 8 epochs on >100k LRC data.
    - Design Motivation: Previous NAR methods required either error-prone manual timestamps or fragile zero-shot LLM prompts to generate word-level timings.

3. **Large-scale Data Pipeline (Three Stages)**:

    - Function: Constructs high-quality training data containing aligned lyrics and hierarchical prompts.
    - Mechanism: (1) Quality filtering—metadata filtering + Audiobox/SongEval aesthetic scoring; (2) Lyric processing—Demucs v4 for vocal separation + FireRedASR/Whisper for transcription + LRC verification; (3) Hierarchical prompt labeling—Audio Flamingo 3 for generating global/segment prompts.
    - Design Motivation: High-quality aligned data is the foundation of segment-level control capabilities.

### Loss & Training
- Conditional Flow Matching (CFM) loss: $\mathcal{L} = \mathbb{E}_{t,q,p} \| v_\theta(t,C,x_t) - u(x_t|x_0,x_1) \|^2$
- Three-stage training: (i) Pre-training (~370k songs, ~27k hours, 20 epochs); (ii) Fine-tuning (~50k songs, ~4k hours, 8 epochs); (iii) Preference alignment (2 iterations of DPO, ~20k pairs per round).
- Inference uses the Euler ODE solver with negative condition CFG: cfg=3, cfg_n=1.
- Global and segment conditions each have 20% dropout to support CFG.

## Key Experimental Results

### Main Results

| Model | PER↓ | AudioBox-CE↑ | SongEval-OM↑ | G-Mulan↑ | Gender Acc↑ | Age Acc↑ |
|------|------|-------------|-------------|---------|------------|---------|
| YuE | 48.5% | 7.16 | 3.22 | 0.29 | 80.7% | 44% |
| LeVo | 29.8% | 7.43 | 3.35 | 0.32 | 90.6% | 50% |
| DiffRhythm++ | 27.4% | 7.55 | 3.76 | 0.47 | 37.5% | 54% |
| ACE-Step | 35.6% | 7.38 | 3.74 | 0.35 | 78.1% | 56% |
| **SegTune-SFT** | **14.5%** | 7.38 | 3.19 | 0.47 | **96.7%** | 57% |
| **SegTune-DPO** | 18.5% | **7.63** | **3.97** | 0.46 | 81.0% | 51% |

### Ablation Study (Prompt Encoder Design)

| Global Encoder | Segment Encoder | G-Mulan↑ | S-Mulan↑ | Gender Acc↑ | SongEval-OM↑ |
|-----------|-----------|---------|---------|------------|-------------|
| MuQ | – | 0.39 | 0.30 | 47.6% | 2.86 |
| Qwen3-Emb | – | 0.40 | 0.33 | 92.2% | 3.12 |
| Qwen3-Emb(G) + MuQ(S) | Concat | 0.44 | 0.37 | 84.4% | 3.34 |
| **Qwen3-Emb + Qwen3-Emb** | **Concat** | **0.47** | **0.38** | **96.7%** | **3.19** |

### Key Findings
- The PER of SegTune-SFT is only 14.5%, significantly lower than all baselines (lowest baseline DiffRhythm++ is 27.4%), indicating superior lyric fidelity and vocal intelligibility.
- Segment-level prompt injection significantly improves instruction-following: S-Mulan increased from 0.33 to 0.38 and Gender accuracy from 92.2% to 96.7% after adding the segment encoder.
- DPO fine-tuning improves musicality (MOS 4.57±0.52), but leads to a decline in gender/age control accuracy due to bias in preference data (dominated by young female voices).
- Subjective MOS evaluation: SegTune-DPO achieved the highest score in musicality at 4.57±0.52 (lowest std dev) and quality at 3.87±0.56 (second highest, lowest std dev).

## Highlights & Insights
- Introduced explicit segment-level text conditioning into NAR song generation for the first time, achieving fine-grained temporal control of musical attributes.
- The LLM duration predictor is an elegant engineering design: fine-tuning Qwen3-4B as an LRC format generator completely eliminates the need for manual timestamps.
- The three-stage training (Pre-training → Fine-tuning → DPO) combined with the data cleaning pipeline forms a complete engineering loop.
- Qwen3-Embedding as a prompt encoder outperforms the music-specific MuQ-MuLan in instruction-following, suggesting that semantic understanding is crucial for controllable generation.

## Limitations & Future Work
- Instruction-following capabilities (gender/age) declined after DPO; the issue of preference data bias remains to be solved, possibly through online policy optimization (dynamic penalties for attribute bias).
- Training data is dominated by Chinese pop songs (>90%), so cross-lingual and cross-genre generalization needs further verification.
- Currently only supports sentence-level duration prediction; finer-grained word-level/phoneme-level control has not been explored.
- Internal datasets and some models are not public, limiting reproducibility.

## Related Work & Insights
- While NAR methods like DiffRhythm / ACE-Step / JAM accelerate generation, they lack fine-grained control; SegTune's segment conditioning paradigm can be generalized to other NAR frameworks.
- Music ControlNet introduced time-varying control signals but was limited to instrumental music; SegTune extends this to full songs (vocals + accompaniment).
- The idea of an LLM duration predictor can inspire other multi-modal generation tasks that require temporal alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ Segment-level text conditioning and LLM duration predictor are innovative designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive objective metrics (PER/AudioBox/SongEval/MuLan/Attribute accuracy), including ablations and subjective MOS.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with a complete logical chain from motivation to method and then to experiments.
- Value: ⭐⭐⭐⭐ Addresses the core problem of lacking fine-grained control in NAR song generation with a complete engineering closed-loop.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)
- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](../../ICML2026/audio_speech/mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](../../NeurIPS2025/audio_speech/levo_high-quality_song_generation_with_multi-preference_alignment.md)
- [\[NeurIPS 2025\] Segment-Factorized Full-Song Generation on Symbolic Piano Music](../../NeurIPS2025/audio_speech/segment-factorized_full-song_generation_on_symbolic_piano_music.md)

</div>

<!-- RELATED:END -->
