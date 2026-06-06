---
title: >-
  [Paper Note] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation
description: >-
  [ICLR 2026][Audio & Speech][multi-track music generation] SyncTrack is proposed with a unified architecture comprising track-shared modules (dual cross-track attention for rhythmic synchronization) and track-specific mod…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "multi-track music generation"
  - "rhythmic synchronization"
  - "diffusion models"
  - "cross-track attention"
  - "evaluation metrics"
date: 2026-05-08
content_hash: 312d36a07356290a
---

# SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation

**Conference**: ICLR 2026
**arXiv**: [2603.01101](https://arxiv.org/abs/2603.01101)  
**Code**: [https://synctrack-v1.github.io](https://synctrack-v1.github.io)  
**Area**: Music Generation / Audio
**Keywords**: multi-track music generation, rhythmic synchronization, diffusion models, cross-track attention, evaluation metrics

## TL;DR
SyncTrack is proposed with a unified architecture comprising track-shared modules (dual cross-track attention for rhythmic synchronization) and track-specific modules (learnable instrument priors for timbre preservation), along with three new rhythmic consistency evaluation metrics (IRS/CBS/CBD), achieving substantial improvements in multi-track music generation quality (FAD: 6.55→1.26, subjective MOS: 3.42 vs. 1.57).

## Background & Motivation

**Background**: Multi-track music generation enables independent control over individual instrument tracks (mixing, re-arrangement); methods such as MSDM and MSG-LD employ diffusion models to learn the joint distribution of multiple tracks.

**Limitations of Prior Work**: Existing methods treat multi-track generation as multivariate time-series or video generation, over-emphasizing inter-track differences while neglecting shared rhythmic structure—resulting in rhythmic instability and inter-track desynchronization. MSDM achieves an FAD of 6.55 and a subjective score of only 1.57/5.0.

**Key Challenge**: Rhythmic information is shared across tracks (all instruments follow the same beat), whereas timbre information is track-independent (bass is low-pitched, piano is bright)—these two types of information must be handled separately.

**Core Idea**: Track-shared modules (shared rhythm) + track-specific modules (independent timbre) + novel evaluation metrics.

## Method

### Overall Architecture
Built upon a latent diffusion model (LDM): multi-track audio → STFT+Mel → VAE-encoded latent representation $z^s \in \mathbb{R}^{C \times T \times F}$ → noising/denoising diffusion (U-Net with track-shared and track-specific modules) → VAE decoding → HiFi-GAN waveform synthesis. Each track (bass, drums, guitar, piano) is generated in parallel.

### Key Designs

1. **Track-Shared Module**:

    - Comprises ResBlock, intra-track attention, global cross-track attention, and time-specific cross-track attention.
    - **Global Cross-Track Attention**: each element $z_{t,f}^s$ attends to all time-frequency positions across all tracks → maintains global rhythmic consistency and stable beat framework.
    - **Time-Specific Cross-Track Attention**: at the same timestep $t$, attention is applied over the frequency dimension across all tracks → achieves fine-grained beat alignment and chord synchronization.
    - **Design Motivation**: rhythmic information is shared across tracks; the two attention mechanisms handle "global rhythmic framework" and "beat-level synchronization" respectively.

2. **Track-Specific Module**:

    - **Learnable Instrument Prior**: one-hot track identifier → positional encoding → two-layer MLP → added to ResBlock output.
    - **Design Motivation**: timbre and pitch range are unique to each track and require independent modeling.

3. **Three New Evaluation Metrics**:

    - **IRS (Inner-track Rhythmic Stability)**: mean standard deviation of intra-track beat intervals; measures rhythmic stability (↓ better).
    - **CBS (Cross-track Beat Synchronization)**: proportion of cross-track beat alignment computed via a sliding tolerance window (↑ better).
    - **CBD (Cross-track Beat Dispersion)**: normalized mean of cross-track beat errors (↓ better).

### Loss & Training
Standard DDPM noise prediction objective $\|\epsilon - \epsilon_\theta(\{z_l^s\}, l)\|^2$. Initialized from MusicLDM pretrained weights; trained for 320K steps, batch size 16, with DDIM 200-step inference.

## Key Experimental Results

### Main Results

| Method | FAD↓ | IRS↓ | CBS↑ | CBD↓ |
|--------|------|------|------|------|
| MSDM | 6.55 | 0.167 | 0.428 | 0.156 |
| MSG-LD | 1.31 | 0.148 | 0.434 | 0.147 |
| **SyncTrack** | **1.26** | **0.125** | **0.487** | **0.120** |
| Ground Truth | - | 0.049 | 0.592 | 0.079 |

SyncTrack achieves the closest performance to Ground Truth across all metrics; improvements in IRS and CBS particularly indicate significant gains in rhythmic consistency.

### Ablation Study

| Configuration | FAD↓ | IRS↓ | CBS↑ | Note |
|---------------|------|------|------|------|
| Full SyncTrack | 1.26 | 0.125 | 0.487 | Complete |
| w/o global cross-track attn | ~1.8 | ~0.14 | ~0.45 | Global rhythm degraded |
| w/o time-specific cross-track attn | ~1.6 | ~0.13 | ~0.46 | Fine-grained alignment degraded |
| w/o instrument prior | ~1.4 | ~0.13 | ~0.47 | Timbre discriminability reduced |

### Key Findings
- FAD reduced from 6.55 (MSDM) to 1.26 (−80%), and from 1.31 (MSG-LD) to 1.26 (−3.8%).
- Per-track FAD shows the largest improvement on Piano (MSG-LD: 2.04→1.11, −45.6%), attributable to the piano's wide pitch range and complex notation.
- IRS reduced from MSG-LD's 0.148 to 0.125; CBS improved from 0.434 to 0.487—indicating significant gains in rhythmic consistency.
- In subjective evaluation, SyncTrack mix MOS: 3.42 vs. MSG-LD: 1.57 vs. GT: 4.48.
- Ablation: global cross-track attention contributes most to global rhythmic stability; time-specific attention contributes most to beat-level synchronization.
- The three proposed metrics correlate strongly with human subjective preferences—temporal structure undetectable by FAD is effectively quantified by IRS/CBS/CBD.

## Highlights & Insights
- The **shared/specific module separation** design principle is generalizable to other multi-channel generation tasks—applicable to any multi-channel scenario exhibiting "shared attributes + independent attributes."
- **Rhythmic evaluation metrics fill a gap**—FAD cannot capture temporal structure; IRS/CBS/CBD provide complementary dimensions.
- The division of labor between the two cross-track attention mechanisms is clear: global cross-track → beat framework; time-specific cross-track → beat-level synchronization.
- The learnable instrument prior requires only one-hot encoding + positional encoding + MLP, making it extremely lightweight yet effective for timbre discrimination.
- The large gap in subjective evaluation (3.42 vs. 1.57) demonstrates that rhythmic synchronization is a core factor in human auditory perception.
- Initialization from MusicLDM pretrained weights effectively leverages existing audio generation knowledge, accelerating convergence.

## Limitations & Future Work
- Validation is limited to Slakh2100 (4 tracks: bass/drums/guitar/piano); more complex ensembles (orchestral, choral) and larger track counts remain untested.
- Conditional generation (text- or melody-guided) is unexplored—the current model is unconditional.
- The IRS/CBS/CBD metrics rely on beat detection algorithms, whose accuracy affects metric reliability.
- The global cross-track attention in the track-shared module may become a computational bottleneck as the number of tracks increases.
- The generated audio sample rate (16 kHz) is well below commercial standards (44.1 kHz); performance at higher sample rates remains to be verified.

## Related Work & Insights
- **vs. MSDM**: MSDM learns the joint multi-track distribution with a unified model without distinguishing shared/specific information; reducing FAD from 6.55 to 1.26 represents a substantial improvement.
- **vs. MSG-LD**: MSG-LD is stronger but still neglects rhythmic synchronization; SyncTrack comprehensively outperforms it on IRS/CBS/CBD.
- **vs. StemGen/JEN-1 Composer**: these methods employ Transformer/LDM but lack explicit cross-track synchronization mechanisms.
- **Insights**: the shared/specific module separation principle applies to all multi-channel generation tasks (e.g., multi-speaker speech generation, multi-instrument arrangement).

## Rating
- Novelty: ⭐⭐⭐⭐ Shared/specific module separation + dual cross-track attention + novel rhythmic evaluation metrics
- Experimental Thoroughness: ⭐⭐⭐⭐ Objective + subjective evaluation + ablation study, with comprehensive metric design
- Writing Quality: ⭐⭐⭐⭐ Motivation clearly articulated, figures intuitive
- Value: ⭐⭐⭐⭐ Fills the gap in multi-track music rhythmic evaluation; model design principles are generalizable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](../../ACL2026/audio_speech/jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[NeurIPS 2025\] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization](../../NeurIPS2025/audio_speech/unifying_symbolic_music_arrangement_track-aware_reconstruction_and_structured_to.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](../../ACL2026/audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)
- [\[ICLR 2026\] Toward Complex-Valued Neural Networks for Waveform Generation](toward_complex-valued_neural_networks_for_waveform_generation.md)
- [\[ICLR 2026\] Discovering and Steering Interpretable Concepts in Large Generative Music Models](discovering_and_steering_interpretable_concepts_in_large_generative_music_models.md)

</div>

<!-- RELATED:END -->
