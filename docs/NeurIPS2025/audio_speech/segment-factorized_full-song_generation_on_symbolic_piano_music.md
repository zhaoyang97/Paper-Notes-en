---
title: >-
  [Paper Note] Segment-Factorized Full-Song Generation on Symbolic Piano Music
description: >-
  [NeurIPS 2025 (AI for Music Workshop)][Audio & Speech][symbolic music generation] This paper proposes the Segmented Full-Song (SFS) model, which decomposes a song into segments and autoregressively generates each segment by attending selectively to structurally relevant context. SFS achieves faster and more structurally coherent full-song piano generation compared to existing methods, while supporting interactive human-AI co-creation.
tags:
  - NeurIPS 2025 (AI for Music Workshop)
  - "Audio & Speech"
  - symbolic music generation
  - full-song generation
  - Transformer
  - structure modeling
  - human-AI co-creation
date: 2026-05-08
content_hash: 5ebba5739c753f43
---

# Segment-Factorized Full-Song Generation on Symbolic Piano Music

**Conference**: NeurIPS 2025 (AI for Music Workshop)
**arXiv**: [2510.05881](https://arxiv.org/abs/2510.05881)
**Code**: [Available](https://github.com/eri24816/segmented-full-song-gen)
**Area**: Music Generation / Object Detection (classification label)
**Keywords**: symbolic music generation, full-song generation, Transformer, structure modeling, human-AI co-creation

## TL;DR
This paper proposes the Segmented Full-Song (SFS) model, which decomposes a song into segments and autoregressively generates each segment by attending selectively to structurally relevant context. SFS achieves faster and more structurally coherent full-song piano generation compared to existing methods, while supporting interactive human-AI co-creation.

## Background & Motivation
Full-song generation from symbolic music is a highly challenging task: models must not only generate long sequences efficiently but also maintain global structural coherence. Human composers typically conceive themes and high-level structure first, place thematic material, and then fill in the remaining sections—a process that is only partially autoregressive, with composers referencing the most relevant context rather than revisiting the entire piece. The existing method WholeSong employs a four-stage coarse-to-fine diffusion approach, but its diffusion backbone is inefficient and does not incorporate high-level concepts such as themes or motifs.

## Method

### Overall Architecture
SFS takes a user-provided song structure $(\\hat{s}_{1:M}, \\hat{e}_{1:M}, \\hat{l}_{1:M})$ and optional seed segments as input, factorizing the joint probability of the full song into a product of segment-level conditional probabilities. Segments can be generated in any order, with each segment attending only to structurally relevant context.

### Key Designs

**Segment Factorization and Selective Attention**: Four context types are defined:
- **Left**: the nearest already-generated segment to the left of the target segment (for smooth continuation)
- **Right**: the nearest already-generated segment to the right of the target segment (for smooth connection)
- **Seed**: the seed segment carrying the main musical idea of the song (global style anchor)
- **Ref**: a reference segment sharing the same label as the target (structural consistency)

Each context type is truncated to at most 8 measures. The model attends to these four context types at the token level, while all generated segments are encoded into a compact representation via a global visual module $G$.

**Global Visual Encoder**: A pretrained VAE encoder converts each generated measure into a measure-level embedding, providing a coarse overview of the song's overall content.

**Generator**: A Transformer decoder that integrates global visual outputs via in-attention and receives token-level information from the four context types through an encoder.

**Frame-Based Music Representation**: Notes are quantized to 1/8-beat frames. Each frame consists of one frame token followed by note tokens (ordered by ascending pitch); each note token is generated sequentially from three sub-tokens: pitch, velocity, and duration.

**Automatic Segmentation Algorithm**: An unsupervised song structure annotation method based on spectral clustering—a measure similarity matrix is computed, adjacency regularization encourages consecutive label assignments, and k-means clustering on spectral embeddings yields segment labels and boundaries.

### Loss & Training
- Negative log-likelihood (NLL) loss, summed (not averaged) over tokens
- Adam optimizer with learning rate decaying exponentially from $1\times10^{-4}$ to $5\times10^{-6}$
- 2 million training steps, 127 hours, single RTX 4090, batch size 12
- All permutations are trained over during training; inference adapts to any user-specified order
- The seed segment is selected as the segment closest to the song's midpoint among the most frequently occurring label

## Key Experimental Results

### Objective Evaluation (Structure Index SI)

| Model | SI₂₋₈ | SI₈₋₁₆ | SI₁₆₊ | Inference Speed |
|-------|-------|--------|-------|----------------|
| Flat | 0.3426 | 0.1990 | 0.0409 | 5.68 beat/sec |
| WholeSong | 0.3234 | 0.2262 | 0.0860 | 0.197 beat/sec |
| **SFS (Ours)** | **0.3286** | **0.2264** | **0.1109** | **2.03 beat/sec** |
| Dataset | 0.4398 | 0.3827 | 0.3300 | - |

### User Study (44 participants, 5-point scale)

| Model | Overall Quality (O) | Adherence to Seed (A) |
|-------|:-------------------:|:---------------------:|
| Flat | **3.36** | 2.34 |
| WholeSong | 3.02 | 3.16 |
| **SFS (Ours)** | 3.14 | **3.59** |
| Dataset | 4.00 | 4.07 |

### Key Findings
- SFS achieves the largest advantage in long-range structural coherence (SI₁₆₊: 0.1109 vs. WholeSong 0.0860), validating the effectiveness of the structural modeling approach
- SFS significantly outperforms WholeSong in seed adherence (3.59 vs. 3.16)
- SFS is approximately 10× faster than WholeSong (2.03 vs. 0.197 beat/sec), enabling real-time streaming output
- The Flat model receives slightly higher overall quality scores due to the fluency afforded by purely forward generation
- A notable gap from real data remains (O: 3.14 vs. 4.00), indicating substantial room for improvement in full-song generation
- Segment-level correspondence is satisfactory, but smooth phrase transitions and progressive full-song development are lacking

## Highlights & Insights
1. **Modeling the human compositional process**: The design of theme-first generation followed by infilling, with selective attention to relevant context, aligns with human creative intuition
2. **Substantial efficiency gains**: Autoregressive Transformer with segment factorization vs. the full diffusion process per segment in WholeSong
3. **Flexible generation order**: Support for arbitrary generation ordering enables interactive human-AI co-creation workflows
4. **Web interface**: An interactive composition tool featuring a structure editor and piano roll editor

## Limitations & Future Work
- Generated music sometimes lacks smooth phrase transitions and emotional development
- Segments are thematically consistent but loosely connected; high-level planning mechanisms to guide phrase-level development are needed
- The automatic segmentation algorithm cannot provide semantic labels (e.g., verse/chorus)
- Consecutive repeated segments cannot be identified

## Related Work & Insights
- Compared to WholeSong's diffusion-based approach, autoregressive generation with selective attention constitutes a more efficient paradigm for full-song modeling
- The seed-conditioning design is generalizable to "thematic consistency" control in other generation tasks
- The training strategy of learning over arbitrary generation orders has broader applicability

## Rating
- Novelty: ⭐⭐⭐⭐ (segment factorization with selective attention for full-song modeling)
- Technical Depth: ⭐⭐⭐⭐ (careful design of segmentation algorithm, positional encoding, and training strategy)
- Experimental Thoroughness: ⭐⭐⭐⭐ (objective metrics + user study with 44 participants)
- Value: ⭐⭐⭐⭐⭐ (real-time generation + interactive web tool + open source)

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](../../ACL2026/audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)
- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](levo_high-quality_song_generation_with_multi-processing_refined_supervision.md)
- [\[NeurIPS 2025\] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization](unifying_symbolic_music_arrangement_track-aware_reconstruction_and_structured_to.md)
- [\[NeurIPS 2025\] From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era](from_generation_to_attribution_music_ai_agent_architectures_for_the_post-streami.md)
- [\[NeurIPS 2025\] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)

<!-- RELATED:END -->
