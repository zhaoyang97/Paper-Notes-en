---
title: >-
  [Paper Note] Semantic Satellite Communications for Synchronized Audiovisual Reconstruction
description: >-
  [CVPR 2026][Video Generation][Satellite communications] This paper proposes an adaptive multimodal semantic transmission system for satellite communications. A dual-stream generative architecture (video-driven audio / au…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Satellite communications"
  - "semantic transmission"
  - "audiovisual synchronization"
  - "cross-modal generation"
  - "LLM-based intelligent decision-making"
date: 2026-05-08
content_hash: 9b37cae913f749bc
---

# Semantic Satellite Communications for Synchronized Audiovisual Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.10791](https://arxiv.org/abs/2603.10791)  
**Code**: N/A  
**Area**: Video Generation
**Keywords**: Satellite communications, semantic transmission, audiovisual synchronization, cross-modal generation, LLM-based intelligent decision-making

## TL;DR

This paper proposes an adaptive multimodal semantic transmission system for satellite communications. A dual-stream generative architecture (video-driven audio / audio-driven video) enables dynamic modality priority switching, combined with a dynamic knowledge base update mechanism and an LLM agent decision module, achieving high-fidelity synchronized audiovisual reconstruction under severe bandwidth constraints.

## Background & Motivation

1. **Background**: Satellite communications face extreme physical-layer limitations (rain fading, large Doppler shifts, hundreds of milliseconds of propagation delay), making it difficult for conventional approaches to support high-bandwidth multimedia streams at kbps-level bitrates. Semantic communication, which transmits only task-relevant semantics, has successfully addressed bandwidth bottlenecks in text and image domains.

2. **Limitations of Prior Work**: (1) Existing multimodal semantic transmission methods fix modality priorities and cross-modal generation paths at design time, lacking the flexibility to adapt to varying task requirements. (2) Knowledge bases lack dynamic update mechanisms, resulting in stale information or wasted resources. (3) Channel adaptation is reactive rather than proactive, lacking forward-looking transmission strategies.

3. **Key Challenge**: In satellite scenarios, bandwidth is extremely scarce and the channel is highly dynamic, while synchronized multimodal transmission inherently demands high bandwidth and stable channels — a fundamental contradiction.

4. **Goal**: To achieve flexible, robust, and high-fidelity audiovisual synchronization under limited satellite bandwidth.

5. **Key Insight**: Exploiting the complementarity of cross-modal generation — transmitting only the most critical modality's semantics and recovering the other modality via a generative model.

6. **Core Idea**: Coordinating dual-stream cross-modal generation (V2A/A2V) and dynamic knowledge base updates via an LLM agent to enable intelligent audiovisual semantic transmission under satellite bandwidth constraints.

## Method

### Overall Architecture

The system comprises three layers: (1) **Effectiveness Layer**: evaluates task-level performance metrics; (2) **Semantic Layer**: selectively extracts and transmits task-relevant features under LLM agent guidance, reconstructing audiovisual content via V2A or A2V dual-stream workflows; (3) **Technical Layer**: manages OFDM physical transmission. A shared semantic knowledge base stores static information such as user reference images.

### Key Designs

1. **Dual-Stream Cross-Modal Generation Architecture**:

    - Function: Dynamically switches between video-first and audio-first transmission-generation paths according to task requirements.
    - Mechanism: **V2A path (video-first)**: transmits 3DMM parameters and text; reconstructs video using a reference image and 3DMM as $\hat{V}_i = f_{VG}(\hat{S}_{i,M}, V_1)$; then generates synchronized audio from video lip features and text. A lip encoder extracts features $E_{\text{lip}} = f_{\text{Lip}}(\hat{V})$; an attention mechanism fuses lip and text features as $E_{\text{lip-text}} = \text{Attention}(E_{\text{lip}}, E_{\text{text}}, E_{\text{text}})$; transposed convolutions then expand the representation to produce a Mel spectrogram and audio waveform. **A2V path (audio-first)**: transmits audio semantics (text, phonemes, and duration); reconstructs audio as $\hat{A} = f_{\text{HiFi}}(f_{\text{Mel}}(\hat{S}_P, \hat{S}_D))$; predicts facial parameters via an audio-to-3DMM module; and finally synthesizes video.
    - Design Motivation: Different tasks assign different priorities to modalities — surveillance emphasizes video fidelity, while emergency voice dispatch prioritizes audio intelligibility. Flexible switching avoids the rigidity of fixed pipelines.

2. **Dynamic Knowledge Base Update Mechanism**:

    - Function: Balances generation quality and transmission overhead under bandwidth constraints.
    - Mechanism: A four-level discrimination mechanism is designed. **L0 (identity consistency)**: evaluates identity matching in facial embedding space via cosine similarity (CSIM) with threshold $\alpha_{\text{CSIM}}=0.7$. **L1 (pixel reconstruction quality)**: evaluates low-level visual consistency via PSNR with threshold $\alpha_{\text{PSNR}}=13$ dB. **L2 (3DMM semantic quality)**: computes a weighted 3DMM parameter distance across expression, rotation, and translation; reference frames are reused if all three components fall below their respective thresholds. **L3 (forced update)**: the reference frame is updated for every video segment, incurring the highest bandwidth cost. Bandwidth scales progressively: L0 requires only 2,785 symbols per segment, while L3 requires 16,384.
    - Design Motivation: Satellite bandwidth is precious; each reference frame update costs 16,384 symbols. Multi-level discrimination ensures updates occur only when genuinely necessary — V2A-L2 achieves near-L3 performance using approximately 50% of L3's bandwidth.

3. **LLM Agent Intelligent Decision Module**:

    - Function: Adaptively optimizes transmission strategies based on task requirements and real-time channel conditions.
    - Mechanism: GPT-4o serves as the decision agent, executing a three-step reasoning process: (1) **Intent understanding**: analyzes task objectives and performance requirements, and assesses current channel quality from environmental information; (2) **Workflow selection**: chooses between V2A and A2V paths; (3) **Resource allocation**: dynamically adjusts compression ratio, bandwidth allocation, and knowledge base update level. The agent is configured via prompt engineering, taking satellite ID, orbital position, and weather conditions as inputs; its decisions directly configure the OFDM transceiver.
    - Design Motivation: Conventional lookup-table approaches face combinatorial state-space explosion and cannot enumerate all scenarios; the LLM's semantic understanding and reasoning capabilities enable more flexible cross-layer decision-making.

### Loss & Training

Semantic codec pairs are jointly trained with MSE (for floating-point data) and cross-entropy (for token sequences) for 400 epochs at a learning rate of 0.001. The V2A backbone network is trained with three L2 losses on pitch, energy, and Mel spectrogram for 1,000 epochs at a learning rate of 0.0001. The satellite channel model follows NTN-TDL-A at altitudes of 300–1200 km.

## Key Experimental Results

### Main Results

Video reconstruction (SNR = 12 dB):

| Method | AKD ↓ | Bandwidth (symbols) |
|--------|-------|---------------------|
| H264+LDPC | N/A (undetectable) | 400,991 |
| H265+LDPC | N/A | 54,390 |
| SVC | 8.36 | 600 |
| V2A (Ours) | 5.41 | 600 |
| A2V (Ours) | 5.85 | 0 (video portion) |

Audio transmission (SNR = 20 dB):

| Method | LSE-C ↑ | LSE-D ↓ | WER ↓ | Bandwidth |
|--------|---------|---------|-------|-----------|
| DeepSC-S | 7.85 | 6.57 | 0.11 | 32,768 |
| A2V (Ours) | 5.85 | 8.69 | 0.11 | 600 |
| V2A (Ours) | 2.22 | 12.16 | 0.11 | 300 |

### Ablation Study

Effect of knowledge base update level (V2A, SNR = 12 dB):

| Update Level | AKD | Avg. Updates / 100 segments | Avg. Bandwidth / segment |
|--------------|-----|-----------------------------|--------------------------|
| L0 | ~8 | 17 | 3,085 |
| L1 | ~6.5 | 27 | 4,727 |
| L2 | ~5.8 | 50 | 8,492 |
| L3 | ~4.8 | 100 | 16,684 |

### Key Findings

- **Extreme compression ratio**: V2A/A2V requires only 600–900 symbols per frame, approximately 600× fewer than H264's 400,000+ symbols.
- **A2V achieves "zero video transmission"**: video is generated entirely from audio semantics without transmitting a single video symbol, which is highly valuable in extreme bandwidth scenarios.
- **Semantic methods greatly outperform conventional methods at low SNR**: traditional H264/H265 degrades rapidly at low SNR, while generative methods remain stable.
- **LLM agent vs. lookup table**: the agent proactively reduces the update level from L3 to L2 and reallocates bandwidth, achieving near-L3 performance at approximately 50% of the bandwidth.
- **V2A-L2 offers the best cost-effectiveness**: AKD of 5.8 vs. 4.8 for L3, at half the bandwidth.

## Highlights & Insights

- **Cross-modal generation as a substitute for transmission**: transmitting semantics of only one modality and recovering the other via generation is particularly meaningful under the extreme bandwidth constraints of satellite communications. The "zero video transmission" design of A2V is a notably bold choice.
- **Elegant multi-level knowledge base update design**: the progressive discrimination — from identity level to pixel level to semantic level — efficiently balances bandwidth savings and generation quality; L0–L2 form a practical quality–bandwidth trade-off toolkit.
- **LLM as a communication system controller**: this approach transcends conventional rule-based adaptation strategies, enabling cross-layer decision-making grounded in task intent and physical constraints, and represents an interesting exploration at the intersection of semantic communications and foundation models.

## Limitations & Future Work

- **High inference latency of generative networks**: V2A/A2V inference latency (0.07–0.1 s/frame) is far higher than conventional methods, potentially unsuitable for ultra-low-latency scenarios.
- **Limited to facial video scenarios**: the current design is built around 3DMM facial parameters and cannot be straightforwardly extended to general video content.
- **Real-time feasibility of the LLM agent**: whether GPT-4o's inference speed can satisfy real-time decision-making demands in satellite communications remains questionable.
- **Single-satellite relay assumption**: multi-satellite cooperation and inter-satellite link scenarios are not considered.

## Related Work & Insights

- **vs. SVC**: SVC transmits video semantics via keypoints and is highly sensitive to channel noise (AKD = 8.36 at 12 dB); the proposed 3DMM parameters are more robust (AKD = 5.41).
- **vs. DeepSC-S**: DeepSC-S achieves high end-to-end speech codec quality but requires 55× more bandwidth than the proposed method.
- **vs. fixed modality-priority schemes**: references [57][58][55][23] fix unidirectional generation paths; the proposed dual-stream switching is considerably more flexible.

## Rating

- Novelty: ⭐⭐⭐⭐ The system-level design integrating dual-stream cross-modal generation with LLM agent decision-making is novel and organically combines multiple research directions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three evaluation dimensions — video, audio, and synchronization — with detailed knowledge base ablations and agent case studies.
- Writing Quality: ⭐⭐⭐ High system complexity makes the paper verbose, and some sections are overly formulaic.
- Value: ⭐⭐⭐⭐ Provides important reference value for the satellite communications and semantic transmission community; the LLM agent-controlled communication system paradigm is forward-looking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)
- [\[CVPR 2026\] SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution](swift_sliding_window_reconstruction_for_few-shot_training-free_generated_video_a.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[ICLR 2026\] Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator](../../ICLR2026/video_generation/text-to-3d_by_stitching_a_multi-view_reconstruction_network_to_a_video_generator.md)
- [\[ICCV 2025\] SweetTok: Semantic-Aware Spatial-Temporal Tokenizer for Compact Video Discretization](../../ICCV2025/video_generation/sweettok_semantic-aware_spatial-temporal_tokenizer_for_compact_video_discretizat.md)

</div>

<!-- RELATED:END -->
