---
title: >-
  [Paper Note] Semantic Satellite Communications for Synchronized Audiovisual Reconstruction
description: >-
  [CVPR 2025][Video Generation][Semantic Communications] This paper proposes an adaptive multimodal semantic transmission system tailored for satellite communication scenarios. By employing a dual-stream generative architecture (Video-to-Audio / Audio-to-Video) that flexibly switches transmission pathways, combined with a dynamic knowledge base update mechanism and an LLM agent decision-making module, high-fidelity synchronized audiovisual reconstruction is achieved under extre…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Semantic Communications"
  - "Satellite Communications"
  - "Synchronized Audiovisual Reconstruction"
  - "LLM Agent"
  - "Cross-Modal Generation"
date: 2026-05-08
content_hash: a14014f94e948ad3
---

# Semantic Satellite Communications for Synchronized Audiovisual Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2603.10791](https://arxiv.org/abs/2603.10791)  
**Code**: None  
**Area**: Video Generation  
**Keywords**: Semantic Communications, Satellite Communications, Synchronized Audiovisual Reconstruction, LLM Agent, Cross-Modal Generation

## TL;DR

This paper proposes an adaptive multimodal semantic transmission system tailored for satellite communication scenarios. By employing a dual-stream generative architecture (Video-to-Audio / Audio-to-Video) that flexibly switches transmission pathways, combined with a dynamic knowledge base update mechanism and an LLM agent decision-making module, high-fidelity synchronized audiovisual reconstruction is achieved under extremely limited satellite bandwidth constraints.

## Background & Motivation

**Background**: Satellite communications serve as critical infrastructure for global connectivity, finding extensive use in maritime, aviation, and disaster relief. However, satellite links suffer from severe physical layer constraints, including rain attenuation, Doppler shifts, and propagation delays on the order of hundreds of milliseconds. Semantic communication has emerged as a new paradigm that enhances transmission efficiency by extracting and transmitting only task-relevant semantic information, achieving success in text and image transmission.

**Limitations of Prior Work**: Existing semantic video transmission methods (such as DeepWiVe, DVST, VISTA) mostly focus on single-modality video transmission, whereas real-world applications often require synchronized audiovisual data. The few works involving cross-modal generation (such as reconstructing facial videos based on 3DMM parameters accompanied by generated audio) suffer from a critical limitation: the modality priority and cross-modal generation pathways are fixed during the design stage and cannot adapt dynamically to task requirements (e.g., prioritizing audio quality in emergency services). Furthermore, existing knowledge-base-driven generative semantic systems lack context-awareness and dynamic update mechanisms, leading to degraded generation quality or wasted bandwidth when the knowledge base becomes outdated.

**Key Challenge**: Under the extreme bandwidth constraints of satellite communications, simultaneously ensuring high-fidelity reconstruction and synchronization of both audio and video requires a flexible cross-modal semantic transmission strategy. However, rigid transmission architectures cannot adapt to dynamically changing channel conditions and diverse task requirements.

**Goal**: To design an adaptive multimodal semantic transmission system for satellite scenarios that addresses three sub-problems: (1) flexible modality priority switching; (2) dynamic knowledge base management to balance quality and bandwidth; and (3) environment-aware intelligent transmission policy planning.

**Key Insight**: The authors observe that different tasks have differing priority requirements for audio and video (e.g., surveillance tasks prioritize video, while voice dispatching prioritizes audio). Therefore, a switchable dual-stream generative architecture is adopted, alongside leveraging the reasoning capabilities of LLMs to achieve proactive transmission strategy planning.

**Core Idea**: Utilizing an LLM-driven agent to unify and coordinate the dual-stream cross-modal generation pathways and knowledge base update strategies, replacing traditional passive fixed-rules, thereby achieving high-fidelity synchronized audiovisual transmission over satellite links.

## Method

### Overall Architecture

The system consists of three core layers: the effects layer (evaluating reconstruction quality), the semantic layer (semantic extraction and cross-modal generation), and the technical layer (physical-layer transmission management), complemented by a shared semantic knowledge base. The input multimodal audiovisual data $\mathbf{M}$ is decomposed into video $\mathbf{V}$ and audio $\mathbf{A}$ for separate semantic extraction and encoding. It is then transmitted over the satellite channel via an OFDM system. The receiver dynamically selects either the V2A (Video-to-Audio) or A2V (Audio-to-Video) workflow for reconstruction based on specific task requirements.

### Key Designs

1. **Dual-stream cross-modal generative architecture**:

    - **Function**: Supports both V2A (video-driven audio generation) and A2V (audio-driven video generation) workflows, dynamically switching based on task demands.
    - **Mechanism**: The V2A pathway transmits 3DMM facial parameters and text, first reconstructing the video, and then extracting lip movement features via a lip encoder $f_{\text{Lip}}$. A multi-head attention mechanism learns the mapping between lip shapes and text embeddings as $\mathbf{E}_{\text{lip-text}} = \text{Attention}(\mathbf{E}_{\text{lip}}, \mathbf{E}_{\text{text}}, \mathbf{E}_{\text{text}})$, which is ultimately synthesized into audio using a Mel-spectrogram generator and a HiFi-GAN vocoder. The A2V pathway transmits audio semantics (text, phonemes, duration) to first reconstruct the audio, predicts facial parameters via a pre-trained Audio-to-3DMM module, and finally synthesizes the video using a video generator.
    - **Design Motivation**: Different application scenarios have varying priority demands for modalities. The flexible dual-stream design allows the system to transmit only the critical modality when bandwidth is restricted, leveraging cross-modal generation to recover the missing modality.

2. **Dynamic knowledge base update mechanism**:

    - **Function**: Minimizes the bandwidth overhead of keyframe updates while maintaining high generative quality.
    - **Mechanism**: A four-level decision mechanism is designed: L0 (user consistency layer) evaluates identity consistency using CSIM cosine similarity; L1 (pixel reconstruction quality layer) evaluates low-level visual consistency using PSNR; L2 (3DMM semantic quality layer) evaluates 3D geometric discrepancy via a weighted 3DMM parameter distance; L3 (forced update layer) enforces full updates directly when bandwidth is abundant. Updates are only triggered when the difference between the current frame and reference frames in the knowledge base exceeds statistical saliency.
    - **Design Motivation**: Static knowledge bases diverge from current content over time, leading to degraded generation quality. However, frequently updating high-dimensional image data over satellite links consumes substantial bandwidth. The multi-level decision mechanism balances quality and bandwidth across different levels of granularity.

3. **LLM intelligent decision-making module**:

    - **Function**: Acts as the central controller to coordinate generation pathway selection and resource allocation.
    - **Mechanism**: The LLM (GPT-4o), acting as an agent, receives task descriptions, user preferences, and real-time environmental data to perform a three-step reasoning process: (1) intent understanding—analyzing task objectives and channel conditions; (2) workflow selection—identifying the optimal V2A/A2V pathway and knowledge base update levels; (3) resource adjustment—dynamically configuring parameters like semantic compression rate and bandwidth allocation. Satellite communication domain knowledge is injected via prompt engineering or fine-tuning.
    - **Design Motivation**: Traditional lookup-table-based methods suffer from combinatorial explosion when task diversity scales, and fail to capture the semantic intent. An LLM possesses semantic reasoning and autonomous planning capabilities, facilitating preemptive cross-layer decisions.

### Loss & Training

- **Semantic Encoder-Decoder Training**: MSE loss is used for floating-point data (3DMM parameters, durations), while cross-entropy loss is applied to token sequences (text, phonemes), with end-to-end training conducted under noisy satellite channel conditions.
- **V2A Audio Generator Training**: The divergence between synthesized and ground-truth audio is minimized. The loss function encompasses pitch, energy, and Mel-spectrogram items: $$\hat{\Theta}_{\text{V2A}} = \arg\min (||\hat{\mathbf{P}} - \mathbf{P}||_2^2 + ||\hat{\mathbf{E}} - \mathbf{E}||_2^2 + ||\hat{\mathbf{F}} - \mathbf{F}||_F^2)$$
- Each semantic encoder-decoder pair is trained for 400 epochs, and the V2A network is trained for 1000 epochs.

## Key Experimental Results

### Main Results

| Method | Transmitted Symbols | Parameters (M) | Execution Time (s/frame) |
|------|-----------|----------|---------------|
| H264+LDPC | 400,991 | - | 0.033 |
| H265+LDPC | 54,390 | - | 0.013 |
| SVC | 600 | 60.11 | 0.019 |
| V2A (Video) | 300 | 172.01 | 0.071 |
| A2V (Video) | 0 | 159.88 | 0.053 |

A2V achieves "zero-symbol" video transmission by entirely driving video generation through audio semantics, requiring no video semantic transmission. Both V2A and A2V achieve orders-of-magnitude bandwidth compression compared to H265.

### Ablation Study

| KB Update Level | Update Frequency | Semantic Symbols | KB Update Symbols |
|--------------|---------|---------|-------------|
| L0 | 17 | 300 | 2,785 |
| L1 | 27 | 300 | 4,427 |
| L2 | 50 | 300 | 8,192 |
| L3 | 100 | 300 | 16,384 |

At 12dB SNR, V2A-L2 achieves an AKD of 5.8, which is close to L3's 4.8, while requiring only 50% of the bandwidth of L3.

### Key Findings

- Under low SNR conditions, traditional methods (H264/H265) suffer severe performance degradation, whereas generative methods (A2V, V2A) demonstrate remarkable robustness.
- V2A achieves continuous video quality improvement as bandwidth increases, whereas A2V exhibits a performance bottleneck where additional bandwidth fails to further improve video reconstruction quality.
- Compared to lookup-table-based methods, the LLM agent achieves comparable performance while reducing bandwidth usage by approximately 50%.

## Highlights & Insights

- First work to introduce an LLM agent into satellite semantic communications, realizing a paradigm shift from "passive adaptation" to "proactive planning".
- The modal decoupling design of the dual-stream generative architecture is elegant, and A2V can even achieve zero-symbol video transmission.
- The multi-level knowledge base update mechanism achieves an excellent trade-off between quality and bandwidth.
- Strong interdisciplinary integration of communication system design and generative AI models.

## Limitations & Future Work

- Generative methods suffer from high inference latency (0.1s/clip for the V2A audio generation part), which may limit their application in highly real-time scenarios.
- Currently limited to facial video scenarios; extending to general video content requires further investigation.
- The computational overhead of the LLM agent is not analyzed in detail.
- Multiple threshold parameters in the knowledge base update mechanism still require manual configuration.

## Related Work & Insights

- The SVC system serves as an important foundation for this work, achieving low-bandwidth video transmission through keypoint transmission and generation.
- Compared to end-to-end semantic communication methods like DeepSC-S, generative methods exhibit significant advantages under extreme bandwidth constraints.
- The application of LLMs in communication systems is a recent hotspot; this work demonstrates the potential of LLMs in resource scheduling.
- This work provides valuable insights for designing generative semantic transmission systems in satellite communications.

## Rating

- **Novelty**: 8/10 — The combination of a dual-stream cross-modal architecture and LLM decision-making shows strong innovation.
- **Experimental Thoroughness**: 7/10 — Ablation studies and case analyses are extensive, but comparison with more recent state-of-the-art methods is lacking.
- **Writing Quality**: 8/10 — Clear system modeling and complete mathematical derivations.
- **Value**: 7/10 — Holds significant application value for satellite multimedia communications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Wav2Sem: Plug-and-Play Audio Semantic Decoupling for 3D Speech-Driven Facial Animation](wav2sem_plug-and-play_audio_semantic_decoupling_for_3d_speech-driven_facial_anim.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[ICLR 2026\] Syncphony: Audio-to-Video Generation with Synchronized Visual Dynamics using Diffusion Transformers](../../ICLR2026/video_generation/syncphony_synchronized_audio-to-video_generation_with_diffusion_transformers.md)
- [\[CVPR 2026\] Cross-Subject EEG-to-Video Reconstruction and Beyond](../../CVPR2026/video_generation/cross-subject_eeg-to-video_reconstruction_and_beyond.md)
- [\[CVPR 2026\] U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](../../CVPR2026/video_generation/u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)

</div>

<!-- RELATED:END -->
