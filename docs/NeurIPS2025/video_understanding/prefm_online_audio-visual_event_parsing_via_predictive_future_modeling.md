---
title: >-
  [Paper Note] PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling
description: >-
  [NeurIPS 2025][Video Understanding][Online audio-visual event parsing] This paper introduces the Online Audio-Visual Event Parsing (On-AVEP) paradigm for the first time, along with the PreFM framework…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Online audio-visual event parsing"
  - "predictive future modeling"
  - "multimodal fusion"
  - "real-time video understanding"
  - "knowledge distillation"
date: 2026-05-08
content_hash: e11c0449ebdd4739
---

# PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2505.23155](https://arxiv.org/abs/2505.23155)  
**Code**: [GitHub](https://github.com/XiaoYu-1123/PreFM)  
**Area**: Video Understanding / Audio-Visual Event Parsing
**Keywords**: Online audio-visual event parsing, predictive future modeling, multimodal fusion, real-time video understanding, knowledge distillation

## TL;DR

This paper introduces the Online Audio-Visual Event Parsing (On-AVEP) paradigm for the first time, along with the PreFM framework, which leverages pseudo-future sequences to enhance current contextual understanding. Combined with modality-agnostic knowledge distillation and focal temporal prioritization, PreFM surpasses offline SOTA methods by +9.3 event-level average F1 score using only 2.7% of their parameter count.

## Background & Motivation

Audio-Visual Event Parsing (AVEP) is a key task in multimodal video understanding, requiring the simultaneous handling of audio-only, visual-only, and audio-visual joint events. Existing methods (e.g., UnAV, UniAV, CCNet) process entire video sequences offline—achieving high accuracy but at the cost of large model sizes and the need for full video input, making them unsuitable for real-time applications such as autonomous driving and wearable devices.

The root cause of the challenge is: in the online setting, models can only access historical and current information; the lack of future context leads to ambiguity (e.g., a person opening their mouth—are they singing or speaking?), while computational efficiency must also be maintained.

The paper's starting point is to generate pseudo-future multimodal cues via predictive modeling, enabling online models to "foresee" upcoming events and thereby achieve stronger contextual understanding while remaining lightweight.

## Method

### Overall Architecture

The PreFM framework receives streaming audio and visual features, taking a feature sequence of length $L_c$ (current window) as input. It consists of three core components: (1) a pseudo-future mechanism that generates predicted future sequences of length $L_f$; (2) temporal-modality cross fusion for cross-temporal and cross-modal feature enhancement; and (3) training-phase strategies including modality-agnostic robust representation and focal temporal prioritization. Event predictions are output at time step $T$.

### Key Designs

1. **Universal Hybrid Attention (UHA)**:

    - Serves as the foundational module for all fusion operations, taking a target query sequence $Q$ and multiple context sets $\{F_i\}$ as input.
    - Aggregates multiple context sources into the query via multi-head attention: $\text{UHA}(Q, \{F_i\}) = \text{FFN}(\text{LN}(Q + \sum_i \text{Attn}(Q, F_i, F_i)))$
    - Flexibly supports unified computation of self-attention, cross-modal attention, and cross-temporal attention.

2. **Pseudo-Future Mechanism**:

    - First performs initial cross-modal fusion of current audio and visual features via UHA.
    - Learnable query tokens $Q^a, Q^v$ then attend to the fused current features to generate pseudo-future sequences $\tilde{F}_f^a, \tilde{F}_f^v$.
    - Design motivation: online inference lacks future information; predictive modeling compensates by supplying critical temporal context.

3. **Temporal-Modality Cross Fusion**:

    - Future enhancement stage: pseudo-future sequences interact via UHA with themselves (self-attention), the other modality's pseudo-future (cross-modal), and the corresponding current features (cross-temporal).
    - Current refinement stage: the enhanced pseudo-future information is fed back into the current representation, granting current features a "forward-looking" perspective.
    - A shared classification head generates event predictions for both the current and future windows.

4. **Modality-agnostic Robust Representation (MRR)**:

    - A frozen OnePeace large model converts event labels into modality-agnostic text features as distillation targets.
    - The student model's joint audio-visual representation is aligned with the teacher features via cosine similarity loss.
    - This acquires generalizable knowledge from the large model in a lightweight manner without adding inference parameters.

5. **Focal Temporal Prioritization (FTP)**:

    - A Gaussian function centered at current time $T$ is used to weight the loss at different time steps.
    - Predictions closer to the current moment receive higher weights, encouraging the model to focus on the most critical current decisions.
    - Distinct Gaussian weights are applied to the current window and the pseudo-future window respectively.

### Loss & Training

The total loss is a combination of weighted BCE loss and MRR distillation loss: $\mathcal{L} = \sum w_c \cdot \mathcal{L}_c + \sum w_f \cdot \mathcal{L}_f + \lambda \sum w \cdot \mathcal{L}_{mrr}$. During training, a random segment sampling strategy is employed, generating target time points by sliding with step $L_c$ plus a random offset within the video, enhancing data diversity. Training runs for 60 epochs with a 10-epoch warmup phase.

## Key Experimental Results

### Main Results

**On-AVEL Task (UnAV-100 Dataset)**:

| Method | Features | Seg-F1 | Seg-mAP | Event Avg F1 | Params | FLOPs |
|--------|----------|--------|---------|-------------|--------|-------|
| CCNet* (offline) | OnePeace | 65.0 | 70.6 | 58.3 | 238.8M | 72.1G |
| UniAV* (offline) | OnePeace | 59.2 | 70.0 | 52.9 | 130.8M | 22.7G |
| PreFM (online) | CLIP+CLAP | 59.1 | 70.1 | 46.3 | **6.5M** | **0.4G** |
| PreFM+ (online) | OnePeace | **62.4** | **70.6** | **51.5** | 13.8M | 0.5G |

**On-AVVP Task (LLP Dataset)**:

| Method | Seg-F1a/F1v/F1av | Event-Avga/Avgv/Avgav | Params |
|--------|------------------|-----------------------|--------|
| MM-CSE | 53.3/56.5/48.9 | 37.7/46.9/36.2 | 6.2M |
| PreFM | **60.0**/59.3/**53.3** | **46.3**/**50.6**/**41.2** | **3.3M** |

### Ablation Study

| Configuration | Event Avg F1 | Note |
|---------------|-------------|------|
| Base model (no prediction) | 42.1 | Current window only |
| + Pseudo-future modeling | 44.5 | Future sequence prediction added |
| + Cross fusion | 45.3 | Cross-temporal modality enhancement |
| + MRR distillation | 45.8 | Knowledge distillation |
| + Focal temporal prioritization | 46.3 | Full PreFM |

### Key Findings
- The online PreFM with only 2.7% of the parameters (6.5M vs. 238.8M) surpasses offline methods such as CCNet—which require full video input—on multiple metrics.
- Inference speed reaches 51.9 FPS (vs. 7.5 FPS for CCNet) with a latency of only 19.3 ms.
- Pseudo-future modeling is the most critical module, contributing the largest performance gain.

## Highlights & Insights

- **Paradigm innovation**: On-AVEP is defined and systematically addressed for the first time, unifying AVEL and AVVP under an online streaming processing framework.
- **Excellent efficiency–performance trade-off**: PreFM achieves performance comparable to or exceeding that of a 238.8M-parameter offline model using only 6.5M parameters in an online setting.
- **Elegant UHA module design**: A flexible context list unifies self-attention, cross-modal attention, and cross-temporal attention, avoiding the overhead of stacking different attention types layer by layer.
- **Cross-modal interaction in pseudo-future sequences**: Rather than generating future predictions for a single modality, audio-visual cross-attention is applied to reduce noise in the pseudo-future representations.

## Limitations & Future Work

- The quality of pseudo-future sequences depends on the information content of the current window; predictions may be inaccurate when abrupt scene changes occur.
- Distillation targets rely on event labels to generate text prompts, potentially limiting performance in weakly supervised settings.
- The current window length $L_c = 10$ seconds and future window $L_f = 5$ seconds are fixed, lacking adaptive adjustment.
- End-to-end fine-tuning of audio and visual feature extractors has not been explored.

## Related Work & Insights

- **vs. CCNet (offline)**: PreFM approaches CCNet's performance in online mode with far fewer parameters, demonstrating that predictive modeling can effectively compensate for the absence of future information.
- **vs. MAT/TPT (online action detection)**: PreFM extends online video understanding from single visual modality to audio-visual joint modeling; UHA's unified attention design is more efficient than designing separate modules for each type of interaction.
- **Implications for other tasks**: The combination of pseudo-future modeling and focal temporal prioritization is transferable to other online perception tasks (e.g., online object detection, streaming dialogue).

## Rating

- Novelty: ⭐⭐⭐⭐ — On-AVEP is defined for the first time; pseudo-future modeling is a novel application in the audio-visual domain.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two datasets, full segment- and event-level evaluation, comprehensive comparison of parameters/computation/speed, and sufficient ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic, intuitive figures, and well-defined problem formulation.
- Value: ⭐⭐⭐⭐ — Significant reference value for real-time multimodal understanding; efficiency advantages make practical deployment feasible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](../../ICCV2025/video_understanding/emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](../../ICCV2025/video_understanding/hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)
- [\[ICML 2026\] AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes](../../ICML2026/video_understanding/avtrack_audio-visual_tracking_in_human-centric_complex_scenes.md)
- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](pass_path-selective_state_space_model_for_event-based_recognition.md)

</div>

<!-- RELATED:END -->
