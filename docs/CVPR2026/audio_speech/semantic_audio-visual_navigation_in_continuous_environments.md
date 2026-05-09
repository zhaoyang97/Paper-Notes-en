---
title: >-
  [Paper Note] Semantic Audio-Visual Navigation in Continuous Environments
description: >-
  [CVPR 2026][Audio & Speech][audio-visual navigation] This paper introduces the SAVN-CE task, extending semantic audio-visual navigation to continuous 3D environments, and proposes MAGNet (Memory-Augmented Goal description Network). By fusing historical context and ego-motion cues, MAGNet achieves robust goal inference after target sounds cease, yielding absolute success rate improvements of up to 12.1%.
tags:
  - CVPR 2026
  - "Audio & Speech"
  - audio-visual navigation
  - continuous environments
  - memory augmentation
  - goal reasoning
  - embodied intelligence
date: 2026-05-08
content_hash: d76456fde69fb9e2
---

# Semantic Audio-Visual Navigation in Continuous Environments

**Conference**: CVPR 2026
**arXiv**: [2603.19660](https://arxiv.org/abs/2603.19660)
**Code**: [https://github.com/yichenzeng24/SAVN-CE](https://github.com/yichenzeng24/SAVN-CE)
**Area**: Embodied Navigation / Audio-Visual
**Keywords**: audio-visual navigation, continuous environments, memory augmentation, goal reasoning, embodied intelligence

## TL;DR
This paper introduces the SAVN-CE task, extending semantic audio-visual navigation to continuous 3D environments, and proposes MAGNet (Memory-Augmented Goal description Network). By fusing historical context and ego-motion cues, MAGNet achieves robust goal inference after target sounds cease, yielding absolute success rate improvements of up to 12.1%.

## Background & Motivation
1. **Background**: Audio-visual navigation (AVN) enables embodied agents to navigate toward sounding targets in unknown environments using auditory and visual cues. Semantic audio-visual navigation (SAVN) further requires the target to be a semantically meaningful object (e.g., "a squeaking chair") rather than an arbitrary location.
2. **Limitations of Prior Work**: Existing methods rely on pre-computed room impulse responses (RIRs), requiring terabytes of storage, and restrict agents to discrete grid points (1-meter resolution, 4 fixed orientations), severely limiting task realism.
3. **Key Challenge**: Discrete environments produce spatially discontinuous observations, preventing free exploration; in continuous environments, target sounds may intermittently or completely cease, resulting in loss of goal information.
4. **Goal**: (a) How to achieve free-movement audio-visual navigation in continuous environments? (b) How to maintain a stable goal representation when target sounds disappear? (c) How to jointly infer the spatial location and semantic category of the target?
5. **Key Insight**: The authors observe that ego-motion cues (previous action + current pose) can be used to infer dynamic changes in relative target position, while episodic memory preserves temporal continuity of goal representations after sounds cease.
6. **Core Idea**: A memory-augmented Transformer encoder fuses binaural audio, ego-motion cues, and episodic memory to enable persistent goal tracking after sounds disappear.

## Method

### Overall Architecture
MAGNet consists of three modules: (1) a **multimodal observation encoder** that encodes RGB-D images, binaural audio, actions, and pose into compact embeddings stored in scene memory; (2) a **Memory-Augmented Goal description Network (GDN)** that fuses binaural features, ego-motion information, and episodic memory to infer a spatial-semantic goal representation; and (3) a **context-aware policy network** based on a Transformer encoder-decoder architecture that leverages scene memory to predict the next action.

### Key Designs

1. **Multimodal Observation Encoder**:
    - Function: Encodes multimodal observations at each timestep into a unified representation.
    - Mechanism: RGB and depth images are encoded by ResNet-18, the previous action by an embedding layer, and the normalized pose $[x/d, y/d, \sin\theta, \cos\theta, t/t_{max}]$ by a fully connected layer. Binaural waveforms are converted to complex spectrograms via STFT, and 4-channel acoustic features (mean magnitude spectrum, sine/cosine of ICP phase difference, ILD) are extracted and encoded by a 3-layer convolutional network. All embeddings are concatenated into an observation representation, maintained in a sliding-window scene memory of capacity $N_s=150$.
    - Design Motivation: Continuous environments yield denser observations (0.25s step interval), requiring efficient multimodal encoding and long-term history retention.

2. **Memory-Augmented Goal Description Network (GDN)**:
    - Function: Maintains a stable goal representation during intermittent or complete absence of target sounds.
    - Mechanism: At each step, binaural audio embeddings, action embeddings, and pose embeddings are fused via an MLP into $m_t$ and stored in an episodic memory of capacity $N_g=128$. The memory sequence, augmented with positional encodings, is fed into a causal Transformer encoder, producing two outputs: (a) a goal embedding $e_t^G$ for the policy network; and (b) an ACCDDOA-format goal description $y_{ct} = [a_{ct}R_{ct}, d_{ct}]$ (encoding sound activity status, direction-of-arrival unit vector, and normalized distance) used as a supervised training signal.
    - Design Motivation: Ego-motion cues enable precise inference of target bearing changes (TurnLeft/TurnRight shifts the azimuth by ±15°; MoveForward affects both azimuth and distance), while episodic memory ensures temporal continuity. The fine-grained action space (0.25m forward / 15° turn) limits inter-step positional displacement, stabilizing goal tracking.

3. **Context-Aware Policy Network**:
    - Function: Predicts the next action based on historical and current observations.
    - Mechanism: A Transformer encoder processes the scene memory $M_{s,t}$ to capture temporal dependencies; a decoder uses the current observation embedding as a query and the encoded memory as keys and values to produce a context-aware latent state $s_t$. This state is passed to separate actor and critic fully connected layers to predict the action distribution and state value, respectively.
    - Design Motivation: The Transformer attention mechanism enables full exploitation of historical information, supporting coherent decision-making in partially observable continuous environments.

### Loss & Training
- GDN is trained online with MSE loss using oracle ACCDDOA labels; causal attention masks prevent future information leakage.
- The policy network is trained with DD-PPO following the two-stage paradigm of SAVi.
- Rewards: +10 for success, intermediate rewards based on change in geodesic distance to goal, and −0.01 time penalty per step.
- Each iteration uses 150-step rollouts; training runs for approximately 14 days on 128 CPUs and 4 A800 GPUs.

## Key Experimental Results

### Main Results

| Method | SR↑ | SPL↑ | SNA↑ | DTG↓ | SWS↑ |
|--------|-----|------|------|------|------|
| AV-Nav | 21.3 | 17.8 | 13.1 | 10.7 | 4.0 |
| SMT+Audio | 24.8 | 21.0 | 16.8 | 10.1 | 5.3 |
| SAVi | 25.6 | 21.2 | 17.3 | 10.1 | 6.0 |
| **MAGNet** | **37.7** | **32.9** | **27.4** | **8.0** | **10.6** |
| Oracle1 | 41.4 | 37.8 | 31.0 | 6.3 | 13.0 |
| Oracle2 | 75.0 | 63.7 | 51.9 | 4.2 | 48.4 |

In the clean environment, MAGNet outperforms SAVi by 12.1% in absolute SR and 4.6% in SWS.

### Ablation Study

| Configuration | SR↑ | SPL↑ | SWS↑ |
|---------------|-----|------|------|
| w/o GDN | 32.4 | 27.9 | 6.3 |
| GDN w/o Memory | 33.9 | 29.8 | 8.9 |
| GDN w/o Self-motion | 34.3 | 30.4 | 7.8 |
| Full MAGNet | **37.7** | **32.9** | **10.6** |

### Key Findings
- Removing GDN reduces SR by 5.3%, yet still surpasses all baselines, indicating that the policy network itself is already strong.
- The contribution of episodic memory (+3.8% SR) exceeds that of ego-motion cues (+1.9% SR), though the combination yields the best performance.
- In distractor environments, all methods degrade; MAGNet's SR drops from 37.7 to 19.3, and the high DSR (up to 7.8%) suggests that acoustically similar distractors are the primary bottleneck.
- The large gap between Oracle2 and Oracle1 (75.0 vs. 41.4) confirms that goal information loss after sounds cease is the central challenge.

## Highlights & Insights
- **ACCDDOA format for unified goal description**: Sound activity status, direction-of-arrival unit vector, and distance are fused into a compact vector representation, elegantly combining SELD with navigation.
- **Physical interpretability of ego-motion cues**: TurnLeft/TurnRight precisely shifts the target azimuth by ±15°; this explicit geometric relationship facilitates learning of goal position updates after sounds cease.
- **Continuous environment dataset construction**: Sound onset times are uniformly sampled from [0, 5]s, durations follow a Gaussian distribution (mean 15s, std 9s), and the test set averages 78.49 oracle actions per episode (vs. only 26.52 in the discrete setting), substantially increasing task difficulty.

## Limitations & Future Work
- The large gap between Oracle2 and MAGNet (75.0 vs. 37.7) indicates considerable room for improvement in GDN.
- High DSR in distractor environments reveals insufficient discriminability against acoustically similar distractors.
- Only single-target navigation is supported; future work may extend to multi-target and dynamic target scenarios.
- High training cost (14 days, 128 CPUs + 4 GPUs) limits large-scale experimentation.

## Related Work & Insights
- **vs. SAVi**: SAVi aggregates historical estimates using a weighting factor λ; the present work replaces this with a Transformer episodic memory, yielding superior performance after sounds cease.
- **vs. VLN-CE**: VLN-CE provides explicit language instructions as goal specification, whereas SAVN-CE requires goal inference from partial perception, posing a greater challenge.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of continuous environments and memory-augmented GDN is relatively novel, though individual component designs are fairly standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ablations are comprehensive, including GDN evaluation, factor analysis, and trajectory visualization.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with well-defined task formulations.
- Value: ⭐⭐⭐⭐ Provides a more realistic continuous-environment benchmark for audio-visual navigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification](../../ACL2026/audio_speech/retrieving_to_recover_towards_incomplete_audio-visual_question_answering_via_sem.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[AAAI 2026\] Generalizing Analogical Inference from Boolean to Continuous Domains](../../AAAI2026/audio_speech/generalizing_analogical_inference_from_boolean_to_continuous_domains.md)
- [\[ACL 2026\] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](../../ACL2026/audio_speech/music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](../../ICCV2025/audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)

</div>

<!-- RELATED:END -->
