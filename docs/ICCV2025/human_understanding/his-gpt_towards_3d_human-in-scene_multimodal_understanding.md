---
title: >-
  [Paper Note] HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding
description: >-
  [ICCV 2025][Human Understanding][human-in-scene understanding] This paper proposes the HIS-QA task, the HIS-Bench benchmark, and HIS-GPT — the first foundation model for joint 3D human-in-scene understanding. Through an Auxiliary Interaction Module (AInt) and a Layout-Trajectory Positional Encoding (LTP), HIS-GPT captures fine-grained human–scene interactions and substantially outperforms GPT-4o and other baselines across 16 sub-tasks.
tags:
  - ICCV 2025
  - Human Understanding
  - human-in-scene understanding
  - 3D multimodal
  - large language models
  - human motion
  - QA benchmark
date: 2026-05-08
content_hash: 49bc080098b1c214
---

# HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding

**Conference**: ICCV 2025
**arXiv**: [2503.12955](https://arxiv.org/abs/2503.12955)
**Code**: [ZJHTerry18/HumanInScene](https://github.com/ZJHTerry18/HumanInScene)
**Area**: 3D Vision
**Keywords**: human-in-scene understanding, 3D multimodal, large language models, human motion, QA benchmark

## TL;DR

This paper proposes the HIS-QA task, the HIS-Bench benchmark, and HIS-GPT — the first foundation model for joint 3D human-in-scene understanding. Through an Auxiliary Interaction Module (AInt) and a Layout-Trajectory Positional Encoding (LTP), HIS-GPT captures fine-grained human–scene interactions and substantially outperforms GPT-4o and other baselines across 16 sub-tasks.

## Background & Motivation

**3D scene-language understanding** and **human motion understanding** have each made notable progress, yet their joint understanding — comprehending human states and behaviors within 3D scenes — remains severely underexplored. This capability is critical for embodied intelligence:

1. **Scene models ignore the human body**: Existing 3D scene LLMs (e.g., LL3DA, Chat-Scene) understand scene layouts and objects but cannot process human motion sequences, precluding recognition of human–object interactions such as "sitting on a chair."

2. **Human body models ignore the environment**: Existing 3D human LLMs (e.g., MotionGPT, AvatarGPT) analyze isolated poses and motions without environmental awareness, making it impossible to answer queries like "facing the television" that require scene context.

3. **No joint evaluation benchmark exists**: Existing benchmarks either address scene QA (SQA3D) or motion description (Motion-X) in isolation; none jointly integrates scene and human modalities for open-ended language understanding.

**Core Idea**: Joint human-in-scene understanding requires two key capabilities — (a) capturing interaction cues between human motion and surrounding objects (e.g., contact, spatial relations), and (b) spatiotemporally aligning scene spatial layout with human motion trajectories. HIS-GPT addresses these two challenges via the AInt and LTP modules, respectively.

## Method

### Overall Architecture

HIS-GPT takes three inputs: a 3D scene point cloud $\mathcal{S} \in \mathbb{R}^{P \times 6}$, a human motion sequence $\mathcal{M} = \{M_i\}_{i=1}^T$ (SMPL pose sequence), and a text instruction $\mathcal{I}$. Scene and motion are encoded by independent encoders; the resulting embeddings are enriched by AInt and LTP, then projected to an LLM for autoregressive text generation.

### Scene Encoder

A pretrained 3D encoder (Uni3D) extracts object-level features. A 3D scene segmentor (Mask3D) first extracts per-object point clouds from the input, which are then encoded as $\{s_i \in \mathbb{R}^d\}_{i=1}^N$, where $N$ is the number of detected objects.

### Motion Encoder

A motion VQ-VAE (MotionGPT) maps the motion sequence to a discrete codebook, yielding motion embeddings $\{m_t \in \mathbb{R}^d\}_{t=1}^T$.

### Auxiliary Interaction Module (AInt)

Since scene and motion embeddings are generated independently, they lack mutual interaction cues. AInt injects interaction information via three auxiliary tasks:

**(1) Activity Classification**: Motion embeddings are fused with features from the spatially nearest $k$ objects, and the resulting representation is used to predict the activity category. Scene-context fusion for frame $t$ is defined as:

$$\tilde{m}_t = m_t + \text{Avg}(s_{t_1}, \ldots, s_{t_k})$$

An MLP with Softmax then performs classification, supervised by cross-entropy loss $\mathcal{L}_{act}$.

**(2) Spatial Relation Detection**: Eight types of human–object spatial relations (e.g., "facing") are defined. The spatial relation between object $i$ and motion frame $t$ is predicted as:

$$\mathcal{L}_{spa} = \sum_{i,t} \text{CE}(p_{it}^s, \text{SM}(W_s^{spa}(s_i) \cdot W_m^{spa}(m_t)))$$

**(3) Contact Detection**: Binary prediction of whether an object is in contact with a specific body part, supervised by BCE loss:

$$\mathcal{L}_{cont} = \sum_{i,t} \text{BCE}(p_{it}^c, \sigma(W_s^{cont}(s_i) \cdot W_m^{cont}(m_t)))$$

### Layout-Trajectory Positional Encoding (LTP)

Conventional positional encodings model only token-sequence relationships, ignoring the complex spatiotemporal structure between humans and scenes. LTP encodes 3D coordinates and temporal information via Spatial Fourier (SF) and Temporal Fourier (TF) transforms:

$$SF(\mu) = \text{sincos}(\phi_{SF} \cdot 2\pi\mu), \quad TF(t) = \text{sincos}(\phi_{TF} \cdot 2\pi t)$$

- **Motion positional encoding**: $e_t^m = SF(\mu_t) + TF(t)$, based on the 3D position and timestamp of frame $t$
- **Scene positional encoding**: $e_i^s = SF(\mu_i) + \frac{1}{T}\sum_t TF(t)$, where the temporal component is averaged since objects persist throughout the entire motion sequence

Final features are $f_i^s = s_i + e_i^s$ and $f_t^m = m_t + e_t^m$.

### Loss & Training

Two-stage training:

- **Stage 1 — Modality Alignment**: HIS descriptions, scene descriptions, and motion descriptions are used for alignment. Total loss: $\mathcal{L} = \mathcal{L}_{llm} + \lambda_{act}\mathcal{L}_{act} + \lambda_{spa}\mathcal{L}_{spa} + \lambda_{cont}\mathcal{L}_{cont}$
- **Stage 2 — HIS Instruction Tuning**: 700K instruction-tuning samples, with only $\mathcal{L}_{llm}$

Training data comprises 60K visual descriptions and 700K instruction-tuning samples, covering 750+ scenes.

## Key Experimental Results

### Main Results: HIS-Bench Evaluation

| Method | Activity | Spatial | HoI | Analysis | Prediction | Dialogue | Planning | **Avg.** |
|--------|----------|---------|-----|----------|------------|----------|----------|----------|
| LL3DA | 6.5 | 9.1 | 8.7 | 11.9 | 5.3 | 4.7 | 0.4 | 6.7 |
| Chat-Scene | — | — | — | — | — | — | — | 8.2 |
| GPT-4o | 30.2 | 25.8 | 36.6 | 35.5 | 20.5 | 36.5 | 35.0 | 31.3 |
| Qwen-VL-max | 28.7 | 17.6 | 37.1 | 13.4 | 14.5 | 33.0 | 21.5 | 23.5 |
| LLaVA-Video | 13.8 | 11.3 | 24.9 | 17.7 | 13.3 | 20.8 | 14.1 | 16.3 |
| **HIS-GPT** | **44.6** | **42.1** | **55.5** | **41.0** | **50.3** | **53.2** | **53.9** | **48.7** |

**Key Findings**:
- HIS-GPT achieves an average score of 48.7, surpassing the strongest baseline GPT-4o (31.3) by 17.4 points
- Advantages are most pronounced on tasks requiring fine-grained spatial understanding (Human Position, Contact Part)
- Vision LLMs benefit from strong instruction-following and visual generalization but still fall far short of HIS-GPT
- 3D scene LLMs perform worst due to the absence of human body data in their training corpora

### Ablation Study

| Configuration | AInt (act/spa/cont) | PE | Act. | Spa. | HoI | **Avg.** |
|---------------|--------------------|----|------|------|-----|----------|
| Baseline | None | sine | 41.8 | 34.7 | 45.8 | 43.0 |
| +AInt | ✓✓✓ | sine | 43.5 | 35.3 | 51.0 | 44.1 |
| +LTP | None | LTP | 43.5 | 38.8 | 50.3 | 46.0 |
| **+AInt+LTP** | ✓✓✓ | LTP | **44.6** | **42.1** | **55.5** | **48.7** |

- **AInt contribution**: +1.1 average; act/spa/cont yield +1.3 on Activity, +0.9 on Spatial, +1.7 on HoI, respectively
- **LTP contribution**: +3.0 average
- **Joint usage**: +5.7 average, demonstrating strong complementarity between the two modules

Training strategy ablation confirms that adding scene and motion caption data in Stage 1 improves average score by 2.9 points, validating the importance of modality alignment.

## Highlights & Insights

1. **Pioneering task definition**: The paper is the first to define the HIS-QA task and construct the HIS-Bench benchmark, filling the gap in joint 3D human-in-scene understanding evaluation, with 3 capability dimensions, 7 core tasks, 16 sub-tasks, and 800 questions.
2. **Elegant interaction modeling**: The AInt module explicitly learns human–scene interaction cues (activity, spatial relations, contact) through three auxiliary tasks, incurring no additional inference overhead.
3. **Unified spatiotemporal alignment**: The LTP module aligns scene spatial layout and motion trajectory into a common coordinate system via Fourier positional encoding, improving cross-modal perception.
4. **Scalable annotation pipeline**: A combination of 3D segmentation tools, video captioning models, and rule-based algorithms enables automatic multi-faceted annotation, reducing manual annotation costs.

## Limitations & Future Work

1. Training data is drawn from only two HIS datasets (PROX and GIMO), limiting diversity in scenes and activity categories.
2. Several HIS-Bench tasks (focused analysis, situational analysis, navigation) still require manual annotation, constraining scalability.
3. The scene encoder and motion encoder are frozen during training, potentially limiting the depth of cross-modal feature fusion.

## Related Work & Insights

- **3D Scene LLMs**: LL3DA and Chat-Scene excel at scene QA and grounding but cannot process the human body modality.
- **3D Human LLMs**: MotionGPT and AvatarGPT support motion understanding and generation but lack environmental context.
- **Situated scene understanding**: SQA3D and similar works determine agent position via text or egocentric vision, but do not incorporate whole-body pose representations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (a complete pioneering contribution — new task, new benchmark, new method)
- **Practicality**: ⭐⭐⭐⭐ (high application value for embodied AI, home robotics, etc.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (multi-baseline comparison and multi-dimensional ablation, though evaluation is confined to the authors' own benchmark)
- **Writing Quality**: ⭐⭐⭐⭐ (clear structure and detailed method description)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [\[ICCV 2025\] PHD: Personalized 3D Human Body Fitting with Point Diffusion](phd_personalized_3d_human_body_fitting_with_point_diffusion.md)
- [\[ICCV 2025\] Bring Your Rear Cameras for Egocentric 3D Human Pose Estimation](bring_your_rear_cameras_for_egocentric_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
