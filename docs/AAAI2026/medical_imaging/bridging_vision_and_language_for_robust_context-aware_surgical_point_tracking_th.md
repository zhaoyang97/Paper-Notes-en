---
title: >-
  [Paper Note] Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark
description: >-
  [AAAI 2026][Medical Imaging][Surgical point tracking] This paper presents VL-SurgPT, the first large-scale multimodal surgical point tracking dataset combining visual coordinates with textual state descriptions…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Surgical point tracking"
  - "vision-language multimodal"
  - "dataset"
  - "text-guided"
  - "surgical video analysis"
date: 2026-05-08
content_hash: e50c5e0a4e22d292
---

# Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark

**Conference**: AAAI 2026
**arXiv**: [2511.12026](https://arxiv.org/abs/2511.12026)
**Code**: [Available](https://szupc.github.io/VL-SurgPT/)
**Area**: Medical Imaging
**Keywords**: Surgical point tracking, vision-language multimodal, dataset, text-guided, surgical video analysis

## TL;DR

This paper presents VL-SurgPT, the first large-scale multimodal surgical point tracking dataset combining visual coordinates with textual state descriptions, and proposes TG-SurgPT, a text-guided tracking method that leverages semantic information to significantly improve tracking accuracy and robustness in complex surgical scenes.

## Background & Motivation

Precise point tracking in surgical environments is critical for computer-assisted surgery, enabling applications such as motion understanding and scene perception. However, surgical scenes introduce extreme visual challenges:

- **Electrosurgical smoke**: Dense smoke occludes tracking targets
- **Specular reflection**: Moist anatomical surfaces generate strong highlights
- **Instrument occlusion**: Surgical instruments dynamically occlude tissue
- **Tissue deformation**: Non-rigid tissue undergoes continuous deformation during surgical manipulation

**Limitations of Prior Work**:

**Lack of semantic context**: Datasets such as TAP-Vid, SurgT, and STIR provide only geometric coordinates and cannot explain the causes of tracking failure.

**Unimodal limitation**: All existing datasets consist of purely visual annotations without textual descriptions.

**Incomplete coverage**: Most datasets focus on either tissue tracking or instrument tracking, but not both.

**Ex vivo settings**: Many datasets are based on ex vivo setups that fail to reflect the complexity of real surgical procedures.

Core hypothesis: **Introducing semantic textual descriptions can provide contextual guidance unavailable to purely visual methods, substantially improving tracking robustness when visual features are unreliable.**

## Method

### Overall Architecture

The paper's contributions comprise two major components: the construction of the VL-SurgPT dataset and the TG-SurgPT tracking method.

#### VL-SurgPT Dataset

Data were collected from a da Vinci Xi robotic surgical system at Shenzhen People's Hospital, covering complex procedures including radical gastrectomy and radical proctectomy. A total of approximately 33 hours of video from 20 surgeries were recorded, of which 115 minutes were selected for annotation.

**Data acquisition pipeline**:

1. ICG (indocyanine green) fluorescent dye is used intraoperatively to mark tracking points; high-precision ground-truth coordinates are obtained under UV mode.
2. The system is switched to white-light mode for 5–10 seconds of normal surgical operation.
3. UV mode is reactivated to record endpoint coordinates.

**Annotation content** (per point):

- 2D coordinates $(x, y)$ or "null" (when invisible)
- Point type: Tissue or Instrument
- **Textual state description**: Clear View / Pulled / Reflection / Smoke Obscuration / Instrument Obscuration / Tissue Obscuration / Out of View / External Occlusion / Self-occlusion
- Instrument-specific labels: instrument type (7 categories), instance ID

**Dataset scale**:

- Tissue tracking subset: 754 video clips, 180.5k frames, 7,117 annotated frames, 1,862 trajectories, 17,171 visible points
- Instrument tracking subset: 154 video clips, 26,490 frames, 1,108 annotated frames, 7 instrument types
- Covers 5 challenging scenarios: tissue deformation, instrument occlusion, camera shake, surface reflection, and electrosurgical smoke

### Key Designs: TG-SurgPT

TG-SurgPT extends the Track-On framework via a dual-branch architecture that integrates visual and textual modalities:

**Text branch**: Two semantic attributes per tracking point (point type + point state) are encoded through a frozen CLIP Text Encoder to produce text features $F_t \in \mathbb{R}^{2 \times 512}$.

**Visual branch**: The frozen Track-On model processes query points and video frames, outputting query point features $F_q$, dense features of the current frame $F_h$, coarse matching position $F_p$, and initial coordinates $(X_N, Y_N)$.

**Attributes Prediction Head**:

- $F_p$ and $F_h$ are fused via a multi-scale correlation module to yield $F_{p\text{-}h}$
- Multi-scale deformable attention is applied with $F_q$ as Query and $F_{p\text{-}h}$ as Key/Value
- Two parallel classification heads: Point Type Head ($2 \times N$) and Point Status Head ($7 \times N$ or $4 \times N$)
- **Key design**: At inference, the model's own predicted state labels replace GT text, enabling fully automatic text-guided tracking

**Text-guided Attention**:

- Text features $F_t$ are projected to the same dimensionality as $F_q$ via a linear transformation
- Standard cross-attention: $F_q$ as Query, $F_t$ as Key/Value → fused representation $F_{t\text{-}q}$
- $F_{t\text{-}q}$ and $F_{p\text{-}h}$ generate refined offsets $(a_N, b_N)$ through deformable attention
- Final prediction: $(\hat{X}_N, \hat{Y}_N) = (X_N + a_N, Y_N + b_N)$

### Loss & Training

The total loss comprises three terms:

$$\mathcal{L} = \sum_{t \in \mathcal{T}} \left( \underbrace{\mathcal{H}_\delta(\hat{\mathbf{p}}_t - \mathbf{p}_t)}_{\mathcal{L}_p} + \underbrace{\|\Delta^2 \hat{\mathbf{p}}_t\|_1}_{\mathcal{L}_s} + \underbrace{\mathcal{L}_{CE}(\hat{\mathbf{s}}_t, \mathbf{s}_t)}_{\mathcal{L}_{text}} \right)$$

- **Point distance loss** $\mathcal{L}_p$: Huber loss measuring the distance between predicted and ground-truth coordinates
- **Trajectory smoothness loss** $\mathcal{L}_s$: Minimizes second-order differences of predicted trajectories to promote temporal smoothness
- **Text classification loss** $\mathcal{L}_{text}$: Cross-entropy loss supervising state label classification

Training characteristic: The model can be trained on **sparsely annotated real surgical data** without relying on dense synthetic data (e.g., MOVi-F), owing to the supplementary supervision provided by textual annotations.

## Key Experimental Results

### Main Results

Comparison of 8 state-of-the-art tracking methods and TG-SurgPT on VL-SurgPT:

**Tissue tracking subset:**

| Method | AJ↑ | $<\delta_{avg}^x$↑ | OA↑ | EPE↓ |
|--------|-----|---------------------|-----|------|
| RAFT | 27.81 | 30.37 | 85.81 | 99.73 |
| BootsTAP | 56.93 | 62.77 | 87.87 | 23.52 |
| MFTIQ | 61.52 | 63.44 | 87.80 | 19.81 |
| Track-On | 58.55 | 66.27 | 88.81 | 13.79 |
| **TG-SurgPT** | **62.88** | **67.77** | **91.04** | **11.02** |

**Instrument tracking subset:**

| Method | AJ↑ | $<\delta_{avg}^x$↑ | OA↑ | EPE↓ |
|--------|-----|---------------------|-----|------|
| Track-On | 46.97 | 59.18 | 85.07 | 41.67 |
| **TG-SurgPT** | **49.52** | **62.94** | **89.79** | **39.14** |

Improvements on instrument tracking are more pronounced: AJ +5.4%, positional accuracy +6.4%. Inference speed reaches 9.72 fps, approaching clinical real-time requirements.

### Ablation Study

| Fine-tune | Clip Length | Text | Tissue AJ↑ | Tissue EPE↓ | Inst. AJ↑ | Inst. OA↑ |
|-----------|-------------|------|------------|-------------|-----------|-----------|
| ✗ | - | ✗ | 58.55 | 13.79 | 46.97 | 85.07 |
| ✓ | short | ✗ | 61.09 | 12.82 | 47.48 | 86.43 |
| ✓ | short | ✓ | **62.88** | **11.02** | **49.52** | **89.79** |

- Fine-tuning improves AJ by +4.3% (tissue); text guidance yields an additional +2.9%
- Short clips (31 frames) outperform long clips (181 frames) by reducing temporal noise
- Text guidance improves instrument tracking OA by +3.9%, with particularly notable gains against occlusion

### Key Findings

- **Scene difficulty hierarchy**: Tissue deformation poses the greatest challenge, while electrosurgical smoke is surprisingly the easiest scenario (smoke produces distinctive spatiotemporal patterns that attention mechanisms can effectively exploit)
- **Text-guided advantage is amplified in difficult scenarios**: TG-SurgPT achieves its largest improvements under instrument occlusion and camera shake conditions
- **Visual state prediction accuracy**: Clear View (78.5%) and Pulled (85.1%) achieve high classification accuracy, while Out of View (48.1%) is the most difficult to predict

## Highlights & Insights

1. **First vision-language surgical tracking dataset**: VL-SurgPT is the first surgical tracking dataset to provide both coordinates and semantic state descriptions, filling an important gap in the field.
2. **Automated inference**: GT textual descriptions are used during training, while the model's own predicted state labels are used at inference, enabling text-guided tracking without manual annotation.
3. **ICG fluorescence for ground-truth acquisition**: The method cleverly leverages ICG marking techniques already present in clinical practice to obtain high-precision ground truth, closely aligned with real surgical workflows.
4. **Sparse annotation training**: The approach does not require dense synthetic data and trains directly on real surgical data, lowering the barrier to clinical deployment.

## Limitations & Future Work

- **Limited surgical types**: Coverage is restricted to gastrointestinal surgery (da Vinci Xi system); extension to other surgical specialties is needed.
- **Insufficient inference speed**: 9.72 fps does not yet meet clinical real-time requirements (typically ≥25 fps).
- **Limited point state prediction accuracy**: Out of View achieves only 48.1% accuracy; occlusion disambiguation capability requires further improvement.
- **2D tracking only**: 3D reconstruction and depth information are not addressed, limiting applicability to 3D surgical navigation.

## Related Work & Insights

- **Track-On**: The visual backbone of TG-SurgPT, balancing efficiency and performance (10.85 fps)
- **SurgMotion**: A recent surgical tracking method, but remains purely visual and unimodal
- **CLIP**: A frozen CLIP text encoder is used for semantic feature extraction, demonstrating that pretrained vision-language models can be effectively transferred to the surgical domain
- The concept of text-guided tracking is generalizable to other application scenarios where explaining tracking failures is important

## Rating

- **Novelty**: ★★★★★ — First work to introduce language into surgical point tracking; both the dataset and the method are pioneering
- **Experimental Thoroughness**: ★★★★★ — Comparison with 8 baselines, detailed per-scenario analysis, and comprehensive ablation study
- **Writing Quality**: ★★★★☆ — Clear structure with thorough description of the dataset construction pipeline
- **Value**: ★★★★☆ — Dataset is of exceptional value; inference speed still requires optimization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Surgical Smoke: A Smoke-Type-Aware Laparoscopic Video Desmoking Method and Dataset](rethinking_surgical_smoke_a_smoke-type-aware_laparoscopic_video_desmoking_method.md)
- [\[CVPR 2026\] Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos](../../CVPR2026/medical_imaging/synergistic_bleeding_region_and_point_detection_in_laparoscopic_surgical_videos.md)
- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [\[AAAI 2026\] DiA-gnostic VLVAE: Disentangled Alignment-Constrained Vision Language Variational AutoEncoder for Robust Radiology Reporting with Missing Modalities](dia-gnostic_vlvae_disentangled_alignment-constrained_vision_language_variational.md)
- [\[AAAI 2026\] Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks](sim4seg_boosting_multimodal_multi-disease_medical_diagnosis_segmentation_with_re.md)

</div>

<!-- RELATED:END -->
