---
title: >-
  [Paper Note] SHANDS: A Multi-View Dataset and Benchmark for Surgical Hand-Gesture and Error Recognition Toward Medical Training
description: >-
  [CVPR 2026][Video Understanding][Surgical Training] SHANDS is the first multi-view RGB video dataset for open surgery training, recording incision and suturing operations from 52 experts/trainees using five synchronized cameras. It provides frame-level annotations for 15 gesture primitives and 8 clinically-validated error categories, establishing benchmarks for mainstream video models across single-view, multi-view, and cross-view protocols.
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Surgical Training"
  - "Multi-View Video"
  - "Gesture Recognition"
  - "Error Detection"
  - "Cross-View Generalization"
date: 2026-05-08
content_hash: 690aba0d43d1c719
---

# SHANDS: A Multi-View Dataset and Benchmark for Surgical Hand-Gesture and Error Recognition Toward Medical Training

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_SHands_A_Multi-View_Dataset_and_Benchmark_for_Surgical_Hand_Gesture_and_CVPR_2026_paper.html)  
**Code**: TBD  
**Area**: Video Understanding / Multi-View Action Recognition / Surgical Skill Assessment  
**Keywords**: Surgical Training, Multi-View Video, Gesture Recognition, Error Detection, Cross-View Generalization

## TL;DR
SHANDS is the first multi-view RGB video dataset for open surgery training, recording incision and suturing operations from 52 experts/trainees using five synchronized cameras. It provides frame-level annotations for 15 gesture primitives and 8 clinically-validated error categories, establishing benchmarks for mainstream video models across single-view, multi-view, and cross-view protocols.

## Background & Motivation

**Background**: The cultivation of surgical skills for medical students has long relied on expert observation and scoring, a process that is expensive, time-consuming, difficult to scale, and concentrated in specialized institutions. Computer vision and AI are expected to provide scalable, objective skill assessments, but progress is currently hindered.

**Limitations of Prior Work**: The bottleneck lies in the data. Existing surgical video datasets only cover half of the problem—robotic surgery datasets (e.g., JIGSAWS) offer synchronized kinematics and video but capture robot-mediated operations, losing the core manual hand-tool coordination of open surgery. Endoscopic datasets (e.g., Cholec80, M2CAI16) provide single-view laparoscopic footage that misses the surgeon’s hands and lacks fine-grained gesture boundaries or clinically validated error labels. Meanwhile, general multi-view action datasets (NTU RGB+D, Assembly101) demonstrate the value of multi-view modeling for complex operations but lack clinical supervision and surgical error taxonomies.

**Key Challenge**: Surgical datasets possess clinical authenticity but are mostly single-view without systematic error annotations; multi-view action datasets offer complementary views but lack surgical expertise. The intersection of "multi-view synchronized acquisition + fine-grained gestures + clinical error labels" remains a vacuum.

**Goal**: Construct an open surgery dataset capable of supporting three types of tasks—(1) gesture recognition, (2) error detection, and (3) cross-view generalization—while providing standard evaluation protocols.

**Key Insight**: Utilize five hardware-synchronized RGB cameras to capture standardized incision/suturing operations on ex-vivo chicken tissue from complementary perspectives. This approach preserves realistic tool-tissue interactions while significantly reducing occlusions and supporting cross-view reasoning via multi-view data.

**Core Idea**: Introduce the "multi-view action recognition" paradigm to surgical skill assessment for the first time. Combined with a dual-layer annotation system (gesture primitives + error types) validated by clinical experts, surgical skill assessment is transformed into a standard benchmark for modern video models.

## Method

As a dataset and benchmark paper, the core contribution is not a specific network but the acquisition system, annotation hierarchy, statistical design, and evaluation protocols.

### Overall Architecture
The construction pipeline for SHANDS includes: five-camera synchronized acquisition → standardized experimental workflow (experts/trainees each performing incision + suturing three times) → dual-layer annotation (15 gesture primitives + 8 error categories with frame-level boundaries) → definition of single/multi/cross-view protocols → benchmark evaluation using mainstream video backbones. The dataset was collected from 52 participants (20 board-certified surgeons, 32 medical trainees). Five Canon cameras recorded at 25 fps with $640 \times 480$ resolution from complementary top-down and oblique angles. Hardware-level frame alignment was achieved using CHDK (Canon Hack Development Kit), eliminating the need for post-hoc registration and enabling cross-view pixel-level correspondence. The dataset contains approximately 900,000 frames, with 520,000 frames annotated (58%). Trainees contributed 90.9% of the duration, and experts 9.1%.

### Key Designs

**1. Five-Camera Hardware-Synchronized Multi-View Acquisition: Offsetting Occlusion with Complementary Perspectives**

A core difficulty in open surgery is that hand-tool-tissue interactions are often occluded by the practitioner, making information loss inevitable in single-view setups. This work uses five static RGB cameras (C1–C5) surrounding the surgical area, combining top-down and oblique angles to cover complementary fields of view. Hardware-level temporal synchronization via CHDK ensures frame-level alignment. This enables pixel-level correspondence across all views without post-hoc registration, making "cross-view reasoning" possible—fusing multi-view data during training to reduce occlusion and testing the model's ability to generalize to unseen camera positions. This distinguishes SHANDS from single-view endoscopic datasets and general datasets without synchronization guarantees.

**2. Dual-Layer Clinical Annotation System: Mapping Both "How to Do" and "What Went Wrong"**

Action categories alone are insufficient for skill feedback; identifying errors is critical. In collaboration with surgical educators, a two-level taxonomy was designed: the first layer consists of 15 gesture primitives, decomposing incision into I1–I5 and suturing into S1–S10 as clinically meaningful units with frame-level boundaries. The second layer includes 8 expert-validated error types (Incision II1–II3, Suturing IS1–IS5), covering improper tool handling, incorrect angles, repeated cutting, needle entry errors, excessive force, and poor knotting techniques. All annotations were completed by a single annotator following educator definitions and reviewed by surgeons to ensure consistency. Background/Idle classes were defined for meaningless segments to maintain temporal integrity.

**3. Three Standardized Evaluation Protocols: Single-View / Multi-View / Cross-View**

To ensure comparability, three cross-subject (test subjects do not appear in training) settings were defined: Single-view (training and testing on a single camera C1–C5); Multi-view (all five synchronized cameras available for both training and testing); and Cross-view generalization (training on C1–C3 and evaluating on unseen C4–C5) to examine view-invariant representation learning. The split is Train/Val/Test = 31/11/10 participants (approx. 60/20/20), maintaining the expert/trainee ratio. Error detection focuses on the trainee subset where error frequency is highest. Gesture recognition uses Top-1 and Macro-F1, while error detection is treated as 8-class multi-label classification using Top-1 and per-class F1.

### Loss & Training
No new loss functions are proposed; instead, mainstream video backbones are fine-tuned. Single-view gesture recognition benchmarks include CNN-based (R3D, X3D-M, SlowFast) and Transformer-based (TimeSformer, ViViT-B, MViTv2-B, VideoMAE-B/L, InternVideo2) models, all pre-trained on Kinetics-400 or ImageNet-21K. Multi-view methods (MVAction, ViewCLR, ViewCon, DVANet) are pre-trained on NTU RGB+D 120 before adaptation to SHANDS.

## Key Experimental Results

### Main Results

Single-view gesture recognition (Top-1, higher is better) shows that Transformers consistently outperform CNNs, with masked pre-training providing further gains. However, absolute accuracy remains moderate, highlighting the difficulty of fine-grained tool-tissue interaction and the domain gap:

| Method | Type | Pre-training | Top-1 (%) |
|------|------|--------|-----------|
| R3D | CNN | K400 | 52.3 ± 0.8 |
| SlowFast | CNN | K400 | 55.7 ± 1.2 |
| TimeSformer | Transformer | K400 | 58.4 ± 1.1 |
| MViTv2-B | Transformer | K400 | 61.8 ± 0.9 |
| VideoMAE-L | Transformer | K400 | 65.9 ± 0.8 |
| InternVideo2 | Transformer | Im21K+K400 | **68.9 ± 0.8** |

Multi-view gesture recognition (five synchronized views) significantly outperforms single-view, with methods explicitly modeling cross-view dependencies performing best:

| Method | Pre-training | Top-1 (%) | Macro-F1 |
|------|--------|-----------|----------|
| MVAction | NTU RGB+D 120 | 70.4 ± 0.9 | 0.704 |
| ViewCLR | NTU RGB+D 120 | 71.7 | 0.717 |
| ViewCon | NTU RGB+D 120 | 72.3 | 0.723 |
| DVANet | NTU RGB+D 120 | **73.6** | **0.736** |

### Ablation Study (View Count Impact & Cross-View Generalization)

Error detection on the trainee subset using the VideoMAE-B backbone shows gains as synchronized cameras increase:

| Config | Top-1 (%) | Description |
|------|-----------|------|
| Single-view VideoMAE-B | 60.4 | Single camera baseline |
| ViewCon (Multi-view) | 66.2 | View consensus fusion |
| DVANet (Five-view) | **68.5** | Decoupled view-invariant representation |

In cross-view generalization (C1–C3 training → C4–C5 testing), DVANet is more robust on unseen views:

| Method | Train View Top-1 (%) | Unseen View Top-1 (%) | Retention |
|------|------|------|------|
| ViewCon | 68.4 (Macro-F1 0.662) | 63.2 | 92.4% |
| DVANet | 72.8 (Macro-F1 0.706) | **68.5** | **94.1%** |

### Key Findings
- **Multi-view is a necessity, not an elective**: Multi-view information consistently improves both gesture recognition and error detection (error detection 60.4% → 68.5%), validating the value of complementary perspectives under occlusion.
- **Decoupled view-invariant representations generalize best**: DVANet decouples view-specific and view-invariant factors, maintaining a 94.1% performance retention during cross-view transfer. This suggests surgical training centers could deploy pre-trained models across different camera setups without re-annotating for new views.
- **Fine-grained errors remain challenging**: While DVANet performs well on "improper tool handling" (77.3%) and "outcome error" (76.2%), subtle categories like "needle entry error" (60.9%) and "excessive force" (61.8%) remain difficult, indicating a need for more precise temporal and contextual modeling.

## Highlights & Insights
- **The dual-layer annotation** of "what is being done" and "what is wrong" is the most valuable design. It directly addresses the real-world demand for skill feedback in surgical education.
- **Hardware synchronization yielding pixel correspondence**: Using CHDK for frame-level synchronization bypasses the need for post-registration. This technical foundation enables valid cross-view protocols and provides a blueprint for low-cost multi-view acquisition.
- **Unlabeled sequences for semi-supervised learning**: Only a portion of trainee and expert footage is annotated, leaving room for research into semi-supervised or self-supervised learning on the remaining data.

## Limitations & Future Work
- **Scale and Task Constraints**: Limited to 10 hours and 52 participants across two operations (incision + suturing). While ex-vivo tissue is standardized, it lacks the complexity of living tissue, bleeding, and deformation found in real clinical scenarios.
- **Labeling Subjectivity**: Annotations were performed by a single individual (then reviewed by experts). The subjective nature of clinical judgment in error categorization may affect label consistency and benchmark upper bounds.
- **Error Distribution**: Error detection was only conducted on the trainee subset due to a lack of positive samples from experts, introducing a data distribution bias.

## Related Work & Insights
- **vs. JIGSAWS / DESK**: These focus on robot-mediated operations. SHANDS targets manual open surgery and provides multi-view RGB data.
- **vs. Cholec80 / M2CAI16**: These are single-view laparoscopic datasets that label phases but miss hand movements and error labels. SHANDS captures hands directly from multiple views.
- **vs. NTU RGB+D / Assembly101**: While these prove the value of multi-view data, they lack surgical expertise. SHANDS applies the multi-view paradigm to a specialized surgical context with validated error systems.

## Rating
- Novelty: ⭐⭐⭐⭐ First multi-view RGB dataset for open surgery with clinical error taxonomy. Filled a clear gap, though methods are based on existing video backbones.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers single/multi/cross-view protocols and multiple backbones, though scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed comparison tables, and well-described protocols.
- Value: ⭐⭐⭐⭐⭐ High potential for AI-driven surgical skill assessment, addressing a high-barrier real-world need.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](savax_egotoexo_imitation_error_detection_via_scene.md)
- [\[CVPR 2026\] SMV-EAR: Bring Spatiotemporal Multi-View Representation Learning into Efficient Event-Based Action Recognition](smv-ear_bring_spatiotemporal_multi-view_representation_learning_into_efficient_e.md)
- [\[CVPR 2026\] TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation](tacsim_a_dataset_and_benchmark_for_football_tactical_style_imitation.md)
- [\[CVPR 2026\] MV-TAP: Tracking Any Point in Multi-View Videos](mv-tap_tracking_any_point_in_multi-view_videos.md)
- [\[CVPR 2026\] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)

</div>

<!-- RELATED:END -->
