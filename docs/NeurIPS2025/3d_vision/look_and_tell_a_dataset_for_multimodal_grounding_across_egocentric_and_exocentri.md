---
title: >-
  [Paper Note] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views
description: >-
  [NeurIPS 2025][3D Vision][Multimodal grounding] Look and Tell introduces a multimodal dataset that synchronously captures gaze, speech, and dual-view video from 25 participants in a kitchen environment using Meta Aria smart glasses and a fixed GoPro camera. Combined with 3D scene reconstruction and a multi-level annotation pipeline, it provides the first benchmark for studying referential communication across egocentric and exocentric perspectives.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Multimodal grounding
  - egocentric/exocentric views
  - gaze tracking
  - referential communication
  - spatial intelligence
date: 2026-05-08
content_hash: 843eff9871e317e7
---

# Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views

**Conference**: NeurIPS 2025
**arXiv**: [2510.22672](https://arxiv.org/abs/2510.22672)
**Code**: None
**Area**: 3D Vision
**Keywords**: Multimodal grounding, egocentric/exocentric views, gaze tracking, referential communication, spatial intelligence

## TL;DR

Look and Tell introduces a multimodal dataset that synchronously captures gaze, speech, and dual-view video from 25 participants in a kitchen environment using Meta Aria smart glasses and a fixed GoPro camera. Combined with 3D scene reconstruction and a multi-level annotation pipeline, it provides the first benchmark for studying referential communication across egocentric and exocentric perspectives.

## Background & Motivation

**State of the Field**: Understanding how humans coordinate gaze, gesture, and speech during referential communication is a fundamental prerequisite for building embodied agents capable of natural interaction. Researchers need to understand visual attention patterns, linguistic strategies, and their temporal relationships when humans describe or identify objects, in order to enable robots to interpret user intent within shared environments.

**Limitations of Prior Work**: Existing multimodal communication datasets suffer from three critical gaps. First, most gaze–speech studies focus on the temporal properties of gaze (e.g., whether gaze precedes speech), while neglecting the effect of spatial representation — the performance difference between 2D image space and 3D reconstructed space for grounding has not been systematically compared. Second, virtually no existing dataset simultaneously provides egocentric and exocentric views of the same interaction, even though collaborative human–robot systems require agents to understand both the user's first-person intent (what the user is looking at) and an objective third-person world model (where objects are located). Third, complete synchronized recordings of gaze, speech, and video in natural settings are extremely scarce — most studies are conducted in controlled laboratory environments with limited ecological validity.

**Root Cause**: Embodied agents must interpret multimodal communication signals across views and spatial representations, yet no dataset simultaneously covers ego/exo perspectives, 2D/3D representations, and synchronized gaze/speech to systematically study their effects on grounding performance.

**Core Idea**: Construct the first referential communication dataset in natural settings that integrates synchronized dual-view capture, 3D scene reconstruction, and a multi-level automated annotation pipeline, providing a standardized benchmark for multimodal grounding under different spatial representations and view combinations.

## Method

### Overall Architecture

Dataset construction proceeds in three stages: (1) synchronized dual-view data collection — 25 participants wearing Meta Aria smart glasses recall recipes and locate ingredients in a kitchen while a fixed GoPro records the exocentric view; (2) a multi-stage automated annotation pipeline — cascading audio transcription, language understanding, visual grounding, and mask propagation to produce frame-level temporally aligned annotations; and (3) 3D scene reconstruction — point clouds extracted from separate recordings via Meta Aria MPS services and normalized to a unified coordinate system.

### Key Designs

1. **Dual-View Synchronized Data Capture System**:

    - Function: Simultaneously acquire the participant's egocentric view (with gaze tracking) and a fixed exocentric view in a natural kitchen setting.
    - Mechanism: Participants memorize recipe steps while wearing Meta Aria smart glasses, then verbally describe and locate ingredients. The Aria glasses provide synchronized gaze vectors (fixation events + gaze vectors), RGB video (1408×1408 @ 30 fps), and audio; a fixed GoPro camera records the environment from the side. Both devices are synchronized via timestamps. Each participant completes 5 recipes, yielding 125 recording sessions totaling 3.67 hours (396,208 RGB frames).
    - Design Motivation: The egocentric view directly reflects the user's visual attention and interaction intent, while the exocentric view provides an objective scene context unaffected by head movement. Together they address the dual demands of "understanding user intent" and "maintaining a world model" in shared autonomy scenarios — a core challenge in human–robot collaboration.

2. **Four-Stage Cascaded Annotation Pipeline**:

    - Function: Automatically transform raw audio-visual data into frame-level referential expression annotations with temporal alignment, target localization, and segmentation masks.
    - Mechanism: The pipeline chains four models: (a) **WhisperX** for word-level speech transcription with precise timestamps; (b) **GPT** for extracting ingredient/object mentions from transcriptions, performing coreference resolution (handling pronouns such as "it" and demonstratives such as "that"), and outputting a structured mention graph with unique IDs and anaphoric links; (c) **Molmo**, a vision–language model, for localizing mentioned objects as 2D coordinates in corresponding frames; (d) **SAM2** for propagating segmentation masks across frames from Molmo seed points. Post-processing includes name normalization, alias expansion, temporal alignment verification, and coreference graph construction.
    - Design Motivation: The fully automated pipeline reduces annotation cost across 2,707 mentions from 25 participants × 5 recipes, while human correction is introduced where Molmo is unreliable (small or visually similar objects) — 747 mentions required manual intervention (106 skipped due to invisible targets, 641 manually annotated due to small object size), ensuring annotation quality.

3. **3D Scene Reconstruction and Unified Coordinate System**:

    - Function: Extract point clouds from separately recorded room videos to construct a 3D model of the kitchen.
    - Mechanism: Image frames are extracted from multiple recording passes at different viewpoints using the Meta Project Aria MPS service, point clouds are generated and normalized to a canonical coordinate system. Alignment across multiple reconstructions ensures spatial consistency across recording sessions, enabling gaze and referential behavior from all participants to be compared within a single 3D space.
    - Design Motivation: Object positions in 2D image space are affected by viewing angle and distance, whereas 3D reconstruction provides a view-independent absolute spatial reference. This allows the dataset to systematically compare the effect of 2D vs. 3D representations on multimodal grounding — an analysis dimension that prior datasets could not support.

### Loss & Training

This is a dataset paper and does not involve model training. Quality assurance strategies include: deterministic post-processing rules applied after GPT extraction (name normalization to canonical recipe/distractor names, alias mapping such as tap→sink, precise timestamp-text alignment, coreference graph integrity verification); a dual-channel automatic-plus-manual verification scheme for Molmo localization; and sub-second synchronization between WhisperX word-level timestamps and Aria fixation events.

## Key Experimental Results

### Main Results

| Statistic | Value |
|-----------|-------|
| Participants | 25 (18 female, 7 male; ages 22–37) |
| Total recording duration | 3.67 hours (220.1 minutes) |
| Total RGB frames | 396,208 (30 fps) |
| Recording sessions | 125 (5 recipes per participant) |
| Mean session duration | 1.8 ± 0.7 min (range 0.7–4.3) |
| Annotated referential expressions | 2,707 (mean 22 per session) |
| Native language diversity | 12 languages (Mandarin: 10, Swedish: 4, Korean/Icelandic/Spanish, etc.) |

### Ablation Study

| Dimension | Value | Note |
|-----------|-------|------|
| Ingredient mentions | 1,680 (62.1%) | Dominant referential type |
| Pronouns/coreference | 614 (22.7%) | Indirect reference accounts for nearly 1/4, highlighting coreference resolution importance |
| Incidental objects | 154 (5.7%) | Environmental objects not in the recipe |
| Distractors | 28 (1.0%) | Ingredients absent from the current recipe |
| Mean coreference chain length | 6.7 | Same object repeatedly mentioned |
| Ingredient coverage | >90% | Nearly all target ingredients mentioned at least once |
| Automatic localization rate | 72% | Molmo reliable for large objects |
| Manual correction rate | 28% (747/2707) | Small and visually similar objects require human annotation |

### Key Findings

- **Gaze–Speech Temporal Relationship**: The mean temporal offset of gaze preceding speech is $\Delta = -189$ ms (median $-102$ ms), indicating that participants typically still fixate the object when naming it. In 41.1% of mentions, gaze precedes speech onset, suggesting that gaze is a predictable signal for linguistic reference.
- **Temporal Overlap**: The mean overlap between gaze and mention is 352 ms (median 367 ms), indicating high synchrony without a strict sequential ordering.
- **Gaze Range**: $\Delta$ spans $[-6159, +531]$ ms, with extreme cases where gaze precedes speech by several seconds or slightly follows it.
- **High Pronoun Rate**: Nearly 23% of references are indirect (pronouns, demonstratives), underscoring the importance of coreference resolution for practical multimodal understanding systems.

## Highlights & Insights

- **Cross-View Design Fills an Important Gap**: Providing simultaneous ego and exo view data from the same interaction enables, for the first time, systematic comparison of how viewing perspective and spatial representation (2D vs. 3D) affect multimodal grounding. Prior datasets offer either ego (e.g., Ego4D) or exo views in isolation, precluding such comparisons.
- **Multi-Level Annotation Pipeline Balances Efficiency and Quality**: The four-stage cascaded pipeline (WhisperX→GPT→Molmo→SAM2) achieves automated annotation from audio to frame-level masks, with human correction providing a quality safety net. This "automation-first with human fallback" strategy offers broadly applicable guidance for annotating medium-scale datasets.
- **Ecological Validity of Natural Settings**: Conducted in a real kitchen with participants freely narrating recipes and locating ingredients, the task is considerably closer to real-world human–robot interaction than instruction-following tasks in controlled laboratory environments. The multilingual participant pool (12 native languages) further enhances linguistic diversity.

## Limitations & Future Work

- The dataset scale is small (25 participants, 3.67 hours), substantially smaller than large-scale datasets such as Ego4D (3,000 hours).
- The single-scene setting (KTH kitchen laboratory only) may limit the generalizability of findings due to environmental bias.
- Participants are predominantly university students and staff (ages 22–37), limiting demographic representativeness.
- The annotation pipeline depends on four external models (WhisperX, GPT, Molmo, SAM2), and errors may propagate across stages.
- The paper provides no quantitative evaluation of any baseline method (e.g., cross-view grounding accuracy), presenting only the dataset and its analysis.
- Gesture annotation is not yet complete; the current release lacks information on pointing and co-speech gestures.

## Related Work & Insights

- **vs. Ego4D**: Ego4D provides large-scale egocentric video but lacks synchronized exocentric views and gaze data. Look and Tell is smaller in scale but richer in modality combination, making it better suited for fine-grained communicative behavior analysis.
- **vs. ScanQA / EmbodiedQA**: These datasets support question answering in 3D reconstructed scenes but lack egocentric gaze and naturalistic speech data. The combination of 3D reconstruction, gaze, and speech in Look and Tell is better suited for studying embodied grounding.
- **vs. VENUS**: VENUS combines speech, facial expression, and body pose but focuses on conversational affect rather than spatial grounding. Look and Tell targets spatial referential tasks and is thus complementary.
- **vs. Kontogiorgos et al. (2018)**: A multimodal collaborative corpus collected in a controlled environment without 3D reconstruction. The dual-view and 3D design of Look and Tell is better suited for studying cross-representational-space grounding.

## Rating

- Novelty: ⭐⭐⭐⭐ The first referential communication dataset integrating ego/exo dual views, gaze, speech, and 3D reconstruction, filling an important data gap.
- Experimental Thoroughness: ⭐⭐⭐ Detailed statistical analysis and valuable gaze–speech synchrony analysis, but no quantitative evaluation of baseline models.
- Writing Quality: ⭐⭐⭐⭐ Dataset construction pipeline and annotation workflow are described clearly; statistical analysis is rigorous.
- Value: ⭐⭐⭐ Useful reference for embodied interaction and human–robot collaboration research, though limitations in scale and scene diversity constrain its direct impact.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants](indego_a_dataset_of_industrial_scenarios_and_collaborative_work_for_egocentric_a.md)
- [\[ICLR 2026\] EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations](../../ICLR2026/3d_vision/egoworld_translating_exocentric_view_to_egocentric_view_using_rich_exocentric_ob.md)
- [\[ICCV 2025\] EgoM2P: Egocentric Multimodal Multitask Pretraining](../../ICCV2025/3d_vision/egom2p_egocentric_multimodal_multitask_pretraining.md)
- [\[NeurIPS 2025\] From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes](from_objects_to_anywhere_a_holistic_benchmark_for_multi-level_visual_grounding_i.md)
- [\[NeurIPS 2025\] CLIPGaussian: Universal and Multimodal Style Transfer Based on Gaussian Splatting](clipgaussian_universal_and_multimodal_style_transfer_based_on_gaussian_splatting.md)

<!-- RELATED:END -->
