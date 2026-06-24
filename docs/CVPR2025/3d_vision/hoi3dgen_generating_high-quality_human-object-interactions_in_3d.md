---
title: >-
  [Paper Note] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D
description: >-
  [CVPR 2025][3D Vision][Human-Object Interaction Generation] The HOI3DGen framework is proposed, which automatically annotates high-quality interaction data via MLLMs, fine-tunes a diffusion model conditioned on view, and performs 3D lifting along with SMPL registration. It is the first to achieve high-quality 3D human-object interaction generation with precise contact-semantic control from text, outperforming baselines by 4-15x in text consistency.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human-Object Interaction Generation"
  - "Text-to-3D"
  - "Contact Semantics"
  - "Data Annotation"
  - "SMPL"
date: 2026-05-08
content_hash: 1e85bfda50d41c97
---

# HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D

**Conference**: CVPR 2025  
**arXiv**: [2603.12126](https://arxiv.org/abs/2603.12126)  
**Code**: [https://virtualhumans.mpi-inf.mpg.de/hoi3dgen/](https://virtualhumans.mpi-inf.mpg.de/hoi3dgen/)  
**Area**: 3D Vision  
**Keywords**: Human-Object Interaction Generation, Text-to-3D, Contact Semantics, Data Annotation, SMPL

## TL;DR
The HOI3DGen framework is proposed, which automatically annotates high-quality interaction data via MLLMs, fine-tunes a diffusion model conditioned on view, and performs 3D lifting along with SMPL registration. It is the first to achieve high-quality 3D human-object interaction generation with precise contact-semantic control from text, outperforming baselines by 4-15x in text consistency.

## Background & Motivation
**Background**: 3D Human-Object Interaction (HOI) generation is crucial for gaming and AR/XR. Existing methods either only generate the human or the object, or rely on SDS (Score Distillation Sampling) to distill 3D assets from text-to-image models.

**Limitations of Prior Work**: SDS-based methods (e.g., InterFusion) suffer from the Janus problem, noisy textures, and severe interpenetration, and they cannot precisely control contact points. Direct 3D generation methods (e.g., TRELLIS) lack interaction awareness. The fundamental cause is the lack of high-quality **paired text-3D interaction data**.

**Key Challenge**: Existing 3D interaction datasets (e.g., BEHAVE) feature rich interactions but lack fine-grained textual descriptions; manual annotation is unscalable; GPT-4V annotation is costly and yields unnatural descriptions.

**Goal**: (1) How to automatically generate high-quality interaction text annotations? (2) How to enable the model to learn interaction generation with minimal data? (3) How to ensure that 3D results have correct contact semantics?

**Key Insight**: Decompose complex annotation tasks into sub-tasks of appearance, action, and contact, allowing open-source MLLMs to solve them separately; fine-tuning the interaction capability with only 400 high-quality samples.

**Core Idea**: Use a decomposed MLLM annotation pipeline + fine-tuning a diffusion model with a minimal set of finely filtered data to achieve text-driven 3D HOI generation with precise contact control.

## Method

### Overall Architecture
The input is a text description (including human appearance, object, action, and contact regions), and the output is segmented human and object textured meshes + an aligned, animatable SMPL model. The pipeline consists of three steps: (1) automated data annotation to construct a high-quality training set; (2) view-conditioned fine-tuning of a text-to-image model to generate 2D interaction images; and (3) 2D-to-3D lifting + interaction segmentation + SMPL registration.

### Key Designs

1. **Decomposed Annotation Pipeline**:

    - **Function**: Automatically generate high-quality text descriptions from the ProciGen 3D interaction dataset.
    - **Mechanism**: Decompose annotation into three sub-tasks: appearance annotation (using InternVL to describe clothing/hairstyle/shoes and object properties from 4 orthogonal views), interaction annotation (selecting action types from a predefined action list + analyzing body parts in contact with the object <4cm using SMPL), and comprehensive description generation (using LLaMA 3.1 70B to integrate all sub-annotations into natural language).
    - **Design Motivation**: To avoid hallucination in end-to-end annotation; decomposing makes each sub-task simpler and manageable for open-source MLLMs; contact information is derived directly from 3D geometric analysis rather than generation, ensuring accuracy.

2. **Data Filtering**:

    - **Function**: Filter 400 high-quality samples from 750k ProciGen samples.
    - **Mechanism**: Group samples into 8 contact configurations (right hand/left hand/both hands/back/right leg/left leg/no contact/others), filter out interpenetrations, implausible actions, and action-contact mismatches, retaining 50 samples per category.
    - **Design Motivation**: Data quality > quantity—a small amount of high-quality, diverse data is sufficient for fine-tuning, preventing forgetting and erroneous learning caused by noisy data.

3. **View-Conditioned Generation**:

    - **Function**: Fine-tune the SANA diffusion model to generate interaction images from specified views.
    - **Mechanism**: Append view descriptions $t_v \in \{\text{front}, \text{left diagonal}, \text{right diagonal}\}$ corresponding to azimuth angles $0°, -45°, +45°$ to the text prompt, and fine-tune with standard diffusion loss. After generation, re-texturize using Flux to enhance quality.
    - **Design Motivation**: Multi-view generation stabilizes the subsequent 3D lifting step—at least 1 of the 3 views completely displays the interaction, improving contact accuracy from 78% (single view) to 90% (3 views).

4. **3D lifting + interaction segmentation + SMPL registration**:

    - **Function**: Lift 2D interaction images into semantically segmented 3D meshes.
    - **Mechanism**: Lift single images to 3D meshes using Hunyuan3D $\rightarrow$ render video frames along the camera trajectory $\rightarrow$ segment human/object using Grounded SAM2 $\rightarrow$ assign vertex labels via multi-view voting (Eq. 2-3) $\rightarrow$ estimate SMPL with CameraHMR + optimize 7DoF via Chamfer distance.
    - **Design Motivation**: To output semanticized 3D interactions end-to-end, providing not only geometry but also contact point semantics and animatable properties.

### Loss & Training
SANA is fine-tuned on 4 H100 GPUs for 24 hours with an effective batch size of 16 and a resolution of 1024x1024. Only 400 samples are used for fine-tuning.

## Key Experimental Results

### Main Results

| Method | GPT Text Consistency ↑ | Contact Accuracy ↑ | User Preference (Text) ↑ | GPT Quality ↑ | User Preference (Quality) ↑ |
|------|--------|--------|---------|------|---------|
| InterFusion | 0.15 | N/A | 5.47% | 0.00 | 3.28% |
| TRELLIS | 0.04 | N/A | 3.44% | 0.21 | 10.16% |
| **Ours** | **0.81** | **90%** | **91.09%** | **0.79** | **85.56%** |

### Ablation Study

| Configuration | GPT ↑ | Contact Accuracy ↑ | Description |
|------|------|----------|------|
| w/o Data Filtering | 0.20 | 0.80 | Full ProciGen data leads to learning incorrect interactions |
| w/o Re-texturizing | 0.05 | 0.85 | GPT score drops significantly |
| Full model | 0.75 | 0.90 | Optimal combination of data filtering and re-texturizing |
| 1-View Lifting | — | 78.3% | Contact is often lost in single-view 3D lifting |
| 3-View Lifting | — | 90.0% | Multi-view significantly improves stability |

### Key Findings
- Only 400 finely filtered samples are needed to grant SANA strong interaction generation capabilities—pre-trained models already possess interaction "potential," requiring only structured data for activation.
- CLIP score is unsuitable for evaluating interaction quality—it is insensitive to fine-grained contact semantics (Fig. 3 shows counterexamples); GPT scoring and contact accuracy are more reasonable.
- The contact accuracy of baseline SANA is only 45.76% and is heavily concentrated in "right hand" and "both feet" configurations; after removing these two classes, it drops to 23%, indicating severe bias.
- The SMPL registration scheme (1.60cm Chamfer distance) significantly outperforms ETCH (4.30cm), as ETCH is trained solely on standing poses.

## Highlights & Insights
- **The Power of Small-Data Fine-Tuning**: Teaching the diffusion model to generate diverse interactions with only 400 samples indicates that pre-trained models already encode rich interaction knowledge; the key is properly activating it with high-quality data. This concept can be generalized to other compositional generation tasks.
- **Smart Decomposed Annotation**: Instead of asking the MLLM to describe complex interactions all at once, splitting it into sub-tasks of appearance, action, and contact before combining them significantly reduces hallucination rates.
- **Contact Semantics Directly Analyzed from 3D Geometry** instead of being conjectured by language models, ensuring ground-truth quality.

## Limitations & Future Work
- Textual descriptions of complex human poses are not precise enough—language descriptions of poses are inherently ambiguous; specialized text-to-pose modules could be incorporated in the future.
- Dependency on Hunyuan3D for 2D-to-3D lifting, where its quality serves as a bottleneck.
- Contact configuration annotation relies on a predefined list, which might miss atypical interaction patterns.
- Training data only originates from ProciGen (100 human bodies + 18 object categories); generalizing to more object categories requires additional data.
- Multi-person interaction scenarios are entirely unaddressed—currently supporting only one person and one object; scaling to multi-person/multi-object combinatorial explosion requires new data strategies.
- The SMPL registration step relies on the initial estimation quality of CameraHMR and may fail on extreme poses (e.g., handstands, crouching).

## Related Work & Insights
- **vs InterFusion**: SDS-based, slow with severe Janus problems, and uncontrollable contacts. Ours is learning-based, fast, and contact-accurate.
- **vs TRELLIS**: Strong general 3D generation but lacks interaction awareness, frequently ignoring the object or producing partial objects. Ours is interaction-aware and contact-controllable.
- **vs ComboVerse**: Compositional 3D generation requires separate meshes, whereas ours generates end-to-end directly from text.
- **Generality of Decomposed Annotation**: The method of decomposing a complex annotation task into sub-tasks for open-source MLLMs to process separately can be extended to other multi-attribute data annotation scenarios (such as scene graphs, action videos).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The concept of decomposed annotation + extreme small-data fine-tuning is inspiring; technical novelty is moderate.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Quantitative + user study + multi-dimensional ablations; highly thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Clear and fluent; the analysis on why CLIP is unsuitable is excellent.
- **Value**: ⭐⭐⭐⭐ Provides a practical push for 3D HOI generation; the data annotation pipeline is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Guiding Human-Object Interactions with Rich Geometry and Relations](guiding_human-object_interactions_with_rich_geometry_and_relations.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes](hearing_hands_generating_sounds_from_physical_interactions_in_3d_scenes.md)
- [\[CVPR 2025\] Material Anything: Generating Materials for Any 3D Object via Diffusion](material_anything_generating_materials_for_any_3d_object_via_diffusion.md)
- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)

</div>

<!-- RELATED:END -->
