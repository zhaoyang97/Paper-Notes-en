---
title: >-
  [Paper Note] GuideDog: A Real-World Egocentric Multimodal Dataset for Blind and Low-Vision Accessibility-Aware Guidance
description: >-
  [ACL 2026][Multimodal VLM][BLV Navigation] GuideDog utilizes an "expert standard-driven silver label generation + manual gold label verification" pipeline to construct 22K egocentric pedestrian scene image-text pairs (in…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "BLV Navigation"
  - "First-person View"
  - "MLLM Evaluation"
  - "Depth Perception"
  - "Human-AI Collaborative Annotation"
date: 2026-05-08
content_hash: 6778477bedf8bac9
---

# GuideDog: A Real-World Egocentric Multimodal Dataset for Blind and Low-Vision Accessibility-Aware Guidance

**Conference**: ACL 2026  
**arXiv**: [2503.12844](https://arxiv.org/abs/2503.12844)  
**Code**: https://jun297.github.io/GuideDog/  
**Area**: Multimodal VLM / Accessibility / Datasets  
**Keywords**: BLV Navigation, First-person View, MLLM Evaluation, Depth Perception, Human-AI Collaborative Annotation

## TL;DR
GuideDog utilizes an "expert standard-driven silver label generation + manual gold label verification" pipeline to construct 22K egocentric pedestrian scene image-text pairs (including an 818-question QA benchmark) from 269 global walking videos. This represents the first large-scale, geographically diverse, and standardized training and evaluation data for MLLMs in BLV (Blind and Low-Vision) navigation tasks.

## Background & Motivation

**Background**: 2.2 billion BLV individuals worldwide face safety challenges during independent travel; approximately 7% of visually impaired people fall at least once a month. Early solutions primarily utilized electronic mobility aids and computer vision for "obstacle detection/avoidance" but could not answer questions like "What environment am I in?" or "Where should I go next?". MLLMs (e.g., GPT-4o, Gemini, Qwen-VL) enable high-level scene understanding, and recent work has begun using MLLMs as BLV visual assistants.

**Limitations of Prior Work**: Existing BLV datasets are either too small (VIALM with 200 images, Merchant with 48), do not use a pedestrian first-person perspective (VizWiz contains casual photos of objects taken by BLV users), or suffer from severe geographic imbalance (WalkVLM covers only 10 locations). Furthermore, visual tasks require "BLV-aware" descriptions; average sighted annotators cannot accurately predict what BLV users truly care about, forcing a reliance on a few experts for manual writing, which limits scale and diversity.

**Key Challenge**: There is a structural contradiction between the "quality" and "scale" of BLV annotation—experts ensure quality but are in short supply; purely automated generation provides scale but often fails to meet real BLV needs; and sighted crowdsourcing lacks professional knowledge, leading to the omission of critical obstacles.

**Goal**: (1) Produce a diverse image-text dataset of at least 10,000 samples that aligns with professional BLV guidance standards using a scalable pipeline; (2) Provide a fine-grained evaluation benchmark to assess MLLM performance in object recognition and relative depth perception in real street scenes; (3) Systematically characterize the strengths and weaknesses of current MLLMs in BLV navigation.

**Key Insight**: Distill official BLV guidance standards (from over ten guidelines including Vision Australia, Be My Eyes, and Wisconsin DHS) into three machine-executable "GuideDog Standards" (S1 Environment / S2 Obstacles / S3 Direction). Use a "generation → verification" approach rather than "generation from scratch." MLLMs, detectors, and depth models first generate "silver labels," which humans then filter and correct to obtain "gold labels," shifting expensive human labor from writing to auditing.

**Core Idea**: A three-stage pipeline comprising "Expert Standard Template + Automatic Silver Label Generation + Manual Gold Label Verification" transforms unscalable expert writing into scalable expert auditing, complemented by a QA benchmark for spatial perception.

## Method

### Overall Architecture
The pipeline consists of four steps: (a) Scene Frame Acquisition—sampling frames from YouTube walking videos based on geographic diversity and removing duplicates; (b) Scene Information Extraction—extracting global (scene/location text) and local (80 categories of objects + bounding boxes + step distance + clock direction) information; (c) Silver Label Generation—inputting global and local info into a BLV-specific prompt for an MLLM to write navigation text following S1/S2/S3 standards; (d) Gold Label Generation—three annotators filter and correct samples based on image usability $c_1$ and standard compliance $c_2$. The final output includes 22K silver labels, 2K gold labels, and an 818-question QA benchmark.

### Key Designs

1.  **Three GuideDog Standards (S1/S2/S3)**:
    - **Function**: Compressing "how to guide" expertise from over ten BLV guidelines into three rigid standards that MLLMs can strictly follow and humans can audit.
    - **Mechanism**: S1 describes the environment (location + key elements), e.g., "You are on a busy pedestrian street with shops on both sides"; S2 describes obstacle type, clock direction, and step distance, e.g., "A signpost is at 12 o'clock, 4 steps ahead"; S3 provides summary directions using intuitive units ("3 steps," "1 o'clock") instead of precise measurements ("3 ft," "5 m"). These serve as prompt templates, manual checklists, and evaluation dimensions.
    - **Design Motivation**: BLV users prefer intuitive directions (steps + clock positions). "Free-form" generation often fails in real use. Explicit standards constrain MLLMs and provide measurable metrics.

2.  **Scene Information Extraction—80 Objects + Clock Directions + Step Distance**:
    - **Function**: Turning images into structured "global description + local obstacle list" for the MLLM to prevent missing critical obstacles or using incorrect terminology.
    - **Mechanism**: GPT-4o filters useless frames (sky/ground/occluded) and extracts scene/location descriptions $\mathcal{X}^{\text{global}}=(t^s,t^l)$. An open-vocabulary detector (Grounding-style) detects 80 BLV-critical object categories $\mathcal{O}_i,\mathcal{B}_i$, while a depth model (like Depth-Anything) generates a depth map $m_i$. Object distance $d_{ij}$ is the median depth of the bounding box, converted to steps (0.7m/step). Direction $l_{ij}$ maps the horizontal center of the box to 10/11/12/1/2 o'clock. These aggregate into $\mathcal{A}_i=\{(o_{ij},b_{ij},d_{ij},l_{ij})\}$ for the MLLM to refine into a "BLV-relevant" subset $\mathcal{X}^{\text{local}}$.
    - **Design Motivation**: Direct MLLM generation leads to synonym confusion, missed small obstacles, and wrong units. Pre-defining 80 classes and step/clock metrics encodes "BLV common sense" into the pipeline, making silver labels more stable.

3.  **Two-Layer Quality Gates: Auto-Silver Labeling + Manual Gold Verification**:
    - **Function**: Converting expensive writing into cheaper auditing to scale the dataset while maintaining quality.
    - **Mechanism**: The first layer uses EgoBlur to hide identities and combines $\mathcal{X}^{\text{global}}$ and $\mathcal{X}^{\text{local}}$ into BLV-aware instructions for MLLMs to produce S1+S2+S3 silver labels. The second layer involves three annotators running two filters: image-level $c_1$ (readability/angle/occlusion) and standard-level $c_2$ (compliance with S1/S2/S3). Non-compliant samples are rejected or corrected. Statistics show 26.5% were rejected for image quality and 8.1% for standard non-compliance.
    - **Design Motivation**: Full manual labeling is unscalable; full automation is unreliable. The "explicit standards + auto-gen + audit rejection" trio ensures a 22K scale with acceptable rejection costs (~30%).

### Loss & Training
As a dataset paper, the model side involves small-scale fine-tuning of Qwen2.5-VL using LoRA. It performs standard next-token prediction on the 22K silver labels to verify if fine-tuning on GuideDog can significantly improve open-source models.

## Key Experimental Results

### Main Results: Navigation Generation (GuideDog)
Evaluation across BLEU/ROUGE/METEOR/GPT-Eval/Gemini-Eval in 0-shot and 3-shot settings.

| Model | 0-shot GPT-Eval | 3-shot GPT-Eval | 0-shot METEOR | 3-shot METEOR |
| :--- | :--- | :--- | :--- | :--- |
| Cambrian-1 (Open) | 0.219 | 0.307 | 0.267 | 0.375 |
| Qwen2.5-VL (Open) | 0.230 | 0.319 | 0.294 | 0.412 |
| Qwen2.5-VL **+ GuideDog LoRA (Ours)** | **0.541** | 0.529 | **0.471** | 0.456 |
| Gemini 2.0 Flash | 0.462 | 0.481 | 0.400 | 0.463 |
| GPT-4o | 0.490 | 0.505 | 0.445 | 0.474 |
| Socratic GPT-4o (Text-only) | 0.384 | 0.391 | 0.419 | 0.417 |

### Ablation Study: Visual Perception QA (GuideDogQA)
Deconstructed "Recognition vs. Depth" performance revealed capacity mismatches in current MLLMs.

| Model | Object Recog. Acc | Rel. Depth Acc | Note |
| :--- | :--- | :--- | :--- |
| Random | 25.0 | 25.0 | 4-way / 2-way tasks |
| Cambrian-1 | 82.3 | 24.3 | Strong recog, random depth |
| Qwen2.5-VL | 85.7 | 22.2 | Depth worse than random |
| Qwen2.5-VL **+ GuideDog LoRA (Ours)** | 83.9 | **41.5** | Recog stable, Depth Gain +19.3 |
| Gemini 2.0 Flash | 65.7 | 53.0 | Weaker recog, better depth |
| GPT-4o | 74.7 | **67.1** | Leading depth perception |

### Key Findings
- **Fine-tuning makes the biggest difference**: Qwen2.5-VL with LoRA jumped from 0.230 to 0.541 in GPT-Eval, surpassing GPT-4o. This proves 22K silver labels are sufficient to push open-source MLLMs to SOTA levels.
- **Depth perception is the real bottleneck**: Relative depth accuracy for open-source models sits at 22–32% (near random), while GPT-4o reaches 67%. BLV navigation urgently needs distance judgments like "something 3 steps ahead," explaining why S2 (obstacle description) scores lowest in user studies.
- **Socratic models prove visual information is irreplaceable**: Converting vision to captions for text-only reasoning (SM pipeline) shows minimal 0-shot/3-shot differences and performs weaker than direct MLLMs, indicating the task requires genuine "seeing."
- **Silver labels are high quality**: In user studies, filtered silver labels received a Likert score of 4.63/5, higher than GPT-4o's 3.90, suggesting the bottleneck lies in spatial perception rather than text quality.

## Highlights & Insights
- **Turning "Professional Guides" into Prompts + Checklist**: By distilling over ten guidelines into S1/S2/S3, the authors ensured "Standard Consistency" across generation, manual verification, and LLM-as-a-judge scoring.
- **Generation → Verification Paradigm**: Transitioning labor from "writing" to "auditing" allows for both scale and quality. This paradigm is transferable to other domains with clear rules but high volume requirements (e.g., medical imaging, legal compliance).
- **Decoupled Evaluation of "Recognition" and "Depth"**: GuideDogQA targets these specific sub-capabilities, exposing the "Strong Recognition / Guessed Depth" asymmetry in open-source MLLMs, providing a roadmap for future spatial perception training.

## Limitations & Future Work
- **Silver label ceiling depends on base models**: The authors admit noise stems from detector hallucinations. The pipeline's performance is capped by open-vocabulary detectors and depth estimators; future work should integrate perception modules into the active learning loop.
- **Image-level vs. Video-level**: Real BLV navigation is a continuous stream. The 22K samples are static frames, lacking evaluation for temporal coherence (instruction consistency across frames, dynamic obstacle tracking).
- **Subjective bias of sighted annotators**: Gold label verification is done by sighted people. Although guided by a checklist, true utility for BLV users requires more extensive validation beyond the 14-person user study conducted.
- **Future Directions**: Including BLV users in the loop, introducing video temporal versions, and jointly training detection/depth/language modules to optimize label quality.

## Related Work & Insights
- **vs VizWiz / VIALM / Merchant**: Prior works relied on manual annotation. VizWiz (31K) is not for navigation; VIALM (200) is professional but tiny. GuideDog uses human-AI collaboration to reach 22K across 183 cities/46 countries.
- **vs WalkVLM / EgoBlind**: Also video-derived, but WalkVLM has only 10 locations and focuses on VideoQA; EgoBlind has only 1.3K snippets. GuideDog is superior in scale and standardized output.
- **vs Socratic Models**: While the original Socratic paper argued text-only reasoning could replace visual models, the SM baseline here proves otherwise for BLV navigation, highlighting spatial tasks as a weakness of the Socratic paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ (Expert standard-driven "Generation-Verification" paradigm is systematically applied to BLV for the first time).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers 7 MLLMs, 0/3-shot, auto/LLM/human metrics, and decoupled QA).
- Writing Quality: ⭐⭐⭐⭐ (Clear pipeline diagrams, statistics, and qualitative comparisons).
- Value: ⭐⭐⭐⭐⭐ (Provides a large-scale standardized foundation for AI assisting 2.2 billion BLV people).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] EDU-CIRCUIT-HW: Evaluating Multimodal Large Language Models on Real-World University-Level STEM Student Handwritten Solutions](edu-circuit-hw_evaluating_multimodal_large_language_models_on_real-world_univers.md)
- [\[CVPR 2026\] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](../../CVPR2026/multimodal_vlm/towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)
- [\[ACL 2026\] DraDDP: A Multimodal Multi-Party Dialogue Discourse Parsing Dataset](draddp_a_multimodal_multi-party_dialogue_discourse_parsing_dataset.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)

</div>

<!-- RELATED:END -->
