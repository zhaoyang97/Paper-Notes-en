---
title: >-
  [Paper Note] Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models
description: >-
  [CVPR 2026][Multimodal VLM][deepfake detection] This paper introduces FAQ (Forensic Answer-Questioning), the first large-scale multiple-choice QA benchmark focused on temporal inconsistencies in deepfake videos (33K QA p…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "deepfake detection"
  - "video forensics"
  - "VLM reasoning"
  - "temporal inconsistency"
  - "multiple-choice benchmark"
  - "instruction tuning"
date: 2026-05-08
content_hash: f8c5b2164184218d
---

# Beyond Static Artifacts: A Forensic Benchmark for Video Deepfake Reasoning in Vision Language Models

**Conference**: CVPR 2026
**arXiv**: [2602.21779](https://arxiv.org/abs/2602.21779)
**Code**: To be confirmed
**Area**: Multimodal VLM / Deepfake Detection
**Keywords**: deepfake detection, video forensics, VLM reasoning, temporal inconsistency, multiple-choice benchmark, instruction tuning

## TL;DR

This paper introduces FAQ (Forensic Answer-Questioning), the first large-scale multiple-choice QA benchmark focused on temporal inconsistencies in deepfake videos (33K QA pairs, ~4,500 videos). Through a three-level progressive task hierarchy (facial perception → temporal localization → forensic reasoning), FAQ systematically enhances VLM forensic capabilities, yielding significant gains both on in-domain benchmarks and cross-dataset detection after fine-tuning (Qwen2.5-VL average accuracy improves from 21.6% to 52.4%).

## Background & Motivation

**Background**: VLMs have shown promise in deepfake detection; methods such as FakeShield and SIDA construct QA datasets to train VLMs for interpretable detection. However, existing methods and datasets almost exclusively focus on spatial artifacts (texture inconsistencies, edge blurring, and other static cues).

**Limitations of Prior Work**: (1) Temporal inconsistencies have long been validated as important detection signals in traditional deepfake detection (e.g., abrupt facial expression changes, edge flickering), yet existing QA datasets (DD-VQA, VLFFD) extract annotations only from static frames, completely ignoring dynamic cues. (2) Existing datasets either handle only images (DD-VQA, VLFFD) or focus exclusively on AI-generated content (Forensics-Bench), without addressing temporal analysis of classical forgery methods such as face swapping.

**Key Challenge**: VLM training data lacks temporal forensic information → models cannot exploit the most informative dynamic forgery cues in video → detection performance and generalization are constrained. How to effectively encode temporal inconsistencies into QA training data remains an open core problem.

**Goal**: Construct the first deepfake QA benchmark centered on temporal inconsistencies, and systematically enhance VLMs' complete forensic capability chain—from perception to reasoning—through progressive task design.

**Key Insight**: Coarse-grained manual spatiotemporal annotations (sparse clicks) are converted into structured multi-level MCQ data via an automated pipeline of spatiotemporal clustering → keypoint extraction → description parsing → QA generation.

**Core Idea**: A three-level progressive MCQ system (static perception → dynamic localization → comprehensive reasoning) injects video temporal inconsistency into VLM training, endowing models with genuine temporal forensic capability.

## Method

### Overall Architecture

The FAQ benchmark construction pipeline proceeds as follows: video collection and quality filtering (FF++ 5,000 fake + 1,000 real videos; ~10% low-quality samples removed by YOLOv8) → manual annotation of 50K+ sparse spatiotemporal clicks → spatiotemporal clustering to obtain 14,392 forgery segments → keypoint extraction (dlib facial landmark tracking) → description parsing (term-frequency analysis + LLM-based atomic annotation) → three-level QA generation → human verification. The final benchmark comprises approximately 33K QA pairs.

### Key Designs

1. **Three-Level Progressive Task Hierarchy**

   - **Function**: Progressively evaluate and train VLM forensic capability from basic perception to complex reasoning.
   - **Mechanism**:
     - **Level 1 – Facial Perception**: Region perception (judging clarity/blur quality of specific facial regions) + edge perception (distinguishing sharp vs. blurred facial boundaries); tests basic visual discrimination.
     - **Level 2 – Temporal Deepfake Localization**: Three sub-tasks—type understanding (given a time window and facial region, identify the artifact type), region localization (given a time segment and artifact type, identify the facial region), and temporal localization (given a facial region and artifact type, identify the time window); tests spatiotemporal localization ability.
     - **Level 3 – Forensic Reasoning**: Forgery analysis (autonomously identify artifact type → localize region → determine time segment without any hints, selecting the best match from carefully designed distractors) + final verdict (determine video authenticity by integrating all evidence).
   - **Design Motivation**: Directly tackling the complex reasoning of Level 3 is too difficult. The progressive design allows models to first establish basic perceptual ability, then learn spatiotemporal localization, and only then engage in comprehensive reasoning. Ablation experiments confirm that training with only static QA is nearly ineffective (LLaVA-NeXT average gain of only 3.5%); temporal information is essential.

2. **Spatiotemporal Clustering and Forgery Trajectory Construction**

   - **Function**: Convert sparse manual click annotations into coherent forgery segments and their spatiotemporal trajectories.
   - **Mechanism**: A spatiotemporal adjacency function is defined as $f(c_i, c_j) = (\|c_i - c_j\|_2 \leq \tau_s) \wedge (\|c_i - c_j\|_1 \leq \tau_t)$, with $\tau_s=4$ and $\tau_t=1$. This clusters 50K+ sparse clicks into 14,392 forgery segments (average duration 2.1 seconds). For each segment, dlib extracts facial landmarks (5 categories: eyes, nose, mouth, jawline, ears); the most relevant facial region is determined by $n^* = \arg\min_n S_n$ based on spatial centroid; concatenating keypoints across all frames forms the forgery motion trajectory.
   - **Design Motivation**: Sparse click annotations are inexpensive but cannot directly generate QA; an automated intermediate representation (segments + trajectories) is needed to bridge manual annotations and QA generation.

3. **Carefully Designed Distractors and Human Verification**

   - **Function**: Ensure MCQ options are sufficiently challenging so that models cannot rely on linguistic priors to guess answers.
   - **Mechanism**: Distractors are restricted to visually and temporally plausible options—adjacent facial regions for region tasks, nearby time windows for temporal localization tasks, and "partially correct" options for forensic reasoning. Each QA pair undergoes human verification (average 1.5 min/question at Level 1; average 5 min/question at Level 3); items that fail are revised or discarded.
   - **Design Motivation**: Prevent models from exploiting LLM linguistic priors rather than genuine visual understanding, ensuring the benchmark truly tests forensic capability.

### Loss & Training

FAQ-IT instruction tuning: the visual encoder is frozen; the visual connector and full LLM parameters are updated. AdamW optimizer with cosine scheduler, lr = 1e-5, batch size 16, 1 epoch, on 4 × H200 GPUs.

## Key Experimental Results

### Main Results (FAQ Benchmark Zero-Shot)

| Model | Level 1 | Level 2 | Level 3 | Average |
|-------|---------|---------|---------|---------|
| GPT-4o | 26.9% | 27.1% | 13.2% | 22.8% |
| Gemini-2.5-Flash | 40.0% | 25.6% | 15.3% | 27.8% |
| ShareGPT4V-7B | **73.8%** | 20.5% | **22.3%** | **39.7%** |
| Qwen3-VL-8B | 45.6% | **29.4%** | 15.0% | 30.3% |
| Qwen2.5-VL-7B | 24.1% | 23.8% | 16.8% | 21.6% |

### Fine-Tuning Results

| Model | Training Data | Level 1 | Level 2 | Level 3 | Average |
|-------|--------------|---------|---------|---------|---------|
| Qwen2.5-VL | Zero-shot | 24.1% | 23.8% | 16.8% | 21.6% |
| Qwen2.5-VL | FAQ-IT♠ (static only) | 31.3% | 21.9% | 17.9% | 23.9% |
| Qwen2.5-VL | **FAQ-IT (full)** | **89.9%** | **41.4%** | **25.8%** | **52.4%** |
| LLaVA-NeXT | Zero-shot | 40.0% | 29.0% | 21.2% | 30.3% |
| LLaVA-NeXT | FAQ-IT♠ (static only) | 49.2% | 28.8% | 23.3% | 33.8% |
| LLaVA-NeXT | **FAQ-IT (full)** | **88.8%** | **45.8%** | **26.5%** | **53.7%** |

### Ablation Study

| Training Configuration | FS | NT | F2F | DF | FSh | Avg MCQ |
|----------------------|-----|-----|-----|-----|-----|---------|
| Qwen2.5-VL + FAQ-IT♠ | 10.5% | 13.3% | 12.8% | 10.4% | 10.7% | 11.5% |
| Qwen2.5-VL + FAQ-IT | **45.9%** | **46.7%** | 24.4% | **45.3%** | **45.3%** | **41.5%** |

### Key Findings

- Closed-source models (GPT-4o, Gemini) underperform some open-source counterparts, possibly due to insufficient forensics-related content in their training data.
- Training with static QA alone yields minimal improvement (Qwen2.5-VL gains only 2.3 points), whereas incorporating temporal data produces a 30.8-point surge, confirming that temporal information is central to forensic capability.
- Level 1 (perception) shows the largest gain (Qwen2.5-VL: 24.1% → 89.9%), while Level 3 (reasoning) shows the smallest (16.8% → 25.8%), indicating that complex reasoning remains difficult to resolve through SFT alone.
- Face2Face (F2F) exhibits the weakest cross-manipulation generalization, as its temporal artifacts are subtler and harder to capture with cross-frame sampling strategies.

## Highlights & Insights

- This work is the first to systematically introduce temporal inconsistency into VLM-based deepfake detection—a signal almost entirely overlooked by prior work. The three-level progressive design is an intuitive and effective strategy, with ablations demonstrating that static QA alone is nearly useless.
- The automated pipeline from sparse clicks to structured QA is elegantly designed—spatiotemporal clustering → landmark tracking → atomic description → MCQ generation—reproducible and scalable without relying on proprietary annotations.
- The distractor design principle of "visually and temporally plausible options" is a transferable insight that prevents models from taking linguistic shortcuts, applicable to other VLM benchmark designs.

## Limitations & Future Work

- The data source is limited to the single FaceForensics++ dataset with a narrow range of forgery types (primarily face swapping); generalization to more advanced AI-generated videos (e.g., Sora, Kling) has not been validated.
- Despite targeted training, Level 3 reasoning improvement remains limited (~9 points), potentially requiring stronger reasoning paradigms such as chain-of-thought or reinforcement learning.
- Poor F2F detection exposes the limitations of cross-frame sampling strategies; forgery types with subtler temporal artifacts require denser frame sampling or optical flow analysis.
- The MCQ evaluation format cannot assess models' forensic reasoning capabilities in free-text generation settings.

## Related Work & Insights

- **vs. DD-VQA / VLFFD**: These are image-level QA datasets focused solely on spatial cues; FAQ is the first video-level temporal QA benchmark, filling the gap in dynamic forensics.
- **vs. Forensics-Bench**: Forensics-Bench contains 63K samples but targets AI-generated content; FAQ addresses temporal inconsistencies in classical forgeries such as face swapping. The two are complementary.
- **vs. FakeShield / SIDA**: These methods employ CLIP or large VLMs for detection but neglect temporal information; FAQ's training data can directly augment these methods with temporal forensic awareness.

## Rating

- Novelty: ⭐⭐⭐⭐ — First temporal forensics QA benchmark; three-level design is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Zero-shot evaluation on 13 VLMs + fine-tuning of 2 models + cross-manipulation generalization + ablations; however, validation beyond FF++ is absent.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure; pipeline description is detailed and reproducible.
- Value: ⭐⭐⭐⭐ — Fills the data and benchmark gap in VLM temporal forensics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[CVPR 2026\] Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models](beyond_recognition_evaluating_visual_perspective_taking_in_vision_language_model.md)
- [\[CVPR 2026\] Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision](pixels_dont_lie_but_your_detector_might_bootstrapping_mllm-as-a-judge_for_trustw.md)
- [\[CVPR 2026\] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)
- [\[AAAI 2026\] CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)

</div>

<!-- RELATED:END -->
