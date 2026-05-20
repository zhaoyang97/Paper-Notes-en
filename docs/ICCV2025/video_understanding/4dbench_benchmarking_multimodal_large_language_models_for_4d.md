---
title: >-
  [Paper Note] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding
description: >-
  [ICCV 2025][Video Understanding][4D understanding] This paper introduces 4D-Bench, the first benchmark for evaluating multimodal large language models (MLLMs) on 4D object understanding (i.e.…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "4D understanding"
  - "multimodal large language models"
  - "benchmark"
  - "multi-view temporal understanding"
  - "4D object QA"
date: 2026-05-08
content_hash: 06d3c1f87d2dabde
---

# 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding

**Conference**: ICCV 2025
**arXiv**: [2503.17827](https://arxiv.org/abs/2503.17827)  
**Code**: [https://4dbench.github.io/](https://4dbench.github.io/)  
**Area**: Video Understanding / Multimodal VLM
**Keywords**: 4D understanding, multimodal large language models, benchmark, multi-view temporal understanding, 4D object QA

## TL;DR
This paper introduces 4D-Bench, the first benchmark for evaluating multimodal large language models (MLLMs) on 4D object understanding (i.e., 3D objects with temporal evolution). It comprises two core tasks: 4D Object QA (751 QA pairs) and 4D Object Captioning (580 objects × 5 annotations). Evaluation reveals that even the state-of-the-art GPT-4o achieves only 63% accuracy compared to 91% for humans, exposing a substantial gap in multi-view spatiotemporal understanding among MLLMs.

## Background & Motivation
Digital 4D assets (dynamic 3D objects) are increasingly important in digital twins, augmented reality, and gaming, creating an urgent need for language-based understanding and interaction with 4D objects. Although existing MLLMs perform well on 2D image/video understanding, they face unique challenges with 4D objects:

1. **Multi-view Ambiguity**: A 4D object presents different appearances from different viewpoints; certain parts may be occluded or unclear from specific angles, requiring cross-view information integration.
2. **Temporal Evolution**: Dynamic object motion requires tracking and reasoning across time steps.
3. **Data Scarcity**: Unlike 2D image-text data abundantly available on the internet, large-scale 4D object-text data is extremely scarce.
4. **Lack of Evaluation Standards**: No public benchmark previously existed for evaluating MLLMs' ability to understand 4D objects.

Limitations of existing benchmarks: 3D language understanding benchmarks (e.g., ScanQA, SceneVerse) focus only on static 3D scenes and ignore motion; 2D video benchmarks (e.g., MVBench, Video-MME) ignore multi-view understanding. Neither can comprehensively assess multi-view spatiotemporal understanding of 4D objects.

## Core Problem
**Can advanced MLLMs be directly extended to 4D object understanding?** Existing MLLMs (GPT-4o, Qwen2-VL, etc.) have learned rich world knowledge from massive text, image, and video data. By representing 4D objects as multi-view videos, one can directly leverage MLLMs for 4D object-language understanding. However, the critical issue is that without a dedicated evaluation benchmark, it is impossible to understand these models' strengths and limitations in 4D object understanding, making it difficult to improve and unlock their potential.

## Method

### Overall Architecture
4D-Bench is an evaluation benchmark rather than a model or method. It contains two core tasks:

**Input**: Multi-view videos of 4D objects (rendered from Objaverse-XL, using $K=3$ uniformly selected viewpoints, sampling $N=6$ frames per viewpoint, totaling $K \times N = 18$ frames fed to the MLLM)
**Task 1 – 4D Object QA**: Given multi-view videos and a multiple-choice question (4 options), evaluate MLLM accuracy.
**Task 2 – 4D Object Captioning**: Given multi-view videos, require the MLLM to generate a description and compare it against 5 human-annotated captions.

### Key Designs

1. **Five Sub-tasks of 4D Object QA**:

    - **Appearance**: Evaluates the ability to describe visual attributes of objects. The challenge lies in synthetic/fictional objects deviating from training data distributions and the need to integrate appearance information across viewpoints.
    - **Action**: Evaluates understanding of object activities and local motion, including typical action recognition, fine-grained motion detection, and motion direction analysis.
    - **Object Counting**: Requires precise counting in dynamic and spatially complex scenes, demanding handling of temporal dynamics (objects appearing/disappearing) and occlusion (requiring cross-view information fusion).
    - **Spatial Relationship**: Evaluates understanding of spatial configurations and object relations across multiple viewpoints.
    - **Temporal Relationship**: Evaluates understanding of object temporal evolution and sequential actions.

2. **4D Object Captioning Design**:

    - Requires simultaneous description of object appearance and action.
    - Appearance descriptions must aggregate visual details from different angles; action descriptions must observe motion sequences from multiple viewpoints.
    - Each 4D object is annotated with 5 independent human-authored captions.

3. **Data Construction Pipeline** (three-stage filtering):

    - **Data Collection**: Multi-view videos of tens of thousands of dynamic 3D objects are rendered from Objaverse-XL (24 viewpoints, up to 125 frames per viewpoint).
    - **Motion Filtering**: Temporal boundaries of object motion are identified via pixel-change detection to ensure the dataset contains only dynamic objects.
    - **Visual Quality Filtering**: A CLIP-based quality classifier is trained on thousands of manually labeled images; majority voting across 8 viewpoints removes low-quality objects.
    - **QA Annotation**: A hybrid annotation strategy is used — professional annotators first manually design 164 high-quality QA pairs; GPT-4o and Qwen2-VL then generate additional QA pairs, verified by Qwen2-VL 7B → blind filtering (text-only LLMs remove QA pairs answerable without visual input) → manual review.
    - **Caption Annotation**: 580 representative 4D objects are manually selected; 5 annotators independently provide captions, with reviewers ensuring quality and diversity.

4. **Evaluation Metric Design**:

    - QA: Per-subtask accuracy and overall accuracy.
    - Captioning: Traditional metrics (BLEU, ROUGE, METEOR, CIDEr) + embedding metrics (BERTScore, Sentence-BERT) + **LLM-based metrics** (GPT-Appearance score, GPT-Action score, GPT-Eval average, on a 0–5 scale).
    - The paper focuses on GPT-based metrics due to their stronger correlation with human judgment.

### Loss & Training
This is a benchmark paper; no model training is involved. Evaluation follows a unified sampling strategy: $K=3$ viewpoints and $N=6$ frames, with input images arranged in viewpoint-first, then temporal order.

## Key Experimental Results

### 4D Object QA Results

| Model | Object Counting | Temporal Rel. | Action | Spatial Rel. | Appearance | Overall |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 44.09% | 59.29% | 63.55% | 69.40% | 77.21% | **62.98%** |
| LLaVA-Video 72B | 54.33% | 58.57% | 57.48% | 66.42% | 77.21% | 62.32% |
| LLaVA-OneVision 72B | 49.61% | 58.57% | 60.75% | 61.19% | 76.47% | 61.38% |
| Gemini 1.5 Pro | 46.46% | 58.57% | 59.35% | 64.18% | 68.38% | 59.52% |
| Qwen2-VL 72B | 45.67% | 55.71% | 58.41% | 61.19% | 72.06% | 58.72% |
| All Models Avg. | 37.29% | 49.29% | 49.37% | 53.57% | 63.92% | 50.69% |
| **Human Baseline** | **88.98%** | **89.29%** | **94.39%** | **91.04%** | **89.71%** | **91.08%** |

### 4D Object Captioning Results (GPT Metrics)

| Model | GPT-Appearance | GPT-Action | GPT-Eval |
|------|:---:|:---:|:---:|
| GPT-4o | 3.507/5 | 3.258/5 | 3.382/5 |
| GPT-4o mini | 3.311/5 | 3.131/5 | 3.221/5 |
| Gemini 1.5 Pro | 3.311/5 | 2.983/5 | 3.147/5 |
| Qwen2-VL 72B | 3.324/5 | 2.791/5 | 3.057/5 |
| All MLLMs Avg. | 3.038/5 | 2.522/5 | 2.780/5 |
| **Human** | **3.772/5** | **3.879/5** | **3.826/5** |

### Ablation Study
- **Effect of Number of Viewpoints**: Increasing from 1 to 3 viewpoints improves QA accuracy from 41.3% to 53.7% (Gemini 1.5 Flash), but performance degrades beyond 3 viewpoints, possibly due to information redundancy exceeding the model's processing capacity.
- **Temporal Sampling Frequency**: Increasing from 2 to 6 frames improves accuracy from 46.3% to 53.7%; improvement is marginal beyond 6 frames.
- **Input Order Robustness**: Viewpoint-first vs. time-first ordering and the inclusion of timestamp information have minimal impact on results (average change < 1%), confirming the robustness of the experimental design.
- **Counterfactual Data Testing**: Synthetic spiders with only 6 legs (vs. 8 in reality) and balls rolling into a downward hole and then rolling back out (violating physical laws) — all advanced MLLMs answer incorrectly, indicating reliance on prior world knowledge rather than genuine understanding of the input.

## Highlights & Insights
- **Pioneering Contribution**: 4D-Bench is the first benchmark to systematically evaluate MLLMs on 4D (dynamic 3D) object understanding, filling the gap left by 3D benchmarks that ignore the temporal dimension and video benchmarks that ignore multi-view information.
- **Blind Filtering Mechanism**: Text-only LLMs are used to filter out QA pairs answerable without visual input, ensuring that questions genuinely require visual understanding.
- **OOD Evaluation Capability**: Synthetic 4D assets include counterfactual objects and motions, enabling out-of-distribution (OOD) evaluation of MLLMs that cannot exploit prior knowledge trained primarily on real-world data.
- **Revealing Performance Hierarchy**: The benchmark clearly exposes a capability gradient among MLLMs — Appearance > Spatial > Temporal ≈ Action >> Counting — providing concrete guidance for future improvement.
- **Localizing Open- vs. Closed-Source Gaps**: The gap in appearance understanding is small, while the gap in action/temporal understanding is large, pointing the open-source community toward priority research directions.

## Limitations & Future Work
- **4D Representation Limitation**: Representing 4D objects as multi-view videos enables evaluation but discards geometric information present in native 3D/4D representations such as point clouds and 4DGS, precluding evaluation of MLLMs' understanding of native 4D representations.
- **Limited Data Scale**: With only 751 QA pairs and 580 captioned objects, the benchmark is relatively small and may not sufficiently represent the diversity of 4D objects.
- **Single-Object Scope**: 4D-Bench evaluates understanding of individual 4D objects and does not cover multi-object interactions or 4D scene-level understanding.
- **Synthetic Data Bias**: All data are sourced from Objaverse-XL (synthetic data); the domain gap relative to real-world 4D scans may limit the transferability of conclusions.
- **No Training Split**: As an evaluation-only benchmark, no training set is provided to drive the development of 4D object-language understanding models.
- **Potential Extensions**: Data scale could be expanded by incorporating 4D generation methods (e.g., data generated via 4D Gaussian Splatting); more complex subtasks such as multi-object interaction and physical reasoning could be added.

## Related Work & Insights

| Dimension | 4D-Bench | ScanQA/SceneVerse (3D Benchmarks) | MVBench/Video-MME (Video Benchmarks) |
|---------|----------|--------------------------|--------------------------|
| Spatial Dimension | Multi-view 3D | 3D point clouds/scenes | Single-view 2D |
| Temporal Dimension | ✓ Dynamic motion | ✗ Static only | ✓ Temporal changes |
| Object Focus | Object-level (single object) | Scene-level | Scene-level |
| Data Type | Synthetic 4D assets | Real 3D scans | Real videos |
| OOD Capability | ✓ Counterfactual objects | ✗ | ✗ |

Compared to T3Bench (which evaluates text-to-3D generation), 4D-Bench focuses on language-based understanding of 4D objects rather than generation.

**Key Insights:**
1. **Temporal Reasoning is the Core Bottleneck**: MLLMs are approaching closed-source performance on appearance understanding, but the gap in temporal/action understanding is substantial. This indicates that the temporal modeling capability of current visual encoders is the primary bottleneck; more advanced temporally-aware visual encoders may be the key breakthrough.
2. **Extremely Weak Counting Ability**: Multi-view object counting accuracy of only 37% suggests that MLLMs are severely limited in cross-view correspondence reasoning, which is directly related to 3D-consistent understanding.
3. **Counterfactual Data as a Strong Evaluation Signal**: Synthetic data can naturally incorporate physically implausible scenarios, providing a powerful means to distinguish genuine understanding from prior knowledge memorization in MLLMs.
4. **From Evaluation to Training**: The capability gaps revealed by 4D-Bench (temporal, counting, counterfactual) can guide the construction of targeted 4D understanding training data.

## Rating
- Novelty: ⭐⭐⭐⭐ The first MLLM benchmark for 4D object understanding, filling an important gap; methodological innovation is relatively limited as a benchmark paper.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluates 14 MLLMs (open- and closed-source), 5 QA subtasks, and multiple captioning metrics; ablation studies are comprehensive (number of viewpoints, sampling frequency, input order, counterfactual analysis).
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clearly presented findings and rich figures and tables.
- Value: ⭐⭐⭐⭐ Reveals systematic weaknesses of MLLMs in 4D understanding, providing important guidance for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](aim_adaptive_inference_of_multi-modal_llms_via_token_merging_and_pruning.md)
- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](distime_distribution-based_time_representation_for_video_large_language_models.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)

</div>

<!-- RELATED:END -->
