---
title: >-
  [Paper Note] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs
description: >-
  [NeurIPS 2025][Multimodal VLM][Spatial reasoning] This paper proposes Struct2D, a perception-guided prompting framework that converts 3D perception outputs into structured 2D representations (BEV images + object labels +…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Spatial reasoning"
  - "bird's-eye view"
  - "multimodal large language models"
  - "instruction tuning"
  - "3D scene understanding"
date: 2026-05-08
content_hash: 8abe5ff4f2d73068
---

# Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs

**Conference**: NeurIPS 2025
**arXiv**: [2506.04220](https://arxiv.org/abs/2506.04220)  
**Code**: [GitHub](https://github.com/neu-vi/struct2d)  
**Area**: Multimodal VLM
**Keywords**: Spatial reasoning, bird's-eye view, multimodal large language models, instruction tuning, 3D scene understanding

## TL;DR

This paper proposes Struct2D, a perception-guided prompting framework that converts 3D perception outputs into structured 2D representations (BEV images + object labels + metadata), enabling MLLMs to perform complex spatial reasoning without explicit 3D input. The authors also construct Struct2D-Set, a large-scale instruction tuning dataset containing 200K QA pairs.

## Background & Motivation

3D spatial reasoning is a core capability for robotic manipulation, autonomous navigation, and visual question answering. Conventional approaches rely on explicit 3D representations such as point clouds, but face two critical challenges: (1) they require large amounts of annotated data and exhibit limited generalization; and (2) effective bridging with language understanding remains difficult, constraining their applicability in embodied AI.

Recent MLLMs have achieved remarkable progress in 2D image and video perception. Some works attempt to align point cloud features into LLMs for 3D understanding, but these methods require point cloud features as input, limiting their flexibility. Humans, in contrast, infer 3D spatial relationships from a continuous stream of 2D visual inputs, which motivates a central question: **Can MLLMs perform spatial reasoning solely from structured 2D representations, without explicit 3D features?**

Prior work such as GPT4Scene has explored BEV images as 2D spatial cues, but tends to neglect object appearance and detailed prior information (e.g., coordinates, categories), which is insufficient for comprehensive 3D understanding. Directly applying video frames for spatial reasoning also suffers from two limitations: incomplete perception (sparse sampling causes the model to miss critical visual evidence) and lack of global context (egocentric viewpoints fail to capture the overall scene layout).

## Method

### Overall Architecture

Struct2D converts 3D perception outputs into structured 2D inputs comprising three core components: (1) a BEV image with filtered object labels, (2) per-object metadata (category, 3D coordinates), and (3) optional egocentric keyframes. The overall pipeline extracts point clouds and detection results from RGB-D video via a perception module, renders BEV images and projects object labels, constructs metadata text, and finally provides these structured 2D inputs to an MLLM for reasoning.

The framework is formally expressed as:

$$\mathbf{T}^{\text{out}} = \mathcal{F}(\text{Struct2D}(\phi_{\text{percept}}(\mathbf{V}), \mathbf{T}^{\text{meta}}, \mathbf{I}_{\text{keyframe}}), \mathbf{T}^{\text{in}})$$

where $\phi_{\text{percept}}$ denotes the perception module, $\mathbf{T}^{\text{meta}}$ is the metadata text, and $\mathbf{I}_{\text{keyframe}}$ represents optional keyframes.

### Key Designs

1. **Struct2D Prompting**: Converts 3D scene perception outputs into 2D representations directly consumable by MLLMs. Key innovations include: (a) query-based filtering of object labels to retain only query-relevant objects, reducing visual clutter; (b) BEV image rotation aligned with the agent's heading direction to facilitate relative direction reasoning; (c) depth-aware 3D projection for keyframe selection rather than uniform sampling, yielding fewer but more informative keyframes. Compared to the GPT4Scene prompting strategy, training time is reduced from 6 hours to 4 hours.

2. **Struct2D-Set**: A large-scale instruction tuning dataset containing 200K QA pairs automatically generated from 6K+ indoor 3D scenes, covering 8 categories of spatial reasoning tasks. Data generation follows two pipelines: (a) global spatial relation tasks inspired by VSI-Bench (spatial relation recognition, egocentric navigation, comparative reasoning), where initial QA pairs are generated from 3D geometric templates and then enriched with reasoning chains via ChatGPT; and (b) scene understanding tasks adapted from existing benchmarks such as ScanQA and SQA3D (attribute recognition, counting, verification). Each QA pair includes a short answer and a long answer with step-by-step reasoning.

3. **Think-Answer Reasoning Mechanism**: For questions involving complex spatial reasoning (e.g., relative direction, path planning), special tokens `<think>` and `</think>` are inserted to guide the model in generating step-by-step reasoning processes, with the final answer enclosed in `<answer>` and `</answer>`. Simple questions are answered directly with short responses.

### Loss & Training

Supervised fine-tuning (SFT) is performed on Qwen2.5VL with a learning rate of 2e-6, cosine annealing, trained for approximately 8 hours on 8×H200 GPUs. Visual inputs are uniformly resized to 480×480. During evaluation, BundleFusion is used to reconstruct point clouds, Mask3D and UniDet are used to detect 3D object bounding boxes, which are then projected into BEV images and 2D labels.

## Key Experimental Results

### Main Results

**Zero-Shot Analysis (GPT-o3 + Struct2D Prompting, VSI-Bench Subset)**

| Setting | # Images | Avg. | Rel. Distance | Rel. Direction | Path Planning |
|---------|----------|------|---------------|----------------|---------------|
| VSI-Bench Original | 16 | 48.6 | 51.0 | 49.4 | 61.9 |
| GPT4Scene | 9 | 50.3 | 50.5 | 47.9 | 58.8 |
| Ours (noisy detection) | 1 | 56.1 | 60.0 | 60.1 | 76.2 |
| Ours (GT objects) | 1 | 83.8 | 96.5 | 94.4 | 80.1 |

**Full VSI-Bench Evaluation (Open-Source Model Comparison)**

| Method | Avg. | Object Count | Abs. Distance | Rel. Distance | Rel. Direction | Path Planning |
|--------|------|--------------|---------------|---------------|----------------|---------------|
| LLaVA-NeXT-Video-7B | 36.3 | 48.5 | 14.0 | 43.5 | 42.4 | 34.0 |
| R1-Zero-VSI+SFT | 38.8 | 44.7 | 27.6 | 34.0 | 35.7 | 33.0 |
| Qwen2.5-VL-3B (SFT) | 41.9 | 46.0 | 34.7 | 35.1 | 44.9 | 33.5 |
| **Qwen2.5-VL-7B (SFT)** | **43.6** | 47.1 | 35.1 | 35.1 | **45.9** | **35.8** |

### Ablation Study

| Configuration | Avg. | Rel. Distance | Rel. Direction | Path Planning | Note |
|---------------|------|---------------|----------------|---------------|------|
| w/o augmented QA | 31.5 | 21.2 | 14.7 | 31.5 | No ChatGPT augmentation |
| w/ augmented QA | 38.0 | 33.3 | 42.2 | 33.0 | ChatGPT-generated reasoning chains |
| w/o `<think>` tokens | 36.2 | 33.3 | 38.6 | 26.3 | No explicit reasoning guidance |
| w/ `<think>` tokens | 36.1 | 31.5 | 42.2 | 33.0 | Explicit reasoning guidance improves complex tasks |

**Prompting Component Ablation (Metadata + Filtered Labels)**

| Metadata | Filtered Labels | Rel. Distance | Rel. Direction | Path Planning |
|----------|-----------------|---------------|----------------|---------------|
| ✗ | ✗ | 67.5 | 82.1 | 74.3 |
| ✗ | ✓ | 72.1 | 88.3 | 78.3 |
| ✓ | ✗ | 75.3 | 89.5 | 50.6 |
| ✓ | ✓ | **96.5** | **94.4** | **80.1** |

### Key Findings

- A single BEV image combined with lightweight metadata suffices for high-quality spatial reasoning, at far lower cost than multi-frame strategies.
- Metadata and filtered labels are complementary; path planning in particular requires both to achieve optimal performance.
- ChatGPT-generated reasoning chains contribute substantially to complex spatial tasks, improving relative direction performance from 14.7 to 42.2.
- On ScanQA and SQA3D, pure 2D input methods are already competitive with models using explicit 3D point clouds.

## Highlights & Insights

- **Core Insight**: Structured 2D representations can effectively substitute explicit 3D representations for spatial reasoning; the key lies in how information is organized rather than the representation modality itself.
- A single BEV image suffices — this dramatically reduces inference cost (\$27 vs. \$105).
- Filtered object labels represent a simple yet effective design: displaying only query-relevant objects reduces visual noise.
- The data construction pipeline is fully automated and highly scalable.

## Limitations & Future Work

- Preprocessing still relies on 3D perception modules (point cloud reconstruction, 3D detection), which may be limiting in latency-sensitive scenarios.
- The dataset focuses on indoor scenes; generalization to outdoor or open-world environments remains unexplored.
- BEV rendering depends on high-quality depth estimation; performance drops from 83.8 to 56.1 under noisy detection.
- Rule-based evaluation metrics may occasionally fail to fully reflect reasoning quality.

## Related Work & Insights

- GPT4Scene first explored BEV images for spatial prompting, but Struct2D introduces critical improvements in filtered labels, metadata-guided reasoning, and keyframe selection.
- R1-Zero-VSI employs GRPO training to enhance spatial reasoning, but covers a narrower range of QA complexity and task types.
- The paradigm of using structured 2D representations as a substitute for 3D input can be generalized to robotic navigation, AR/VR, and other domains.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of replacing 3D input with structured 2D representations is innovative, and the prompting strategy is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Zero-shot analysis, SFT, multiple benchmarks, and detailed ablations — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured with in-depth analysis.
- Value: ⭐⭐⭐⭐ Provides a practical and low-cost alternative for 3D spatial reasoning; dataset and code are publicly released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [\[NeurIPS 2025\] MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](mmperspective_do_mllms_understand_perspective_a_comprehensive_benchmark_for_pers.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[ACL 2026\] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](../../ACL2026/multimodal_vlm/unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)

</div>

<!-- RELATED:END -->
