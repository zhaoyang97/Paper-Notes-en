---
title: >-
  [Paper Note] The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights
description: >-
  [ACL 2025][VLM Reasoning][Multimodal Mathematical Reasoning] This study systematically reveals that existing multimodal mathematical reasoning models utilize visual information to an extremely limited extent—shuffling or removing training images has negligible impact on model performance—and proposes the HC-M3D benchmark to genuinely test visual dependency, showing that mainstream models fail to identify subtle variations in images.
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Multimodal Mathematical Reasoning"
  - "Visual Dependency"
  - "Benchmark Evaluation"
  - "Image Encoder"
  - "Dataset Construction"
date: 2026-05-08
content_hash: bf815742df5a0d30
---

# The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights

**Conference**: ACL 2025  
**arXiv**: [2503.04167](https://arxiv.org/abs/2503.04167)  
**Code**: [GitHub](https://github.com/Yufang-Liu/visual_modality_role)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Mathematical Reasoning, Visual Dependency, Benchmark Evaluation, Image Encoder, Dataset Construction

## TL;DR

This study systematically reveals that existing multimodal mathematical reasoning models utilize visual information to an extremely limited extent—shuffling or removing training images has negligible impact on model performance—and proposes the HC-M3D benchmark to genuinely test visual dependency, showing that mainstream models fail to identify subtle variations in images.

## Background & Motivation

Multimodal mathematical reasoning has emerged as a hotspot in recent LVLM research, yet a critical question has been overlooked: **Does the model actually utilize image information for mathematical reasoning?**

Most existing methods (G-LLaVA, MathLLaVA, MAVIS, MultiMath) focus on enhancing the diversity and quality of training data, reporting performance gains across various benchmarks. However, the authors reveal a startling phenomenon:

1. During the mathematical SFT stage, **after shuffling the image-text correspondences**, the model performance drops by only 0-4 percentage points, and even improves on certain datasets.
2. **After completely removing the images**, the performance drop remains similarly negligible.
3. This contrasts sharply with general VQA tasks, where shuffling images causes a sudden 30-40% performance plunge.

This indicates that existing multimodal mathematical models inherently rely primarily on text rather than images for reasoning, and the role of the visual modality has been severely overestimated.

## Method

### Overall Architecture

The workflow of this study consists of three progressive levels:
1. **Visual Perturbation Experiments**: Reproducing mainstream methods under a unified architecture to verify visual dependency by shuffling/removing images.
2. **Analysis of Issues in Existing Benchmarks**: Revealing the two major issues of overly rich textual information and option-leaked answers.
3. **HC-M3D Benchmark Construction & Evaluation**: Building a dataset that genuinely requires visual dependency and testing mainstream models.

### Key Designs

1. **Visual Modality Perturbation Experiments**: Under the unified LLaVA three-stage framework (pre-training, general SFT, mathematical SFT), visual information is altered solely during the mathematical SFT stage: ① correct image-text pairs (baseline), ② shuffled image-text correspondence (image distribution remains unchanged but correspondence is randomized), and ③ complete removal of images. A unified base model (DeepSeek-Math-RL-7B + CLIP-ViT-L-14-336) is used to ensure a fair comparison. Experimental results of four methods (G-LLaVA, MathLLaVA, MAVIS, MultiMath) on five mathematical benchmarks consistently demonstrate that visual perturbation has minimal impact.

2. **HC-M3D Benchmark Dataset**: A manually constructed multimodal mathematical benchmark containing 1,851 samples, following three principles: ① data correctness (questions can be solved based on both text and image, and the ground-truth answers are correct), ② visual dependency (answers must depend on the image), and ③ high image-answer correlation (modified similar images are provided for 429 questions, where changing the image alone alters the correct answer). Data sources include GeoQA (48.4%), MathVista (14.7%), and Jingyou Net (37.0%).

3. **Multi-Image Encoder Experiments**: To validate the popular practice of combining multiple image encoders to enhance mathematical reasoning, various combinations of encoders like CLIP-B, CLIP-L, SigLIP, and DINOv2 are tested (using two fusion methods: concatenating hidden features and voting). The results show that while general VQA performance improves, mathematical reasoning performance may actually degrade.

### Loss & Training

- A unified LLaVA three-stage training pipeline is adopted: Stage 1 pre-training projector, Stage 2 SFT on general data, and Stage 3 SFT on mathematical data.
- Base Language Model: DeepSeek-Math-RL-7B
- Image Encoder: CLIP-ViT-L-14-336
- Option Perturbation Experiment: Shuffling the order of multiple-choice options to observe the consistency of predictions.

## Key Experimental Results

### Main Results

Impact of visual perturbation on mathematical reasoning (average accuracy across 5 datasets):

| Method | Correct Image | Shuffled Image | No Image |
|------|---------|---------|--------|
| G-LLaVA | 35.2 | 34.6 (-0.6) | 35.4 (+0.2) |
| MathLLaVA | 36.1 | 32.5 (-3.6) | 32.7 (-3.4) |
| MAVIS | 37.1 | 34.7 (-2.4) | 34.2 (-2.9) |
| MultiMath | 40.1 | 37.2 (-2.9) | 36.4 (-3.7) |

Comparison with general VQA tasks (LLaVA-1.5):

| Setting | VQAv2 | MMBench |
|------|-------|---------|
| Correct Image | 79.2 | 66.8 |
| Shuffled Image | 46.2 (-33.0) | 26.6 (-40.2) |
| No Image | 60.8 (-18.4) | 54.3 (-12.5) |

### HC-M3D Benchmark Evaluation

| Model | Params | ALL↑ | DI↑ | BC↑ | AG↓ |
|------|--------|------|-----|-----|-----|
| G-LLaVA | 7B | 45.4 | 41.5 | 15.2 | 52.2 |
| MultiMath | 7B | 49.2 | 44.8 | 16.6 | 56.9 |
| InternVL2 | 8B | 41.9 | 38.3 | 16.6 | 34.0 |
| Qwen2-VL | 72B | 51.8 | 48.3 | 20.3 | 51.5 |
| GPT-4o | — | 49.0 | 45.8 | 19.1 | 42.0 |

### Key Findings

- **Extremely low BC (accuracy on both similar images) metric**: Even GPT-4o achieves only 19.1%, indicating that models cannot identify subtle differences between images.
- **High AG (agreement) metric**: This indicates that models tend to output the original answer even after the image changes, failing to genuinely perceive visual variations.
- Pure-text LLMs (e.g., Qwen-2.5-Math-7B) achieve an average score of 33.3 on mathematical benchmarks, which is highly competitive with the multimodal model G-LLaVA's 35.2.
- In the option-shuffling experiments, MultiMath's BC/CR on GeoQA is only 9.5%, exposing the option-leakage issue.
- Combining multiple encoders is effective for general VQA but ineffective or even detrimental for mathematical reasoning.

## Highlights & Insights

- Ingenious experimental design: Through the simple action of shuffling/removing images, the study strongly challenges the implicit assumption of "multimodal = utilizing images".
- The "modify image, keep question" methodology of HC-M3D is highly elegant—directly verifying visual understanding using a controlled variable approach.
- The contrast with general VQA experiments is striking, indicating that this issue is unique to the mathematical domain.
- Implies a crucial insight: Current evaluation benchmarks for multimodal mathematical reasoning themselves possess systematic flaws.

## Limitations & Future Work

- The scale of the HC-M3D dataset is relatively small (1,851 samples) and primarily focuses on plane geometry.
- The paper mainly exposes the issues without proposing an effective solution—how to genuinely enhance visual dependency remains an open question.
- Several directions are proposed but not validated: ① pre-training data describing differences between two images, ② more fine-grained image encoders, ③ loss functions that enhance visual dependency.
- There is no analysis of why CLIP encoders struggle to capture subtle differences in mathematical diagrams.

## Related Work & Insights

- MathVerse also emphasizes the balance between textual and visual information, but HC-M3D is more direct by modifying images.
- Echoes the limitations of CLIP in fine-grained visual understanding (Liu et al. 2024b; Tong et al. 2024a).
- Inspiration: Multimodal benchmarks need to design "counterfactual" samples (altering only one modality to observe model behavior) to genuinely measure modality contribution.

## Rating

- Novelty: ⭐⭐⭐⭐ — Unique perspective, uncovering key problems overlooked by the community.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ingeniously designed perturbation experiments and rigorous dataset construction.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, step-by-step presentation, and powerful argumentation.
- Value: ⭐⭐⭐⭐ — The HC-M3D benchmark will drive the community to rethink and improve multimodal mathematical reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](mathcoder-vl_bridging_vision_and_code_for_enhanced_multimodal_mathematical_reaso.md)
- [\[ACL 2025\] We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?](wemath_knowledge_reasoning.md)
- [\[NeurIPS 2025\] When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning](../../NeurIPS2025/vlm_reasoning/when_one_modality_sabotages_the_others_a_diagnostic_lens_on_multimodal_reasoning.md)
- [\[ICLR 2026\] We-Math 2.0: A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning](../../ICLR2026/vlm_reasoning/we-math_20_a_versatile_mathbook_system_for_incentivizing_visual_mathematical_rea.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](../../ACL2026/vlm_reasoning/a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)

</div>

<!-- RELATED:END -->
