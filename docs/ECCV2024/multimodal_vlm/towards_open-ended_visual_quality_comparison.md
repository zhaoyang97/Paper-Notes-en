---
title: >-
  [Paper Note] Towards Open-ended Visual Quality Comparison
description: >-
  [ECCV 2024][Multimodal VLM][Image Quality Assessment] This work proposes Co-Instruct, the first large multimodal model for open-ended visual quality comparison. By constructing a 562K instruction-tuning dataset from two "weakly supervised sources" (LLM-merged single-image descriptions + GPT-4V pseudo-labels), Co-Instruct achieves higher accuracy in multi-image quality comparison than its teacher model, GPT-4V, and introduces MICBench, the first multi-image comparison benchmar…
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Image Quality Assessment"
  - "Multi-image Comparison"
  - "Large Multimodal Models"
  - "Instruction Tuning"
  - "Visual Quality"
date: 2026-05-08
content_hash: caa420ca39fddf9f
---

# Towards Open-ended Visual Quality Comparison

**Conference**: ECCV 2024  
**arXiv**: [2402.16641](https://arxiv.org/abs/2402.16641)  
**Code**: [https://huggingface.co/q-future/co-instruct](https://huggingface.co/q-future/co-instruct)  
**Area**: Multimodal VLMs  
**Keywords**: Image Quality Assessment, Multi-image Comparison, Large Multimodal Models, Instruction Tuning, Visual Quality

## TL;DR

This work proposes Co-Instruct, the first large multimodal model for open-ended visual quality comparison. By constructing a 562K instruction-tuning dataset from two "weakly supervised sources" (LLM-merged single-image descriptions + GPT-4V pseudo-labels), Co-Instruct achieves higher accuracy in multi-image quality comparison than its teacher model, GPT-4V, and introduces MICBench, the first multi-image comparison benchmark.

## Background & Motivation

Image Quality Assessment (IQA) is an important field in visual computing. Recently, Large Multimodal Models (LMMs) have been explored to extend IQA from outputting scalar scores to open-ended scenarios, where they can answer open-ended questions and provide reasoning-based explanations.

**Limitations of Prior Work**: Existing open-ended IQA methods are based on single-image evaluation, which faces a fundamental issue — **the ambiguity of absolute evaluation**. Different observers hold different standards for attributes such as exposure and clarity of the same image, leading to inconsistent absolute evaluations. However, under a comparative setting (e.g., "Which image is brighter?"), responses tend to align consistently.

**Limitations of Prior Work**:
1. Existing datasets and methods only support simple comparisons of overall quality and have not expanded to open-ended scenarios.
2. Open-source LMMs are typically fine-tuned on single-image data, lacking multi-image comparison capabilities.
3. Collecting comparison datasets from human annotations is extremely costly.

**Core Idea**: This work proposes a "Collaborative Instruction Tuning" strategy (Co-Instruct), utilizing two imperfect weakly supervised sources to work complementarily: (1) an LLM "merges" single-image quality descriptions into comparison text; (2) GPT-4V generates pseudo-labels on unlabeled data. The two complementary sources form a 562K training dataset.

## Method

### Overall Architecture

Co-Instruct adopts mPLUG-Owl2 as the base model, which contains a CLIP-ViT-L14 visual encoder, a visual abstractor (compressing visual tokens from $1025$ to $65$), and a LLaMA-2 LLM. It handles multi-image inputs via an interleaved image-text format and is fine-tuned on the Co-Instruct-562K dataset.

### Key Designs

1. **Merge2Compare (LLM-based Merging for Comparison)**: Starting from individual image quality descriptions of 19K images in the Q-Pathway dataset, images are randomly paired/grouped into 100K groups (containing 2 to 4 images). The E5-Mistral text embedding model is utilized to filter out the most similar description pairs, and then an LLM "merges" multiple independent descriptions into comparison text. The core mechanism is to "convert" existing single-image evaluations into comparative evaluations, achieving a human-verified accuracy of 96%.

2. **Teach2Compare (GPT-4V Teacher Comparison)**: A diverse set of 9K unlabeled images (including in-the-wild, artificially degraded, and AI-generated images) is collected and randomly grouped into 30K sets. These are fed into GPT-4V to obtain two types of responses: (a) overall quality comparison descriptions; (b) Q&A pairs for specific quality attributes (clarity, color, etc.), totaling 230K pairs. The accuracy of GPT-4V is approximately 94%, which is slightly lower than Merge2Compare but contains richer content information.

3. **Interleaved Image-Text Format and Visual Token Compression**: To handle multi-image inputs, a visual abstractor is employed to compress the number of tokens per image from $1025$ to $65$, and an interleaved image-text format is designed: `"The first image: <img₀> The second image: <img₁> ... <query>"`, allowing the model to clearly distinguish information from each image. Experiments show that this format significantly outperforms simple concatenation or learnable separators.

4. **MICBench Benchmark**: The first multi-image quality comparison benchmark is constructed, consisting of 2,000 multiple-choice questions (MCQs) covering quality comparisons of 3 or 4 images. This includes "Which" questions (60%), "Yes-or-No" questions (22%), and other types (18%). The images are sourced from LLVisionQA and unlabeled databases, annotated and cross-validated by 10 human experts.

### Loss & Training

- Fine-tuning is performed based on the released checkpoint of mPLUG-Owl2.
- The learning rate is set to $2\times 10^{-5}$, the batch size is $192$, and training is conducted for $2$ epochs.
- All parameters are updated, with a total training time of approximately $25$ hours on $8\times \text{A100}$ GPUs.
- Images are padded to a square and then resized to $448\times 448$.

## Key Experimental Results

### Main Results

| Dataset | Metric | Co-Instruct | GPT-4V | Gain |
|--------|------|-------------|--------|------|
| Q-Bench^PAIR-A1 | Overall Accuracy | 80.18% | 78.07% | +2.7% |
| Q-Bench^PAIR-A1 | Compare Subset | 74.22% | 68.00% | +6.2% |

On Q-Bench^PAIR-A1, Co-Instruct achieves:
- **64%** higher than the base model mPLUG-Owl2
- **51%** higher than the variant trained without multi-image comparison data
- **23%** higher than the best open-source LMM (InternLM-XComposer2)
- **Surpasses non-expert human performance** (80.18% vs 80.12%), being the only LMM to achieve this

### Ablation Study

| Configuration | Overall Accuracy | Description |
|------|-----------------|------|
| W/o multi-image comparison data | 53.15% | Baseline |
| + Merge2Compare | Significant improvement | LLM merging is effective |
| + Teach2Compare | Further improvement | The two sources are complementary |
| Simple image concatenation | Lower | Confounded image information |
| Interleaved image-text format | Highest | Clearly distinguishes each image |

### Key Findings

1. The student (Co-Instruct) surpasses the teacher (GPT-4V): Although the MCQ training data is derived from GPT-4V, Co-Instruct outperforms its teacher in MCQ evaluations, demonstrating the effectiveness of the collaborative teaching strategy.
2. The complementarity of the two weakly supervised sources is crucial: Merge2Compare exhibits higher accuracy but lacks fine-grained comparison, while Teach2Compare is slightly less accurate but contains richer content information and greater Q&A diversity.
3. The interleaved image-text format significantly outperforms simple image concatenation or learnable special token separators.

## Highlights & Insights

- **The concept of weakly supervised collaboration** is highly ingenious: instead of directly collecting expensive human comparison annotations, it complementarily learns from two imperfect sources—existing single-image descriptions and GPT-4V pseudo-labels.
- **Comparison outperforms absolute evaluation**: This is a classic cognitive insight from psychophysics, which is systematically introduced into the LMM domain in this work.
- **The "student surpassing the teacher"** phenomenon: Co-Instruct learns from both LLM-merged data (high accuracy) and GPT-4V (high diversity), outperforming either individual source after integration.
- The visual token compression strategy makes multi-image inputs feasible, addressing the practical issue of limited context windows in models like LLaVA.

## Limitations & Future Work

- MICBench only evaluates MCQ formats and does not cover the evaluation of open-ended responses.
- The reliance on GPT-4V as a teacher limits the data quality to the visual perception capabilities of GPT-4V.
- Compression by the visual abstractor ($1025 \rightarrow 65$ tokens) may result in the loss of some fine-grained details.
- Scenarios involving larger numbers of images (e.g., more than $5$) have not been explored.
- Stronger visual encoders (e.g., InternViT) could be explored to replace CLIP-ViT.

## Related Work & Insights

- The Q-Bench/Q-Instruct/Q-Align series (from the same team) laid the groundwork for single-image quality assessment in this study.
- The data construction strategy (Merge2Compare) can be generalized to other scenarios where comparative data is needed but human annotation is prohibitively expensive.
- The idea of collaborative weak supervision is applicable to comparison tasks in other modalities (e.g., audio and video quality comparison).

## Rating

- Novelty: ⭐⭐⭐⭐ The first to systematically extend LMMs to open-ended multi-image quality comparison, with an innovative data construction strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple benchmarks, detailed ablation studies, and comparison with humans.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete structure, and rich illustrations.
- Value: ⭐⭐⭐⭐ Makes significant contributions to both image quality assessment and multi-image understanding in LMMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/multimodal_vlm/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[CVPR 2026\] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](../../CVPR2026/multimodal_vlm/guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)
- [\[ICML 2026\] ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation](../../ICML2026/multimodal_vlm/eca_efficient_continual_alignment_for_open-ended_image-to-text_generation.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](../../ACL2025/multimodal_vlm/craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)
- [\[CVPR 2025\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](../../CVPR2025/multimodal_vlm/opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)

</div>

<!-- RELATED:END -->
