---
title: >-
  [Paper Note] VLM2-Bench: A Closer Look at How Well VLMs Implicitly Link Explicit Matching Visual Cues
description: >-
  [ACL 2025][Multimodal VLM][Visual cue matching] This paper proposes VLM2-Bench, a benchmark designed to evaluate the capability of visual-language models (VLMs) in cross-image/frame "visual cue association". It covers 9 subtasks divided into 3 major categories (general, object-centric, and person-centric cues) with a total of 3000+ test samples. The study reveals that even state-of-the-art commercial models lag behind humans by over $30\%$ on this task…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Visual cue matching"
  - "Visual-Language Models"
  - "Multi-image understanding"
  - "benchmark"
  - "Cross-image association"
date: 2026-05-08
content_hash: 52ca6db945ec7d7f
---

# VLM2-Bench: A Closer Look at How Well VLMs Implicitly Link Explicit Matching Visual Cues

**Conference**: ACL 2025  
**arXiv**: [2502.12084](https://arxiv.org/abs/2502.12084)  
**Code**: [VLM2-Bench Project Page](https://vlm2-bench.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: Visual cue matching, Visual-Language Models, Multi-image understanding, benchmark, Cross-image association

## TL;DR
This paper proposes VLM2-Bench, a benchmark designed to evaluate the capability of visual-language models (VLMs) in cross-image/frame "visual cue association". It covers 9 subtasks divided into 3 major categories (general, object-centric, and person-centric cues) with a total of 3000+ test samples. The study reveals that even state-of-the-art commercial models lag behind humans by over $30\%$ on this task, highlighting a significant gap in foundational visual matching capabilities.

## Background & Motivation
Humans continuously match and associate visual cues in daily life—such as identifying the same individual across different photographs via facial features and clothing, or tracking the same object across different scenes based on its appearance. This capability is fundamental and requires no external background knowledge.

However, existing VLM benchmarks exhibit the following blind spots:

**Not requiring explicit association of visual cues across images/frames** (e.g., MMDU, etc.)

**Relying on external knowledge instead of evaluating visual cue association capabilities** (e.g., certain knowledge-based QA benchmarks)

**Focusing on macro visual comparison rather than specific cue matching** (e.g., Img-Diff, etc.)

**Focusing on retrieval tasks rather than direct visual cue association**

**Key Challenge**: Though VLMs possess vast knowledge, can they perform well on such foundational tasks that require "no knowledge, only visual matching"?

**Core Idea**: **To return to the basics and evaluate whether VLMs can associate cross-image information by matching visual cues like humans, rather than relying on knowledge reserves**.

## Method

### Overall Architecture
VLM2-Bench is constructed around three categories of visual cues:
- **GC (General Cue)**: Matching/tracking across edited image pairs
- **OC (Object-centric Cue)**: Recognition of everyday objects across different scenes
- **PC (Person-centric Cue)**: Identification of the same individual across images/videos

### Key Designs

1. **GC: General Cue (2 subtasks)**:

    - **Matching**: Given an original image and an edited image, determine whether a certain visual element remains consistent between the two images.
    - **Tracking**: Track changes in specific visual cues before and after editing.
    - Data Source: Reuses image editing datasets, including instance-level and environment-level modifications.
    - QA Construction Three-Stage Pipeline: Manual review $\rightarrow$ Saliency sampling (using LLMs to calculate a saliency score to filter out overly simple cases) $\rightarrow$ Paired answer generation.
    - The saliency score formula leverages the log probability of the editing description $P$ given the image description context. Samples below a threshold of $-2.0$ are retained.

2. **OC: Object-centric Cue (3 subtasks)**:

    - **Comparison**: Determine whether objects in different images are the same. Consistency pair validation is introduced (generating both a positive and negative statement for each judgment, requiring both to be answered correctly to count as correct).
    - **Counting**: Identify the number of unique objects while avoiding duplicate counting.
    - **Grouping**: The most challenging task, requiring the identification of all instances belonging to the same object.
    - Data Collection: Manually collected multi-category images of everyday objects (pets, cups, etc.), with 4 different scene images per object + 4 distractor images of similar but different objects.

3. **PC: Person-centric Cue (4 subtasks)**:

    - Comparison, Counting, and Grouping similar to OC.
    - Additional **VID (Video Identity Describing)**: Evaluates whether the model can correctly describe the identity of people appearing in videos.
    - Distractor images are selected based on the highest CLIP similarity for different individuals.
    - VID constructs two types of video sequences: $P_i \rightarrow \neg P_i$ and $P_i \rightarrow \neg P_i \rightarrow P_i$.

4. **Evaluation Metrics Design**:

    - **T/F Tasks (Mat, Trk, Cpr)**: Paired evaluation, recorded as correct only if both positive and negative statements are answered correctly.
    - **Numerical Tasks (Counting)**: Normalized error + number of images weight + error amplification factor $\alpha$.
    - **Multiple-Choice Tasks (Grouping)**: Standard accuracy.
    - **Open-ended Tasks (VID)**: GPT-4o + rule-based prompt scoring, with a manual validation accuracy of $98.89\%$.

### Loss & Training
This paper acts as an evaluation benchmark and does not involve model training.

## Key Experimental Results

### Main Results

| Model | GC-Mat | GC-Trk | OC-Cpr | OC-Cnt | PC-Cpr | PC-Grp | Overall Avg | Δ_human |
|--------|------|------|------|------|------|------|------|------|
| Human | 95.06 | 98.11 | 96.02 | 94.23 | 97.08 | 91.17 | 94.44 | 0.0 |
| GPT-4o(0806) | 37.45 | 39.27 | 74.17 | 80.62 | 50.00 | 47.00 | 59.56 | -34.88 |
| Claude-3.7 | 33.72 | 36.41 | 74.44 | 73.02 | 67.50 | 60.00 | 59.57 | -34.87 |
| Qwen2.5-VL-7B | 35.91 | 43.38 | 71.39 | 41.72 | 80.00 | 69.00 | 55.86 | -38.58 |
| Chance-Level | 25.00 | 25.00 | 50.00 | 34.88 | 50.00 | 25.00 | 33.72 | -61.44 |

### Ablation Study (Visual Bias Check)

| Resolution | Qwen2.5-VL Mat | Qwen2.5-VL Trk | Qwen2.5-VL OC-Cpr |
|------|---------|------|------|
| Original Image | 35.91 | 43.38 | 71.39 |
| ↓×4 | 19.69 | 33.33 | 52.78 |
| ↓×16 | 9.27 | 18.72 | 34.17 |

### Key Findings
- **Finding I**: Simple tasks where humans easily achieve near-perfect scores present a significant challenge for VLMs. The strongest commercial models lag behind humans by over $30\%$.
- **Finding II**: Within GC, the Matching task is most difficult for "swap" category changes (requiring matching of all other cues first), while the Tracking task is most difficult for "add/remove" changes (requiring association of elements that do not appear).
- **Finding III**: Models perform better on person-centric cues than on object-centric cues (with average score improvements of $7.65\%$, $9.75\%$, and $11.83\%$ for Cpr, Cnt, and Grp, respectively), potentially because individuals in the training data have explicit name anchors.
- **Finding IV**: Chain-of-Thought (CoT) language reasoning helps in associating visual cues logically.
- **Finding V**: The effectiveness of Visual prompting (VP-grid) depends on the model's dual understanding capability of both the prompt cues and the visual content.
- **Finding VI**: The open-ended nature of language can hinder object grouping tasks.
- **Finding VII**: Zooming in on object cues (VP-zoom) benefits strong models and does no harm to weak models.
- **Finding VIII**: Neither CoT nor visual prompting can improve the association of highly abstract human facial features.

## Highlights & Insights
- Accurately pinpoints a blind spot in VLM capabilities: pure visual matching tasks that require no knowledge turn out to be a major weakness for VLMs.
- The consistency pair validation mechanism eliminates the confounding effect of model binary decision bias.
- Resolution ablation experiments demonstrate that the benchmark truly tests fine-grained visual perception rather than superficial biases.
- Comprehensive analysis of CoT and visual prompting provides practical guidelines on when language reasoning is beneficial versus when it is harmful.
- The eight findings form a complete analytical framework, offering clear guidance for future VLM improvements.

## Limitations & Future Work
- The scale of the benchmark is relatively limited ($3060$ QA pairs), meaning model performance may not fully generalize to all real-world scenarios.
- Limitations of automated evaluation result in a relatively small proportion of open-ended questions in the benchmark.
- Complex multi-hop visual cue associations (such as chain associations: $A \leftrightarrow B \leftrightarrow C$) are not covered.
- Future directions to explore: Utilizing self-play to train model visual matching capabilities; synthetic data pre-training to boost fine-grained visual perception; and evaluation via multi-turn dialogues.
- The current VLM training paradigm overemphasizes vision-language association, neglecting the cultivation of reasoning capabilities purely within the visual domain.

## Related Work & Insights
- Difference from multi-image benchmarks like NaturalBench and MMDU: VLM2-Bench focuses on visual cue association rather than knowledge-based QA.
- Limitations of similarity evaluations like CLIP: Similarity scores cannot capture true visual matching capabilities.
- Inspiration for VLM training paradigms: A transition is needed from "vision-language alignment" to a "vision-vision reasoning" paradigm.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic evaluation of the visual cue association capability of VLMs, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation of 12 models and multiple prompting methods, though the benchmark scale could be larger.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The eight findings are well-structured, the conclusions are highly instructive, and the figures/tables are intuitively designed.
- **Value**: ⭐⭐⭐⭐ Reveals foundational capability defects in VLMs, offering key inspirations for future VLM training directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](../../ICCV2025/multimodal_vlm/hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)
- [\[ACL 2025\] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](speaking_beyond_language.md)
- [\[ACL 2025\] Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval](maximal_matching_matters_preventing_representation_collapse_for_robust_cross-mod.md)
- [\[ICCV 2025\] Visual Interestingness Decoded: How GPT-4o Mirrors Human Interests](../../ICCV2025/multimodal_vlm/visual_interestingness_decoded_how_gpt-4o_mirrors_human_interests.md)

</div>

<!-- RELATED:END -->
