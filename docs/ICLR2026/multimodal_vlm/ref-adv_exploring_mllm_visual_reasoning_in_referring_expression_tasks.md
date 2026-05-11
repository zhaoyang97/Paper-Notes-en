---
title: >-
  [Paper Note] Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks
description: >-
  [ICLR 2026][Multimodal VLM][Referring Expression Comprehension] This paper introduces the Ref-Adv benchmark, constructed via a pipeline of **hard distractor pairing + LLM-assisted minimally sufficient expression generati…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Referring Expression Comprehension"
  - "Visual Grounding"
  - "Hard Distractors"
  - "Benchmark"
  - "Shortcut Suppression"
date: 2026-05-08
content_hash: 31b46e2c3ce2e922
---

# Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks

**Conference**: ICLR 2026
**arXiv**: [2602.23898](https://arxiv.org/abs/2602.23898)
**Code**: [https://ref-adv.github.io/](https://ref-adv.github.io/)
**Authors**: Qihua Dong, Kuo Yang, Lin Ju, Handong Zhao, Yitian Zhang, Yizhou Wang, Huimin Zeng, Jianglin Lu, Yun Fu
**Area**: Multimodal VLM — Referring Expression Comprehension, Visual Reasoning
**Keywords**: Referring Expression Comprehension, Visual Grounding, Hard Distractors, Benchmark, Shortcut Suppression

## TL;DR

This paper introduces the Ref-Adv benchmark, constructed via a pipeline of **hard distractor pairing + LLM-assisted minimally sufficient expression generation + three-annotator unanimous verification**. The benchmark eliminates "grounding shortcuts" present in classical REC datasets. Across 13 contemporary MLLMs — including GPT-4o, Gemini 2.5, and Qwen2.5-VL-72B — accuracy drops dramatically from 90%+ on RefCOCO(+/g) to 50–68% on Ref-Adv, systematically exposing severe deficiencies in complex visual reasoning and precise grounding.

## Background & Motivation

**Background**: Referring Expression Comprehension (REC) is a classical task that localizes a natural language description to a specific region in an image. RefCOCO, RefCOCO+, and RefCOCOg are the standard benchmarks, on which state-of-the-art MLLMs (e.g., Qwen2.5-VL-72B, InternVL-3) have achieved 90%+ accuracy, approaching saturation.

**Limitations of Prior Work**: Classical REC benchmarks suffer from three systematic flaws: ① expressions are extremely short (averaging only 3.6 words in RefCOCO/RefCOCO+), imposing minimal language understanding demands; ② distractors are sparse (most images in RefCOCO(+/g) contain only one instance of the same category), making classification sufficient for localization; ③ "grounding shortcuts" exist, as many redundant descriptors allow models to succeed by matching only a subset of the description without comprehending the full expression.

**Key Challenge**: High scores do not equate to genuine visual reasoning. Experiments show that replacing expressions with the fixed phrase "the one," shuffling word order into a bag-of-words representation, or deleting a single descriptor causes only marginal performance drops on RefCOCO(+/g). This indicates that benchmark scores substantially overestimate models' true reasoning and grounding capabilities.

**Goal**: To construct a modern REC benchmark satisfying the following criteria: ① every expression requires multi-step textual reasoning (understanding the target description and its distinguishing attributes relative to distractors); ② fine-grained visual reasoning is required to distinguish the target among multiple highly similar candidates; ③ shortcuts that bypass reasoning are eliminated.

**Key Insight**: REC is reframed as a coupled multi-step reasoning task involving **textual reasoning + visual reasoning**. By mandatorily introducing "hard distractors" (same-category objects that partially but not fully match the target) and using LLMs to generate expressions containing only **minimally sufficient descriptors**, every descriptor becomes necessary for localization, eliminating shortcuts at the source.

**Core Idea**: A data construction pipeline combining hard distractor pairing and minimally sufficient expression generation builds a REC benchmark in which every descriptor is indispensable for localization, thereby genuinely evaluating MLLMs' visual reasoning capability.

## Method

### Overall Architecture

Ref-Adv proposes a new **data construction pipeline and benchmark dataset** rather than a new model. The pipeline consists of four stages: input preparation → similarity judgement (finding hard distractor pairs) → expression generation (minimally sufficient descriptions) → human verification (unanimous three-annotator approval). The final dataset contains 5,000 high-quality referring expression–target pairs, with a publicly released subset, Ref-Adv-s, comprising 1,142 instances.

### Key Designs

1. **Hard Distractor Pressure**:

    - *Function*: Ensures each image contains distractors that are highly similar to the target but not identical, compelling models to perform fine-grained visual discrimination.
    - *Mechanism*: Images containing **≥3 instances of the same category** (based on COCO and OpenImages v7 panoptic instance annotations) are first filtered. GPT-4o then partitions candidate instances into Group A (target + hard distractor) and Group B (other distractors), identifying distinguishing attributes between groups and subtle differences within Group A.
    - *Design Motivation*: Over 70% of images in RefCOCO(+/g) contain 0–1 same-category distractor, allowing models to localize via category recognition alone. Enforcing ≥3 same-category distractors elevates the task from a "classification problem" to a "fine-grained discrimination problem."

2. **Two-Stage Minimally Sufficient Expression Generation**:

    - *Function*: Generates expressions that are both natural and contain only the descriptors necessary for localization, eliminating shortcuts introduced by redundant descriptions.
    - *Mechanism*: **Stage 1** (Similarity Judgement): GPT-4o outputs a list of attributes distinguishing Group A from Group B, and attributes distinguishing the two instances within Group A, producing multiple candidate descriptor sets. **Stage 2** (Expression Generation): Starting from the minimally sufficient subset of these descriptors, the LLM generates natural expressions using two strategies — positive descriptors of the target, or **negated forms** of the hard distractor's descriptors (introducing negation reasoning). The paper explicitly abandons single-step direct generation, as GPT-4o tends to produce overspecified expressions containing many redundant descriptors in single-step mode.
    - *Design Motivation*: Ref-Adv data statistics show an average expression length of 11.5 words, an average of 4.01 distractors, and a negation ratio of 21.25% — far exceeding RefCOCO's 3.6 words / 3.99 distractors / 0.99% negation — while **every descriptor remains indispensable**.

3. **Three-Annotator Verification Protocol**:

    - *Function*: Filters hallucinations and ambiguities in LLM-generated outputs, ensuring annotation quality.
    - *Mechanism*: Three annotators independently make two judgements: ① whether the expression is correct and unambiguous (first localizing the target independently on an unlabeled image, then reflecting with reference to the ground truth before confirming); ② whether hard distractors are genuinely present in the image. A sample is retained only when **all three annotators agree unanimously**. The acceptance rate of LLM-generated expressions is only 18.7%, reflecting extremely rigorous quality control.
    - *Design Motivation*: LLM-assisted annotation inevitably introduces hallucinations; human verification is a necessary step to ensure benchmark credibility.

### Benchmark Quality Validation

Three ablation tests are designed to validate the quality of Ref-Adv:

- **Model Bias Test**: All expressions are replaced with the fixed phrase "the one" to assess whether models can localize via statistical bias alone. Qwen2.5-VL-72B still achieves 35.1% on RefCOCO but only 21.4% on Ref-Adv (Δ=−13.7%), demonstrating that Ref-Adv is less susceptible to data bias.
- **Bag-of-Words Test**: Expression word order is shuffled. On Ref-Adv, Qwen2.5-VL-72B drops by 16.8% (58.3→41.5%), significantly more than the 9.9% drop on RefCOCO, confirming that Ref-Adv demands genuine textual understanding.
- **Descriptor Deletion Test**: One descriptor is randomly removed. On Ref-Adv the drop is 6.4% (58.3→51.9%), exceeding RefCOCO's 4.7%, indicating that each descriptor in Ref-Adv is more necessary.

## Key Experimental Results

### Table 1: Ref-Adv Benchmark Statistics vs. Classical REC Benchmarks

| Benchmark | Images | Instances | Avg. Expr. Length | Avg. Distractors | Negation Ratio | Vocabulary |
|-----------|--------|-----------|-------------------|------------------|----------------|------------|
| RefCOCO | 3,000 | 7,596 | 3.6 | 3.99 | 0.99% | 3,525 |
| RefCOCO+ | 3,000 | 7,578 | 3.6 | 3.96 | 3.36% | 4,387 |
| RefCOCOg | 3,900 | 7,596 | 8.4 | 1.64 | 1.41% | 5,050 |
| **Ref-Adv** | **2,833** | **5,000** | **11.5** | **4.01** | **21.25%** | **5,308** |

Ref-Adv comprehensively surpasses classical benchmarks in expression length, vocabulary diversity, and negation reasoning ratio, while maintaining a high distractor density.

### Table 2: Main Results on Full Ref-Adv (Representative Models)

| Model | CoT | SoM | Acc@0.5 | Acc@0.75 | Acc@0.9 | mAcc | ≥7 Distractors Δ |
|-------|-----|-----|---------|----------|---------|------|------------------|
| GPT-4o | ✗ | ✓ | 52.3 | 31.2 | 13.4 | 27.8 | −0.6 |
| GPT-4o | ✓ | ✓ | **63.7** | **38.4** | **19.7** | **34.1** | −3.2 |
| Claude-3.5 Sonnet | ✗ | ✓ | 40.8 | 22.1 | 3.8 | 22.4 | −3.4 |
| Gemini 2.5-Flash | ✓ | ✗ | 59.4 | 35.1 | 16.3 | 30.6 | −3.8 |
| Gemini 2.5-Pro | ✓ | ✗ | 59.1 | 32.6 | 14.2 | 28.3 | −3.2 |
| InternVL-3-78B | ✓ | ✗ | 58.4 | 47.9 | 29.6 | 41.2 | −3.0 |
| Qwen2.5-VL-72B | ✓ | ✗ | 58.3 | 47.8 | 29.5 | 41.1 | −2.7 |
| GLM-4.5V | ✓ | ✗ | 56.9 | 46.6 | 28.8 | 40.2 | −2.3 |
| CogVLM-Grounding | ✗ | ✗ | 51.5 | 41.2 | 23.4 | 35.0 | −0.7 |

Compared to 90%+ accuracy on RefCOCO(+/g), no model exceeds 64% Acc@0.5 on Ref-Adv. **The strongest configuration, GPT-4o+CoT+SoM, achieves only 63.7%**, revealing a substantial reasoning gap. The gap widens at higher IoU thresholds: Acc@0.9 peaks at only 29.6% (InternVL-3-78B).

### Table 3: Effect of Model Scale and Thinking Mode on Ref-Adv-s (Qwen Series)

| Model | CoT/Thinking | Acc@0.5 | ≥7 Distractors Acc@0.5 | ≥7 Distractors Δ |
|-------|-------------|---------|------------------------|------------------|
| Qwen2.5-VL-3B | ✗ | 23.8 | 17.1 | −6.8 |
| Qwen2.5-VL-72B | ✓ | 52.4 | 38.8 | −13.6 |
| Qwen3-VL-2B-Thinking | ✓ | 44.4 | 31.0 | −13.4 |
| Qwen3-VL-8B-Thinking | ✓ | 59.5 | 47.3 | −12.2 |
| Qwen3-VL-32B-Thinking | ✓ | 65.6 | 52.7 | −12.9 |
| Qwen3-VL-235B-A22B-Thinking | ✓ | 67.1 | 56.6 | −10.5 |
| Qwen3.5-397B-A17B | ✓ | **68.0** | 56.6 | −11.4 |

Two key findings emerge: ① the Thinking mode substantially outperforms the same-scale Instruct mode (e.g., Qwen3-VL-8B-Thinking 59.5 vs. Instruct 47.2, +12.3); ② even the largest model, Qwen3.5-397B, achieves only 68.0%, and performance in high-distractor settings remains significantly degraded (−11.4).

### Key Findings

- **CoT is effective on Ref-Adv but not on RefCOCO**: Ref-Adv requires multi-step reasoning, and CoT helps models progressively eliminate distractors; on RefCOCO, where localization is straightforward, CoT instead introduces unnecessary redundancy and errors.
- **Distractor count is a performance bottleneck**: All models perform significantly worse in the ≥7-distractor group than overall, with a maximum drop of −19.3% (Qwen3-VL-235B-A22B-Instruct), indicating that handling multiple highly similar candidates is a core weakness of current MLLMs.
- **Models frequently select the hard distractor**: Qualitative analysis reveals that even with CoT, models often select the hard distractor rather than the true target — due to visual perception errors or misinterpretation of the expression — mid-way through the reasoning chain.
- **Acc@0.9 is extremely low**: Even when localization succeeds (Acc@0.5), precise bounding box regression remains poor, with the best model reaching only ~35% Acc@0.9.

## Highlights & Insights

1. **Systematic diagnosis of the "shortcut" problem**: The paper is the first to unify three systematic flaws of REC benchmarks (short expressions, sparse distractors, redundant descriptors) into a single "grounding shortcut" framework, quantitatively validated through three ablation tests (bias / word order / descriptor deletion), providing a methodological template for future benchmark design.

2. **Two-stage generation outperforms single-step generation**: Abandoning direct single-step LLM expression generation in favor of first extracting discriminative attributes and then composing expressions from the minimal subset is a transferable design insight — in any scenario requiring LLMs to generate "precise and non-redundant" text, analysis-then-composition outperforms direct end-to-end generation.

3. **Introduction of negation reasoning**: The 21.25% negation expression ratio (vs. only 0.99% in RefCOCO) tests models' ability to understand "not X," an underexplored but important reasoning dimension.

4. **Clear advantage of the Thinking mode**: Qwen3-VL-2B-Thinking (44.4%) even outperforms Qwen2.5-VL-32B-Instruct (48.0%), demonstrating that a smaller model with reasoning training can surpass a larger model with standard fine-tuning on tasks requiring complex reasoning.

## Limitations & Future Work

1. **Limited data sources**: Only images from COCO and OpenImages v7 are used, limiting scene diversity. More complex real-world scenarios (e.g., dense urban street scenes, industrial inspection) are not covered, and generalizability remains to be validated.

2. **SoM dependency affects fairness**: GPT-4o and Claude are evaluated using Set-of-Marks (SoM) + Semantic-SAM, while open-source models directly output coordinates. SoM converts localization into a selection problem, making the two evaluation paradigms not fully comparable and potentially overestimating the reasoning capability of SoM-based models.

3. **LLM expression generation ceiling**: The 18.7% acceptance rate means 81.3% of LLM-generated expressions are discarded, and retained samples may be biased toward simpler scenes that LLMs can correctly interpret, introducing potential selection bias.

4. **Absence of segmentation-level evaluation**: Ref-Adv evaluates only bounding-box-level localization (IoU), without extension to referring expression segmentation (RES), where pixel-level grounding imposes even higher demands on precise understanding.

5. **No coverage of video or 3D scenes**: REC is equally important in video understanding and 3D scene localization; evaluation on static images alone is insufficient to comprehensively measure models' reasoning capabilities.

## Related Work & Insights

- **vs. RefCOCO(+/g)**: Classical benchmarks feature short expressions, sparse distractors, and permissive shortcuts; Ref-Adv systematically strengthens every dimension. RefCOCO nonetheless retains value for assessing models' basic grounding capability.

- **vs. Cops-Ref / FineCops-Ref**: These benchmarks also introduce compositional reasoning and distractors, but generate expressions using fixed GQA scene graph templates, yielding lower naturalness compared to Ref-Adv's LLM + human pipeline.

- **vs. HC-RefLoCo / Ref-L4**: These works push toward longer and more natural expressions but do not fundamentally address grounding shortcuts introduced by descriptor redundancy. HC-RefLoCo's average length of 90+ words actually introduces more shortcuts.

- **Relationship to VQA/reasoning benchmarks**: Ref-Adv's design philosophy — eliminating shortcuts and enforcing reasoning — parallels adversarial benchmarks in VQA (e.g., VQA-CP, Winoground), and can be regarded as an adversarial benchmark for the REC domain.

## Rating

- Novelty: ⭐⭐⭐⭐ The benchmark construction methodology (hard distractors + minimally sufficient expressions + triple verification) demonstrates systematic innovation, though the core idea represents an engineering solution to a known problem.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation covers 13 models, three ablation tests, multiple IoU thresholds, distractor-count-stratified analysis, CoT comparisons, and scaling experiments — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The problem statement is clear and the ablation design is elegant, though the paper is somewhat lengthy and certain analyses could be more concise.
- Value: ⭐⭐⭐⭐ Significant reference value for the MLLM community; exposes the false prosperity behind saturated RefCOCO scores and advances more realistic evaluation standards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](../../ICCV2025/multimodal_vlm/dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[CVPR 2026\] CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning](../../CVPR2026/multimodal_vlm/codedance_a_dynamic_tool-integrated_mllm_for_executable_visual_reasoning.md)
- [\[CVPR 2026\] Downscaling Intelligence: Exploring Perception and Reasoning Bottlenecks in Small VLMs](../../CVPR2026/multimodal_vlm/downscaling_intelligence_exploring_perception_and_reasoning_bottlenecks_in_small.md)
- [\[ICLR 2026\] Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)
- [\[ICCV 2025\] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs](../../ICCV2025/multimodal_vlm/are_they_the_same_exploring_visual_correspondence_shortcomings_of_multimodal_llm.md)

</div>

<!-- RELATED:END -->
