---
title: >-
  [Paper Note] AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis
description: >-
  [ACL 2026][Multimodal VLM][Sentiment Analysis] This paper proposes AICA-Bench, a comprehensive benchmark covering three dimensions — Emotion Understanding (EU), Emotion Reasoning (ER)…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Sentiment Analysis"
  - "Vision-Language Models"
  - "Benchmark"
  - "Affective Reasoning"
  - "Prompt Engineering"
date: 2026-05-08
content_hash: 5a5d31a364525ee8
---

# AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis

**Conference**: ACL 2026
**arXiv**: [2604.05900](https://arxiv.org/abs/2604.05900)  
**Code**: None  
**Area**: Multimodal VLM / Affective Computing
**Keywords**: Sentiment Analysis, Vision-Language Models, Benchmark, Affective Reasoning, Prompt Engineering

## TL;DR
This paper proposes AICA-Bench, a comprehensive benchmark covering three dimensions — Emotion Understanding (EU), Emotion Reasoning (ER), and Emotion-Guided Content Generation (EGCG) — to evaluate 23 VLMs. The evaluation reveals two systematic deficiencies: intensity calibration failure and shallow description, and introduces a training-free framework, GAT Prompting, to mitigate these issues.

## Background & Motivation

**Background**: VLMs have achieved remarkable progress in perceptual capabilities, and existing benchmarks primarily assess factual correctness, semantic grounding, and visual reasoning. Some benchmarks evaluating affective capabilities of VLMs have recently emerged (e.g., EVE, AffectGPT, EEmo-Bench), but they mainly focus on basic emotion classification tasks.

**Limitations of Prior Work**: Existing affective benchmarks suffer from three critical shortcomings: (1) they incorporate only a limited number of image emotion datasets with restricted coverage; (2) they predominantly focus on multiple-choice emotion classification, lacking evaluation of affective reasoning and emotion-guided generation; and (3) they lack a holistic evaluation framework spanning the full "understanding–reasoning–generation" pipeline. Affective intelligence requires not only recognizing emotional cues but also reasoning about their causes and producing appropriate emotional expressions.

**Key Challenge**: The absence of a comprehensive AICA benchmark constitutes a critical bottleneck in advancing affective intelligence — systematic evaluation is a prerequisite for effective improvement.

**Goal**: To construct a holistic affective image content analysis benchmark covering the three dimensions of understanding, reasoning, and generation, and to identify systematic deficiencies of VLMs on affective tasks.

**Key Insight**: Drawing from affective psychology, the authors conceptualize affective intelligence as comprising three layers — perception, attribution, and expression — which correspond to the three evaluation tasks in the benchmark design.

**Core Idea**: AICA-Bench, consisting of 9 datasets and 18,124 instructions, comprehensively evaluates the affective capabilities of VLMs, revealing two systemic problems — "intensity hallucination" and "shallow description" — and proposes GAT Prompting to mitigate them via visual anchoring and hierarchical reasoning.

## Method

### Overall Architecture
AICA-Bench comprises three core tasks: EU (Emotion Understanding, identifying expressed and evoked emotions in images), ER (Emotion Reasoning, explaining why an image evokes specific emotions), and EGCG (Emotion-Guided Content Generation, generating consistent descriptions based on an image and a target emotion). Dataset construction involves two stages: image filtering and automatic instruction generation via GPT-4o.

### Key Designs

1. **Three-Dimensional Evaluation Task Design**:

    - Function: Comprehensively evaluate the affective intelligence of VLMs.
    - Mechanism: EU is evaluated using weighted F1 under both basic and Chain-of-Thought prompting modes; ER and EGCG are scored by a scoring model fine-tuned from QwenVL2.5-7B, assessing three dimensions — emotional alignment, descriptive richness, and causal plausibility — achieving Pearson correlations of 0.88/0.90 with human annotations.
    - Design Motivation: Conventional automatic metrics (e.g., BLEU) fail to capture critical affective dimensions, necessitating a dedicated affective scoring model.

2. **Diagnostic Error Analysis**:

    - Function: Reveal systematic deficiencies in VLM affective capabilities.
    - Mechanism: Analysis identifies two major issues — intensity errors account for 72.25% of misclassifications (models can distinguish positive from negative affect but struggle to calibrate intensity, e.g., misclassifying Amusement as Contentment); in open-ended tasks, emotional alignment scores are high (median ~4.1) while descriptive scores are low (median ~3.0), indicating a "safe response trap."
    - Design Motivation: Rather than simply reporting accuracy, the goal is to understand the cognitive mechanisms underlying failures.

3. **GAT Prompting (Grounded Affective Tree)**:

    - Function: Training-free enhancement of VLM affective capabilities.
    - Mechanism: Composed of two components — a visual scaffold (using image segmentation to generate visual anchor regions, guiding the model to scan region-by-region and extract objective visual elements) and AffectToT reasoning (with fixed search depth $d=3$ and breadth $k=3$, generating 3 competing emotion–intensity hypotheses, each citing specific region IDs as evidence, followed by a verification phase that prunes inconsistent hypotheses).
    - Design Motivation: Visual anchoring forces the model to attend to concrete visual evidence rather than relying on linguistic priors, and hypothesis competition eliminates intensity hallucinations.

### Loss & Training
The scoring model is fine-tuned on QwenVL2.5-7B using 10,000 question-answer pairs (generated by GPT-4o and rated by 5 annotators, Krippendorff's α = 0.78).

## Key Experimental Results

### Main Results

| Model | EU Avg. | ER Avg. | EGCG Avg. | Overall |
|-------|---------|---------|-----------|---------|
| Gemini-2.5-Pro | 67.27 | 79.08 | 74.13 | 73.49 |
| GPT-4o | 64.93 | 77.81 | 75.73 | 72.82 |
| Qwen2.5VL-7B | 56.84 | 74.50 | 66.00 | 65.78 |
| LLaVA-1.6-13B | 41.80 | 73.57 | 64.51 | 59.96 |

### GAT Prompting Gains

| Model | EU Gain | ER Gain | EGCG Gain |
|-------|---------|---------|-----------|
| Gemini-2.5-Pro | +4.18 | +3.37 | +4.12 |
| GPT-4o | +2.98 | +3.69 | +3.27 |
| Average (all models) | +6.15 | +3.54 | +3.96 |

### Key Findings
- All models exhibit a "top-heavy" pattern: reasoning and generation scores exceed understanding scores by 15–30%, suggesting that models infer affect through linguistic priors rather than genuine visual perception.
- Scaling models from 8B to 16B yields marginal gains, indicating the bottleneck lies in visual encoding quality rather than model scale.
- Masking facial regions causes an 11.1% drop in F1, revealing that models heavily rely on facial expressions as a visual shortcut.

## Highlights & Insights
- The **separation analysis of intensity vs. polarity errors** is particularly illuminating — 72.25% of errors stem from intensity rather than positive/negative polarity, indicating that emotional granularity is the true bottleneck.
- The **discovery of the "safe response trap"**: models tend to generate templated, safe responses in open-ended tasks rather than engaging in in-depth analysis, a phenomenon commonly observed across other open-ended evaluations.
- The **design philosophy of GAT Prompting** is transferable to any VLM task requiring fine-grained visual grounding.

## Limitations & Future Work
- The scoring model is fine-tuned from QwenVL2.5-7B, which may introduce bias toward the models being evaluated.
- Evaluation is limited to static images; dynamic affective changes in video are not addressed.
- GAT Prompting increases inference complexity, and deployment costs warrant practical consideration.

## Related Work & Insights
- **vs. EVE**: EVE evaluates only 7 models on classification and explanation; AICA-Bench evaluates 23 models across the full understanding–reasoning–generation pipeline.
- **vs. EEmo-Bench**: EEmo-Bench focuses solely on image-evoked emotions; AICA-Bench distinguishes between expressed emotions and evoked emotions.

## Rating
- Novelty: ⭐⭐⭐⭐ First affective VLM benchmark covering the three-dimensional understanding–reasoning–generation pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 models, 9 datasets, 18K+ instructions.
- Writing Quality: ⭐⭐⭐⭐ Diagnostic analysis is particularly in-depth.
- Value: ⭐⭐⭐⭐ Provides a solid benchmark for affective multimodal research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)
- [\[CVPR 2026\] Rethinking VLMs for Image Forgery Detection and Localization](../../CVPR2026/multimodal_vlm/rethinking_vlms_for_image_forgery_detection_and_localization.md)
- [\[ACL 2026\] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)
- [\[AAAI 2026\] Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection](../../AAAI2026/multimodal_vlm/multi-agent_vlms_guided_self-training_with_pnu_loss_for_low-resource_offensive_c.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](../../ICCV2025/multimodal_vlm/hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)

</div>

<!-- RELATED:END -->
