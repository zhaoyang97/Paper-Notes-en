---
title: >-
  [Paper Note] AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis
description: >-
  [ACL 2026][Multimodal VLM][Affective Analysis] This paper proposes AICA-Bench, a comprehensive benchmark covering three dimensions: Affective Understanding (EU), Affective Reasoning (ER)…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Affective Analysis"
  - "Visual Language Models"
  - "Benchmark"
  - "Affective Reasoning"
  - "Prompt Engineering"
date: 2026-05-08
content_hash: b98984f9dd934c5c
---

# AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.05900](https://arxiv.org/abs/2604.05900)  
**Code**: None  
**Area**: Multimodal VLM / Affective Computing  
**Keywords**: Affective Analysis, Visual Language Models, Benchmark, Affective Reasoning, Prompt Engineering

## TL;DR
This paper proposes AICA-Bench, a comprehensive benchmark covering three dimensions: Affective Understanding (EU), Affective Reasoning (ER), and Emotion-Guided Content Generation (EGCG). After evaluating 23 VLMs, the study identifies two major defects: intensity calibration failure and shallow descriptions. It further proposes the training-free GAT Prompting framework to mitigate these issues.

## Background & Motivation

**Background**: VLMs have achieved significant progress in perceptual capabilities. Existing benchmarks primarily evaluate factual correctness, semantic localization, and visual reasoning. Recently, some benchmarks assessing the affective capabilities of VLMs (e.g., EVE, AffectGPT, EEmo-Bench) have emerged, but they focus mainly on basic affective classification tasks.

**Limitations of Prior Work**: Existing affective benchmarks suffer from three key deficiencies: (1) limited coverage with few image affective datasets; (2) a primary focus on multiple-choice affective classification, lacking evaluation of affective reasoning and emotion-guided generation; (3) the absence of a holistic evaluation framework for the "understanding-reasoning-generation" pipeline. Affective intelligence requires not only identifying affective cues but also reasoning about emotional causes and producing appropriate affective expressions.

**Key Challenge**: The lack of a comprehensive AICA benchmark is a critical bottleneck for advancing affective intelligence—the inability to perform systematic evaluation means an inability to improve effectively.

**Goal**: To construct a holistic affective image content analysis benchmark covering understanding, reasoning, and generation dimensions, and to identify systematic deficiencies in VLMs regarding affective tasks.

**Key Insight**: Drawing from affective psychology, affective intelligence consists of three levels: perception, attribution, and expression, which correspond to the three designed evaluation tasks.

**Core Idea**: AICA-Bench, containing 9 datasets and 18,124 instructions, is used to comprehensively evaluate the affective capabilities of VLMs. It reveals two systematic issues: "intensity hallucination" and "shallow description." These are addressed using GAT Prompting through visual anchors and hierarchical reasoning.

## Method

### Overall Architecture
AICA-Bench encompasses three core tasks: EU (Affective Understanding: identifying expressed and evoked emotions), ER (Affective Reasoning: explaining why an image evokes a specific emotion), and EGCG (Emotion-Guided Content Generation: generating consistent descriptions based on the image and a target emotion). Dataset construction involves two stages: image filtering and automated instruction generation via GPT-4o.

### Key Designs

1. **Three-Dimensional Evaluation Task Design**:
    - Function: Comprehensively evaluates the affective intelligence of VLMs.
    - Mechanism: EU is evaluated using weighted F1 across basic and CoT prompting modes. ER and EGCG utilize a scoring model fine-tuned on QwenVL2.5-7B to score across three dimensions: affective alignment, description richness, and causal rationality (achieving a Pearson correlation of 0.88/0.90 with human annotation).
    - Design Motivation: Traditional automatic metrics (e.g., BLEU) cannot capture critical affective dimensions, necessitating a specialized affective scoring model.

2. **Diagnostic Error Analysis**:
    - Function: Reveals systematic flaws in VLM affective capabilities.
    - Mechanism: Analysis shows that intensity errors account for 72.25% of misclassifications (models distinguish positive/negative valence but struggle to calibrate intensity, e.g., misjudging Amusement as Contentment). In open-ended tasks, affective alignment scores are high (median ~4.1) but descriptive scores are low (median ~3.0), indicating a "safe response trap."
    - Design Motivation: To understand the cognitive mechanisms of failure rather than just reporting accuracy.

3. **GAT Prompting (Grounded Affective Tree)**:
    - Function: A training-free method to improve VLM affective capabilities.
    - Mechanism: Consists of two components: Visual Scaffolding (using image segmentation to generate visual anchor regions, guiding the model to scan and extract objective visual elements region-by-region) and AffectToT Reasoning (using a fixed search depth $d=3$ and width $k=3$ to generate three competitive affective-intensity hypotheses, each citing specific region IDs as evidence, followed by a verification phase to prune inconsistent hypotheses).
    - Design Motivation: Forces the model to focus on concrete visual evidence via visual anchors rather than relying on linguistic priors, and eliminates intensity hallucinations through hypothesis competition.

### Loss & Training
The scoring model was fine-tuned on QwenVL2.5-7B using 10,000 QA pairs (generated by GPT-4o and scored by 5 annotators, with a Krippendorff's α=0.78).

## Key Experimental Results

### Main Results

| Model | EU Avg. | ER Avg. | EGCG Avg. | Overall |
|------|---------|---------|-----------|---------|
| Gemini-2.5-Pro | 67.27 | 79.08 | 74.13 | 73.49 |
| GPT-4o | 64.93 | 77.81 | 75.73 | 72.82 |
| Qwen2.5VL-7B | 56.84 | 74.50 | 66.00 | 65.78 |
| LLaVA-1.6-13B | 41.80 | 73.57 | 64.51 | 59.96 |

### Performance Gain with GAT Prompting

| Model | EU Gain | ER Gain | EGCG Gain |
|------|--------|--------|----------|
| Gemini-2.5-Pro | +4.18 | +3.37 | +4.12 |
| GPT-4o | +2.98 | +3.69 | +3.27 |
| Average (All Models) | +6.15 | +3.54 | +3.96 |

### Key Findings
- All models exhibit a "top-heavy" pattern: reasoning and generation scores are 15-30% higher than understanding scores, indicating that models infer emotion based on linguistic priors rather than true visual perception.
- Gains from scaling models from 8B to 16B are minimal; the bottleneck lies in visual encoding quality rather than model scale.
- Masking faces leads to an 11.1% drop in F1, revealing a heavy reliance on facial expressions as a visual shortcut.

## Highlights & Insights
- The **separated analysis of intensity vs. polarity errors** is highly insightful—72.25% of errors stem from intensity rather than valence, proving that affective granularity is the true challenge.
- **Discovery of the "Safe Response Trap"**: Models tend to generate templated safe responses in open-ended tasks rather than providing in-depth analysis, a phenomenon common in other open-ended evaluations.
- The **design philosophy of GAT Prompting** is transferable to any VLM task requiring fine-grained visual grounding.

## Limitations & Future Work
- The scoring model is fine-tuned on QwenVL2.5-7B, which may introduce biases relative to the models being evaluated.
- Evaluation is limited to static images; dynamic affective changes in video are not covered.
- GAT Prompting increases inference complexity, and the cost of practical deployment needs consideration.

## Related Work & Insights
- **vs. EVE**: While EVE only evaluates classification and explanation for 7 models, AICA-Bench evaluates the full "understanding-reasoning-generation" pipeline across 23 models.
- **vs. EEmo-Bench**: While EEmo-Bench focuses only on evoked emotions, AICA-Bench distinguishes between expressed and evoked emotions.

## Rating
- Novelty: ⭐⭐⭐⭐ First affective VLM benchmark covering the understanding-reasoning-generation dimensions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 23 models, 9 datasets, 18K+ instructions.
- Writing Quality: ⭐⭐⭐⭐ Deep diagnostic analysis.
- Value: ⭐⭐⭐⭐ Provides a solid benchmark for multimodal affective research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](cnsl-bench_benchmarking_the_sign_language_understanding_capabilities_of_mllms_on.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)
- [\[CVPR 2026\] Rethinking VLMs for Image Forgery Detection and Localization](../../CVPR2026/multimodal_vlm/rethinking_vlms_for_image_forgery_detection_and_localization.md)
- [\[AAAI 2026\] Multi-Agent VLMs Guided Self-Training with PNU Loss for Low-Resource Offensive Content Detection](../../AAAI2026/multimodal_vlm/multi-agent_vlms_guided_self-training_with_pnu_loss_for_low-resource_offensive_c.md)
- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)

</div>

<!-- RELATED:END -->
