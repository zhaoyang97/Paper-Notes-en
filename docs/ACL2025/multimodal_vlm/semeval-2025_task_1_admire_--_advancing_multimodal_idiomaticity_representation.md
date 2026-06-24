---
title: >-
  [Paper Note] SemEval-2025 Task 1: AdMIRe -- Advancing Multimodal Idiomaticity Representation
description: >-
  [ACL 2025 (SemEval Workshop)][Multimodal VLM][Idiom Understanding] The SemEval-2025 AdMIRe shared task is designed to evaluate model comprehension of idiomatic expressions in multimodal (text + image) and multilingual (English + Brazilian Portuguese) contexts through two subtasks: image ranking and image sequence completion. The best-performing system achieves near-human performance using Mixture-of-Experts and multi-query smoothing strategies.
tags:
  - "ACL 2025 (SemEval Workshop)"
  - "Multimodal VLM"
  - "Idiom Understanding"
  - "Multimodal"
  - "Vision-Language Models"
  - "Shared Task"
  - "Mixture of Experts"
date: 2026-05-08
content_hash: 1f3c85eef70f9961
---

# SemEval-2025 Task 1: AdMIRe -- Advancing Multimodal Idiomaticity Representation

**Conference**: ACL 2025 (SemEval Workshop)  
**arXiv**: [2503.15358](https://arxiv.org/abs/2503.15358)  
**Code**: Yes (Dataset: doi.org/10.15131/shef.data.28436600.v1, CC-BY-4.0)  
**Area**: Multimodal/VLM  
**Keywords**: Idiom Understanding, Multimodal, Vision-Language Models, Shared Task, Mixture of Experts

## TL;DR

The SemEval-2025 AdMIRe shared task is designed to evaluate model comprehension of idiomatic expressions in multimodal (text + image) and multilingual (English + Brazilian Portuguese) contexts through two subtasks: image ranking and image sequence completion. The best-performing system achieves near-human performance using Mixture-of-Experts and multi-query smoothing strategies.

## Background & Motivation

Idiomatic expressions represent a major challenge in natural language processing because their meaning cannot be directly inferred from the literal definitions of their constituent words. For example, "eager beaver" does not refer to an enthusiastic semi-aquatic rodent, but rather describes a highly enthusiastic and diligent person. Although large language models (LLMs) perform exceptionally well on general benchmarks, their handling of figurative language remains unstable.

Existing idiom comprehension datasets (e.g., NCTTI, FLUTE, MAGPIE) mostly focus on text-only domains. However, research suggests that these tasks may not truly require models to possess robust semantic representations of idioms. The introduction of the visual modality allows for a more rigorous test of whether a model truly "understands" the meaning of an idiom. Specifically, models must distinguish between different visual scenes corresponding to the literal versus figurative meanings of the idiom.

Furthermore, errors in handling idioms can lead to severe consequences in practical applications. As a real-world example, a translation error in machine translation once caused the Israeli Prime Minister to "praise" a Eurovision winner as "a real cow" due to an incorrect idiomatic translation.

## Method

### Overall Architecture

AdMIRe establishes two subtasks:
- **Subtask A (Static Image Selection)**: Given a context sentence containing a potential idiomatic noun-compound and 5 images, models must rank the images according to how well they match the idiomatic/literal meaning in the sentence.
- **Subtask B (Image Sequence Completion)**: Given the first 2 images of a sequence representing either the literal or figurative meaning of an idiom and 4 candidate images, models must select the correct 3rd image to complete the sequence and determine if the sequence depicts a literal or idiomatic expression.

### Key Designs

1. **Target Compound Selection**: Noun compounds with duality—possessing both a plausible, imaginable literal meaning and a figurative meaning—were filtered from existing datasets such as NCTTI, FLUTE, and MAGPIE. For instance, "silver bullet" can literally describe a bullet made of silver or figuratively refer to an all-encompassing solution. Purely compositional expressions (e.g., "olive oil") and idioms that are difficult to visualize (e.g., "kangaroo court") were excluded.

2. **Five-Tiered Image Design (Subtask A)**: For each expression, 5 images are generated representing: strong figurative, weak figurative, weak literal, strong literal, and a distractor. Annotators wrote visual descriptions for each scenario, which were then used to generate cartoon-style images with Midjourney. A unified style reference was applied to ensure visual consistency.

3. **Three-Frame Narrative Sequence (Subtask B)**: Similar to a three-frame comic strip, annotators described 3 visual scenes along with 2 alternative endings, testing the model's understanding of idiomatic meaning unfolding over time.

4. **Multilingual Support**: The dataset covers English (100 expressions) and Brazilian Portuguese (55 expressions). In Portuguese, hyphens (which act as formal markers to distinguish literal and idiomatic meanings) were deliberately omitted to prevent giving the models an unfair shortcut.

5. **Textual Alternative Modality**: Image descriptions generated by LLaVA were provided to allow text-only models to participate, thereby lowering the barrier to entry.

### Evaluation Metrics

- Subtask A: Top-1 Accuracy + DCG (Discounted Cumulative Gain), with ranking weights of [3,1,0,0,0].
- Subtask B: Image Completion Accuracy + Sentence Type (idiomatic/literal) Identification Accuracy.

## Key Experimental Results

### Subtask A Leaderboard (English, Text+Image)

| Rank | Team | Test Set Top-1 Acc | Test Set DCG | Extended Set Top-1 Acc | Extended Set DCG |
|------|------|-----------------|-----------|-----------------|-----------|
| 1 | PALI-NLP | **0.93** | 3.52 | **0.83** | 3.43 |
| 2 | dutir914 | 0.93 | 3.46 | 0.79 | 3.28 |
| 3 | AlexUNLP-NB | 0.93 | 3.45 | 0.72 | 3.22 |
| 4 | AIMA | 0.87 | 3.44 | 0.48 | 2.90 |
| 5 | daalft | 0.87 | 3.43 | 0.81 | 3.35 |

### Human Evaluation vs. System Performance (Extended Evaluation Set)

| Evaluator | Top-1 Acc | DCG |
|--------|-----------|------|
| Annotator Average | 0.71 | 3.22 |
| Best Individual | 0.86 | 3.41 |
| Expert Pool (Rank Aggregation) | 0.83 | 3.39 |
| PALI-NLP (Best System) | **0.83** | **3.43** |

### Key Findings

1. **Best Systems Achieve Human-Level Performance**: PALI-NLP's Top-1 accuracy (0.83) on the extended evaluation set matched the expert pool method (0.83), and its DCG (3.43) even surpassed the expert pool (3.39).

2. **Mixture-of-Experts as a Crucial Strategy**: Four teams adopted Mixture-of-Experts approaches, leveraging multiple models/prompt variants to smooth out inconsistencies across models in idiom comprehension—no single model was capable of mastering the idiomatic phenomena entirely.

3. **LLM Bias Toward Idiomatic Meaning**: Three teams utilizing generative LLMs observed that models tended to classify almost all expressions as idiomatic. PALI-NLP mitigated this by adopting a strategy of "prompting the LLM to generate literal usage examples first," which improved classification accuracy from 91.4% to 98.6%.

4. **AlexUNLP's Synonym Replacement Strategy**: When a compound was detected as an idiom, it was replaced with a compositional synonym (e.g., "dirty money" -> "illegal money") to bypass the VLM's inherent bias toward literal interpretations.

5. **Extended Evaluation Set is More Robust**: Models that performed well on the test set often saw a significant drop on the extended set, suggesting a risk of overfitting.

6. **Unexpectedly Strong Portuguese Performance**: The performance gap between English and Portuguese tasks was narrower than expected, indicating that multilingual LLMs are improving in their comprehension of figurative language in non-English tongues.

## Highlights & Insights

- **Ingenious Task Design**: The granularity of the five-tiered image selection (ranging from strong figurative to distractors) allows evaluation to go beyond binary classification, assessing fine-grained comprehension of semantic distance.
- **Unique Value of Visual Modality**: Images require models to understand the actual semantics of the idiom rather than relying on surface-level textual pattern matching, exposing comprehension flaws more effectively than text-only benchmarks.
- **Cost-Sensitive Design**: Providing textual descriptions as alternatives to images allows resource-constrained teams to participate.
- **Well-Defined Human Baseline**: The task is challenging even for humans (average annotator Top-1 accuracy was only 71%), demonstrating that this is indeed a highly formidable research problem.

## Limitations & Future Work

- The dataset size for Subtask B is relatively small (only 30 expressions in English), which consequently attracted fewer participating teams.
- Generating images using Midjourney might introduce biases inherent to the generative model itself.
- The "expected order" in the five-tier ranking (such as rankings under idiomatic conditions) possesses some degree of subjectivity—specifically regarding the definite relative order between literal images and distractors.
- The quality of Portuguese translations generated by Google Translate might affect the fairness of the multilingual evaluation.
- The image descriptions were automatically generated by LLaVA, which could potentially miss critical visual cues present in the images.

## Related Work & Insights

- Multimodal idiom processing is an important yet under-explored research direction in natural language understanding (NLU).
- The debiasing strategy of "generating literal usages before classification" provides valuable insights for other generative tasks plagued by bias.
- The incorporation of image sequences (temporal modality) adds a new dimension to idiom comprehension, which can be further extended to the video modality.

## Rating

| Metric | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4.5 |
| Value | 4 |

As a shared task overview paper, it is exceptionally detailed in terms of dataset design, evaluation metric formulation, and analysis of participating systems. The multimodal modeling of idiomatic phenomena holds significant research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs](../../NeurIPS2025/multimodal_vlm/t-rex_task-adaptive_spatial_representation_extraction_for_robotic_manipulation_w.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](../../CVPR2025/multimodal_vlm/task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[ACL 2025\] ChartCoder: Advancing Multimodal Large Language Model for Chart-to-Code Generation](chartcoder_chart_to_code.md)
- [\[ACL 2025\] COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models](coling-unia_at_scivqa_2025_few-shot_example_retrieval_and_confidence-informed_en.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)

</div>

<!-- RELATED:END -->
