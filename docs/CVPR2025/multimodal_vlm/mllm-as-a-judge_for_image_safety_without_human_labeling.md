---
title: >-
  [Paper Note] MLLM-as-a-Judge for Image Safety without Human Labeling
description: >-
  [CVPR 2025][Multimodal VLM][image safety] Proposes the CLUE framework, which achieves zero-shot image safety judgment without human labeling through rule objectification, CLIP relevance scanning, precondition chain decomposition, and debiased token probability analysis, significantly outperforming baselines across multiple MLLMs.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "image safety"
  - "MLLM"
  - "zero-shot judgment"
  - "safety constitution"
  - "debiased token probability"
  - "content moderation"
date: 2026-05-08
content_hash: c15cb6534fb541db
---

# MLLM-as-a-Judge for Image Safety without Human Labeling

**Conference**: CVPR 2025  
**arXiv**: [2501.00192](https://arxiv.org/abs/2501.00192)  
**Code**: Not open-sourced  
**Area**: Multimodal VLM  
**Keywords**: image safety, MLLM, zero-shot judgment, safety constitution, debiased token probability, content moderation

## TL;DR

Proposes the CLUE framework, which achieves zero-shot image safety judgment without human labeling through rule objectification, CLIP relevance scanning, precondition chain decomposition, and debiased token probability analysis, significantly outperforming baselines across multiple MLLMs.

## Background & Motivation

**Background**: In the era of online platforms and AIGC, image content moderation is crucial. Existing solutions mainly rely on two types of methods: (1) traditional classifiers (e.g., Q16, NSFW Detector), and (2) fine-tuned MLLMs (e.g., LLaVA Guard). Both rely on human-labeled data.

**Key Challenge**: Human labeling is expensive and difficult to scale; safety rules may be updated frequently, while fine-tuning-based methods require re-labeling and re-training each time a rule changes. This leads to the **Core Problem**: **Can pre-trained MLLMs perform image safety judgment under a zero-shot setting solely based on a predefined safety constitution?**

**Three Challenges of Directly Querying MLLMs**:

1. **Rule Subjectivity**: Vague rules like "should not contain pornographic content" make it difficult even for human experts to judge boundary cases.
2. **Difficulty in Long-Rule Reasoning**: Current MLLMs struggle to perform accurate reasoning on complex, lengthy safety rules.
3. **Inherent Model Biases**: Including language prior bias (model's tendency in answering) and image non-central region bias (cropped top $\rightarrow$ model biases towards assuming the lower body is also bare).

## Method

### Overall Architecture

CLUE (Constitutional MLLM JUdgE) is a multi-stage reasoning framework:

Input image $\rightarrow$ CLIP relevance scanning $\rightarrow$ Check rule-by-rule $\rightarrow$ Precondition decomposition $\rightarrow$ Debiased token probability judgment $\rightarrow$ (At low confidence) Cascaded CoT reasoning $\rightarrow$ Output safety label + list of violated rules

### Key Designs

#### 1. Rules Objectification

Translates subjective/vague safety rules into objective, actionable rules:

- Uses LLM-as-an-Optimizer to evaluate the objectivity of each rule (score 1-10)
- Rules scoring below 9 are iteratively revised until they meet the threshold
- For example: "should not contain pornographic content" $\rightarrow$ refined into multiple specific rules, such as "any part of the female breast region not fully covered by opaque clothing is not allowed"
- Allows users to adjust key parameters (e.g., angle threshold of 90°)

#### 2. Relevance Scanning

Utilizes CLIP's text-image similarity to quickly filter out rules obviously irrelevant to the current image:

$$\text{relevant if } \cos(\mathbf{I}(x), \mathbf{T}(r)) > t$$

with threshold $t=0.22$. The parameter scale of the CLIP encoder is much smaller than that of MLLMs, significantly improving overall inference efficiency.

#### 3. Precondition Extraction

Automatically decomposes complex rules into simplified precondition chains, judging a violation only when all preconditions are met:

**Example**: Rule "there should be no humans or animals suffering visible bloody injuries leading to imminent death"
$\rightarrow$ Precondition chain: [[human visible] OR [animal visible]] AND [body has visible bloody injury] AND [injury is severe enough to cause imminent death]

This decomposition: (1) reduces the reasoning complexity of a single MLLM query, and (2) allows for early exit (skips subsequent checks if a certain precondition is not met).

### Loss & Training

**Debiased Token Probability Judgment**:

For each precondition query "Yes/No", the precondition score is calculated as (probability of Yes / (probability of Yes + probability of No)).

**Strategy 1 — Language Prior Debiasing**:
Compares the token probability difference with and without the image:
- $\mathcal{M}(x, c) - \mathcal{M}(\text{None}, c) < \alpha_1$ $\rightarrow$ Precondition not satisfied
- $\mathcal{M}(x, c) - \mathcal{M}(\text{None}, c) > \alpha_2$ $\rightarrow$ Precondition satisfied

**Strategy 2 — Image Non-Central Region Debiasing**:
Uses OWLv2 to detect the central object and compares the probability difference between the original image and after removing the central region:
- $\mathcal{M}(x, c) - \mathcal{M}(x \ominus i, c) > \beta$ $\rightarrow$ Precondition satisfied

The two strategies are used in combination. Low-confidence samples proceed to the cascaded CoT reasoning stage.

## Key Experimental Results

### Main Results

| Method | Model | Recall | Accuracy | F-1 |
|------|------|--------|----------|-----|
| Prior Knowledge + Yes/No | InternVL2-76B | 62.6% | 71.8% | 0.691 |
| Entire Constitution + Yes/No | InternVL2-76B | 79.7% | 85.5% | 0.846 |
| Entire Constitution + CoT | InternVL2-76B | 75.3% | 82.2% | 0.809 |
| **CLUE (Ours)** | **InternVL2-76B** | **95.9%** | **94.8%** | **0.949** |
| **CLUE (Ours)** | InternVL2-8B-AWQ | 91.2% | 87.4% | 0.879 |
| **CLUE (Ours)** | Qwen2-VL-7B | 88.9% | 86.3% | 0.866 |

### Comparison with Fine-Tuning Methods

| Method | Type | Generalizability |
|------|------|--------|
| Q16, SD Filter, NSFW Detector, LLaVA Guard| Fine-tuned | Poor (only effective on trained rules) |
| **CLUE** | **Zero-shot** | **Strong (rules can be updated without re-labeling/re-training)** |

CLUE significantly outperforms all fine-tuned baselines under the zero-shot setting, verifying the inherent limitations of fine-tuning methods in rule generalization.

### Key Findings

1. **Rule Objectification is Fundamental**: Elevating raw subjective rules to an objectivity score $\ge 9$ significantly improves the judgment capability of MLLMs.
2. **Debiasing Mechanism is Crucial**: After removing language priors and image non-central region biases, the accuracy of token probability-based judgment is significantly enhanced.
3. **Precondition Decomposition Outperforms Direct Reasoning**: Even GPT-4o struggles to directly reason about complex rules, but can correctly judge decomposed preconditions.
4. **Highly Efficient CLIP Relevance Filtering**: Filters out a large number of irrelevant rules at extremely low computational cost, increasing inference speed several times.
5. **Cross-Model Generalization**: Hyperparameters ($\alpha_1, \alpha_2, \beta$) do not require tuning across different MLLMs.

## Highlights & Insights

1. **Completely Zero-Shot**: Requires no human-labeled data. Rule updates only require text modification, greatly reducing deployment and maintenance costs.
2. **Systematically Addressing MLLM Biases**: Debiases from two dimensions—language priors and visual attention, presenting a novel and generalizable approach.
3. **Multi-Stage Cascaded Design**: A cascaded strategy combining fast token probability judgment with deep CoT reasoning, balancing efficiency and accuracy.
4. **Constructed OS Bench**: The first image safety evaluation benchmark labeled based on objective rules, filling an evaluation gap.

## Limitations & Future Work

1. **Safety Constitution Requires Human Definition**: Although human effort for labeled images is eliminated, experts are still needed to write detailed safety rules.
2. **Reliance on CLIP's Perceptual Capabilities**: Relevance scanning is limited by CLIP's understanding of safety-related concepts.
3. **Still High Inference Cost**: MLLM queries are required for each precondition of every relevant rule, resulting in multiple forward passes.
4. **OS Bench Uses AI-Generated Images**: The test set is generated by text-to-image models, which may deviate from the distribution of real user-uploaded content.
5. **Hyperparameter Thresholds**: Although claimed to be robust across models, there are still multiple thresholds that need to be manually set.

## Related Work & Insights

- **LLaVA Guard**: Fine-tuning-based MLLM safety judgment, which this paper proves can be outperformed in a zero-shot manner.
- **Constitutional AI (Bai et al.)**: The source of the safety constitution concept, which this paper extends to the visual domain.
- **VCD (Visual Contrastive Decoding)**: The inspiration source for the debiasing approach.
- **Insight**: The idea of decomposing complex judgment tasks into simple precondition chains combined with debiased token probability is highly generalizable and can be extended to other rule-based judgment scenarios (e.g., content compliance, copyright detection).

## Rating ⭐

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Engineering Practicality | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Calibrating MLLM-as-a-Judge via Multimodal Bayesian Prompt Ensembles](../../ICCV2025/multimodal_vlm/calibrating_mllm-as-a-judge_via_multimodal_bayesian_prompt_ensembles.md)
- [\[CVPR 2025\] HEIE: MLLM-Based Hierarchical Explainable AIGC Image Implausibility Evaluator](heie_mllm-based_hierarchical_explainable_aigc_image_implausibility_evaluator.md)
- [\[ICLR 2026\] Human Uncertainty-Aware Data Selection and Automatic Labeling in Visual Question Answering](../../ICLR2026/multimodal_vlm/human_uncertainty-aware_data_selection_and_automatic_labeling_in_visual_question.md)
- [\[CVPR 2025\] It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](its_a_blind_match_towards_vision-language_correspondence_without_parallel_data.md)
- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](efficient_motion-aware_video_mllm.md)

</div>

<!-- RELATED:END -->
