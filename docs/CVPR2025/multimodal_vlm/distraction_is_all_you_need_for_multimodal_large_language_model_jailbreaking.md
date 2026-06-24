---
title: >-
  [Paper Note] Distraction is All You Need for Multimodal Large Language Model Jailbreaking
description: >-
  [CVPR 2025][Multimodal VLM][MLLM Safety] Proposed the "distraction hypothesis"—generating out-of-distribution (OOD) effects by constructing high-contrast, multi-subgraph composite inputs to increase visual complexity, which, combined with query decomposition and carefully designed benign instructions, achieves black-box jailbreaking with attack success rates of 42-64% against closed-source MLLMs like GPT-4o.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "MLLM Safety"
  - "Jailbreak Attack"
  - "Distraction"
  - "Contrastive Subgraphs"
  - "Black-box Attack"
date: 2026-05-08
content_hash: 80dc2f3de54654a5
---

# Distraction is All You Need for Multimodal Large Language Model Jailbreaking

**Conference**: CVPR 2025  
**arXiv**: [2502.10794](https://arxiv.org/abs/2502.10794)  
**Code**: [https://github.com/TeamPigeonLab/CS-DJ](https://github.com/TeamPigeonLab/CS-DJ)  
**Area**: Multimodal VLM  
**Keywords**: MLLM Safety, Jailbreak Attack, Distraction, Contrastive Subgraphs, Black-box Attack

## TL;DR
Proposed the "distraction hypothesis"—generating out-of-distribution (OOD) effects by constructing high-contrast, multi-subgraph composite inputs to increase visual complexity, which, combined with query decomposition and carefully designed benign instructions, achieves black-box jailbreaking with attack success rates of 42-64% against closed-source MLLMs like GPT-4o.

## Background & Motivation

**Background**: The safety alignment mechanisms of MLLMs remain vulnerable to meticulously designed jailbreak attacks. Existing attack methods such as FigStep (converting harmful text into images) and Hades (hybridizing harmful text and images) generally yield low attack success rates (<20% ASR) on closed-source models.

**Limitations of Prior Work**: Existing visual jailbreak methods mainly rely on "content-level" adversarial perturbations—using harmful image content to bypass safety detection. However, models like GPT-4o have increasingly stronger detection capabilities for harmful content, leading to a continuous decline in the effectiveness of content-based attacks. More fundamentally, there is a lack of mechanistic understanding regarding why certain visual inputs can bypass safety alignment.

**Key Challenge**: The safety mechanism of MLLMs relies on a holistic understanding of the input to assess harmfulness. However, when the visual complexity of the input exceeds the model's processing capacity, the safety detection mechanism may fail. This hypothesis has not been systematically validated or exploited.

**Goal**: Validate the hypothesis that "visual complexity (rather than content) is the key to jailbreaking," and design an efficient black-box jailbreak framework accordingly.

**Key Insight**: Construct multiple high-contrast (maximizing distance in CLIP space) subgraphs to form a composite input, combined with decomposing the query into multiple sub-query images. This creates multi-level distractions, preventing the model from effectively focusing on the detection of harmful content.

**Core Idea**: Generate visual complexity using contrastive subgraphs that maximize semantic variance in CLIP space, bypassing safety mechanisms by distracting the model's attention rather than hiding harmful content.

## Method

### Overall Architecture
A three-step pipeline: (1) Query Decomposition: decompose the harmful query $Q$ into $m$ sub-queries using a small model and convert them into images; (2) Contrastive Subgraph Retrieval: greedily retrieve $k$ images from an image database that have the lowest CLIP similarity to the query and to each other; (3) Jailbreak Execution: assemble the sub-query images and contrastive subgraphs into a grid-structured composite image, and feed it into the MLLM along with three-part benign instructions.

### Key Designs

1. **Query Decomposition (Structured Distraction)**:

    - **Function**: Fragment harmful intent to reduce the probability of a single sentence triggering safety detection.
    - **Mechanism**: Use Qwen2.5-3B-Instruct to split the harmful query into $m=3$ sub-queries, and then convert each sub-query into an image using typographical transformations (Super Moods font, red, size 50). The sub-query images are embedded into the composite image.
    - **Design Motivation**: Direct text queries are intercepted nearly 100% of the time, but when decomposed, each sub-query appears ambiguous or benign on its own. Three sub-queries represent the optimal sweet spot—while 6 is better, 9 degrades performance (as over-decomposition may expose the original intent).

2. **Contrastive Subgraph Retrieval (Visual-Enhanced Distraction)**:

    - **Function**: Create visual complexity using highly semantically unrelated images.
    - **Mechanism**: Encode queries using CLIP-ViT-L/14 and greedily retrieve images with the lowest cosine similarity from the LLaVA pre-training dataset (10K subset). The selection strategy maximizes the pairwise L2 distance between all nodes (queries and subgraphs): $DL = \sum_{i}\sum_{j \neq i} \|v_i - v_j\|_2$. By default, 9 contrastive subgraphs are retrieved.
    - **Design Motivation**: The contrastive distance is strongly positively correlated with the attack success rate—9 identical similar images yield only 24.66% ASR, 9 most similar images yield 32.53% ASR, while 9 most contrastive images achieve 43.73% ASR. A key finding: random noise (20% ASR) is largely ineffective because MLLMs can easily identify low-information inputs.

3. **Three-Part Benign Instruction Design**:

    - **Function**: Guide the model to process all content in the composite image rather than focusing on safety detection.
    - **Mechanism**: The instructions consist of three parts—role guidance (setting the scene), task guidance (requiring simultaneous multitasking), and visual guidance (implying the subgraphs are relevant to the task). Even using only task guidance (32% ASR) far outperforms Hades (5.51%), with the combination of all three parts reaching 43.73%.
    - **Design Motivation**: The primary role of the instructions is to convince the model that all subgraphs in the composite image are relevant to the task, thereby dispersing its attention across multiple unrelated images and reducing its capability to detect harmful content within the sub-query images.

### Loss & Training
A completely training-free black-box method—requiring neither model gradients nor parameter access, only utilizing CLIP for retrieval and a small model for query decomposition.

## Key Experimental Results

### Main Results

| Model | Hades ASR | CS-DJ ASR | Hades EASR | CS-DJ EASR |
|------|-----------|-----------|------------|------------|
| GPT-4o-mini | 6.08% | **57.80%** | 12.00% | **77.20%** |
| GPT-4o | 5.51% | **42.24%** | 10.13% | **65.86%** |
| GPT-4V | 20.33% | **45.44%** | 45.06% | **71.86%** |
| Gemini-1.5-Flash | 9.20% | **64.11%** | 14.79% | **81.46%** |

### Ablation Study

| Configuration | GPT-4o ASR | Description |
|------|-----------|------|
| Raw Query Image (RQ) | 3.20% | Baseline |
| 3 Sub-queries (3SQ) | 18.80% | Decomposition works +15.6 |
| 3SQ + 9 Identical Similar Images | 24.66% | Low contrast is ineffective |
| 3SQ + 9 Most Similar Images | 32.53% | Diversity helps |
| 3SQ + 9 Random Noise | 20.00% | Noise is largely ineffective |
| 3SQ + 9 Most Contrastive Images | **43.73%** | Contrast is key |

### Key Findings
- **Complexity, Not Content, Drives Jailbreaking**: 9 random noise images barely increase ASR (20% vs 18.8% without subgraphs), whereas 9 semantically rich contrastive images more than double the ASR (43.73%), supporting the distraction hypothesis.
- **Strong Correlation Between Contrastive Distance and ASR**: The most contrastive group with $DL = 1878.83$ corresponds to 43.73% ASR, the most similar group with $DL = 1378.77$ corresponds to 32.53%, and the identical single image with $DL = 399.08$ corresponds to 24.66%.
- **Black-box Efficiency**: Without accessing model parameters, the ASR on GPT-4o increases from Hades' 5.51% to 42.24%, representing a 7.7x gain.
- **Attention Visualization Validation**: Under Hades, model attention is concentrated on the harmful image (enabling detection), whereas under CS-DJ, attention is dispersed across multiple subgraphs.

## Highlights & Insights
- **The "Distraction Hypothesis" provides a mechanistic explanation for jailbreak attacks**: Rather than "hiding harmful content," it "overwhelms the processing capacity of safety mechanisms." This implies that current safety alignment may depend on "inputs being sufficiently simple to be effectively detected."
- **The Contrastive Subgraph Retrieval Strategy is Simple and Effective**: It constructs distracting candidates by maximizing semantic distance in the CLIP space without requiring adversarial training or gradient optimization. It can serve as a standard red-teaming tool for safety assessment.
- **The Role of Instruction Design Cannot Be Overlooked**: Visual distraction alone (without specific instructions) has limited efficacy; the synergy between the three-part instructions and contrastive subgraphs is critical.

## Limitations & Future Work
- The contrastive distance metric ($DL$) is based on the CLIP space, while MLLMs use different vision encoders; thus, CLIP distance may not completely align with the actual distraction effect.
- Defense countermeasures are not discussed; if MLLMs incorporate pre-processing steps for input complexity detection, this method may become ineffective.
- Ethical concerns of the method: Although this is safety research, publishing the complete attack framework risks potential malicious exploitation.
- The performance gain on open-source models is relatively small (LLaVA-OneVision +7.07%), suggesting that the safety alignment of open-source models might inherently be weaker.

## Related Work & Insights
- **vs FigStep / MM-SafetyBench**: These approaches bypass safety filters by rendering harmful text into images, but yield very low ASR (3.73% / 5.86%) as models can still "read" the text in the image. CS-DJ does not rely on content disguise but on complexity-based distraction.
- **vs Hades**: Hades employs hybrid harmful images and text but remains a single-image input, allowing the model to focus its attention on detection. CS-DJ's multi-subgraph strategy fundamentally disperses model attention.
- **Safety Implications**: Current MLLM safety mechanisms may assume that "inputs are relatively simple single-image + text formats"; when facing composite multi-image inputs, their defensive capabilities degrade significantly.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "distraction hypothesis" represents a brand-new perspective, and the contrastive subgraph retrieval strategy is highly ingenious, explaining jailbreaking from a mechanistic level rather than a brute-force approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ The ablation study is comprehensive (number of decompositions, contrastive degrees, instruction components), though it lacks defense evaluation and validation on more models.
- Writing Quality: ⭐⭐⭐⭐ The hypothesis-validation argumentative structure is clear, and the contrastive distance analysis is highly convincing.
- Value: ⭐⭐⭐⭐ Offers significant implications for MLLM safety research, revealing a foundational weakness in safety alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/multimodal_vlm/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)
- [\[CVPR 2026\] Foundation Encoders Are All You Need for Preference-Aware Personalization](../../CVPR2026/multimodal_vlm/foundation_encoders_are_all_you_need_for_preference-aware_personalization.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](../../ICCV2025/multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[CVPR 2025\] Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model](period-llm_extending_the_periodic_capability_of_multimodal_large_language_model.md)
- [\[CVPR 2025\] UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation](upme_an_unsupervised_peer_review_framework_for_multimodal_large_language_model_e.md)

</div>

<!-- RELATED:END -->
