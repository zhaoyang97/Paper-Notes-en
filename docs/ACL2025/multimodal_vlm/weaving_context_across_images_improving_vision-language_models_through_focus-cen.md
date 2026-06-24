---
title: >-
  [Paper Note] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains
description: >-
  [ACL 2025][Multimodal VLM][Multi-image Reasoning] This paper proposes the Focus-Centric Visual Chain multi-image reasoning paradigm, achieving cross-image reasoning through question decomposition and stepwise focusing on key visual information. It constructs the VISC-150K dataset, leading to consistent performance improvements of 2-3% across seven multi-image benchmarks.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multi-image Reasoning"
  - "Vision-Language Models"
  - "Data Synthesis"
  - "Chain-of-Thought"
  - "Multimodal Reasoning"
date: 2026-05-08
content_hash: 694bd8846ec36cee
---

# Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains

**Conference**: ACL 2025  
**arXiv**: [2504.20199](https://arxiv.org/abs/2504.20199)  
**Code**: [GitHub - VISC](https://github.com/zhangjuntian/VISC)  
**Area**: Multimodal VLMs  
**Keywords**: Multi-image Reasoning, Vision-Language Models, Data Synthesis, Chain-of-Thought, Multimodal Reasoning

## TL;DR

This paper proposes the Focus-Centric Visual Chain multi-image reasoning paradigm, achieving cross-image reasoning through question decomposition and stepwise focusing on key visual information. It constructs the VISC-150K dataset, leading to consistent performance improvements of 2-3% across seven multi-image benchmarks.

## Background & Motivation

Vision-Language Models (VLMs) have reached human-level performance on single-image tasks but suffer from significant performance degradation in multi-image scenarios. Two main challenges of multi-image tasks are:

**Cross-image correlations**: Diverse relationships (temporal, spatial, semantic) exist across images, requiring a holistic understanding of their contextual connections.

**Visual discontinuity**: Information is distributed fragmentedly across images, making it difficult to accurately capture cross-image relations.

Limitations of existing solutions:
- Generating reasoning chains directly with multimodal models lacks reliability; even GPT-4o exhibits unstable performance on multi-image tasks.
- Knowledge distillation from stronger models is costly and difficult to scale.
- Existing multi-image reasoning datasets are extremely scarce.

## Method

### Overall Architecture

The proposed method consists of two parts: (1) The Focus-Centric Visual Chain reasoning paradigm, which performs multi-step reasoning through question decomposition and stepwise focusing; and (2) The Focus-Centric Data Synthesis (FCDS) framework, which synthesizes high-quality multi-image reasoning data in a bottom-up manner.

### Key Designs

1. **Focus-Centric Visual Chain Reasoning Paradigm**: Given an image collection $\mathcal{G} = \{I_k\}$ and a question $Q$, the model $\mathcal{M}$ incrementally constructs a reasoning chain $\mathcal{R}$. At step $i$, the model generates a sub-question $q_i$ and identifies the corresponding focus-centric visual subset $G_i$ (a minimized subset of visual information), obtaining an intermediate answer $a_i$ by jointly analyzing $q_i$ and $G_i$. The model dynamically decides whether to terminate the reasoning (via a stop signal $z_i$) and finally synthesizes the ultimate answer from all QA pairs. The core idea is to decompose complex multi-image tasks into a sequence of sub-tasks focused on localized visual inputs.

2. **Feature Extraction Module**: Constructs a detailed textual profile for each image, comprising four elements: global view, background description, object attributes, and object interactions. LLaVA-OneVision-7B serves as the base model for the Extractor, generating profiles through three components: the visual encoder $f_e$, the vision-language connector $f_c$, and the LLM $f_\phi$.

3. **Pair Connection Module**: Determines associations between image nodes based on two criteria: (1) object-oriented context (images sharing the same objects) and (2) event-oriented context (images describing related events). Qwen2.5-7B-Instruct is utilized as the Connector to identify valid pairwise connections based on the profile collection.

4. **Relevance Annotation Module**: Classifies the relationships between image pairs into three categories:

    - **Temporal**: Images depicting chronological sequences.
    - **Spatial**: Visual elements presenting geometric and positional correlations.
    - **Semantic**: Abstract connections involving themes, logic, and causal associations.
   LLaVA-OneVision-7B acts as the Annotator to annotate the relationship for each pair of connected images.

5. **Question Generation Module**: Samples continuous node chains along the reasoning path, and generates sub-questions for each pair of connected images based on their relevance annotations and profiles, eventually synthesizing a comprehensive question. Qwen2.5-7B-Instruct is employed as the Questioner. This bottom-up design ensures data quality while maintaining computational efficiency, relying solely on open-source models throughout the pipeline.

### Loss & Training

- LoRA fine-tuning is conducted based on LLaVA-OneVision-7B and Qwen2-VL-7B-Instruct.
- Trained for 1 epoch on VISC-150K, with a batch size of 8 and a learning rate of 1e-5.
- Warmup ratio of 0.05 with a cosine scheduler.
- Maximum context length of 32,768.

## Key Experimental Results

### Main Results

| Model | MMIU | MuirBench | MIRB | BLINK | NLVR2 | Mantis-Eval | MVBench |
|------|------|-----------|------|-------|-------|-------------|---------|
| LLaVA-OneVision-7B | 40.32 | 41.77 | 51.18 | 48.20 | 89.40 | 64.20 | 56.70 |
| +VISC-150K | **46.52**(↑6.20) | **49.62**(↑7.85) | **53.02**(↑1.84) | **50.24**(↑2.04) | **89.88**(↑0.48) | **66.36**(↑2.16) | **58.23**(↑1.53) |
| Qwen2-VL-7B | 50.00 | 39.12 | 58.67 | 53.20 | 86.42 | 69.60 | 67.00 |
| +VISC-150K | **52.76**(↑2.76) | **44.50**(↑5.38) | **60.16**(↑1.49) | **55.34**(↑2.14) | **89.82**(↑3.40) | 69.12(↓0.48) | **68.01**(↑1.01) |

### Ablation Study

| Experimental Question | Key Results | Explanation |
|---------|---------|------|
| Impact of Data Scale (RQ1) | Rapid improvement from 0 to 25K, gradually converging from 125K to 150K | 25K data is sufficient to activate multi-image reasoning capabilities |
| Sub-task Performance (RQ2) | Significant improvement in 8 out of 12 MuirBench sub-tasks | Similarity analysis and comparative reasoning show the largest improvement |
| Number of Input Images (RQ3) | Most significant improvement with 3-8 images; slight degradation with 15+ images | Medium-scale image sets benefit the most |
| Impact on General Capabilities (RQ4) | Performance is maintained or slightly improved on 4 single-image benchmarks | Does not sacrifice general vision-language capabilities |
| Data Quality (RQ5) | 97.5% overall accuracy (evaluated by 3 human reviewers) | Fleiss' $\kappa = 0.637$, indicating high reliability |

### Key Findings

- LLaVA-OneVision improves by 6.20% on MMIU and 7.85% on MuirBench.
- Establishes new SOTA results on 4 out of 7 benchmarks (MMIU, MIRB, BLINK, NLVR2).
- Even the already powerful Qwen2-VL achieves an average improvement of 2.24%.
- The method also yields improvements on the video benchmark MVBench, demonstrating the domain-agnostic nature of the paradigm.
- The entire data synthesis process employs open-source models, with a 97.5% accuracy rate validating the reliability of the method.

## Highlights & Insights

- **Complete Closed Loop from Reasoning Paradigm to Data Synthesis**: The reasoning paradigm (top-down decomposition) and data synthesis (bottom-up construction) form a dual design, which is logically elegant.
- **Purely Open-Source Solution**: Data synthesis entirely relies on 7B-level open-source models, keeping costs controllable and reproducible.
- **Cross-Architecture Consistency**: Improvements are observed across two distinct architectures, LLaVA-OneVision and Qwen2-VL, demonstrating the general value of the data.
- **No Compromise on General Capabilities**: Performance is maintained or even slightly improved on single-image benchmarks such as HallusionBench and MMStar.
- **25K Data Activation Effect**: A small amount of data can unlock the model's multi-image reasoning potential, suggesting that this is a process of "capability activation" rather than learning brand-new capabilities.

## Limitations & Future Work

- The quadratic complexity of pairwise image relevance annotation limits the scalability of the image set size.
- The dataset mainly covers real photos and comics, leaving its effectiveness on structured visual content, such as charts and code screenshots, unverified.
- The number of reasoning steps is constrained by the intrinsic capabilities of the backbone language models.
- Complex spatial dynamic understanding and domain-specific visual tasks remain weak points.
- Performance slightly degrades when input images exceed 15, indicating room for improvement in long-sequence image processing.

## Related Work & Insights

- Similar to the success of Chain-of-Thought in textual reasoning, the Visual Chain introduces step-by-step reasoning to multi-image visual understanding.
- The bottom-up design of the data synthesis framework avoids the high costs and unreliability associated with relying on closed-source models.
- Insight: The core of multi-image reasoning lies in dynamic selective attention, and the Focus-Centric mechanism essentially implements this capability.
- Future Direction: Extending this paradigm to video understanding (which has been initially validated) and longer sequence image understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ The Focus-Centric reasoning paradigm and bottom-up data synthesis represent a novel combined innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, encompassing 7 benchmarks, 2 architectures, 5 research questions, and human quality evaluation.
- Writing Quality: ⭐⭐⭐⭐ Systematic description of the method and clear formulation of equations, though some notations are somewhat redundant.
- Value: ⭐⭐⭐⭐ The high-quality 150K dataset and effective reasoning paradigm make a significant contribution to multi-image VLM research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Vision-Language Models Struggle to Align Entities across Modalities](vision-language_models_struggle_to_align_entities_across_modalities.md)
- [\[ACL 2025\] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](jailbreak_large_vision-language_models_through_multi-modal_linkage.md)
- [\[ACL 2025\] Symmetrical Visual Contrastive Optimization: Aligning Vision-Language Models with Minimal Contrastive Images](symmetrical_visual_contrastive_optimization_aligning_visionlanguage.md)
- [\[ACL 2025\] Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models](performance_gap_in_entity_knowledge_extraction_across_modalities_in_vision_langu.md)
- [\[ACL 2025\] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback](improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)

</div>

<!-- RELATED:END -->
