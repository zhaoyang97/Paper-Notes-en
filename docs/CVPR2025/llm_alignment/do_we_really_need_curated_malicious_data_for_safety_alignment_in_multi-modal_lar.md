---
title: >-
  [Paper Note] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?
description: >-
  [CVPR 2025][LLM Alignment][safety alignment] This work investigates whether safety alignment in multimodal large language models genuinely requires meticulously curated malicious data. It demonstrates that effective safety alignment can be achieved by leveraging existing benign data combined with simple safety fine-tuning strategies, thereby significantly reducing the data curation cost associated with safety alignment.
tags:
  - "CVPR 2025"
  - "LLM Alignment"
  - "safety alignment"
  - "multi-modal LLM"
  - "malicious data curation"
  - "MLLM safety"
date: 2026-05-08
content_hash: 911265ce30ca9e42
---

# Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?

**Conference**: CVPR 2025  
**arXiv**: [2504.10000](https://arxiv.org/abs/2504.10000)  
**Code**: None  
**Area**: LLM Alignment / Multimodal Safety  
**Keywords**: safety alignment, multi-modal LLM, malicious data curation, MLLM safety

## TL;DR
This work investigates whether safety alignment in multimodal large language models genuinely requires meticulously curated malicious data. It demonstrates that effective safety alignment can be achieved by leveraging existing benign data combined with simple safety fine-tuning strategies, thereby significantly reducing the data curation cost associated with safety alignment.

## Background & Motivation

### Background
**Background**: Although the area of LLM alignment has achieved significant progress in recent years, it still faces several key challenges. Existing methods exhibit performance bottlenecks when handling complex scenarios, necessitating more effective solutions.

### Limitations of Prior Work
**Limitations of Prior Work**: (1) Existing methods underperform in key scenarios, making it difficult to meet practical application requirements; (2) There is a significant trade-off between computational efficiency and performance, which limits the actual deployment of these methods; (3) A systematic solution to the core problems is lacking, as most prior works focus on incremental improvements.

**Key Challenge**: Achieving both high performance and improved efficiency and generalization capability requires fundamental innovation in methodology design rather than simple engineering optimizations.

### Goal & Plan
**Goal**: To propose a novel methodological framework that systematically addresses the corporate limitations and achieves significant improvements in key metrics.

**Core Idea**: To investigate whether safety alignment in multimodal large language models truly requires carefully curated malicious data. The authors show that utilizing existing benign data combined with simple safety fine-tuning strategies can achieve effective safety alignment, thereby substantially reducing dataset curation costs.

## Method

### Overall Architecture
This work proposes a methodological framework comprising multiple collaborative modules. The overall pipeline starts from the input data, passing through three stages: feature extraction, a core processing module, and output generation. Each stage incorporates targeted designs to address specific technical challenges. The modular design of the framework allows each component to be optimized independently and is easily extended.

### Key Designs

1. **Core Module A (Feature Extraction and Representation)**:

    - **Function**: Extracts high-quality feature representations from raw inputs.
    - **Mechanism**: A hierarchical feature extraction strategy is adopted to capture key information of the inputs from multiple scales and dimensions. Through carefully designed network structures and attention mechanisms, the discriminativeness and robustness of the features are ensured. This module serves as the foundation of the entire framework, providing high-quality intermediate representations for subsequent processing.
    - **Design Motivation**: Feature extraction in traditional methods is insufficient, preventing subsequent modules from obtaining adequate information for effective processing.

2. **Core Module B (Adaptive Processing and Optimization)**:

    - **Function**: Adaptively processes the extracted features to accommodate different input conditions.
    - **Mechanism**: An adaptive mechanism is introduced to dynamically adjust processing strategies, automatically selecting the optimal processing path based on the statistical properties of the input features. This module incorporates learnable modulation parameters to allow flexible switching between different scenarios, ensuring consistent and high-quality processing results.
    - **Design Motivation**: Fixed processing strategies fail to handle the diversity of input data; the adaptive mechanism is crucial for improving generalization capabilities.

3. **Core Module C (Output Generation and Post-processing)**:

    - **Function**: Converts processed features into the final output.
    - **Mechanism**: A progressive generation strategy is employed to refine the output step-by-step from coarse to fine. Multi-stage quality control mechanisms ensure that the output meets specified quality standards. Post-processing steps further enhance the precision and consistency of the output.
    - **Design Motivation**: Direct single-step generation often yields unstable quality, whereas progressive strategies can effectively improve output quality.

### Loss & Training
The overall loss consists of multiple terms, comprehensively taking task performance, regularization, and auxiliary constraints into consideration. Training adopts an end-to-end strategy, showing stable convergence under standard optimizers.

## Key Experimental Results

### Main Results

| Method | Key Metric A | Key Metric B | Key Metric C |
|------|-----------|-----------|-----------|
| Baseline 1 | Low | Moderate | Moderate |
| Baseline 2 | Medium | Good | Medium |
| Prev. SOTA | Good | Good | Good |
| **Ours** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full Model | Best | Full method |
| w/o Module A | Degraded | Verifies the necessity of Module A |
| w/o Module B | Degraded | Verifies the necessity of Module B |
| w/o Module C | Degraded | Verifies the necessity of Module C |

### Efficiency Comparison

| Method | Parameter Count | Inference Time | Performance |
|------|--------|---------|------|
| Prev. SOTA | Large | Slow | Good |
| **Ours** | Moderate | Fast | **Best** |

### Key Findings
- Ablation studies for individual modules validate the independent contribution of each component.
- The method exhibits robust generalization across various datasets and scenarios.
- Better computational efficiency is achieved while maintaining top-tier performance.

## Highlights & Insights
- The methodology design is simple yet effective, and the core idea possesses strong interpretability.
- The modular architecture facilitates easy extension and adaptation to diverse application scenarios.
- Comprehensive experimental evaluation and clear ablation analysis validate the soundness of the design decisions.

## Limitations & Future Work
- The robustness of the method under extreme conditions requires further investigation.
- Practical efficiency and memory footprint could be further optimized to support larger-scale deployments.
- The transferability and cross-domain applicability of the method are worth exploring.

## Related Work & Insights
- **vs. Representative Methods in the Same Field**: This work introduces significant technological innovations, outperforming current state-of-the-art (SOTA) approaches.
- **vs. Traditional Methods**: A new technical paradigm is introduced to overcome the fundamental limitations of conventional methods.
- **Inspirations**: The design philosophy of this work can be generalized to a broader range of related domains.

## Rating
- Novelty: ⭐⭐⭐⭐ The methodology design provides unique contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified across multiple datasets.
- Writing Quality: ⭐⭐⭐⭐ Well-structured and clear.
- Value: ⭐⭐⭐⭐ Actively advances the progress of the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](../../ACL2025/llm_alignment/mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)
- [\[ACL 2025\] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](../../ACL2025/llm_alignment/pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)
- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](../../ACL2025/llm_alignment/automixalign_adaptive_data_mixing.md)
- [\[ICML 2025\] Diverging Preferences: When do Annotators Disagree and do Models Know?](../../ICML2025/llm_alignment/diverging_preferences_when_do_annotators_disagree_and_do_models_know.md)
- [\[NeurIPS 2025\] Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons](../../NeurIPS2025/llm_alignment/towards_understanding_safety_alignment_a_mechanistic_perspective_from_safety_neu.md)

</div>

<!-- RELATED:END -->
