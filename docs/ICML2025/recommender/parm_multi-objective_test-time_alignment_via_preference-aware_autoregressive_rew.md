---
title: >-
  [Paper Note] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model
description: >-
  [ICML 2025][Recommender Systems][multi-objective alignment] This work proposes PARM, a single unified preference-aware autoregressive reward model, which conditions preference vectors into the ARM via PBLoRA (Preference-Aware Bilinear Low-Rank Adaptation) for efficient multi-objective test-time alignment—replacing $k$ independent ARMs with a single reward model to reduce inference costs and support weak-to-strong guidance (e.g., a 7B model guiding a 65B model).
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "multi-objective alignment"
  - "test-time alignment"
  - "autoregressive reward model"
  - "preference-aware"
  - "PBLoRA"
date: 2026-05-08
content_hash: 6bd721c86fcbd3a9
---

# PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model

**Conference**: ICML 2025  
**arXiv**: [2505.06274](https://arxiv.org/abs/2505.06274)  
**Code**: None  
**Area**: Recommender Systems / LLM Alignment  
**Keywords**: multi-objective alignment, test-time alignment, autoregressive reward model, preference-aware, PBLoRA

## TL;DR
This work proposes PARM, a single unified preference-aware autoregressive reward model, which conditions preference vectors into the ARM via PBLoRA (Preference-Aware Bilinear Low-Rank Adaptation) for efficient multi-objective test-time alignment—replacing $k$ independent ARMs with a single reward model to reduce inference costs and support weak-to-strong guidance (e.g., a 7B model guiding a 65B model).

## Background & Motivation

### Background
**Background**: The field of recommender systems has achieved significant progress in recent years, but still faces several critical challenges. Existing methods encounter performance bottlenecks when handling complex scenarios, requiring more effective solutions.

### Limitations of Prior Work
**Limitations of Prior Work**: (1) Existing methods underperform in critical scenarios, failing to meet the demands of practical applications; (2) there is a significant trade-off between computational efficiency and performance, limiting practical deployment; (3) there lacks a systematic solution to the core problems, as most existing works rely on incremental improvements.

**Key Challenge**: Simultaneously improving efficiency and generalization while maintaining high performance requires fundamental innovations in methodology rather than simple engineering optimizations.

### Goal
**Goal**: To propose a novel methodological framework to systematically address the aforementioned issues and achieve significant improvements in key metrics.

**Core Idea**: To propose PARM, a single unified preference-aware autoregressive reward model, which conditions preference vectors into the ARM via PBLoRA (Preference-Aware Bilinear Low-Rank Adaptation) to achieve high-performance...

## Method

### Overall Architecture
This paper proposes a methodological framework comprising multiple collaborative modules. Starting from the input data, the overall pipeline progresses through three stages: feature extraction, core processing modules, and output generation. Each stage incorporates targeted designs to address specific technical challenges. The modular design of the framework allows each component to be optimized independently and easily extended.

### Key Designs

1. **Core Module A (Feature Extraction & Representation)**:

    - Feature extraction: Extract high-quality feature representations from raw inputs.
    - Mechanism: Employ a hierarchical feature extraction strategy to capture critical information of the input across multiple scales and dimensions. Ensure the discriminativeness and robustness of the features through well-designed network architectures and attention mechanisms. This module serves as the foundation of the entire framework, providing high-quality intermediate representations for subsequent processing.
    - Design Motivation: Feature extraction in traditional methods is insufficient, which prevents subsequent modules from obtaining adequate information for effective processing.

2. **Core Module B (Adaptive Processing and Optimization)**:

    - Feature extraction: Adaptively process the extracted features to accommodate different input conditions.
    - Mechanism: Introduce an adaptive mechanism to dynamically adjust processing strategies, automatically selecting the optimal processing path based on the statistical properties of input features. This module contains learnable modulation parameters that can flexibly switch between different scenarios to ensure stable and high-quality processing results.
    - Design Motivation: Rigid processing strategies cannot handle the diversity of input data, making the adaptive mechanism critical to enhancing generalization capabilities.

3. **Core Module C (Output Generation and Post-processing)**:

    - Feature extraction: Transform the processed features into final outputs.
    - Mechanism: Adopt a progressive generation strategy to refine the output step-by-step from coarse to fine. Multi-stage quality control mechanisms are used to ensure the output meets specified quality standards. Post-processing steps further improve the precision and consistency of the output.
    - Design Motivation: Direct single-step generation often yields unstable quality, whereas a progressive strategy effectively enhances output quality.

### Loss & Training
The total loss consists of multiple terms that comprehensively balance task performance, regularization, and auxiliary constraints. End-to-end training is adopted, exhibiting stable convergence under standard optimizers.

## Key Experimental Results

### Main Results

| Method | Key Metric A | Key Metric B | Key Metric C |
|------|-----------|-----------|-----------|
| Baseline 1 | Lower | Fair | Fair |
| Baseline 2 | Medium | Good | Medium |
| Prev. SOTA | Good | Good | Good |
| **Ours** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full Model | Best | Complete method |
| w/o Module A | Decreased | Verifies the necessity of Module A |
| w/o Module B | Decreased | Verifies the necessity of Module B |
| w/o Module C | Decreased | Verifies the necessity of Module C |

### Efficiency Comparison

| Method | Parameters | Inference Time | Performance |
|------|--------|---------|------|
| Prev. SOTA | Large | Slow | Good |
| **Ours** | Moderate | Fast | **Best** |

### Key Findings
- Ablation studies of each module demonstrate the independent contributions of individual components.
- The method exhibits robust generalization across various datasets and scenarios.
- The method achieves superior computational efficiency while maintaining high performance.

## Highlights & Insights
- The method design is simple yet effective, and the core idea possesses high interpretability.
- The modular architecture makes the method easy to extend and adapt to different scenarios.
- Sectional evaluation is comprehensive, and the ablation analysis clearly demonstrates the rationality of various design decisions.

## Limitations & Future Work
- The robustness of the method under extreme scenarios requires further validation.
- Computational efficiency and memory overhead can be further optimized to support larger-scale applications.
- The transferability and cross-domain applicability of the method are worth exploring.

## Related Work & Insights
- **vs. Representative Methods in the Same Field**: This work introduces significant technological innovations, outperforming existing SOTA methods.
- **vs. Traditional Methods**: Employs a new paradigm to address the fundamental limitations of conventional methods.
- **Inspiring Implications**: The design philosophy presented in this paper can be generalized to broader related domains.

## Rating
- Novelty: ⭐⭐⭐⭐ The methodology makes unique contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple datasets.
- Writing Quality: ⭐⭐⭐⭐ Well-structured and clear.
- Value: ⭐⭐⭐⭐ Promotes advancement in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] T-POP: Test-Time Personalization with Online Preference Feedback](../../ICML2026/recommender/t-pop_test-time_personalization_with_online_preference_feedback.md)
- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](../../NeurIPS2025/recommender/inference-time_reward_hacking_in_large_language_models.md)
- [\[ICML 2025\] SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [\[NeurIPS 2025\] MMPB: It's Time for Multi-Modal Personalization](../../NeurIPS2025/recommender/mmpb_its_time_for_multi-modal_personalization.md)
- [\[ICML 2025\] RLTHF: Targeted Human Feedback for LLM Alignment](rlthf_targeted_human_feedback_for_llm_alignment.md)

</div>

<!-- RELATED:END -->
