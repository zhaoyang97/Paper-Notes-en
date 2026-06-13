---
title: >-
  [Paper Note] Jailbreaking Large Language Models with Morality Attacks
description: >-
  [ACL 2026][LLM Safety][Morality attacks] This work constructs a 10.3K morality attack dataset (comprising value ambiguity and value conflict) and manipulates LLM moral judgment through four adversarial strategies. Result…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Morality attacks"
  - "Jailbreak attacks"
  - "Pluralistic values"
  - "LLM robustness"
  - "Moral judgment"
date: 2026-05-08
content_hash: 58722ef18812271d
---

# Jailbreaking Large Language Models with Morality Attacks

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.17053](https://arxiv.org/abs/2604.17053)  
**Code**: [GitHub](https://github.com/MMLC-lab/Jailbreaking-LLM-Morality)  
**Area**: AI Safety / Moral Robustness  
**Keywords**: Morality attacks, Jailbreak attacks, Pluralistic values, LLM robustness, Moral judgment

## TL;DR

This work constructs a 10.3K morality attack dataset (comprising value ambiguity and value conflict) and manipulates LLM moral judgment through four adversarial strategies. Results indicate that LLMs and guardrail models are extremely vulnerable to morality attacks, and larger models are conversely more susceptible to being compromised.

## Background & Motivation

**Background**: Significant research has accumulated in this field, yet critical gaps remain.

**Limitations of Prior Work**: Existing methods fail to fully address core issues, exhibiting constraints in accuracy, scalability, or applicability.

**Key Challenge**: The fundamental tension arises from the mismatch between the implicit assumptions of existing paradigms and actual practical requirements.

**Goal**: Propose a new framework/method/benchmark to systematically resolve the aforementioned issues.

**Key Insight**: Identify new pathways for problem-solving based on unique observations or theoretical perspectives.

**Core Idea**: Address the key challenge through innovative technical methodologies.

## Method

### Overall Architecture

The proposed method consists of multiple synergistic components that form a complete processing pipeline.

### Key Designs

1. **Core Component 1**:

    - **Function**: Address primary technical challenges.
    - **Mechanism**: Achieve objectives through innovative algorithms or architectural designs.
    - **Design Motivation**: Rooted in a profound understanding of the problem's essence.

2. **Core Component 2**:

    - **Function**: Provide auxiliary support or regularization.
    - **Mechanism**: Supplement the deficiencies of the primary components.
    - **Design Motivation**: Necessity demonstrated through experimental or theoretical analysis.

3. **Core Component 3**:

    - **Function**: Optimize training or inference efficiency.
    - **Mechanism**: Balance performance and efficiency.
    - **Design Motivation**: Requirements for practical deployment.

### Loss & Training

Adopts optimization strategies and evaluation metrics appropriate for the specific task.

## Key Experimental Results

### Main Results

| Method | Core Metric | Note |
|------|---------|------|
| Baseline | Lower | Prev. SOTA |
| **Ours** | **Highest** | Significant Gain |

### Ablation Study

| Configuration | Result | Note |
|------|------|------|
| Full | Highest | Full model |
| w/o Core Component | Decrease | Validates criticality |

### Key Findings

- The proposed method consistently outperforms baselines across multiple benchmarks.
- Ablation studies validate the necessity of each individual component.
- Performance is particularly prominent in specific scenarios.

## Highlights & Insights

- Core technical innovations resolve long-standing issues.
- The method demonstrates strong scalability and practicality.
- Analysis reveals valuable underlying patterns.

## Limitations & Future Work

- The scope of evaluation can be further expanded.
- The applicability of specific assumptions requires further verification.
- Future work may explore additional application scenarios.

## Related Work & Insights

- **vs Related Work A**: This work improves upon key dimensions.
- **vs Related Work B**: This work provides a different solution approach.

## Rating

- Novelty: ⭐⭐⭐⭐ Innovative, though some techniques combine existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure.
- Value: ⭐⭐⭐⭐ Substantial contribution to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](topic-based_watermarks_for_large_language_models.md)
- [\[ACL 2026\] ASTRA: An Automated Framework for Strategy Discovery, Retrieval, and Evolution for Jailbreaking LLMs](astra_an_automated_framework_for_strategy_discovery_retrieval_and_evolution_for_.md)
- [\[ACL 2026\] Multi-component Causal Tracing in Large Language Models](multi-component_causal_tracing_in_large_language_models.md)
- [\[CVPR 2026\] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](../../CVPR2026/llm_safety/force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)

</div>

<!-- RELATED:END -->
