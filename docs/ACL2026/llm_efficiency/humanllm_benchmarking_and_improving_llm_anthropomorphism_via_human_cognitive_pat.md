---
title: >-
  [Paper Note] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns
description: >-
  [ACL 2026][LLM Efficiency][Anthropomorphism] HumanLLM models 244 psychological patterns as interacting causal forces, with dual-layer checklist evaluation achieving r=0.90 human alignment; HumanLLM-8B surpasses Qwen3-32B in multi-pattern dynamics at 4x fewer parameters.
tags:
  - ACL 2026
  - LLM Efficiency
  - Anthropomorphism
  - Cognitive Patterns
  - Multi-Pattern Dynamics
  - Role-Playing Agent
content_hash: 5718c12227744d16
---

# HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns

**Conference**: ACL 2026
**arXiv**: [2601.10198](https://arxiv.org/abs/2601.10198)
**Code**: [GitHub](https://github.com/YJGoodbye2024/HumanLLM)
**Area**: Role-Playing / Personality Simulation
**Keywords**: Anthropomorphism, Cognitive Patterns, Multi-Pattern Dynamics, Role-Playing Agent, Psychological Modeling

## TL;DR
HumanLLM models 244 psychological patterns (100 personality traits + 144 social cognitive patterns) as interacting causal forces rather than isolated labels, constructs 11,359 multi-pattern interaction scenarios, achieves $r=0.90$ human alignment through dual-layer checklist evaluation, and HumanLLM-8B surpasses Qwen3-32B in multi-pattern dynamics at 4x fewer parameters.

## Method

### Key Designs

1. **Literature-Based Psychological Pattern Construction**: Each pattern is backed by ~50 academic papers, structured into definition, core mechanism, and real-world manifestations.

2. **Multi-Pattern Interaction Scenario Generation**: Each scenario contains 2-5 interacting patterns covering enhancement, conflict, and conditional modulation. Dialogues include three-dimensional expression: inner thoughts (brackets), physical behavior (parentheses), and verbal expression.

3. **Dual-Layer Checklist Evaluation**: Pattern-level (12-15 universal behavioral indicators per pattern) + scenario-level (2-6 situation-specific behavioral expectations). Achieves $r=0.90$ vs traditional holistic metrics' $r=0.43$.

## Key Experimental Results

| Model | IPE | MPD |
|-------|-----|-----|
| GPT-5 | 15.5 | 43.4 |
| **HumanLLM-8B** | **25.7** | **70.3** |
| Qwen3-32B | 26.0 | 65.8 |

## Highlights & Insights
- Modeling psychological patterns as "interacting causal forces" rather than "isolated labels" represents a conceptual breakthrough
- Discovery of normative confusion: LLM judges equate social desirability with simulation accuracy; checklist methods effectively decouple the two

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GeoCodeBench: Benchmarking PhD-Level Coding in 3D Geometric Computer Vision](../../CVPR2026/llm_efficiency/benchmarking_phd-level_coding_in_3d_geometric_computer_vision.md)
- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](../../ICLR2026/llm_efficiency/understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [\[AAAI 2026\] InterMoE: Individual-Specific 3D Human Interaction Generation via Dynamic Temporal-Selective MoE](../../AAAI2026/llm_efficiency/intermoe_individual-specific_3d_human_interaction_generation_via_dynamic_tempora.md)
- [\[NeurIPS 2025\] Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads](../../NeurIPS2025/llm_efficiency/let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e.md)
- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)

<!-- RELATED:END -->
