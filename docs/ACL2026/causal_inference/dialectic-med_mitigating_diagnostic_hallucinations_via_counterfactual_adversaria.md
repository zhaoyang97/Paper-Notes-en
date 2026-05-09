---
title: >-
  [Paper Note] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate
description: >-
  [ACL 2026][Causal Inference][Medical Hallucination] Dialectic-Med, inspired by Popperian falsificationism, uses three-agent adversarial dialectical reasoning (proposer for diagnostic hypotheses, opponent with visual falsification module for proactively retrieving contradictory visual evidence, and mediator with weighted consensus graph), achieving SOTA on MIMIC-CXR-VQA, VQA-RAD, and PathVQA with 12.5% explanation faithfulness improvement.
tags:
  - ACL 2026
  - Causal Inference
  - Medical Hallucination
  - Multi-Agent Debate
  - Counterfactual Reasoning
  - Visual Falsification
  - Confirmation Bias
content_hash: 7602d2805b194d0b
---

# Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate

**Conference**: ACL 2026
**arXiv**: [2604.11258](https://arxiv.org/abs/2604.11258)
**Code**: N/A
**Area**: Causal Inference
**Keywords**: Medical Hallucination, Multi-Agent Debate, Counterfactual Reasoning, Visual Falsification, Confirmation Bias

## TL;DR
Dialectic-Med, inspired by Popperian falsificationism, uses three-agent adversarial dialectical reasoning (proposer for diagnostic hypotheses, opponent with visual falsification module for proactively retrieving contradictory visual evidence, and mediator with weighted consensus graph), achieving SOTA on MIMIC-CXR-VQA, VQA-RAD, and PathVQA with 12.5% explanation faithfulness improvement.

## Method

### Key Designs

1. **Visual Falsification Module (VFM)**: Given hypothesis $H_t$, the opponent generates counterfactual probe queries $Q_{cf}$ and uses PubMedCLIP to compute attention maps $M_{cf}$ locating contradictory evidence in images.

2. **Dynamic Consensus Graph**: Nodes represent diagnostic hypotheses or visual evidence; edges encode support/refute logical relations with confidence weights. Includes cycle detection to prevent hypothesis loops.

3. **Attack Strength Threshold Termination**: Debate terminates when $S_{attack} < \theta_{thresh}$, indicating the current hypothesis has withstood falsification attempts.

## Key Experimental Results

- SOTA across all three medical VQA benchmarks
- 12.5% explanation faithfulness improvement
- Visual falsification is the key differentiator vs pure semantic debate

## Highlights & Insights
- Operationalizing Popperian falsificationism as an AI system design principle: actively seeking disconfirming evidence rather than just supporting evidence
- VFM grounds debate in concrete image regions rather than language games
- Direct value for medical AI safety as a safeguard layer before clinical deployment

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MUG: Multi-agent Undercover Gaming — Hallucination Removal via Counterfactual Test for Multimodal Reasoning](../../AAAI2026/causal_inference/multi-agent_undercover_gaming_hallucination_removal_via_coun.md)
- [\[NeurIPS 2025\] A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](../../NeurIPS2025/causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [\[ICLR 2026\] AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems](../../ICLR2026/causal_inference/agenttrace_causal_graph_tracing_for_root_cause_analysis_in_deployed_multi-agent_.md)
- [\[CVPR 2026\] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](../../CVPR2026/causal_inference/fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)
- [\[ICLR 2026\] Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models](../../ICLR2026/causal_inference/flattery_fluff_and_fog_diagnosing_and_mitigating_idiosyncratic_biases_in_prefere.md)

</div>

<!-- RELATED:END -->
