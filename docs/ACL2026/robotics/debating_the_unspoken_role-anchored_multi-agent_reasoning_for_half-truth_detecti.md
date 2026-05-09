---
title: >-
  [Paper Note] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection
description: >-
  [ACL 2026][Robotics][Half-Truth Detection] RADAR uses role-anchored (politician vs scientist) multi-agent debate to detect half-truths — statements that are factually correct but misleading due to omitted context — with dual-threshold adaptive early stopping, consistently outperforming single-agent and traditional multi-agent baselines under noisy retrieval conditions.
tags:
  - ACL 2026
  - Robotics
  - Half-Truth Detection
  - Multi-Agent Debate
  - Omission Reasoning
  - Role Anchoring
  - Adaptive Termination
content_hash: f91cbebebef5caff
---

# Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection

**Conference**: ACL 2026
**arXiv**: [2604.19005](https://arxiv.org/abs/2604.19005)
**Code**: [https://github.com/tangyixuan/RADAR](https://github.com/tangyixuan/RADAR)
**Area**: Fact Verification / Misinformation Detection
**Keywords**: Half-Truth Detection, Multi-Agent Debate, Omission Reasoning, Role Anchoring, Adaptive Termination

## TL;DR
RADAR uses role-anchored (politician vs scientist) multi-agent debate to detect half-truths — statements that are factually correct but misleading due to omitted context — with dual-threshold adaptive early stopping, consistently outperforming single-agent and traditional multi-agent baselines under noisy retrieval conditions.

## Method

### Key Designs

1. **Role-Anchored Debate Protocol**: Politician agent constructs the most persuasive supporting narrative (confirmatory reasoning); scientist agent probes for missing, weak, or selectively presented information (analytical reasoning). The contrast naturally models half-truth creation and detection mechanisms.

2. **Dual-Threshold Adaptive Early Stopping**: Terminates only when stop margin $s \geq \tau_s$ AND maximum label confidence $c \geq \tau_v$ are both met, preventing premature stopping on uncertain cases.

3. **Retrieval-Anchored Evidence Sharing**: All agents share the same evidence pool, grounding arguments in retrieved evidence rather than parametric knowledge.

## Key Experimental Results

| Method | Accuracy | F1_macro | F1_HalfTrue |
|--------|----------|---------|-------------|
| D2D (MAD) | 63.0 | 50.9 | 39.7 |
| **RADAR_multi** | **77.7** | **63.3** | **56.5** |

## Highlights & Insights
- The "politician-scientist" role metaphor is ingenious — half-truths are common in political discourse, and using agents that model this discourse strategy to detect them creates a "fighting fire with fire" design philosophy
- Paradigm shift from "finding contradictions" to "discovering omissions" opens new directions for fact verification

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection](../../CVPR2026/robotics/probabilistic_concept_graph_reasoning_for_multimodal_misinformation_detection.md)
- [\[AAAI 2026\] Adaptive Theory of Mind for LLM-based Multi-Agent Coordination](../../AAAI2026/robotics/adaptive_theory_of_mind_for_llm-based_multi-agent_coordination.md)
- [\[AAAI 2026\] EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](../../AAAI2026/robotics/evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)
- [\[AAAI 2026\] A Computable Game-Theoretic Framework for Multi-Agent Theory of Mind](../../AAAI2026/robotics/a_computable_game-theoretic_framework_for_multi-agent_theory_of_mind.md)
- [\[AAAI 2026\] Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](../../AAAI2026/robotics/shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)

</div>

<!-- RELATED:END -->
