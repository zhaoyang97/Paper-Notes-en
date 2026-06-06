---
title: >-
  [Paper Note] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection
description: >-
  [ACL 2026][Social Computing][Half-Truth Detection] RADAR uses role-anchored (politician vs scientist) multi-agent debate to detect half-truths — statements that are factually correct but misleading due to omitted context…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Half-Truth Detection"
  - "Multi-Agent Debate"
  - "Omission Reasoning"
  - "Role Anchoring"
  - "Adaptive Termination"
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

- [\[ACL 2026\] MM-StanceDet: Retrieval-Augmented Multi-modal Multi-agent Stance Detection](mm-stancedet_retrieval-augmented_multi-modal_multi-agent_stance_detection.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/social_computing/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ICML 2026\] MIND: Multi-Rationale Integrated Discriminative Reasoning Framework for Multi-Modal Fake News](../../ICML2026/social_computing/mind_multi-rationale_integrated_discriminative_reasoning_framework_for_multi-mod.md)
- [\[CVPR 2026\] Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection](../../CVPR2026/social_computing/probabilistic_concept_graph_reasoning_for_multimodal_misinformation_detection.md)
- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](../../ICLR2026/social_computing/stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
