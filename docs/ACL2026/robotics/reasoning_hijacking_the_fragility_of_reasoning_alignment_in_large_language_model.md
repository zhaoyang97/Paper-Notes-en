---
title: >-
  [Paper Note] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models
description: >-
  [ACL 2026][Robotics][Reasoning Hijacking] This paper introduces "Reasoning Hijacking," a new attack paradigm that manipulates LLM reasoning logic by injecting false decision criteria into the data channel rather than changing task goals, achieving high attack success rates while bypassing intent-detection-based defenses.
tags:
  - ACL 2026
  - Robotics
  - Reasoning Hijacking
  - Indirect Prompt Injection
  - Criteria Attack
  - LLM Safety
  - Alignment Fragility
content_hash: 7f362b3e1c109734
---

# Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2601.10294](https://arxiv.org/abs/2601.10294)
**Code**: [GitHub](https://github.com/Yuan-Hou/criteria_attack)
**Area**: Robotics & Embodied AI
**Keywords**: Reasoning Hijacking, Indirect Prompt Injection, Criteria Attack, LLM Safety, Alignment Fragility

## TL;DR
This paper introduces "Reasoning Hijacking," a new attack paradigm that manipulates LLM reasoning logic by injecting false decision criteria into the data channel rather than changing task goals, achieving high attack success rates while bypassing intent-detection-based defenses.

## Method

### Key Designs

1. **Label-Conditioned Criteria Mining**: Extracts label-associated decision criteria from datasets, clusters and deduplicates via text embeddings + k-means.

2. **Refutable Criteria Identification**: Identifies criteria that the target sample does not satisfy — even though the sample clearly belongs to its true class, heuristic criteria are correlative rather than necessary.

3. **Misleading Reasoning Trace Synthesis**: Packages refutable criteria as authoritative decision rules via natural language templates, presenting a structured reasoning process leading to incorrect conclusions.

## Key Experimental Results

| Defense | Criteria Attack ASR (Spam) | Combined Attack ASR (Spam) |
|---------|---------------------------|---------------------------|
| None | 92.7% | 100.0% |
| Instruction | 86.9% | 64.2% |
| Sandwich | 94.2% | 79.0% |

- Highly stable under prompt-level defenses; SecAlign and StruQ also ineffective
- Cross-model generalization: >80% ASR on at least one task for each of 5 victim LLMs
- Fake reasoning traces are the key mechanism: removing them causes the largest ASR drop

## Highlights & Insights
- Reveals a critical blind spot in safety research: all existing defenses assume attacks manifest as goal deviation; reasoning hijacking proves that even with aligned goals, the reasoning process itself can be manipulated
- Exploits LLMs' "reasoning shortcut preference" — models tend to adopt ready-made structured reasoning rather than performing semantic analysis from scratch

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](grasprune_global_gating_for_budgeted_structured_pruning_of_large_language_models.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[ICLR 2026\] SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models](../../ICLR2026/robotics/synthworlds_controlled_parallel_worlds_for_disentangling_reasoning_and_knowledge.md)
- [\[ICLR 2026\] Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](../../ICLR2026/robotics/sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)
- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](../../ICLR2026/robotics/juli_jailbreak_large_language_models_by_self-introspection.md)

</div>

<!-- RELATED:END -->
