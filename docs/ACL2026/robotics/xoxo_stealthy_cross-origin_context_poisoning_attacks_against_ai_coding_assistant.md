---
title: >-
  [Paper Note] XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants
description: >-
  [ACL 2026][Robotics][Adversarial Attack] This paper reveals a design vulnerability in AI coding assistants' automatic context collection and proposes Cross-Origin Context Poisoning (XOXO) attacks: poisoning shared codeba…
tags:
  - "ACL 2026"
  - "Robotics"
  - "Adversarial Attack"
  - "AI Coding Assistant"
  - "Context Poisoning"
  - "Semantics-Preserving Transform"
  - "Code Security"
content_hash: 8335cd78b16d88ad
---

# XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants

**Conference**: ACL 2026
**arXiv**: [2503.14281](https://arxiv.org/abs/2503.14281)
**Code**: [https://github.com/adamstorek/cross-origin-context-poisoning](https://github.com/adamstorek/cross-origin-context-poisoning)
**Area**: Robotics & Embodied AI
**Keywords**: Adversarial Attack, AI Coding Assistant, Context Poisoning, Semantics-Preserving Transform, Code Security

## TL;DR
This paper reveals a design vulnerability in AI coding assistants' automatic context collection and proposes Cross-Origin Context Poisoning (XOXO) attacks: poisoning shared codebases via semantics-preserving code transformations (e.g., variable renaming) causes assistants like GitHub Copilot to unknowingly generate vulnerable code, achieving 73.20% average ASR across 8 SOTA models.

## Method

### Key Designs

1. **XOXO Attack Model**: Exploits three properties: (a) automatic context collection without source discrimination; (b) greedy decoding or low-temperature sampling for reproducible effects; (c) reverse-engineerable prompt templates and sampling parameters via network traffic analysis.

2. **Confidence Monotonicity and Greedy Cayley Graph Search (GCGS)**: Discovers that combining confidence-decreasing transforms further decreases confidence (verified at $p < 1.7 \times 10^{-10}$). GCGS greedily explores transform combinations along the confidence-decreasing direction.

3. **End-to-End GitHub Copilot Attack Verification**: Renaming `USE_RAW_QUERIES` to `RAW_QUERIES` (semantically identical) causes Copilot to generate SQL injection vulnerabilities. Attack succeeds consistently across multiple sessions.

## Key Experimental Results

| Model | HumanEval+ ASR | CWEval Vulnerability Rate |
|-------|---------------|--------------------------|
| Claude 3.5 Sonnet v2 | 92.00% | 40.00% |
| GPT 4.1 | 81.82% | 50.00% |
| Llama 3.1 8B | 97.11% | 54.00% |

- Triggers 17 different vulnerability types (CWEs)
- All 7 surveyed mainstream coding assistants share the same architectural vulnerability

## Highlights & Insights
- Attack stealth is extremely high — semantics-preserving variable renaming is virtually undetectable in code review, far more dangerous than traditional prompt injection
- Confidence monotonicity reveals LLMs' over-reliance on code surface form rather than semantics — a fundamental architectural flaw
- Directly points to a design improvement: coding assistants should discriminate context source trustworthiness

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences](can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[CVPR 2026\] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](../../CVPR2026/robotics/force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](../../NeurIPS2025/robotics/understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[NeurIPS 2025\] MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](../../NeurIPS2025/robotics/mip_against_agent_malicious_image_patches_hijacking_multimod.md)

</div>

<!-- RELATED:END -->
