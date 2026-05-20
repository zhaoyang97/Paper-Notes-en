---
title: >-
  [Paper Note] SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution
description: >-
  [NeurIPS 2025][Reinforcement Learning][Software Evolution Data] This work is the first to apply reinforcement learning (RL) to real-world software engineering tasks (GitHub PR/Issue resolution)…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Software Evolution Data"
  - "GRPO"
  - "SWE-bench"
  - "Reasoning Generalization"
  - "Code Editing"
  - "Pull Request"
date: 2026-05-08
content_hash: 8799b01836f36e08
---

# SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution

**Conference**: NeurIPS 2025
**arXiv**: [2502.18449](https://arxiv.org/abs/2502.18449)  
**Code**: [facebookresearch/swe-rl](https://github.com/facebookresearch/swe-rl)  
**Area**: LLM Reinforcement Learning / Software Engineering
**Keywords**: Reinforcement Learning, Software Evolution Data, GRPO, SWE-bench, Reasoning Generalization, Code Editing, Pull Request

## TL;DR

This work is the first to apply reinforcement learning (RL) to real-world software engineering tasks (GitHub PR/Issue resolution), training Llama-3.3-70B exclusively with a rule-based sequence-similarity reward. It achieves a 41.0% resolve rate on SWE-bench Verified (SOTA among medium-scale models). Notably, although RL training is conducted solely on issue-solving data, it elicits emergent generalization in out-of-domain tasks including code reasoning, mathematics, and general language understanding.

## Background & Motivation

**SWE-bench as a core real-world SE challenge**: SWE-bench requires models to resolve real GitHub issues, involving full repository comprehension, bug localization, and patch generation—far exceeding the difficulty of competitive programming. Existing approaches rely heavily on closed-source models such as GPT-4o and Claude-3.5, while open-source models lag significantly behind.

**DeepSeek-R1 demonstrates that RL can enhance reasoning**: DeepSeek-R1 substantially improves LLM reasoning on competitive programming and mathematics via RL with rule-based rewards. However, its 671B parameter scale is difficult to reproduce, and its effectiveness on SE tasks remains limited.

**Limitations of Prior Work**: Existing RL training for reasoning depends on execution feedback from test cases, which is readily available in competitive programming but prohibitively expensive to obtain at scale for real-world SE tasks due to complex dependencies and inconsistent execution environments.

**Prior open-source SE models rely on SFT with closed-source distillation**: Models such as Lingma-SWE-GPT, SWE-Gym, and SWE-Fixer all incorporate distillation outputs from GPT-4o or Claude-3.5-Sonnet in their training data, and are entirely SFT-based without exploring the RL paradigm.

**Key Challenge**: SFT fits the model to a specific task distribution, making it prone to performance degradation on out-of-domain tasks (e.g., mathematics, general understanding), and requires carefully engineered data mixing to maintain generalization.

**Software evolution data as a natural RL training ground**: GitHub hosts a vast collection of high-quality PR data, where each PR contains an issue description, code context, and an oracle patch, enabling natural RL training signals without any execution environment.

## Method

### Overall Architecture

The SWE-RL pipeline proceeds as follows: (1) collect 273K high-quality PRs from GitHub as seed data; (2) each instance comprises an issue description, code context (full file contents), and an oracle patch; (3) the policy LLM generates code edits in search/replace format via chain-of-thought reasoning; (4) a rule-based reward computes a similarity score; (5) the policy is optimized via GRPO.

### Key Designs

- **Lightweight rule-based reward**: Rather than relying on code execution, the method uses Python `difflib.SequenceMatcher` to compute the sequence similarity (continuous value in $[0, 1]$) between the predicted patch and the oracle patch. Malformed outputs receive a penalty of $-1$. This continuous reward captures partial correctness, outperforming sparse discrete exact-match signals.
- **Full-file context**: Input prompts include complete contents of relevant files, implicitly requiring the model to reason about fault localization (where to edit) before generating a fix, thereby training both diagnostic and repair capabilities jointly.
- **PR seed filtering heuristics**: PRs are required to be associated with at least one issue, the issue must describe a bug-fix request, and the code changes must involve programming files, ensuring data quality.
- **Format constraint**: Model outputs must conform to the search/replace code editing format (following Agentless conventions); format violations incur a $-1$ penalty, encouraging the model to quickly internalize formatting norms.

### Loss & Training

- **Base model**: Llama-3.3-70B-Instruct
- **Optimization algorithm**: GRPO (Group Relative Policy Optimization), with 32 problems per batch, 16 rollouts sampled per problem (group size $G=16$), and a global batch size of 512
- **Context window**: 16K tokens
- **Training**: 1,600 steps on 512 H100 GPUs, approximately 32 hours wall-time
- **Inference framework — Agentless Mini**: A simplified variant of Agentless performing only file-level localization, delegating detailed reasoning to the repair step. Supports scaled sampling and reranking using multiple reproduction tests.

## Key Experimental Results

### Main Results: SWE-bench Verified (pass@1)

| Model | Scaffold | Resolve Rate |
|-------|----------|-------------|
| GPT-4o | Agentless | 38.8% |
| o1-preview | Agentless | 41.3% |
| DeepSeek-R1 (671B) | Agentless | 49.2% |
| Claude-3.5-Sonnet | OpenHands | 53.0% |
| SWE-Fixer-72B | SWE-Fixer | 32.8% |
| Llama3-SWE-SFT-70B | Agentless Mini | 36.2% |
| **Llama3-SWE-RL-70B** | **Agentless Mini** | **41.0%** |

→ SOTA among medium-scale ($<$100B) models, on par with GPT-4o (38.8%) and o1-preview (41.3%), and substantially outperforming all open-source models at the same scale.

### Ablation Study: Repair Capability (Oracle File Localization, Greedy Decoding)

| Model | Format Accuracy | Repair Success Rate |
|-------|----------------|-------------------|
| Llama-3.3-70B-Instruct (greedy) | 12.2% | 5.4% |
| Llama-3.3-70B-Instruct (majority vote) | 44.6% | 16.6% |
| Llama3-SWE-SFT-70B | 96.2% | 29.6% |
| **Llama3-SWE-RL-70B** | **95.6%** | **34.8%** |

→ The RL model substantially outperforms SFT in repair performance (+5.2%); the base model achieves only 12.2% format accuracy.

### Out-of-Domain Generalization (Zero-shot Greedy Decoding)

| Task | Llama-3.3-70B | SFT-70B | **RL-70B** |
|------|---------------|---------|------------|
| HumanEval+ | 76.2 | 73.2 | **79.9** |
| CRUXEval-I | 60.5 | 68.4 | **71.6** |
| CRUXEval-O | 61.9 | 75.1 | **75.5** |
| MATH (strict) | 63.2 | 54.0 | **73.7** |
| MMLU | 86.49 | 85.26 | **86.82** |

→ The RL model outperforms both the base model and SFT across all five out-of-domain tasks; SFT degrades performance on tasks such as MATH and HumanEval.

### Key Findings

- **Continuous vs. discrete reward**: The continuous reward (34.8%) substantially outperforms discrete exact-match reward (29.0%), as real-world patches are highly diverse and exact-match signals are prohibitively sparse.
- **Sampling scaling**: Expanding repair samples from 20 to 160 yields significant gains (33.6%→40.0%), with diminishing returns beyond 160.
- **Emergent "aha moment"**: After RL training, the model spontaneously exhibits self-reflection, exploration of alternative solutions, and divide-and-conquer reasoning strategies.

## Highlights & Insights

- 🔑 **First work to apply RL to real-world SE tasks**: Demonstrates that execution environments are not required — patch similarity alone suffices for effective training, substantially lowering the barrier to applying RL in the SE domain.
- 🔑 **Clear evidence that RL > SFT**: Given the same base model and data source, RL not only surpasses SFT on in-domain tasks but comprehensively outperforms it on out-of-domain tasks, whereas SFT leads to average performance degradation.
- 🔑 **Cross-domain emergence of reasoning capabilities**: RL training solely on issue-solving elicits emergent improvements in mathematical reasoning (+10.5), code reasoning (+11.1), and general language understanding, extending the aha moment findings of DeepSeek-R1 to the SE domain.
- 🔑 **No reliance on closed-source model distillation**: Training data is derived entirely from public PRs, with no GPT-4o/Claude distillation, representing the only "autonomous evolution" pathway among models of comparable scale.
- 🔑 **Continuous reward > discrete reward**: The careful reward design demonstrates that partial-correctness gradient signals are critical for effective RL learning.

## Limitations & Future Work

1. **Reward based on sequence similarity rather than semantic equivalence**: Functionally equivalent but differently expressed patches may be penalized, limiting the policy's exploration of semantically correct diverse solutions.
2. **File-level localization is overly coarse**: Agentless Mini performs only file-level localization, lacking function- or line-level precision, which constrains efficiency in large-file scenarios.
3. **Pipeline architecture limits holistic reasoning**: Each step reasons independently; the model cannot learn from interactive feedback or globally coordinate localization, repair, and verification.
4. **Non-trivial training cost**: 512 H100 GPUs for 32 hours remains a high barrier for academic research groups.
5. **Smaller models not explored**: Experiments are conducted only at the 70B scale; the effectiveness of SWE-RL at the 7B/13B scale remains unknown.
6. **No integration with agentic methods**: Whether pipeline-based and agent-based approaches can be unified through RL remains unexplored.

## Related Work & Insights

- **DeepSeek-R1**: The pioneer of RL + rule-based rewards for enhancing reasoning; this work extends the paradigm from competitive programming/mathematics to real-world SE tasks.
- **Agentless**: A representative pipeline-based SE tool; this work adopts a simplified variant (Agentless Mini) as its inference framework.
- **GRPO (DeepSeekMath)**: The RL optimization algorithm adopted in this work, computing advantage estimates via within-group normalization without requiring a critic network.
- **Magicoder (OSS-Instruct)**: Provides inspiration for SFT baseline data generation; this work demonstrates the superiority of the RL pathway over SFT.
- **SWE-Gym / SWE-Fixer / Lingma-SWE-GPT**: Representative works following the SFT + closed-source distillation paradigm; SWE-RL surpasses them without relying on distillation.
- **Insights**: This work opens a new paradigm for autonomous LLM evolution driven by large-scale software evolution data (commits, PRs, code reviews), suggesting that "learning to fix bugs" is itself an efficient training signal for general reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First successful application of RL to real-world SE tasks, opening an entirely new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage of main experiments, ablations, scaling analysis, and out-of-domain generalization, though error category analysis is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, well-motivated contributions, and highly informative figures and tables.
- Value: ⭐⭐⭐⭐⭐ — Significant contribution to both the SE and RL communities, demonstrating the feasibility of emergent reasoning capabilities driven by software evolution data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning](open_vision_reasoner_transferring_linguistic_cognitive_behavior_for_visual_reaso.md)
- [\[NeurIPS 2025\] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](rl_tango_reinforcing_generator_and_verifier_together_for_lan.md)

</div>

<!-- RELATED:END -->
