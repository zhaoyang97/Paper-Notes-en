---
title: >-
  [Paper Note] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting
description: >-
  [ACL 2026][LLM Reasoning][To be supplemented] To be supplemented after thorough reading.
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "To be supplemented"
date: 2026-05-08
content_hash: 73adda83c76b555e
---

# Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting

**Conference**: ACL 2026
**arXiv**: [2604.05540](https://arxiv.org/abs/2604.05540)
**Code**: [https://github.com/FredJDean/CoT2Edit](https://github.com/FredJDean/CoT2Edit)
**Area**: LLM Reasoning / Knowledge Editing
**Keywords**: Knowledge Editing, Chain-of-Thought, GRPO, RAG, Multi-hop Reasoning

## TL;DR

CoT2Edit proposes a new paradigm for teaching LLMs to perform knowledge editing via CoT reasoning. It constructs CoT instruction data for both structured and unstructured edits, trains with SFT warm-start followed by GRPO optimization, and retrieves edited facts via RAG at inference time. A single training run achieves SOTA across 6 editing benchmarks with strong generalization.

## Background & Motivation

**State of the Field**: Knowledge editing aims to update outdated or incorrect knowledge in LLMs. Mainstream approaches include in-context editing (ICE/IKE), parameter modification (ROME/MEMIT/AlphaEdit), and train-then-retrieve paradigms (LTE/EditCoT).

**Limitations of Prior Work**: (1) Locate-and-edit methods (ROME/MEMIT) directly modify model parameters, making them incompatible with frozen production LLMs and prone to "rote memorization"—accurate on exact queries but failing on semantically equivalent ones; (2) LTE does not explicitly model reasoning paths, and requiring single-step answer generation is prone to hallucination; (3) EditCoT requires a multi-model pipeline (one for CoT generation, one for edit execution), which is complex and non-scalable; (4) all existing methods handle only structured fact triples, ignoring unstructured knowledge such as news articles.

**Root Cause**: Existing methods frame knowledge editing as a memorization problem—"remember the new fact"—rather than a reasoning problem—"understand and reason over the new fact." SFT tends to overfit the training distribution and generalizes poorly to OOD editing data.

**Paper Goals**: Develop a knowledge editing method that, after a single training run, generalizes to diverse editing scenarios (structured/unstructured, single-hop/multi-hop).

**Starting Point**: Knowledge editing is reformulated as a two-stage function $f_{\theta'}(e,q) = g_{\theta'}(h_{\theta'}(e,q))$—first generating an interpretable reasoning chain $h$, then producing the answer $g$ based on that reasoning. SFT provides the cold start, and GRPO provides generalization.

**Core Idea**: An LLM agent is used to generate CoT instructions for both structured and unstructured editing data. SFT instills the editing reasoning paradigm, GRPO enhances generalization to unseen editing scenarios, and RAG retrieves relevant edited facts at inference time.

## Method

### Overall Architecture

Three stages: (1) **Data Construction**—CoT instruction data is generated from MQuAKE (structured) and MQuAKE-uns (unstructured), with additional training data augmented via HotpotQA entity relations; (2) **Training**—Phase 1 SFT cold-start to learn the editing reasoning pattern, Phase 2 GRPO on the combined dataset to enhance generalization; (3) **Inference**—RAG retrieves relevant edited facts, and the model generates answers via learned CoT reasoning.

### Key Designs

1. **CoT Instruction Data Construction**:

    - Function: Teaches the model to perform step-by-step reasoning starting from edited facts.
    - Mechanism: For structured data, an LLM agent generates reasoning chains based on edited facts $\mathcal{E}$ and multi-hop questions $\mathcal{Q}$: $\text{Agent}(\mathcal{Q}, \mathcal{E}, \mathcal{T}) \to \text{CoT}, \mathcal{A}$. For unstructured data, relevant facts are first extracted from the edit context $\mathcal{C}$ before reasoning: $\text{Agent}(\mathcal{Q}, \mathcal{C}, \mathcal{T}) \to \mathcal{E}, \text{CoT}, \mathcal{A}$. Data augmentation synthesizes ~10K additional instruction samples via HotpotQA entity relations.
    - Design Motivation: Covers both structured and unstructured editing scenarios; explicit CoT reasoning paths reduce hallucination.

2. **Two-Stage Training (SFT + GRPO)**:

    - Function: SFT provides a cold start for editing reasoning; GRPO enhances OOD generalization.
    - Mechanism: Phase 1 trains the model autoregressively on CoT instruction data via SFT. Phase 2 optimizes with GRPO on the combined dataset using a reward function $\mathcal{R} = \mathcal{R}_{acc} + \mathcal{R}_{format}$ (accuracy + format). A self-evolution strategy is employed, collecting high-reward samples each round for the next: $\mathcal{D}_{t+1} = \mathcal{D}_t \cup \{s \mid \mathcal{R}(s) > \theta\}$.
    - Design Motivation: Pure SFT overfits to seen editing patterns; GRPO improves generalization by exploring diverse reasoning paths. The self-evolution strategy accelerates convergence.

3. **RAG-based Knowledge Injection at Inference**:

    - Function: Dynamically retrieves relevant edited facts at inference time without retraining.
    - Mechanism: The most relevant edited facts are retrieved as context for the user query; the model applies its learned CoT reasoning to answer based on the retrieved facts.
    - Design Motivation: Decouples knowledge storage from reasoning ability—the knowledge base can be updated at any time, while the model only needs to learn once "how to reason given facts."

### Loss & Training

SFT: standard autoregressive cross-entropy. GRPO: accuracy reward + format reward (enforcing think/answer tags and keywords). Validated on Llama-3.1-8B, Qwen-2.5-7B, and DeepSeek-R1-Distill-Qwen-7B.

## Key Experimental Results

### Main Results (Comprehensive Performance across 6 Editing Benchmarks)

| Method | Edit Succ | Paraphrase | Neighborhood | Scope |
|--------|-----------|------------|--------------|-------|
| AlphaEdit | 88.78 | ~81 | ~70 | Structured only |
| EditCoT | 86.13 | 83.55 | ~70 | Structured only |
| **CoT2Edit** | **93.17** | **89** | **93** | Structured + Unstructured |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| SFT only | Overfitting, poor OOD | Cold start but insufficient generalization |
| SFT + GRPO | Comprehensive improvement | GRPO is the core contribution |
| w/o data augmentation | Insufficient GRPO training | ~10K augmented samples are critical |
| w/o RAG | Performance degradation | Retrieval provides essential edited facts |

### Key Findings

- A single training run generalizes to 6 unseen editing benchmarks, demonstrating that the model learns a general "reasoning over facts" capability.
- Unstructured knowledge editing accuracy reaches 92%, approximately 20% higher than IKE.
- At large-scale editing (20K–30K facts vs. conventional 2K–3K), the model maintains 89% paraphrase and 93% neighborhood success rates.
- GRPO is applied to knowledge editing for the first time; the self-evolution strategy accelerates convergence.
- Results confirm that RL-based training outperforms pure SFT for editing generalization.

## Highlights & Insights

- Reframing knowledge editing from a "memorization problem" to a "reasoning problem" is a fundamental paradigm shift—the model does not need to memorize all edited facts, but only needs to learn how to reason given those facts.
- The two-stage training strategy of SFT cold-start followed by GRPO generalization is transferable to other tasks requiring OOD generalization.
- The self-evolution strategy (collecting high-reward samples for subsequent training rounds) is a simple yet effective form of data augmentation.

## Limitations & Future Work

- Retrieval quality directly affects editing performance; failure to retrieve relevant facts may lead to incorrect answers.
- Training data scale is approximately 13K; scaling behavior under larger datasets remains unvalidated.
- Validation is limited to 7–8B models; performance on larger models may differ.
- Conflict resolution among mutually inconsistent edited facts is not explicitly addressed.

## Related Work & Insights

- **vs. AlphaEdit/MEMIT**: Parameter modification methods are incompatible with frozen models and prone to rote memorization. CoT2Edit generalizes to semantic variants through reasoning.
- **vs. EditCoT**: EditCoT requires two separate LLMs (CoT generation + edit execution); CoT2Edit accomplishes this with a single model and additionally supports unstructured editing.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of GRPO to knowledge editing; reasoning paradigm replaces memorization paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 benchmarks, 3 models, multiple editing scenarios with comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagram is clear; method description is complete.
- Value: ⭐⭐⭐⭐ High practical value from single-training generalization across multiple scenarios.

**Code**: To be confirmed
**Area**: llm_reasoning
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning](crisp_compressing_redundancy_in_chain-of-thought_via_intrinsic_saliency_pruning.md)
- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)
- [\[ACL 2026\] Taming Actor-Observer Asymmetry in Agents via Dialectical Alignment](taming_actor-observer_asymmetry_in_agents_via_dialectical_alignment.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ACL 2026\] Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning](step-grpo_internalizing_dynamic_early_exit_for_efficient_reasoning.md)

</div>

<!-- RELATED:END -->
