---
title: >-
  [Paper Note] Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities
description: >-
  [NeurIPS 2025][LLM Pretraining][Length Generalization] Through systematic study of a family of retrieval and copying tasks, this paper reveals that large-scale pretraining introduces a directional bias into Transformers (rightward/forward over leftward/backward), while failing to overcome fundamental architectural limitations on non-unique tasks. Fine-tuning can eliminate the directional bias but cannot surpass the boundaries of architectural expressiveness.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - Length Generalization
  - Transformer Architectural Limitations
  - Induction Heads
  - Pretraining Bias
  - Reliability
date: 2026-05-08
content_hash: c74a7a16d1c525d6
---

# Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities

**Conference**: NeurIPS 2025
**arXiv**: [2505.21785](https://arxiv.org/abs/2505.21785)
**Code**: [https://github.com/lacoco-lab/always_a_transformer](https://github.com/lacoco-lab/always_a_transformer)
**Area**: LLM Pretraining
**Keywords**: Length Generalization, Transformer Architectural Limitations, Induction Heads, Pretraining Bias, Reliability

## TL;DR
Through systematic study of a family of retrieval and copying tasks, this paper reveals that large-scale pretraining introduces a directional bias into Transformers (rightward/forward over leftward/backward), while failing to overcome fundamental architectural limitations on non-unique tasks. Fine-tuning can eliminate the directional bias but cannot surpass the boundaries of architectural expressiveness.

## Background & Motivation

**Background**: Transformers are theoretically subject to length generalization limitations (e.g., inability to express certain sequence tasks), and experiments on small models have confirmed the existence of such limitations. Whether large-scale pretrained LLMs are subject to the same constraints, however, has remained an open question.

**Limitations of Prior Work**: Existing studies either validate theoretical limitations only on small models (lacking experiments on pretrained models), or observe emergent capabilities in LLMs without connecting them to theoretical analysis. Pretraining may introduce entirely new inductive biases (e.g., ICL, CoT) that render theoretical limitations moot.

**Key Challenge**: Pretraining reshapes model parameters and subcircuits. It remains unclear whether the length generalization failures observed in small models transfer directly to pretrained large models, and whether scale and data diversity can overcome inherent architectural limitations.

**Goal**: (a) Which architectural capabilities are enhanced by pretraining, and which limitations persist? (b) Are there biases unique to pretraining? (c) Can fine-tuning eliminate these biases?

**Key Insight**: Retrieval and copying are chosen as the testbed — they are fundamental building blocks for practical capabilities such as ICL and RAG, and have clear theoretical characterizability via the C-Rasp[pos] framework.

**Core Idea**: Pretraining causes Transformers to be "born preferring" rightward/forward operations (stronger induction circuits), while the inherent length generalization ceiling imposed by the architecture always remains.

## Method

### Overall Architecture

The study designs a matrix of variants across two task families — Retrieval and Copying:
- **Retrieval tasks**: Given a context sequence and a query token, return the token to the left or right of the query token. Tasks are divided into Unique (query appears once) and Non-unique (query appears multiple times) variants; Non-unique tasks are further split into first and last occurrence, yielding 6 subtasks (UR, UL, NRFirst, NRLast, NLFirst, NLLast).
- **Copying tasks**: Given an input sequence, the model is required to copy it. Four variants are defined: unique forward (UF), unique backward (UB), non-unique forward (NF), and non-unique backward (NB).

The study proceeds in three progressive stages: (1) theoretical analysis providing length generalization guarantees, (2) ICL experiments on pretrained models revealing biases, and (3) fine-tuning combined with mechanistic analysis to explain the origins of those biases.

### Key Designs

1. **C-Rasp[pos] Theoretical Framework**:

    - Function: Provides theoretical guarantees (or refutations) of length generalization for each task variant.
    - Mechanism: Leverages the framework of Huang et al. (2025) — if a task can be expressed as a C-Rasp[pos] program, then a Transformer can length-generalize on it. C-Rasp[pos] restricts the use of positional information and arithmetic operations.
    - Theoretical conclusions: UR/UL/NRFirst/NLFirst/UF/UB are expressible in C-Rasp[pos] (generalizable); NRLast/NLLast/NF/NB are not (non-generalizable). Crucially, **left and right variants have no difference in expressiveness** (Theorem 2), so any directional asymmetry observed in pretrained models is purely a product of pretraining bias.
    - Design Motivation: To establish a theoretical baseline that distinguishes "architecturally impossible" from "not learned during pretraining."

2. **Induction vs. Anti-Induction Circuits**:

    - Function: Explains the mechanistic origins of directional bias.
    - Mechanism: The induction head is a two-layer circuit — the first layer (previous-token head) writes the embedding of the preceding token into the residual stream, and the second layer (induction head proper) matches the current token and copies the subsequent token. Anti-induction heads are theoretically symmetric and should support leftward retrieval. However, inductive patterns (bigram continuation) are far more prevalent in pretraining data than anti-inductive ones, causing induction heads to be selectively strengthened.
    - Causality is verified via patching experiments: removing induction heads reduces UF accuracy to zero while leaving UB unaffected; removing anti-induction heads reduces UB to zero while leaving UF unaffected.

3. **Natural-Setting Validation**:

    - Function: Transfers findings from synthetic tasks to real-world application scenarios.
    - **Lorem Ipsum Copying Experiment**: Models are asked to copy a ~500-token Lorem Ipsum passage. Alignments computed via the Needleman-Wunsch algorithm reveal that virtually all copying errors occur at **ambiguous tokens** (non-unique tokens whose subsequent tokens are uncertain), validating the theoretical limitation on non-unique copying.
    - **Git History Operation Experiment**: Models are asked to output commit hashes in forward order (git revert) or reverse order (git cherry-pick). Forward order is near-perfect; reverse order degrades substantially — demonstrating that directional bias manifests in real-world tasks.

4. **Effect of Fine-Tuning on Bias Elimination/Retention**:

    - Function: Distinguishes "biases introduced by pretraining" from "inherent architectural limitations."
    - Supervised fine-tuning is performed on GPT-2 1.5B for each task using a position-offset trick to ensure positional encodings are sufficiently trained. Evaluation is conducted on OOD test sets (sequences twice as long as the training set).
    - Results: For tasks expressible in C-Rasp[pos] (UR/UL/UF/UB/NRFirst/NLFirst), directional bias disappears after fine-tuning and OOD accuracy reaches ~100%. For non-expressible tasks (NRLast/NLLast/NF/NB), even fine-tuning fails to achieve perfect generalization.

### Loss & Training
- Pretrained model evaluation: Llama-3.1 (8B, 70B) and Qwen2.5 (7B, 32B), including both completion and instruction-tuned variants.
- Fine-tuning: GPT-2 1.5B (APE), with loss computed only over the answer span. Training length $[\ell_{min}, 100]$; test length $[101, 200]$.

## Key Experimental Results

### Main Results: ICL Accuracy on Pretrained Models

| Task | Direction | Llama-3 70B | Qwen2.5-32B | Theoretically Generalizable? |
|------|-----------|-------------|-------------|------------------------------|
| UR (Unique Right) | Right | High | High | ✓ |
| UL (Unique Left) | Left | Significantly lower | Significantly lower | ✓ |
| NRFirst | Right | Low | Low | ✓ |
| NLFirst | Left | Even lower | Even lower | ✓ |
| UF (Unique Forward Copy) | Forward | Near-perfect | Near-perfect | ✓ |
| UB (Unique Backward Copy) | Backward | Degraded | Degraded | ✓ |
| NF (Non-unique Forward) | Forward | Near-perfect | Near-perfect | ✗ |
| NB (Non-unique Backward) | Backward | Degraded | Degraded | ✗ |

Core finding: **Directional bias** is consistently present across all models, tasks, and lengths — rightward/forward > leftward/backward.

### Ablation Study: OOD Generalization after Fine-Tuning

| Task | C-Rasp[pos] Expressible | Directional Bias after Fine-Tuning | OOD Accuracy |
|------|-------------------------|------------------------------------|--------------|
| UR / UL | ✓ | Eliminated | ~100% |
| UF / UB | ✓ | Eliminated | ~100% |
| NRFirst / NLFirst | ✓ | Eliminated | ~100% |
| NRLast / NLLast | ✗ | Eliminated | Significantly degraded |
| NF / NB | ✗ | Eliminated | Significantly degraded |

### Key Findings
- **Directional bias is a product of pretraining**: Models trained from scratch exhibit no left-right asymmetry, and fine-tuning can eliminate it. Induction heads are selectively strengthened during pretraining; anti-induction heads are not.
- **Architectural limitations are hard constraints**: Fine-tuning eliminates directional bias but cannot enable perfect OOD generalization for tasks outside C-Rasp[pos]. Failures on non-unique copying/retrieval are inherent to the architecture.
- **Practical implications**: In the Lorem Ipsum experiment, 100% of copying errors occur at ambiguous tokens; in the Git experiment, reverse-order accuracy is substantially lower than forward-order.
- Patching experiments precisely localize the causal mechanism: induction heads and anti-induction heads are the respective "necessary pathways" for forward and backward tasks.

## Highlights & Insights
- The **analysis of induction–anti-induction symmetry** is particularly elegant: the two circuits are theoretically symmetric, yet pretraining creates an asymmetry — directly explaining why LLMs perform worse when "looking backward." This finding has practical implications for RAG (which requires precise copying of context spans) and ICL (which relies on induction heads).
- The application of the **C-Rasp[pos] framework as a generalizability classifier** is rigorous: it provides a yes/no theoretical verdict for each task variant, and experimental results match perfectly.
- The **Lorem Ipsum experimental design** is an excellent example of grounding theoretical limitations in a concrete setting: the ambiguous-token analysis precisely pinpoints the micro-level mechanism underlying non-unique copying failures.

## Limitations & Future Work
- Only two open-source model families (Llama-3 and Qwen2.5) are tested; closed-source models (GPT-4o, Claude, etc.) and recurrent architectures (Mamba, etc.) are not covered.
- Fine-tuning experiments are conducted only on GPT-2 1.5B and are not validated on larger models.
- The C-Rasp[pos] theory is formally proven only for APE positional encoding; empirical validation for RoPE is provided but without rigorous proof.
- The paper does not analyze the precise statistical distribution of inductive patterns in pretraining data (their prevalence over anti-inductive patterns is inferred rather than measured).
- Real-world application scenarios could be further expanded — e.g., quantifying the impact of directional bias in code generation and document summarization.

## Related Work & Insights
- **vs. Liu et al. (2024a) Flip-Flop task**: NRLast is equivalent to Flip-Flop; this paper systematizes it into a task matrix and augments it with theoretical analysis.
- **vs. Zhou et al. (2024)**: Zhou focuses on generalization in small models trained from scratch; this paper is the **first systematic study of pretrained models** on these tasks.
- **vs. Olsson et al. (2022) induction heads**: This paper extends the induction head analysis by introducing the anti-induction head concept and verifying causality through patching.
- Implications for retrieval-augmented generation (RAG): LLMs may "slip" at ambiguous tokens when precisely copying context, suggesting the need for tool-assisted copying.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic task design with triple-layer theory–experiment–mechanism validation, though the core theoretical tools are drawn from prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 pretrained models + fine-tuning + patching + natural-setting validation — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain is clear, progressing step by step from theory to experiment.
- Value: ⭐⭐⭐⭐ Provides practical guidance for understanding the capability boundaries of LLMs, especially in RAG and precise copying scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning in Compact Spaces with Approximately Normalized Transformer](learning_in_compact_spaces_with_approximately_normalized_transformer.md)
- [\[AAAI 2026\] PrefixGPT: Prefix Adder Optimization by a Generative Pre-trained Transformer](../../AAAI2026/llm_pretraining/prefixgpt_prefix_adder_optimization_by_a_generative_pre-trained_transformer.md)
- [\[NeurIPS 2025\] Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [\[NeurIPS 2025\] Understanding and Enhancing Mask-Based Pretraining towards Universal Representations](understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)
- [\[NeurIPS 2025\] The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation](the_atlas_of_in-context_learning_how_attention_heads_shape_in-context_retrieval_.md)

<!-- RELATED:END -->
