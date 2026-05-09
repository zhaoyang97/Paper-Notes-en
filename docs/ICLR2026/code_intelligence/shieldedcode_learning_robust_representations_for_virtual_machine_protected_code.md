---
title: >-
  [Paper Note] ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code
description: >-
  [ICLR 2026][virtual machine protection] This paper proposes ShieldedCode — the first protection-aware code representation learning framework. By introducing hierarchical dependency modeling (three levels: intra-instruction, preceding-instruction, and inter-instruction) and joint functional-aware and protection-aware contrastive learning, the framework enables LLMs to generate, compare, and reason about VM-protected code. ShieldedCode surpasses existing methods on both VM code generation (Pass@1 26.95% vs. GPT-4o 22.58%) and binary similarity detection.
tags:
  - ICLR 2026
  - virtual machine protection
  - code representation learning
  - contrastive learning
  - polymorphic generation
  - software security
date: 2026-05-08
content_hash: f20144ff801f43ed
---

# ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code

**Conference**: ICLR 2026
**arXiv**: [2601.20679](https://arxiv.org/abs/2601.20679)
**Code**: None
**Area**: Code Intelligence
**Keywords**: virtual machine protection, code representation learning, contrastive learning, polymorphic generation, software security

## TL;DR

This paper proposes ShieldedCode — the first protection-aware code representation learning framework. By introducing hierarchical dependency modeling (three levels: intra-instruction, preceding-instruction, and inter-instruction) and joint functional-aware and protection-aware contrastive learning, the framework enables LLMs to generate, compare, and reason about VM-protected code. ShieldedCode surpasses existing methods on both VM code generation (Pass@1 26.95% vs. GPT-4o 22.58%) and binary similarity detection.

## Background & Motivation

1. LLMs have achieved remarkable progress in code generation, yet their potential in software protection remains largely unexplored.
2. Reverse engineering continues to threaten software security, while traditional virtual machine protection (VMP) relies on rigid rule-based transformations that are costly to design and susceptible to automated analysis.
3. Conventional VMP systems produce highly regular virtual machine structures and instruction patterns, making them targets for rule-based and semantic attacks.
4. Advances in machine learning for binary similarity detection and neural decompilation have accelerated the automation of semantic recovery.
5. Existing models designed for compiler-level assembly (e.g., Nova, LLMCompiler) handle structurally stable O0–O3 optimized code, whereas VMP bytecode undergoes polymorphic expansion, virtual register renaming, and interpreter-driven semantic transformations — resulting in a substantial domain gap.
6. Core thesis: protection mechanisms must evolve from fixed transformation rules toward mechanisms that embed semantic diversity and dynamic behavior, thereby resisting both human and AI-assisted analysis.

## Method

### Overall Architecture

Initialized from CodeLlama 34B, the framework constructs a large-scale paired dataset of source code and standardized VM implementations. It is trained via hierarchical dependency modeling with a joint contrastive and language modeling objective, and supports VM code generation and binary code representation learning through a two-stage pipeline (continual pre-training followed by fine-tuning).

### Key Designs

#### 1. Training Data Construction

- Starting from AnghaBench and The Stack source code: compilation (O0–O3) → protection via commercial VMP tools → disassembly to extract VM implementations.
- A normalization operator $\mathcal{N}$ applies four-step canonicalization: removal of debug symbols, whitespace insertion for stable tokenization, symbolic replacement of virtual addresses, and canonical label substitution (`[VINST-1]`, `[VINST-2]`, …).

#### 2. Hierarchical Dependency Modeling (Polymorphic VM Generation)

Three levels of hierarchical masking are applied to define the visible context for token $x_t^k$:

- **Intra-instruction**: tokens of the current virtual instruction together with its `[VINST]_t` marker, treated as a coherent semantic unit.
- **Preceding-instruction**: conditioned on the `[VINST]_{t-1}` marker of the immediately preceding instruction, capturing local execution patterns such as register reuse and operand flow.
- **Inter-instruction**: aggregates all prior markers $\{[\text{VINST}]_1, \ldots, [\text{VINST}]_{t-1}\}$, injecting long-range contextual information and capturing polymorphic transformations and dispersed control-flow dependencies.

$$\mathcal{M}(x_t^k) = \underbrace{\{x_t^1,...,x_t^m, [\text{VINST}]_t\}}_{\text{intra}} \cup \underbrace{\{[\text{VINST}]_{t-1}\}}_{\text{preceding}} \cup \underbrace{\{[\text{VINST}]_1,...,[\text{VINST}]_{t-1}\}}_{\text{inter}}$$

#### 3. Joint Contrastive and Language Modeling

- **Functional Contrastive Learning (FCL)**: pulls together embeddings of the same function across different representations (source code + protection levels L0–L3), using adaptive weights $w_{s,t} = \exp(-|s-t|/\tau_{\text{fcl}})$ such that closer protection levels receive higher weights.
- **Protection Contrastive Learning (PCL)**: enforces proportional separation of embeddings from different protection-level variants via a soft-margin constraint: $d(e_f^s, e_f^t) \geq \beta(t-s) - m$.
- **Total objective**: $L_{\text{vmp}} = L_{\text{lm}} + \lambda(L_{\text{fcl}} + L_{\text{pcl}})$

### Loss & Training

- **Two-stage training**:
  1. Continual pre-training: alternating optimization of contrastive + language modeling ($L_{\text{vmp}}$) and protection effect optimization ($L_{\text{peo}}$), using AnghaBench + The Stack + VirtuCorp (3M samples).
  2. Fine-tuning: optimizing $L_{\text{vmp}}$ only, using 2.5M source–VMP pairs (850M tokens).
- Polymorphic generation is applied to half of the attention heads to balance effectiveness and retention of pre-trained knowledge.
- Protection Effect Optimization (PEO) employs a hard negative mining strategy: $\kappa_i = 1 + \lambda_h \cdot \text{rank}_i$.

## Key Experimental Results

### Main Results

**VM Code Generation (HumanEval_compile)**:

| Model | Pass@1 (L0) | Pass@1 (L1) | Pass@1 (L2) | Pass@1 (L3) |
|-------|------------|------------|------------|------------|
| CodeLlama | 7.84 | 3.26 | 5.19 | 2.79 |
| DeepSeekCoder-7B | 10.28 | 6.89 | 7.94 | 6.17 |
| GPT-4o | 22.58 | 17.43 | 15.26 | 11.89 |
| **ShieldedCode** | **26.95** | **18.47** | **19.23** | **14.71** |

**Binary Similarity Detection (BinaryCorp-VA)**:

| Model | Recall@1 O0+L1 | Recall@1 O0+L3 | MRR O0+L1 |
|-------|-----------|-----------|-----------|
| jTrans (Linear Probe) | 0.333 | 0.404 | 0.245 |
| Trex | 0.118 | 0.148 | 0.073 |
| **ShieldedCode** | **0.488** | **0.272** | **0.575** |

### Ablation Study

| Configuration | Pass@1 Avg. | Pass@10 Avg. |
|--------------|-----------|------------|
| ShieldedCode$^{-\text{CL-PG}}$ (LM only) | 15.78 | 27.41 |
| ShieldedCode$^{-\text{PG}}$ (+ contrastive learning) | 21.86 | 35.25 |
| **ShieldedCode** (all components) | **25.17** | **38.30** |

**Granite 128K Long-Context Ablation**:

| Configuration | Pass@1 Avg. | Pass@10 Avg. |
|--------------|-----------|------------|
| Granite 3B 128K | 4.62 | 6.44 |
| + Standard Fine-Tuning | 12.84 | 19.41 |
| + ShieldedCode Approaches | **17.91** | **25.25** |

### Key Findings

1. **Surpasses GPT-4o on VM code generation**: Pass@1 improves by 4.37 percentage points at L0 (26.95% vs. 22.58%), with a more pronounced gain at L2 (19.23% vs. 15.26%).
2. **Hierarchical dependency modeling contributes the most**: the transition from ShieldedCode$^{-\text{PG}}$ to the full model yields an average Pass@1 improvement of 3.31 percentage points.
3. **Reverse engineering resistance**: manual reverse analysis success rate drops to 17% (vs. 67% for VMProtect), with average analysis time increasing to 14.7 hours (vs. 3.4 hours); pattern-matching attack success rate reaches 0%.
4. **Orthogonal complementarity with long-context techniques**: applying ShieldedCode methods to Granite 128K yields an additional 5.07% Pass@1 improvement.

## Highlights & Insights

1. This work is the first to formalize software protection as a representation learning problem, opening a new direction for learning-based software defense.
2. The three-level hierarchical attention masking is elegantly designed — unlike the flat causal masking of standard Transformers, it introduces inductive biases aligned with the structured dependencies inherent in VM-protected code.
3. The mathematical compatibility of FCL and PCL is noteworthy: the exponential decay weights of FCL and the linearly scaled margin constraints of PCL operate synergistically, achieving a stable equilibrium between functional clustering and protection stratification (supported by a formal theorem).
4. The reverse engineering user study is rigorously designed — cross-validated by 12 graduate students and 3 professional reverse engineers, providing credible security evaluation.

## Limitations & Future Work

1. The model is based on CodeLlama 34B; the large model scale incurs high inference costs for practical deployment.
2. Training data covers only C language on x86-64 architecture; generalization to other languages and ISAs (ARM, RISC-V) remains unverified.
3. Only a single commercial VMP tool is used; variation in protection styles across different VMP systems may affect model generalizability.
4. The candidate pool size for the PEO task (K=50–500) is relatively limited; further evaluation is needed for larger-scale retrieval scenarios.

## Related Work & Insights

- **jTrans**: Transformer-based binary code similarity detection via linear probe fine-tuning, but not designed for VMP code.
- **Nova**: Hierarchical modeling for compiler-level assembly, but targets structurally stable O0–O3 code; VMP bytecode poses substantially greater difficulty.
- **LLMCompiler**: Meta's pre-trained model on LLVM IR and assembly (401B tokens), but not designed for VMP code.
- **CodeArt**: Regularized attention for assembly representation, but does not incorporate protection-aware objectives.
- Broader insight: LLMs are not merely code generators but catalysts for rethinking program representation and protection; the hierarchical attention masking approach may generalize to other code with hierarchical structure (e.g., compiled IR, WebAssembly).

## Rating

- ⭐ Novelty: 4.5/5 — First protection-aware code representation learning framework; the three-level hierarchical dependency modeling and joint FCL/PCL optimization are both original contributions.
- ⭐ Experimental Thoroughness: 4/5 — Covers three tasks (generation, detection, PEO) with ablations and a reverse engineering user study, though some baselines are estimated values.
- ⭐ Writing Quality: 3.5/5 — Technically solid but lengthy, with some inconsistency in mathematical notation.
- ⭐ Value: 4/5 — Opens a new direction for learning-based software protection with significant implications for the security community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)
- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](dro-instructzero_distributionally_robust_prompt_optimization_for_instruction_fol.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[ICLR 2026\] Learning to Reason without External Rewards](learning_to_reason_without_external_rewards.md)
- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](../../NeurIPS2025/code_intelligence/mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)

</div>

<!-- RELATED:END -->
