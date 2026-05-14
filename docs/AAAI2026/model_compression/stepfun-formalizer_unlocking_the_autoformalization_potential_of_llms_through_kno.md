---
title: >-
  [Paper Note] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion
description: >-
  [Model Compression] This paper proposes the ThinkingF pipeline, which enhances LLMs' formal language domain knowledge via large-scale knowledge distillation and their informal-to-formal reasoning ability via template-gui…
tags:
  - "Model Compression"
date: 2026-05-08
content_hash: 26b8eb2338cdb2b6
---

# StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion

- **Conference**: AAAI 2026
- **arXiv**: [2508.04440](https://arxiv.org/abs/2508.04440)
- **Code**: [stepfun-ai/StepFun-Formalizer](https://github.com/stepfun-ai/StepFun-Formalizer)
- **Area**: Autoformalization / Mathematical Reasoning
- **Keywords**: Autoformalization, Lean 4, Knowledge Distillation, Reasoning Templates, RLVR, SFT, Formal Verification

## TL;DR

This paper proposes the ThinkingF pipeline, which enhances LLMs' formal language domain knowledge via large-scale knowledge distillation and their informal-to-formal reasoning ability via template-guided reasoning trajectory synthesis. These two capabilities are then integrated through a two-stage SFT followed by RLVR. The resulting 7B/32B models achieve state-of-the-art performance on FormalMATH-Lite and ProverBench.

## Background & Motivation

Autoformalization aims to translate natural language mathematical statements into verifiable formal language expressions such as Lean 4. Existing approaches suffer from two typical deficiencies:

1. **General-purpose LLMs lack formal language knowledge**: For example, Claude4-thinking is unfamiliar with Lean 4-specific definitions (e.g., the formal expression of the Euler function), leading to incorrect implementations.
2. **Specialized models lack reasoning ability**: For example, Kimina-Autoformalizer fails to correctly interpret context and align informal-to-formal representations when handling complex natural language problems.

The authors used GPT-4o to categorize errors across approximately 10K outputs, finding that Kimina-Autoformalizer exhibits a disproportionately high error rate in "problem misunderstanding" and "informal-to-formal alignment errors," confirming that insufficient reasoning capability is the core bottleneck.

## Method

### Overall Architecture: The ThinkingF Pipeline

ThinkingF consists of four stages: knowledge distillation and filtering → reasoning data synthesis → two-stage SFT → RLVR.

### Key Design 1: Knowledge Distillation with Three-Level Quality Filtering

**Goal**: Construct large-scale, high-quality informal–formal data pairs to supplement the model's formal language knowledge.

1. **Informal problem preparation**: Approximately 256K informal mathematics problems are filtered from the NuminaMath-1.5 dataset.
2. **Formal statement generation**: Kimina-Autoformalizer generates 16 candidate formal statements $\{y_{ij}\}_{j=1}^{16}$ for each problem.
3. **Three-level filtering**:
    - **Syntax checking**: Lean4 REPL is used to filter syntactically invalid statements, retaining correct ones $\{y_{ij}^*\}_{j=1}^{m_i}$.
    - **Majority voting**: Statements are grouped into equivalence classes via BEq verification, and the best formalization is selected from the largest equivalence class:

$$y_i^{**} = \arg\max_{y_{ij}^*} \sum_{k=1}^{m_i} \mathbb{1}(y_{ik}^* \sim y_{ij}^*)$$

   - **LLM validity assessment**: DeepSeek-V3 is used to filter statements that are trivially simple or internally contradictory.

Approximately 183K high-quality knowledge data pairs are retained after this process.

### Key Design 2: Template-Guided Reasoning Trajectory Synthesis

**Goal**: Inject informal-to-formal reasoning capability into the model, compensating for the poor effectiveness of directly distilling reasoning processes.

The reasoning template consists of two core components:

1. **Informal problem comprehension**: Restate the original problem → analyze high-level logical structure → decompose mathematical concepts and objects.
2. **Informal-to-formal analysis**: Anticipate potential pitfalls in formalization → apply a divide-and-conquer paradigm to map each natural language mathematical object to its corresponding Lean 4 formal expression.

Based on human-annotated informal–formal pairs $\{(\hat{x}_i, \hat{y}_i)\}$ (sourced from MiniF2F, ProofNet, PutnamBench, etc., totaling approximately 5.8K entries), Claude 3.7 Sonnet is used to generate reasoning trajectories $\hat{c}_i$ following the template.

**Key finding**: Directly distilling reasoning trajectories from general-purpose reasoning models (e.g., Claude4-thinking) yields significantly inferior results, as such models tend to "solve" rather than "translate" during the formalization process.

### Key Design 3: Two-Stage SFT

- **Stage 1**: Fine-tune on knowledge data $\{(x_i, y_i)\}$, with empty `<think></think>` tags inserted before the output to maintain format consistency.
- **Stage 2**: Fine-tune on reasoning data $\{(\hat{x}_i, \hat{c}_i, \hat{y}_i)\}$, with reasoning trajectories wrapped in `<think>...</think>`.

The base models are DeepSeek-R1-Distill-Qwen (7B / 32B), selected for their strong performance on informal mathematics and coding tasks.

### Key Design 4: RLVR Reinforcement Learning

BEq equivalence verification is used as a verifiable reward signal, with the reward function defined as:

$$R(y_i, \hat{y}_i) = \begin{cases} 1, & \text{if } y_i \sim \hat{y}_i \\ 0, & \text{otherwise} \end{cases}$$

The GRPO algorithm is adopted, incorporating dynamic sampling from DAPO and token-level loss improvements. After training the 7B model for 450 steps and the 32B model for 350 steps, the reward increases from 0.232 to 0.347, and the mean BEq@1 improves from 0.258 to 0.303.

## Key Experimental Results

### Main Results Comparison (BEq@1 / BEq@16, %)

| Model | FormalMATH-Lite | ProverBench | CombiBench |
|------|:-:|:-:|:-:|
| OpenAI o3-pro | 22.6 / 35.5 | 24.7 / 36.2 | 9.0 / 16.0 |
| Claude4-thinking | 20.8 / 32.2 | 24.4 / 35.6 | 9.7 / 18.0 |
| DeepSeek-R1-671B | 18.4 / 31.3 | 23.5 / 34.5 | 8.1 / 20.0 |
| Kimina-Autoformalizer-7B | 35.1 / 60.2 | 13.3 / 25.3 | 2.6 / 6.0 |
| **StepFun-Formalizer-7B** | **38.3 / 61.2** | **25.1 / 37.9** | **5.2 / 11.0** |
| **StepFun-Formalizer-32B** | **40.5 / 59.3** | **26.7 / 38.5** | **6.9 / 14.0** |

### Ablation Study (OOD Benchmarks, BEq@1 / BEq@16, %)

| Configuration | ProverBench | CombiBench |
|------|:-:|:-:|
| ThinkingF (full) | 25.1 / 37.9 | 5.2 / 11.0 |
| w/o knowledge data | 24.5 / 37.4 | 3.9 / 10.0 |
| w/o reasoning data | 21.8 / 25.3 | 5.3 / 6.0 |

## Key Findings

1. **Reasoning data is the primary contributor**: Removing reasoning data causes a substantial drop in BEq@16 (37.9→25.3), indicating that reasoning capability is critical to the performance ceiling.
2. **Knowledge data plays a complementary role**: While its individual contribution is modest, its combination with reasoning data yields significant gains.
3. **Template-guided synthesis outperforms direct distillation**: Directly distilling reasoning trajectories from Claude4-thinking reduces BEq@1 on ProverBench from 25.1 to 21.8.
4. **RL remains effective**: Continued RL training on the same 5.8K data already used in SFT still yields consistent improvements.
5. **End-to-end theorem proving**: When evaluated with Kimina-Prover on 10K problems, StepFun-Formalizer-7B produces 4,940 provable statements, surpassing Kimina-Autoformalizer-7B's 4,549.

## Highlights & Insights

- **The first autoformalization training pipeline that jointly addresses domain knowledge and informal-to-formal reasoning**, bridging the capability gaps of both general-purpose and specialized models.
- The **template-guided reasoning synthesis** is an elegant design that avoids the "solving rather than translating" pitfall of general-purpose reasoning models.
- **A 7B model outperforms a 671B general-purpose model** (DeepSeek-R1) on ProverBench, demonstrating remarkable parameter efficiency.
- Using BEq equivalence verification as the RL reward signal enables a reliable closed-loop verifiable feedback mechanism.

## Limitations & Future Work

1. **Limited data scale**: Only 5.8K reasoning data pairs and 183K knowledge pairs are used; the authors note that the 32B model requires more data to further surpass the 7B model.
2. **Restricted to Lean 4**: Generalizability to other formal languages such as Coq and Isabelle has not been validated.
3. **CombiBench performance remains low**: Combinatorial mathematics problems involving complex real-world scenarios and long contexts remain challenging (7B achieves only 5.2%).
4. **Dependence on external models**: Knowledge distillation relies on Kimina-Autoformalizer and reasoning synthesis relies on Claude 3.7 Sonnet, limiting the reproducibility of the pipeline.
5. **Evaluation methodology limitations**: BEq uses the `exact?` tactic for equivalence verification, which is the most stringent approach but may miss semantically equivalent cases.

## Related Work & Insights

- **Autoformalization**: LeanFormalizer (SFT/PPO), Kimina-Autoformalizer (expert iteration), Goedel-Formalizer (human annotation scaling), Herald-Translator, Mathesis (first to incorporate RL but without a reasoning process).
- **Reasoning-augmented LLMs**: DeepSeek-R1, domain specializations under the OpenAI o1 paradigm — Fin-R1 (finance), Table-R1 (tables), CodeV-R1 (code verification), R1-Code-Interpreter.
- **Theorem proving**: DeepSeek-Prover-V2, Kimina-Prover, Goedel-Prover.

## Rating

⭐⭐⭐⭐ — The method is systematically designed and effective, with a clear dual-data pipeline for knowledge and reasoning; experiments are thorough. The 7B model outperforming a 671B general-purpose model is impressive. Limitations include the relatively small data scale and exclusive support for Lean 4.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](../../ICLR2026/model_compression/uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](../../ICLR2026/model_compression/sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)

</div>

<!-- RELATED:END -->
