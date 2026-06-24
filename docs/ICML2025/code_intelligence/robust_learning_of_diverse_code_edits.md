---
title: >-
  [Paper Note] Robust Learning of Diverse Code Edits (NextCoder)
description: >-
  [ICML 2025][Code Intelligence][Code Editing] This work proposes a synthetic code editing data generation pipeline alongside a robust adaptation algorithm SeleKT (Selective Knowledge Transfer). By performing periodic top-k sparse projections of task vectors during fine-tuning, the model is equipped with strong specialized code editing capabilities while preserving its original code generation and general reasoning capacities. The resulting NextCoder model family outperforms sa…
tags:
  - "ICML 2025"
  - "Code Intelligence"
  - "Code Editing"
  - "Synthetic Data"
  - "Robust Fine-tuning"
  - "Selective Knowledge Transfer"
  - "Catastrophic Forgetting"
  - "SeleKT"
date: 2026-05-08
content_hash: da158bd3f840470b
---

# Robust Learning of Diverse Code Edits (NextCoder)

**Conference**: ICML 2025  
**arXiv**: [2503.03656](https://arxiv.org/abs/2503.03656)  
**Code**: [aka.ms/nextcoder](https://aka.ms/nextcoder)  
**Area**: NLP Generation / Code Editing  
**Keywords**: Code Editing, Synthetic Data, Robust Fine-tuning, Selective Knowledge Transfer, Catastrophic Forgetting, SeleKT

## TL;DR

This work proposes a synthetic code editing data generation pipeline alongside a robust adaptation algorithm SeleKT (Selective Knowledge Transfer). By performing periodic top-k sparse projections of task vectors during fine-tuning, the model is equipped with strong specialized code editing capabilities while preserving its original code generation and general reasoning capacities. The resulting NextCoder model family outperforms same-sized or even larger models across five code-editing benchmarks.

## Background & Motivation

### Problem Definition

Code editing is one of the most fundamental operations in software engineering—modifying existing code based on natural language instructions (e.g., bug fixing, performance optimization, security enhancement). Existing code LMs (especially open-source models under 16B parameters) exhibit sub-optimal performance on code editing tasks, primarily limited by:

**Poor Training Data Quality**: Existing methods rely heavily on GitHub commit data, which often contains vague commit messages, inconsistent quality, and a lack of diverse editing types.

**Catastrophic Forgetting**: Fine-tuning on code editing data significantly degrades the model's pre-trained capabilities, such as code generation, instruction following, and mathematical reasoning.

**Limited Editing Granularity**: Existing synthetic methods (e.g., InstructCoder) only cover function-level Python code snippets and cannot handle class-level/file-level multi-lingual scenarios.

### Key Challenge

How to find the optimal balance between specialized code editing performance and pre-trained general capabilities:
- Full Parameter Fine-Tuning (SFT) $\rightarrow$ Overfitting + Catastrophic Forgetting
- PEFT methods like LoRA $\rightarrow$ Fixes trainable parameters a priori, preventing dynamic adjustments based on the task
- Model Merging (TIES) $\rightarrow$ Acts as a one-time post-processing method lacking dynamic constraints during training

## Method

### Overall Architecture

The proposed method consists of two major modules: **(1) A diverse synthetic data generation pipeline** + **(2) The SeleKT robust adaptation algorithm**.

### Module 1: Synthetic Code Editing Data Generation

The pipeline takes seed code as input and generates high-quality code editing training samples through four stages:

#### Stage 1: Problem Description + Source Code Generation
- Uses GPT-4o or Llama-3.3-70B as text generators.
- Input: Seed code (from the StarCoder dataset, filtered for files with >10 lines containing loops, functions, conditions, or classes) + code granularity requirements (function, class, or file level) + aspects of improvement (bug fixing, latency optimization, security, etc.).
- Output: Source code with **preset defects** + defect metadata.

#### Stage 2: Target Code Generation
- Generates the corrected target code based on the problem description, source code, and defect metadata from Stage 1.
- Simultaneously outputs editing explanations to provide a basis for subsequent instruction generation.

#### Stage 3: Instruction Generation
- Generates natural language editing instructions based on source code, target code, and editing explanations.
- **Four Styles**: Concise, Detailed, Human, Conversational.
- Four fine-tuning data entries of different styles are generated for each sample.

#### Stage 4: Quality Filtering
- LLMs score the samples from 0 to 10 across 5 dimensions: editing correctness, instruction consistency, code quality, instruction quality, and fine-tuning value.
- Retention Standard: Average score $\ge 7$ and each metric $> 5$.

#### Data Scale

Ultimately, **127K synthetic samples (229M tokens)** across 8 programming languages were generated, which are combined with 127K real commit entries from CommitPackFT.

| Language | GPT-4o | Llama-3.3-70B | Total | Tokens (M) |
|------|--------|---------------|------|-----------|
| Python | 8,406 | 6,963 | 15,279 | 28.63 |
| C | 7,039 | 10,114 | 17,153 | 33.48 |
| C++ | 6,272 | 11,065 | 17,337 | 30.93 |
| Java | 6,447 | 9,881 | 16,328 | 27.61 |
| JavaScript | 7,367 | 8,663 | 16,030 | 25.92 |
| Rust | 4,701 | 11,737 | 16,438 | 30.43 |
| Go | 4,503 | 10,701 | 15,204 | 28.56 |
| Kotlin | 3,470 | 9,802 | 13,272 | 22.16 |
| **Total** | **48,205** | **78,926** | **127,041** | **227.72** |

### Module 2: SeleKT Robust Adaptation Algorithm

#### Core Idea

The key insight of SeleKT is that **which parameters should be updated should not be pre-determined, but dynamically evaluated during training based on target task difficulty**.

The algorithm alternately performs two steps:
1. **Dense Gradients**: Performs full-parameter fine-tuning on all parameters to obtain the optimal update directions.
2. **Sparse Projection**: Calculates the task vector $\tau = \theta - \theta_{\text{base}}$, retains the top-$\alpha N$ parameters with the largest change magnitudes, and resets the remaining parameters back to the base model weights.

#### Mathematical Formulation

Robust adaptation is modeled as an optimization problem with $L_0$ constraints:

$$\arg\min_{\theta} \mathcal{L}(\theta) \quad \text{s.t.} \quad \|\theta - \theta_{\text{base}}\|_0 \leq c$$

where $\mathcal{L}$ is the next-token prediction cross-entropy loss, and $c$ controls the number of updateable parameters.

#### Algorithmic Flow (Algorithm 1)

```text
Input: Base model θ_base, Training data D, Total epochs E, Period M, Sparsity α
1. Initialize θ ← θ_base
2. for epoch e = 1 to E:
3.   for each minibatch D[s]:
4.     θ ← TrainStep(θ, D[s])      # Dense gradient update
5.     if s mod M == 0:
6.       τ ← θ - θ_base             # Calculate task vector
7.       γ[i] = 1 if i ∈ top-k(|τ|, ⌊αN⌋) else 0  # Build mask
8.       θ ← θ_base + γ ⊙ τ         # Sparse projection
9. return θ as θ_FT
```

#### Key Design Details

- **Sparsity $\alpha = 0.05$** (per layer): Keeps updates for only 5% of the parameters per layer, which empirically proves to be the optimal choice.
- **Period $M = 1$ epoch**: Performs projection at the end of each epoch; excessive frequency proves counterproductive.
- **Global Selection**: The top-k selection is global and not restricted to specific layers or structural components, outperforming manual layer specification.
- **Fixed Base Model**: Always computes the task vector anchored on the original base model (instead of a sliding baseline), ensuring a strict $L_0$ upper bound on the distance between the final model and the pre-trained weights.

### Loss & Training

- **Base Models**: QwenCoder-2.5-Instruct (3B/7B/14B/32B), DeepSeekCoder-6.7B-Instruct
- **Optimizer**: AdamW, learning rate $10^{-5}$, WarmupLR (warmup ratio 0.1)
- **Training Epochs**: 3 epochs
- **Sequence Length**: 8192 for DeepSeekCoder; 16384 for QwenCoder (using sample packing)
- **Hardware**: 8 × NVIDIA H100 80GB, ~6 hours per epoch

## Key Experimental Results

### Evaluation Benchmarks

The evaluations cover 9 benchmarks across code editing, code generation, and general capabilities:

| Benchmark | Task Type | Granularity | Sample Count |
|------|---------|------|--------|
| CanItEdit | Bug Fix | Class-level | 210 |
| HumanEvalFix | Bug Fix | Function-level | 164 |
| NoFunEval | Code Improvement | File-level | 397 |
| Aider | Bug Fix (Conversational) | File-level | 133 |
| Aider Polyglot | Bug Fix (Polyglot) | File-level | 225 |
| HumanEval+ | Code Generation | Function-level | 164 |
| MBPP+ | Code Generation | Function-level | 378 |
| GSM8K | Math Reasoning | - | 1,320 |
| MMLU | Multi-domain Knowledge | - | 3,150 |

### Main Results: Code Editing Performance (Selected from Table 4)

| Model | HumanEvalFix | CanItEdit | Aider |
|------|-------------|-----------|-------|
| GPT-4o | 90.2 | 59.5 | 74.4 |
| QwenCoder-2.5-32B | 90.2 | 60.9 | 75.2 |
| Llama-3-70B-Inst | 77.4 | 56.7 | 51.1 |
| DeepSeekCoder-33B | 74.4 | 49.5 | 58.6 |
| QwenCoder-2.5-7B | 73.8 | 48.1 | 59.4 |
| QwenCoder-2.5-7B-SFT | 70.1 | 36.7 | 48.9 |
| QwenCoder-2.5-7B-LoRA | 70.7 | 44.3 | 40.6 |
| QwenCoder-2.5-7B-TIES | 79.5 | 47.0 | 60.2 |
| **NextCoder-7B (SeleKT)** | **81.1** | **50.5** | **65.7** |

**Key Findings**:
- NextCoder-7B ranks best among all 7B-level models, even outperforming DeepSeekCoder-V2-16B (with 2× parameters).
- Fine-tuning using SFT and LoRA leads to worse results than the original base model, confirming the severity of catastrophic forgetting.
- SeleKT gains +6.3% on Aider compared to the original model, and +25.1% compared to LoRA.

### Performance Across Different Model Sizes (Table 5)

| Model | HumanEvalFix | CanItEdit | Aider | Aider Polyglot |
|------|-------------|-----------|-------|----------------|
| QwenCoder-2.5-3B | 73.2 | 37.1 | 36.8 | - |
| NextCoder-3B | 75.6 | 42.4 | 37.6 | - |
| QwenCoder-2.5-14B | 87.8 | 58.1 | 66.9 | 9.3 |
| NextCoder-14B | 89.8 | 60.2 | 72.2 | 12.2 |
| QwenCoder-2.5-32B | 90.2 | 61.0 | 72.9 | 16.4 |
| NextCoder-32B | 88.9 | 62.4 | 74.7 | 21.9 |

NextCoder-32B outperforms GPT-4o (18.2%) on Aider Polyglot, reaching 21.9%.

### Preservation of Pre-training Capabilities (Table 6)

| Model | HumanEval+ | MBPP+ |
|------|-----------|-------|
| QwenCoder-2.5-7B | 85.4 | 72.5 |
| QwenCoder-2.5-7B-SFT | 79.3 | 67.2 |
| QwenCoder-2.5-7B-LoRA | 81.7 | 70.9 |
| QwenCoder-2.5-7B-TIES | 82.3 | 71.7 |
| **NextCoder-7B** | **84.8** | **72.0** |

SFT performance drops by 6.1% on HumanEval+, whereas SeleKT only degrades by 0.6%, almost entirely preserving code generation capabilities.

### Ablation Study on Sparsity (Table 9)

| $\alpha$ (Sparsity) | HumanEvalFix | CanItEdit | Aider |
|------------|-------------|-----------|-------|
| 0.05 | 81.1 | **50.5** | **65.7** |
| 0.2 | 76.8 | 45.7 | 53.4 |
| 0.5 | 81.7 | 43.3 | 54.9 |

$\alpha = 0.05$ (the sparsest) achieves the overall best performance, validating that a tighter $L_0$ constraint is more effective at preventing overfitting.

## Highlights & Insights

1. **The "Dense Update, then Sparse Projection" Paradigm**: Contrary to PEFT methods like LoRA that "fix parameters first, then train a subset," SeleKT uses full-parameter gradients to find the optimal direction first and then truncates using sparse projection, achieving both high expressiveness and strong regularization.
2. **Dynamic Parameter Selection**: The top-k masks are updated periodically during training. Different stages can opt for different parameter subsets, making it vastly more adaptive than static methods.
3. **Theoretical Guarantees**: SeleKT guarantees a strict mathematical upper bound (Lemma 1) on the distance between the final model and the base model under the $L_0$ norm, a property that LoRA/SFT lacks.
4. **Multi-dimensional Data Diversity**: Synthetic data achieves full coverage across code granularities (function/class/file), edit categories (6 types), instruction styles (4 styles), and programming languages (8 languages).
5. **Synergy of Synthetic vs. Real Data**: Using CommitPackFT alone yields limited performance (only 21.1% on Aider). Adding synthetic data boosts it to 33.8% (+60%), and combining both yields the best results.

## Limitations & Future Work

1. **High Computational Cost**: Although the sparse projection itself is highly efficient, the dense gradient step requires full-parameter fine-tuning, which is less efficient during training compared to LoRA (requires ~18 hours of training on 8×H100).
2. **Reliance of Synthetic Data on Strong LLMs**: Data generation depends heavily on GPT-4o and Llama-3.3-70B; the quality is bounded by the generators' capabilities, posing risks of data homogenization.
3. **Limited Evaluation Scope**: The study primarily focuses on code editing and has not validated the robust fine-tuning effects of SeleKT in other domains, such as mathematical reasoning or natural language tasks.
4. **Vulnerability to Hyperparameters**: $\alpha$ and $M$ exert significant impact on performance (e.g., a 12.3% difference on Aider between $\alpha=0.05$ and $0.2$), incurring extra tuning costs.
5. **Blind Spots in Editing Scenarios**: Does not cover cross-repository edits (such as repo-level fixes in SWE-Bench) or refactoring edits.

## Related Work & Insights

- **Code Editing Models**: OctoCoder and EditCoder are fine-tuned on commit data; SWE-Fixer is fine-tuned for GitHub issues. This work achieves better generalization through synthetic data and a robust algorithm.
- **Code Data Synthesis**: OSS-Instruct (Magicoder) and Self-CodeAlign base their generation on seed code but only cover function-level granularities; InstructCoder is the only prior synthetic method targeting code editing, but it is limited to short Python snippets. This work is the first to achieve multi-lingual, multi-granularity, and multi-style synthetic dataset generation for code editing.
- **Countering Catastrophic Forgetting**: LoRA adds low-rank adapters while freezing the base model parameters; TIES applies sparse pruning during model merging; sparse adaptation by Nguyen et al. selects parameters a priori and only trains sparse gradients. The core difference of SeleKT lies in the "dense gradients + periodic sparse projection" mechanism.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | SeleKT's "dense training $\rightarrow$ sparse projection" paradigm is simple but highly effective, going against the mainstream PEFT direction. |
| Technical Depth | 4 | The data pipeline design is rigorous, the algorithm is theoretically grounded ($L_0$ bound), and ablation studies are comprehensive. |
| Experimental Thoroughness | 5 | Compared across 9 benchmarks, 5 model sizes, 4 fine-tuning methods, with multiple ablation studies. |
| Practical Value | 4 | Open source model + data + code, NextCoder-7B can be deployed directly for code editing scenarios. |
| Writing Quality | 4 | Clear structure, rich figures and tables, smooth flow from motivation to method and HTML structures. |
| **Total Score** | **4.2** | A solid piece of systems-oriented work, offering a simple yet effective solution for code editing fine-tuning. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](../../ICLR2026/code_intelligence/shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)
- [\[ICML 2026\] UniRTL: Unified Code and Graph for Robust RTL Representation Learning](../../ICML2026/code_intelligence/unirtl_unifying_code_and_graph_for_robust_rtl_representation_learning.md)
- [\[NeurIPS 2025\] Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](../../NeurIPS2025/code_intelligence/principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)
- [\[ICLR 2026\] EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits](../../ICLR2026/code_intelligence/edit-bench_evaluating_llm_abilities_to_perform_real-world_instructed_code_edits.md)
- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](../../NeurIPS2025/code_intelligence/qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)

</div>

<!-- RELATED:END -->
