---
title: >-
  [Paper Note] QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation
description: >-
  [AAAI 2026][Reinforcement Learning][GPU kernel generation] This paper proposes MTMC (Macro Thinking Micro Coding), a hierarchical framework that decouples GPU kernel generation into two stages: a lightweight RL-trained L…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "GPU kernel generation"
  - "LLM code generation"
  - "reinforcement learning policy"
  - "hierarchical optimization"
  - "high-performance computing"
date: 2026-05-08
content_hash: b31976da02bd5ecf
---

# QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation

**Conference**: AAAI 2026
**arXiv**: [2511.20100](https://arxiv.org/abs/2511.20100)
**Code**: None (paper states model and dataset will be released)
**Area**: Reinforcement Learning
**Keywords**: GPU kernel generation, LLM code generation, reinforcement learning policy, hierarchical optimization, high-performance computing

## TL;DR

This paper proposes MTMC (Macro Thinking Micro Coding), a hierarchical framework that decouples GPU kernel generation into two stages: a lightweight RL-trained LLM generates high-level optimization actions (Macro Thinking), while a general-purpose LLM incrementally implements each action (Micro Coding). This design separates correctness from performance concerns, achieving near-100% accuracy and a 2.2× speedup over expert-optimized PyTorch Eager kernels on KernelBench.

## Background & Motivation

High-performance GPU kernel development is foundational to AI and scientific computing, yet remains heavily dependent on expert manual tuning. Implementing FlashAttention on Hopper architecture took years of effort and lacks cross-platform portability. Even with DSLs such as Triton, hardware-specific optimization strategies still require expert design.

**The dilemma of existing LLM-based approaches**:

**Direct generation with general-purpose LLMs**: These models lack hardware understanding, producing kernels that are unsatisfactory in both correctness and performance. Even the SOTA Gemini 2.5 Pro achieves only 36% accuracy on KernelBench-L3.

**Fine-tuned specialized LLMs**: Models such as Meta's KernelLLM and Stanford's Kevin-32B suffer from data scarcity and poor generalization. KernelLLM's accuracy drops sharply from 40–50% on its training benchmark to 2–4% on TritonBench.

**Core challenge**: The GPU kernel optimization space is enormous (approximately $10^9$ optimization configurations per subgraph), and implementation details are complex (hundreds to thousands of lines of code), where minor errors cause performance degradation or failure. Existing methods attempt to generate complete optimized kernels in a single step, requiring simultaneous exploration of both the optimization strategy space and the implementation detail space — an intractable task for LLMs.

**Core insight**: Human experts develop high-performance kernels in stages — first designing high-level optimization strategies (e.g., tiling schemes, fusion strategies), then implementing them incrementally. MTMC decouples this process into two manageable subtasks.

## Method

### Overall Architecture

MTMC consists of two hierarchical levels:

1. **Macro Thinking (high-level strategy)**: An RL-trained lightweight LLM (e.g., DeepSeek-Coder-1.3B) iteratively generates semantic optimization actions.
2. **Micro Coding (low-level implementation)**: A general-purpose LLM (e.g., Gemini 2.5 Pro) incrementally implements each optimization action.

The input is unoptimized PyTorch code; the output is a high-performance GPU kernel. The overall pipeline is iterative: Macro Thinking proposes an optimization action → Micro Coding implements it → the updated code is fed back to Macro Thinking → the next optimization action is proposed → … until termination.

### Key Designs

#### 1. Semantic Optimization Action Space

Each optimization action consists of two components:
- **Optimization type**: Four optimization principles grounded in GPU hardware characteristics:
    - **Tiling**: Partitioning data into tiles to fit shared memory capacity.
    - **Fusion**: Merging operations to reduce memory access overhead.
    - **Pipeline**: Overlapping computation and data movement.
    - **Reordering**: Permuting loops to improve memory access locality.
- **Code region**: A syntactically and semantically valid code segment identified via dataflow and AST analysis.

An example action is: *"fusing the linear and max in line 15 to 20,"* indicating the fusion of adjacent operations to reduce memory accesses. The action space is designed to be both representative of hardware optimization principles and effective in constraining the search space.

#### 2. Macro Thinking Policy Training

Lightweight pre-trained LLMs (DeepSeek-Coder-1.3B, Llama-3.2-1B, Qwen2.5-1.5B) serve as policy models. The semantic optimization action $a_k$ is a token sequence whose sampling probability equals the joint token probability:

$$P_{\text{token}}(a_k|s) = \prod_{i=1}^{N_k} P(w_k^i | s, w_k^1, \ldots, w_k^{i-1})$$

Training employs the TWOSOME framework with PPO. The RL environment is constructed as a tree structure based on 60k offline expert trajectories, avoiding the high latency of real-time LLM interactions.

**Reward shaping** (progressive curriculum from easy to hard):
1. Successful compilation → basic reward
2. Correct execution without errors → intermediate reward
3. Performance improvement over the previous step → high reward

A step-proportional decay mechanism is incorporated to prevent the policy from degenerating into cyclic behavior.

#### 3. Micro Coding Incremental Implementation

Micro Coding receives action prompts from Macro Thinking, each containing three elements:
- The current kernel code
- The optimization action (type + region)
- In-context examples corresponding to the optimization type

Since each step implements only a single atomic optimization with an explicitly specified type and code region, Micro Coding can fully exploit in-context learning to maximize the probability of generating correct code.

### Loss & Training

- **Policy training**: PPO objective with clipping and KL penalty.
- **Data efficiency**: Only 60k offline trajectories are used (no benchmark instances), covering single operators, subgraphs, and complete neural networks.
- **Model efficiency**: The policy model contains only 1.3B/1B/1.5B parameters, while Micro Coding leverages off-the-shelf general-purpose LLMs.

## Key Experimental Results

### Main Results

KernelBench results (H100, Gemini 2.5 Pro as Micro Coding backend):

| Level | Metric | MTMC | Gemini 2.5 Pro (standalone) | Kevin-32B | KernelLLM | Gain |
|-------|--------|------|----------------------------|-----------|-----------|------|
| L1 | Accuracy | **100%** | 63% | 68% | 41% | +37% vs Gemini |
| L1 | fast₁/fast₂ | 67%/13% | 31%/7% | 9%/2% | 11%/2% | +36%/+6% |
| L1 | Mean Speedup | **2.08** | 1.26 | 0.71 | 0.38 | 1.65× |
| L2 | Accuracy | **99%** | 57% | 68% | 35% | +42% vs Gemini |
| L2 | Mean Speedup | **1.28** | 0.77 | 0.58 | 0.41 | 1.66× |
| L3 | Accuracy | **70%** | 36% | 48% | 10% | +34% vs Gemini |
| L3 | Mean Speedup | **0.77** | 0.27 | 0.35 | 0.09 | 2.85× |

TritonBench results (A100, Gemini 2.5 Flash + MTMC):

| Benchmark | Metric | MTMC | Gemini 2.5 Flash | KernelLLM | Gain |
|-----------|--------|------|-----------------|-----------|------|
| TRITONBENCH-G | Call Acc | **32.61%** | 11.41% | 2.17% | +21.2% |
| TRITONBENCH-G | Exec Acc | **22.83%** | 8.70% | 1.09% | +14.13% |
| TRITONBENCH-T | Call Acc | **64.46%** | 14.46% | 4.82% | +50.00% |
| TRITONBENCH-T | Exec Acc | **54.82%** | 9.04% | 4.22% | +45.78% |
| TRITONBENCH-T | Mean Speedup | **0.64** | 0.15 | 0.02 | 4.67× |

### Ablation Study

| Ablation Dimension | Configuration | L1 Acc/Speedup | L2 Acc/Speedup | L3 Acc/Speedup |
|-------------------|---------------|----------------|----------------|----------------|
| Hierarchical generation | GF-2.5 w/o Hier (one-shot generation) | 60%/1.38 | 32%/0.43 | 10%/0.09 |
| Hierarchical generation | GF-2.5 + MTMC (incremental generation) | **94%/2.14** | **97%/1.21** | **64%/0.69** |
| Target language | MTMC (Triton) | 1.38ms | 4.43ms | 37.88ms |
| Target language | MTMC (CUDA) | 1.38ms | 1.34ms | 26.52ms |

Macro Thinking ablation (validating the necessity of policy learning and action space):

| Configuration | Policy | Action Space | L1 | L2 | L3 |
|---------------|--------|-------------|-----|-----|-----|
| DS-Coder 1.3B | ✓ RL | ✓ | 90%/1.10 | 100%/1.16 | 100%/1.82 |
| Llama 1B | ✓ RL | ✓ | 100%/1.17 | 80%/0.86 | 80%/0.74 |
| No RL policy | × | ✓ | Degraded | Degraded | Degraded |
| No RL, no action space | × | × | Further degraded | Further degraded | Further degraded |

### Key Findings

1. **Decoupled design is essential**: The accuracy jump from 60% to 94% with incremental generation confirms that current LLMs cannot complete complex kernel generation in a single step.
2. **Cross-hardware generalization**: Consistent performance across V100/A100/H100 generations indicates that Macro Thinking learns transferable optimization strategies.
3. **Generalization catastrophe in fine-tuned LLMs**: KernelLLM's accuracy collapses from 40–50% on KernelBench to 2–4% on TritonBench, whereas MTMC remains stable.
4. **Smaller policy models generalize better**: The smallest model, DS-Coder-1.3B, yields the best results, demonstrating the high efficiency of the policy training paradigm.
5. **RL training is indispensable**: Directly employing an LLM for Macro Thinking without RL training leads to significant performance degradation.

## Highlights & Insights

- **Systematizing the human expert strategy**: The paper formalizes the decision-making process of human GPU optimization experts — strategy first, then implementation — into a trainable pipeline.
- **Extreme data efficiency**: Effective optimization policies are learned from only 60k offline trajectories, without requiring large-scale kernel datasets.
- **Joint optimization of correctness and performance**: The framework breaks the conventional assumption that high performance necessarily sacrifices correctness.
- **Strong practical significance**: A 2.2× speedup over expert-optimized PyTorch Eager kernels demonstrates genuine deployment value.

## Limitations & Future Work

1. **Micro Coding depends on powerful general-purpose LLMs**: Reliance on closed-source models such as Gemini 2.5 Pro/Flash introduces cost and availability concerns.
2. **Limitations of the offline RL environment**: The tree-structured environment is built on pre-collected trajectories and cannot explore optimization paths not covered by the offline data.
3. **Limited scalability of CUDA kernel generation**: LLMs are less familiar with CUDA than Triton, constraining the scalability of CUDA-targeted generation.
4. **Fixed optimization action space**: The four optimization principles may not cover all optimization techniques, such as quantization and sparse optimization.
5. **Absence of comparison with traditional compilers**: Evaluation is limited to LLM-based methods, with no comparison against traditional auto-tuning compilers such as TVM or Halide.

## Related Work & Insights

- The QiMeng series focuses on LLM-based generation for specific operators; MTMC represents an important step toward general high-performance kernel generation within this line of work.
- Compared to agent-based methods such as AI CUDA Engineer, MTMC learns optimization strategies via RL rather than relying on manually designed agent pipelines.
- The hierarchical decoupling idea is potentially generalizable to other code generation tasks that require co-optimization of strategy and implementation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The hierarchical paradigm that decouples GPU kernel optimization into strategy and implementation is entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 3 hardware platforms, 13 LLMs, and 2 widely used benchmarks with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear, though the large number of tables slightly reduces readability.
- Value: ⭐⭐⭐⭐⭐ — Exceptionally high practical value; this is the first work enabling LLM-generated kernels to surpass expert-optimized code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Estimation of Kernel Surrogate Models for Task Attribution](../../ICLR2026/reinforcement_learning/efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [\[NeurIPS 2025\] Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification](../../NeurIPS2025/reinforcement_learning/kimina_lean_server_a_high-performance_lean_server_for_large-scale_verification.md)
- [\[AAAI 2026\] TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction](tadarag_task_adaptive_retrieval-augmented_generation_via_on-the-fly_knowledge_gr.md)
- [\[AAAI 2026\] TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents](towermind_a_tower_defence_game_learning_environment_and_benchmark_for_llm_as_age.md)
- [\[AAAI 2026\] In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback](in-token_rationality_optimization_towards_accurate_and_concise_llm_reasoning_via.md)

</div>

<!-- RELATED:END -->
