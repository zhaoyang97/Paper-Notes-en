---
title: >-
  [Paper Note] Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning
description: >-
  [ICML 2026][LLM Reasoning][Optimal Control] LLM reasoning is modeled as an optimal control problem (Linear Quadratic Regulator, LQR) in latent space. The proposed Test-Time Control (TTC) layer performs finite-horizon pla…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Optimal Control"
  - "LQR"
  - "Test-Time Planning"
  - "State-Space Models"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: e397f68b71d2f520
---

# Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning

**Conference**: ICML 2026  
**arXiv**: [2603.09221](https://arxiv.org/abs/2603.09221)  
**Code**: https://vita-group.github.io/TTC-Net (Project Page)  
**Area**: LLM Reasoning  
**Keywords**: Optimal Control, LQR, Test-Time Planning, State-Space Models, Mathematical Reasoning  

## TL;DR

LLM reasoning is modeled as an optimal control problem (Linear Quadratic Regulator, LQR) in latent space. The proposed Test-Time Control (TTC) layer performs finite-horizon planning during the forward pass and decodes optimal control actions as next-token representations. Combined with a symplectic iteration CUDA solver, it serves as an adapter for pre-trained LLMs, achieving gains of up to +27.8% on MATH-500 and a 2-3x Pass@8 improvement on AMC/AIME.

## Background & Motivation

**Background**: Current mainstream sequence models (Transformers, SSMs, linear RNNs) share a core design principle—prediction based on associative memory. Attention retains the full KV cache for retrieval via query matching, while linear RNNs compress historical context into a fixed-size latent state for decoding. Both are essentially System 1-style fast pattern matching.

**Limitations of Prior Work**: Pure memory paradigms are limited in tasks requiring discovery, reasoning, and problem-solving. While Reinforcement Learning (RL) can make models more goal-oriented, RL serves only as an external training/post-training process and is absent during forward inference. Models learn "what to optimize" but do not learn "how to reason through planning" during the computation process.

**Key Challenge**: Memory architectures correspond to System 1 thinking, whereas System 2-style deliberation, multi-step planning, and long-range reasoning require specialized architectural support. RL training cannot break the reasoning ceiling imposed by memory architectures; planning capability remains external to the model.

**Goal**: Directly internalize planning into the model architecture, enabling LLMs to perform goal-oriented reasoning during the forward pass rather than relying on external training procedures.

**Key Insight**: The authors observe that LQR (Linear Quadratic Regulator) is an analytically solvable subclass of MDPs, and linear dynamical systems have been proven to express a wide family of MDPs. By modeling the next-token prediction of each layer as a differentiable finite-horizon LQR problem, planning can be performed natively during forward inference.

**Core Idea**: Replace pure memory retrieval with LQR planning from optimal control, allowing the model to "think about future trajectories" before prediction, thus realizing architectural System 2 reasoning.

## Method

### Overall Architecture

TTC-Net is a hybrid architecture: a TTC layer is inserted every 8 layers between the Attention and MLP modules of a pre-trained Transformer. Input token features are linearly projected to an initial latent state $\boldsymbol{h}_0$. The TTC layer constructs a finite-horizon LQR problem on this state and solves for the optimal first-step action $\boldsymbol{u}_1^*$ as the output. After normalization and linear projection, this is added back to the residual flow. The entire process is end-to-end differentiable, supporting training from scratch or fine-tuning on pre-trained models.

### Key Designs

1.  **Test-Time Control (TTC) Layer**:
    - **Function**: Performs finite-horizon optimal control planning during the forward pass, decoding memory states into optimal decisions.
    - **Mechanism**: Given the initial state $\boldsymbol{h}_0$ encoding the context, it constructs linear state transitions $\boldsymbol{h}_t = \boldsymbol{A}_t \boldsymbol{h}_{t-1} + \boldsymbol{B}_t \boldsymbol{u}_t$ and a quadratic cost function $\sum_{t=1}^{T}(\boldsymbol{h}_t^\top \boldsymbol{Q}_t \boldsymbol{h}_t + \boldsymbol{u}_t^\top \boldsymbol{R}_t \boldsymbol{u}_t)$. The optimal first-step action $\boldsymbol{u}_1^* = \boldsymbol{K}_1^* \boldsymbol{h}_0$ is solved via Riccati iteration. LQR parameters are dynamically generated from the context $\boldsymbol{h}_0$ via linear layers (contextualization) and parameterized with time-modulation coefficients $\boldsymbol{\Gamma}_\Box^t$ to achieve time-heterogeneous dynamics and costs. Backpropagation is fully differentiable by solving the dual LQR via the KKT system.
    - **Design Motivation**: Existing memory layers (Attention/SSM) can only recall information from past context. The TTC layer optimizes future trajectories and minimizes long-range costs, endowing each sequence block with an intrinsic value function $V_t(\boldsymbol{h}_t) = -\frac{1}{2}\boldsymbol{h}_t^\top \boldsymbol{P}_t \boldsymbol{h}_t$.

2.  **Symplectic Iteration Solver**:
    - **Function**: Replaces sequential matrix inversions in classical Riccati recursion with parallelizable matrix product chains, achieving over 10x throughput improvement.
    - **Mechanism**: Exploits the symplectic structure of LQR dynamics to reformulate the Riccati recursion as cumulative matrix products of symplectic matrices $\boldsymbol{\Sigma}_t$. Matrix inversions $\boldsymbol{A}_t^{-1}$ and $\boldsymbol{R}_t^{-1}$ at each timestep are independent and fully parallelizable, leaving only matrix multiplications (充分利用 Tensor Cores). By diagonalizing $\boldsymbol{A}_t$ and $\boldsymbol{R}_t$, dense matrix inversion is reduced from $O(T)$ to $O(1)$. Further fused into a CUDA kernel with row-level tiling and SRAM streaming, with row normalization for numerical stability.
    - **Design Motivation**: Classical Riccati solvers require sequential backward iteration for $T$ steps, each involving matrix inversion ($O(Td^3)$), which poorly matches GPU accelerators. Symplectic iteration shifts the bottleneck from inversion to multiplication.

3.  **Hybrid Architecture and Test-time Scaling**:
    - **Function**: Integrates the TTC layer as a lightweight adapter into pre-trained LLMs and supports flexible adjustment of the planning horizon during inference to enhance performance.
    - **Mechanism**: One TTC layer is inserted after every 8 Attention layers (8:1 interleaving ratio) using a multi-head structure (head size 16). During training, the planning horizon is sampled from a truncated Poisson log-normal distribution (mean $T_\mu=8$, max 32) to avoid distribution shifts caused by fixed horizons. During testing, the planning horizon $T_{test}$ can be arbitrarily increased; the model generalizes beyond the training maximum of 32 to $T=64$ with sustained performance gains. The output projection $\boldsymbol{W}_{out}$ is zero-initialized during fine-tuning to ensure the initial model matches the original backbone.
    - **Design Motivation**: TTC layers require rich memory states as input and must be interleaved with Attention. The mixed-horizon training strategy adapts the model to different planning depths, exposing an architecturally native test-time compute scaling axis.

### Loss & Training

A mixed-horizon training strategy is used: in each iteration, the planning horizon $T_{train}$ is sampled from a Poisson log-normal distribution with mean $T_\mu = 8$, log-standard deviation $T_\sigma = 0.1$, and a cap of 32. When fine-tuning on pre-trained models, the OpenThoughts2-114K dataset plus 800K self-collected reasoning samples are used for SFT, equivalent to imitation learning + inverse reinforcement learning.

## Key Experimental Results

### Main Results — Mathematical Reasoning (Finetuned on Llama-3-Instruct-7B)

| Model | MATH-500 | AMC Acc@8 | AMC Pass@8 | AIME24 Acc@8 | AIME24 Pass@8 | AIME25 Pass@8 |
|------|----------|-----------|------------|-------------|---------------|---------------|
| Base model | 25.00 | 6.63 | 31.32 | 0.00 | 0.00 | 0.00 |
| Full Finetuning | 46.80 | 20.78 | 46.98 | 1.67 | 6.67 | 0.00 |
| + Attention | 47.00 | 20.48 | 44.58 | 0.42 | 3.33 | 6.67 |
| + Mamba | 44.80 | 22.29 | 44.58 | 0.83 | 3.33 | 3.33 |
| + GDN | 47.80 | 17.77 | 37.35 | 0.42 | 3.33 | 6.67 |
| + MesaNet | 47.40 | 12.65 | 27.71 | 1.25 | 10.00 | 0.00 |
| **Ours (TTC-Net)** | **52.80** | **23.34** | **54.22** | **3.33** | **20.00** | **20.00** |

### Ablation Study — MATH-500

| Configuration | $T_{test}=8$ | $T_{test}=16$ | Description |
|------|------------|-------------|------|
| Time-homogeneous parameterization | 48.40 | 45.70 | Removing time modulation; performance drops when increasing horizon |
| Fixed training horizon | 50.60 | 31.50 | Fails to generalize to larger test horizons |
| Uniformly sampled horizon | 50.80 | 51.00 | Similar effect but doubles training cost |
| Attn:TTC = 4:1 | 53.00 | — | More TTC layers improve performance but increase computation |
| Attn:TTC = 16:1 | 47.20 | — | Performance drops with too few TTC layers |
| **Ours (PLN + 8:1)** | **52.80** | **53.60** | Optimal balance, generalizes up to $T=64$ |

## Highlights & Insights

- **Architectural Paradigm Shift**: For the first time, reasoning is redefined from "memory retrieval" to "optimal control," providing an architectural implementation of System 2 cognition for LLM reasoning.
- **New Axis for Test-time Scaling**: The planning horizon $T$ provides a compute scaling axis orthogonal to the number of generated tokens; increasing $T$ consistently improves reasoning accuracy without retraining.
- **Breaking Reasoning Ceilings**: TTC-Net achieves a 0% to 20% Pass@8 breakthrough on AIME, indicating that control objectives provide an inductive bias that memory layers cannot reach.
- **Symplectic Iteration Solver**: Achieves over 10x throughput gain via algorithm-hardware co-design, making optimal control layers practically usable in large-scale LLMs.

## Limitations & Future Work

- Theoretical understanding of the joint dynamical behavior of multi-layer TTC is lacking, and the interaction mechanism between layers is unclear.
- Currently only validated on 7B models; the effects on larger models and full-stage training (Pre-training + RL) remain unknown.
- The linear dynamics and quadratic costs of LQR still have expressivity limits; non-linear MDP formulations might provide further improvements.
- The linear layers for parameter contextualization are relatively simple; more rich world model parameterizations are worth exploring.

## Related Work & Insights

- Contrast with TTT (Test-Time Training) series: TTT is test-time memory (self-supervised regression), while TTC is test-time decision-making (optimal control).
- Complementary to RL for LLMs (e.g., DeepSeek-R1): RL provides training-time objectives, while TTC internalizes these objectives into the architecture's forward pass.
- The design of the symplectic iteration solver can be generalized to other scenarios requiring optimization layers embedded within neural networks.
- Memory architectures like Titans and DeltaNet can be hybridized with TTC to explore richer memory-planning interactions.

## Rating

- Novelty: 9/10 — Paradigmatic innovation embedding optimal control as an architectural component in LLMs.
- Experimental Thoroughness: 7/10 — Sufficiently validated on Sudoku and math reasoning, but limited to 7B models and lacks NLP/code tasks.
- Writing Quality: 9/10 — Coherent narrative from cognitive science to control theory with rigorous mathematical derivation.
- Value: 8/10 — Opens a new architectural direction for LLM reasoning, though large-scale validation is needed for practical adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)
- [\[ICML 2026\] Conformal Thinking: Risk Control for Reasoning on a Compute Budget](conformal_thinking_risk_control_for_reasoning_on_a_compute_budget.md)
- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)
- [\[ICML 2026\] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)

</div>

<!-- RELATED:END -->
