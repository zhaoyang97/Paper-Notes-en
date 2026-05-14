---
title: >-
  [Paper Note] VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models
description: >-
  [CVPR 2026][LLM Reasoning][visual refocusing] This paper proposes VisRef, a training-free visual refocusing framework that, during inference in multimodal large reasoning models (MLRMs)…
tags:
  - "CVPR 2026"
  - "LLM Reasoning"
  - "visual refocusing"
  - "test-time scaling"
  - "multimodal reasoning"
  - "DPP"
  - "visual token selection"
  - "training-free"
date: 2026-05-08
content_hash: 751e318900d8c7e4
---

# VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models

**Conference**: CVPR 2026
**arXiv**: [2603.00207](https://arxiv.org/abs/2603.00207)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: visual refocusing, test-time scaling, multimodal reasoning, DPP, visual token selection, training-free

## TL;DR

This paper proposes VisRef, a training-free visual refocusing framework that, during inference in multimodal large reasoning models (MLRMs), adaptively selects a semantically relevant and visually diverse subset of tokens at each reasoning step via Determinantal Point Processes (DPP) and reinjects them into the context. An entropy-based stopping criterion prevents overthinking. Under a fixed compute budget, VisRef improves visual reasoning accuracy by up to 6.4%.

## Background & Motivation

**Background**: MLRMs such as InternVL-3.5, Qwen-3-VL, and SAIL-VL2 have achieved significant progress by extending Chain-of-Thought reasoning to vision-language tasks. However, recent work (Chu et al., Yang et al.) identifies a critical issue: as reasoning chain length increases, model attention to visual tokens progressively decays, and the model increasingly relies on textual priors rather than image content.

**Limitations of Prior Work**: (1) RL fine-tuning approaches (e.g., Look-Back) can teach models to autonomously "look back" at visual inputs, but require 60 GPU-hours of fine-tuning and large-scale annotated dataset construction, limiting scalability. (2) Existing test-time scaling methods (e.g., Budget Forcing, L1) are purely text-oriented—they prompt the model to continue reasoning via tokens such as "Wait"/"Think more" but do not actively maintain visual grounding, allowing visual information to continue decaying. (3) Naively reinjecting all visual tokens is computationally infeasible—InternVL-3.5-8B generates approximately 1,772 visual tokens per image versus approximately 615 text tokens per step, and full reinjection incurs a 2.3× inference latency overhead.

**Key Challenge**: Humans naturally alternate between "viewing" and "reasoning" when solving multimodal problems, yet current MLRMs do not revisit visual tokens after initial processing—the longer the reasoning chain, the weaker the visual grounding. Training-based solutions are effective but costly; purely textual test-time scaling does not address the fundamental problem.

**Goal**: Can visual grounding be fully restored at test time without any retraining or fine-tuning?

**Key Insight**: At each reasoning step, adaptively select a small subset (30%) of visual tokens that are most relevant to the current reasoning context and maximally diverse in visual coverage, then reinject them using a DPP framework that jointly optimizes relevance and diversity.

**Core Idea**: Formalize visual token selection as an optimization problem that maximizes the DPP determinant, enabling adaptive visual refocusing during reasoning without training—plug-and-play on any pretrained MLRM.

## Method

### Overall Architecture

Given a vision-language input $x_{\text{input}} = [I, T]$ and visual token set $\mathcal{V} = \{v_1, \ldots, v_N\}$, after each reasoning step $k$ produces a textual reasoning step $z_k$, VisRef: (1) selects a subset $V_k$ of $m$ visual tokens from $\mathcal{V}$ via DPP; (2) injects $V_k$ into the context for the next step; (3) checks whether the entropy of the model's response distribution falls below a threshold $\delta_{\text{entropy}}$ to determine termination. The final answer is $y \sim \pi_\theta(\cdot | x_{\text{input}}, \tau_{1:k})$, where $\tau_{1:k} = \{(z_1, V_1), \ldots, (z_k, V_k)\}$.

### Key Designs

1. **DPP-Based Visual Token Selection**
    - **Function**: At each reasoning step, select a token subset that is both relevant to the current reasoning state and diverse in visual coverage.
    - **Mechanism**: Define the text subspace geometry $M_k = \sum_{j=1}^{T_k} z_k^{(j)}(z_k^{(j)})^\top$, and construct the kernel $L_k(v_i, v_j) = v_i^\top M_k v_j$. The optimization objective is to maximize the kernel matrix determinant: $\tilde{V}_k = \arg\max_{V_k \subseteq \mathcal{V}} \det(L_k^{V_k})$. This determinant naturally decomposes into two terms:

$$\log\det(L_k^{V_k}) = \underbrace{\sum_{v_i \in V_k} \log(r_i^2)}_{\text{relevance}} + \underbrace{\log\det(\bar{L}_k^{V_k})}_{\text{diversity}}$$

where $r_i^2 = \sum_{j=1}^{T_k}(v_i^\top z_k^{(j)})^2$ measures relevance.
    - **Design Motivation**: Naively injecting all visual tokens incurs 2.3× latency; selecting only relevant tokens leads to redundancy; DPP jointly optimizes relevance and diversity for an optimal trade-off.

2. **Greedy Approximation**
    - **Function**: Efficiently solve the NP-hard subset selection problem.
    - **Mechanism**: Starting from the empty set, greedily select the token with the largest marginal gain at each iteration: $v_{k,i} = \arg\max_{v \in \mathcal{V} \setminus V_k^{(i-1)}} \log\frac{\det(L_k^{V_k^{(i-1)} \cup \{v\}})}{\det(L_k^{V_k^{(i-1)}})}$, repeated for $m$ iterations. The token budget is set to $m = \lfloor 0.3|\mathcal{V}| \rfloor$.
    - **Design Motivation**: Exact maximum-determinant subset selection is NP-hard; the greedy algorithm provides a $(1-1/e)$ approximation guarantee.

3. **Entropy-Based Adaptive Stopping Criterion**
    - **Function**: Prevent overthinking or under-thinking.
    - **Mechanism**: After each reasoning step $k$, compute the entropy of the model's response distribution: $H_k = -\mathbb{E}_{y \sim \pi_\theta}[\log \pi_\theta(y | x_{\text{input}}, \tau_{1:k})]$. Terminate when $H_k < \delta_{\text{entropy}} = 0.25$, indicating convergence to a high-confidence answer. A maximum step count $K_{\max} = 10$ prevents indefinite reasoning.
    - **Design Motivation**: Simple problems quickly reach low entropy and terminate early (saving compute); complex problems utilize more reasoning steps. $\delta_{\text{entropy}} = 0.25$ is consistently optimal across all models.

### Loss & Training

VisRef requires no training whatsoever. All operations are performed at inference time: visual token selection via DPP greedy algorithm, reinjection via context sequence modification, and stopping via entropy computation. The method is plug-and-play and applicable to any pretrained MLRM.

## Key Experimental Results

### Main Results (Three Visual Reasoning Benchmarks × Three MLRMs)

| Model | Method | MathVision | MathVista | MM-Star |
|-------|--------|-----------|-----------|---------|
| InternVL3.5-8B | Standard Thinking | 39.2 | 68.1 | 57.2 |
| InternVL3.5-8B | Textual Self-Reflection | 40.1 | 73.9 | 58.3 |
| InternVL3.5-8B | **VisRef** | **44.6 (+5.4)** | **79.3 (+11.2)** | **63.1 (+5.9)** |
| Qwen3-VL-8B | Standard Thinking | 53.8 | 74.1 | 66.5 |
| Qwen3-VL-8B | Textual Self-Reflection | 54.3 | 74.2 | 65.9 |
| Qwen3-VL-8B | **VisRef** | **56.6 (+2.8)** | **77.1 (+3.0)** | **69.1 (+2.6)** |
| SAIL-VL2-8B | Standard Thinking | 29.8 | 73.1 | 47.7 |
| SAIL-VL2-8B | Textual Self-Reflection | 31.9 | 73.8 | 48.9 |
| SAIL-VL2-8B | **VisRef** | **37.3 (+7.5)** | **78.2 (+5.1)** | **55.3 (+7.6)** |

### Ablation Study

Relevance vs. Diversity Ablation (InternVL-3.5-8B):

| Relevance | Diversity | MathVista | MathVision | MM-Star |
|-----------|-----------|-----------|-----------|---------|
| ✓ | ✗ | 75.6 | 43.3 | 61.0 |
| ✗ | ✓ | 77.4 | 42.9 | 62.8 |
| ✓ | ✓ | **79.3** | **44.6** | **63.1** |

Comparison with Training-Based Method Look-Back (InternVL-3.5-8B):

| Method | MathVista | MathVision | MM-Star |
|--------|-----------|-----------|---------|
| Standard Thinking | 68.1 | 39.2 | 57.2 |
| Look-Back (requires 60 GPU-hr) | 80.8 | 44.2 | 63.7 |
| VisRef (training-free) | 79.3 | 44.6 | 63.1 |
| Look-Back + VisRef | **83.1** | **48.2** | **66.0** |

### Key Findings

- Purely textual self-reflection (TSR) yields inconsistent gains (0.1%–2.1%) and even degrades performance by 0.6% on Qwen3-VL-8B's MM-Star, indicating that extending purely textual reasoning provides limited benefit for visual tasks.
- VisRef consistently outperforms baselines across all 9 configurations (3 models × 3 benchmarks), with a maximum gain of 11.2% (InternVL3.5 on MathVista).
- Relevance-only selection underperforms diversity-only selection (MathVista 75.6 vs. 77.4), demonstrating that diversity is critical for comprehensive visual coverage.
- Training-free VisRef achieves performance close to Look-Back (60 GPU-hr fine-tuning) on MathVista (79.3 vs. 80.8), and the two methods are orthogonal—their combination further improves performance to 83.1.
- Token budget $m=30\%$ is the optimal trade-off: 20% is insufficient (76.1%), 30% is optimal (79.2%), and 40% yields no additional benefit.
- Under a fixed token budget (e.g., 14K thinking tokens) in parallel chain settings, VisRef achieves approximately 6% higher accuracy than parallel inference without visual refocusing.

## Highlights & Insights

- **Theoretical Elegance**: Formalizing visual token selection as DPP determinant maximization and demonstrating its natural decomposition into relevance and diversity terms provides a mathematically clean justification for why DPP is well-suited to this problem.
- **Plug-and-Play**: No training data, no fine-tuning, and no architectural modifications are required—VisRef can be immediately applied to any pretrained MLRM, which is highly valuable for practical deployment.
- **Complementarity with Training-Based Methods**: The VisRef + Look-Back combination outperforms either method alone on all benchmarks, indicating that the two approaches capture distinct visual grounding signals.
- **Attention Visualization**: Figure 5 intuitively demonstrates how VisRef progressively refocuses attention from a diffuse state to task-critical visual regions across reasoning steps.

## Limitations & Future Work

- Each step requires DPP selection and reinjection computation; although only 30% of tokens are selected, context length and compute still increase.
- The current DPP kernel $L_k$ is based on a simple text subspace projection; more sophisticated cross-modal alignment may yield further improvements.
- Although the entropy threshold $\delta_{\text{entropy}} = 0.25$ is consistent across models, finer-grained adjustment may be needed for problems of varying difficulty and domain.
- Evaluation is limited to the 8B parameter scale; scaling behavior on larger models (e.g., 70B+) remains unverified.
- Gains on Qwen3-VL-8B are relatively modest (2–3%), possibly because that model already exhibits stronger visual grounding.

## Related Work & Insights

- **Budget Forcing (Muennighoff et al.)**: A test-time scaling method that extends reasoning chains via "Wait" prompts, but is purely text-oriented.
- **Look-Back (Chu et al.)**: Uses RL fine-tuning to teach models to autonomously revisit visual inputs; effective but costly (60 GPU-hr).
- **L1 (Aggarwal & Welleck)**: Length-controllable policy optimization for precise control of reasoning chain length.
- **Insight**: VisRef demonstrates that actively maintaining visual grounding is more effective than passively extending reasoning. The core insight is that for visual tasks, the problem is not insufficient thinking but insufficient looking.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (DPP framework for visual token selection is a first; theoretical derivation is elegant)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (3 models × 3 benchmarks; extensive ablations covering relevance/diversity decomposition, comparison with training-based methods, token budget analysis, and test-time scaling curves)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Problem motivation is clear, method derivation is rigorous, figures are information-dense)
- **Value**: ⭐⭐⭐⭐⭐ (Training-free plug-and-play + consistent and significant gains + orthogonal complementarity with training-based methods; high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](../../ACL2026/llm_reasoning/parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](../../NeurIPS2025/llm_reasoning/does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
