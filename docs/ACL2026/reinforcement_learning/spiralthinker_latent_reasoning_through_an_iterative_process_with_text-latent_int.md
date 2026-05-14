---
title: >-
  [Paper Note] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving
description: >-
  [ACL 2026][Reinforcement Learning][Latent Reasoning] This paper proposes SpiralThinker, a framework for implicit reasoning that performs iterative updates in the latent representation space interleaved with explicit text…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Latent Reasoning"
  - "Iterative Refinement"
  - "Text-Latent Interleaving"
  - "Progressive Alignment"
  - "Implicit Chain-of-Thought"
date: 2026-05-08
content_hash: 7317ce701456baf1
---

# SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving

**Conference**: ACL 2026
**arXiv**: [2511.08983](https://arxiv.org/abs/2511.08983)
**Code**: [GitHub](https://github.com/shengminp/SpiralThinker)
**Area**: Reinforcement Learning
**Keywords**: Latent Reasoning, Iterative Refinement, Text-Latent Interleaving, Progressive Alignment, Implicit Chain-of-Thought

## TL;DR

This paper proposes SpiralThinker, a framework for implicit reasoning that performs iterative updates in the latent representation space interleaved with explicit text reasoning steps. A progressive alignment objective is introduced to ensure latent representations remain consistent with explicit reasoning throughout the iterative process. SpiralThinker surpasses all latent reasoning baselines on mathematical, logical, and commonsense reasoning tasks.

## Background & Motivation

**Background**: Advances in large reasoning models have been primarily driven by reinforcement learning and test-time compute scaling. A parallel line of research explores *latent reasoning*—allowing reasoning to unfold in high-dimensional hidden representations rather than generating explicit text. Existing latent reasoning methods (e.g., Coconut, iCoT, Pause Token) have demonstrated preliminary feasibility.

**Limitations of Prior Work**: (1) Existing methods lack mechanisms to ensure stable reasoning dynamics in the latent space—most treat latent representations as token-level inputs processed in a single forward pass, forcing them to encode all reasoning steps simultaneously. (2) There is no systematic scheme for interleaving implicit and explicit reasoning—pure text reasoning leads to overthinking, while pure latent reasoning sacrifices interpretability and controllability. (3) Existing iterative methods rely solely on standard language modeling objectives, providing no direct supervision over latent reasoning dynamics.

**Key Challenge**: Unconstrained iterative updates in the latent space cause drift; unregulated iteration can even degrade performance—ablations show that adding iteration alone without alignment constraints drops accuracy from 98.0% to 97.4% on ProsQA.

**Goal**: Design a stable iterative latent reasoning framework in which latent representations are progressively enhanced across iterations while remaining consistent with textual reasoning.

**Key Insight**: Iterative processes naturally correspond to multi-step reasoning (theoretically, $T$ iterations can simulate $T$ reasoning steps), but explicit alignment signals are required to prevent latent representations from drifting away from the reasoning trajectory.

**Core Idea**: Model latent reasoning as an iterative refinement process. A progressive alignment objective constrains latent representations at each iteration to remain consistent with the corresponding text reasoning steps, and a structured annotation scheme enables text-latent interleaving.

## Method

### Overall Architecture

SpiralThinker employs a two-stage training pipeline: (1) **Explicit Reasoning Stage**—standard SFT to acquire step-by-step reasoning capabilities; (2) **Implicit Reasoning Stage**—text reasoning steps at even (or odd) positions are replaced by $N$ `<latent>` tokens, whose representations are iteratively updated subject to a progressive alignment constraint. At inference time, the model automatically alternates between text steps and latent steps.

### Key Designs

1. **Iterative Latent Update (Iterative Process)**:

    - Function: Progressively deepen the reasoning capacity of latent representations through multiple iterations.
    - Mechanism: At each iteration $k$, the latent-token representations $\mathbf{H}^{(L,k-1)}_{\texttt{<latent>}}$ are extracted from the final hidden states of the previous iteration, transformed by a mapping module $g_\phi(\cdot)$ (a lightweight adapter), and written back into the corresponding positions of the embedding sequence. A full forward pass then yields updated hidden states. This process is repeated for $K$ iterations.
    - Design Motivation: A single forward pass forces latent representations to encode all reasoning information simultaneously. Multiple iterations allow the model to progressively deepen its reasoning, with each iteration focusing on different aspects of the reasoning process.

2. **Latent Adapter**:

    - Function: Align final-layer hidden states back to the embedding space.
    - Mechanism: Composed of a residual MLP + RMSNorm + scaling: $\tilde{\mathbf{h}} = \text{norm}(\mathbf{h} + W_2 \text{SiLU}(W_1 \mathbf{h})) \cdot \text{target\_rms}$, where $\text{target\_rms}$ is computed from the root-mean-square statistics of the pretrained embedding matrix.
    - Design Motivation: Final-layer hidden states and input embeddings reside in different subspaces; direct substitution causes distributional mismatch. The adapter ensures that mapped latent representations are consistent with the embedding space distribution.

3. **Progressive Alignment Objective**:

    - Function: Provide progressively strengthened supervision signals for the iterative latent reasoning process.
    - Mechanism: (a) Within each iteration, hidden states at `<eol>` (end of latent step) and `<eot>` (end of text step) positions are aligned: $\mathcal{L}_{\text{align}} = \frac{1}{L}\sum_{l=1}^{L}\frac{\|\mathbf{H}^{(l)}_{\texttt{<eol>}} - \mathbf{H}^{(l)}_{\texttt{<eot>}}\|_1}{\sigma^{(l)}}$; (b) Across iterations, softmax-weighted aggregation is applied, $\mathbf{v} = \text{softmax}(\alpha[1,...,K])$, assigning larger weights to later iterations—encouraging diverse exploration in early iterations and precise alignment in later ones.
    - Design Motivation: Unconstrained iteration causes drift (ablation: iteration alone degrades ProsQA by 0.6%). Progressive weighting avoids over-constraining the exploration phase in early iterations.

### Loss & Training

The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{align\_prog}}$. `<latent>` tokens have no explicit textual form and their positions are excluded from the CE loss. The backbone model is Llama-3.2-1B, fine-tuned with LoRA on 4×A100 GPUs.

## Key Experimental Results

### Main Results

| Method | GSM8K-Aug (%) | ProsQA (%) | StrategyQA (%) |
|--------|--------------|-----------|---------------|
| iCoT-KD | 24.11 | 98.00 | 62.88 |
| Coconut | 49.85 | 97.80 | 60.00 |
| CODI | 51.02 | 80.80 | 60.70 |
| Pause Token | 53.37 | 95.80 | 57.64 |
| **SpiralThinker** | **56.56** | **99.40** | **63.32** |

### Ablation Study

| Alignment | Iteration | GSM8K-Aug | ProsQA | StrategyQA |
|-----------|-----------|-----------|--------|------------|
| ✗ | ✗ | 45.49 | 98.00 | 59.39 |
| ✓ | ✗ | 48.67 (+3.18) | 98.60 (+0.60) | 61.14 (+1.75) |
| ✗ | ✓ | 45.72 (+0.23) | 97.40 (-0.60) | 58.08 (-1.31) |
| ✓ | ✓ | **56.56 (+11.07)** | **99.40 (+1.40)** | **63.32 (+3.93)** |

### Key Findings

- The joint effect of iteration and alignment far exceeds the sum of their individual contributions—on GSM8K-Aug, individual gains are 0.23 and 3.18 respectively, while the joint gain is 11.07, indicating strong synergy.
- **Iteration alone without alignment degrades performance** (ProsQA −0.6%, StrategyQA −1.31%), confirming that unconstrained iteration indeed causes drift.
- Optimal latent token count and iteration number are dataset-specific: GSM8K-Aug favors $N{=}5$/$K{=}5$; StrategyQA favors $N{=}6$/$K{=}3$.
- Qualitative analysis shows that latent tokens progressively converge to correct intermediate results across iterations—the third token stores intermediate computation values, while the first token encodes operators.

## Highlights & Insights

- The ablation result that "unconstrained iteration is harmful" strongly validates the necessity of the alignment objective—iteration and alignment are complementary rather than redundant.
- The design of progressive alignment is elegant: early iterations permit exploration while later iterations enforce convergence, mirroring the cognitive process of "divergent then convergent" reasoning.
- The text-latent interleaving scheme provides a viable formalization of when to reason implicitly versus explicitly.

## Limitations & Future Work

- The current approach uses a fixed number of iterations for all reasoning steps, without dynamically adjusting based on difficulty.
- The text-latent alternation pattern (every other step) is fixed; the model does not learn when to switch to the latent mode.
- Experiments are conducted only on a 1B parameter model; effectiveness at larger scales remains unknown.
- Interpretability of latent reasoning remains limited—while embedding similarity analysis is possible, it is far less transparent than textual chain-of-thought.

## Related Work & Insights

- **vs. Coconut**: Coconut reasons in continuous space but uses a single forward pass without iterative refinement; SpiralThinker introduces iteration and alignment.
- **vs. Pause Token**: Pause Token inserts learnable delay tokens but provides no alignment supervision, resulting in limited performance.
- **vs. CODI**: CODI aligns latent and text representations but lacks iteration, and performs poorly on ProsQA (80.8% vs. 99.4%).
- **vs. Universal Transformer**: Universal Transformer iterates over text tokens, whereas SpiralThinker iterates over latent representations interleaved with text reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of iterative latent reasoning + progressive alignment + text interleaving is novel, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three reasoning types, detailed ablations, hyperparameter analysis, and qualitative analysis are provided, but only a 1B model is evaluated.
- Writing Quality: ⭐⭐⭐⭐⭐ Method motivation is clear, and ablation design precisely validates the contribution of each component.
- Value: ⭐⭐⭐⭐ Provides a viable path for iterative latent reasoning; ablations reveal the necessity of alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[ACL 2026\] Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions](controlling_multimodal_conversational_agents_with_coverage-enhanced_latent_actio.md)
- [\[ACL 2026\] AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](attnpo_attention-guided_process_supervision_for_efficient_reasoning.md)
- [\[CVPR 2026\] Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](../../CVPR2026/reinforcement_learning/seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)
- [\[ICLR 2026\] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](../../ICLR2026/reinforcement_learning/thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)

</div>

<!-- RELATED:END -->
