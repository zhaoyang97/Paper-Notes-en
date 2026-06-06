---
title: >-
  [Paper Note] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving
description: >-
  [ACL 2026][Reinforcement Learning][Latent Reasoning] This paper proposes SpiralThinker, a framework that implements implicit reasoning through iterative updates in the latent representation space interleaved with text re…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Latent Reasoning"
  - "Iterative Refinement"
  - "Text-Latent Interleaving"
  - "Progressive Alignment"
  - "Implicit Chain-of-Thought"
date: 2026-05-08
content_hash: c7b0ef60998bbf29
---

# SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving

**Conference**: ACL 2026  
**arXiv**: [2511.08983](https://arxiv.org/abs/2511.08983)  
**Code**: [GitHub](https://github.com/shengminp/SpiralThinker)  
**Area**: Reinforcement Learning  
**Keywords**: Latent Reasoning, Iterative Refinement, Text-Latent Interleaving, Progressive Alignment, Implicit Chain-of-Thought

## TL;DR

This paper proposes SpiralThinker, a framework that implements implicit reasoning through iterative updates in the latent representation space interleaved with text reasoning steps. It introduces a progressive alignment objective to ensure that latent representations maintain consistency with explicit reasoning during the iterative process, outperforming all latent reasoning baselines on math, logic, and commonsense reasoning tasks.

## Background & Motivation

**Background**: Progress in large reasoning models is primarily driven by reinforcement learning and test-time compute scaling. Another research direction explores "latent reasoning"—allowing reasoning to unfold in high-dimensional hidden representations rather than generating explicit text. Existing latent reasoning methods (e.g., Coconut, iCoT, Pause Token) have demonstrated preliminary feasibility.

**Limitations of Prior Work**: (1) Existing methods lack mechanisms to ensure stable reasoning dynamics in latent space—most treat latent representations as token-level inputs processed in a single forward pass, forcing them to encode all reasoning steps simultaneously; (2) There is a lack of systematic interleaving schemes for implicit and explicit reasoning—pure text reasoning leads to overthinking, while pure latent reasoning sacrifices interpretability and controllability; (3) Existing iterative methods rely solely on standard language modeling objectives with no direct supervision for latent reasoning dynamics.

**Key Challenge**: Unconstrained iterative updates in latent space suffer from drift. Unrestricted iteration may even degrade performance—ablation studies show that adding iteration without alignment constraints drops accuracy on ProsQA from 98.0% to 97.4%.

**Goal**: To design a stable iterative latent reasoning framework where latent representations are progressively enhanced across multiple iterations while maintaining consistency with text reasoning.

**Key Insight**: The iterative process naturally corresponds to multi-step reasoning (theoretically $T$ iterations can simulate $T$ reasoning steps), but explicit alignment signals are required to prevent latent representations from deviating from the reasoning trajectory.

**Core Idea**: Latent reasoning is modeled as an iterative refinement process. A progressive alignment objective constraints latent representations at each iteration to align with corresponding text reasoning steps, implemented through a structured tagging scheme for text-latent interleaving.

## Method

### Overall Architecture

SpiralThinker is trained in two stages: (1) **Explicit Reasoning Phase**—standard SFT to learn step-by-step reasoning; (2) **Implicit Reasoning Phase**—replacing text reasoning steps at even (or odd) positions with $N$ `<latent>` tokens. These latent representations are iteratively updated under progressive alignment constraints. During inference, the model automatically alternates between text and latent steps.

### Key Designs

1.  **Iterative Process**:
    *   **Function**: Gradually enhances the reasoning depth of latent representations through multiple iterations.
    *   **Mechanism**: In each iteration $k$, the representation $\mathbf{H}^{(L,k-1)}_{\text{<latent>}}$ corresponding to the latent tokens is extracted from the final hidden states of the previous iteration. It is transformed via a mapping module $g_\phi(\cdot)$ (a lightweight adapter), written back to the corresponding positions in the embedding sequence, and a full forward pass is executed. This is repeated for $K$ iterations.
    *   **Design Motivation**: A single forward pass forces latent tokens to encode all reasoning information at once; multiple iterations allow the model to deepen reasoning progressively, focusing on different aspects of reasoning in each iteration.

2.  **Latent Adapter**:
    *   **Function**: Aligns final-layer hidden states back to the embedding space.
    *   **Mechanism**: Composed of a residual MLP + RMSNorm + scaling: $\tilde{\mathbf{h}} = \text{norm}(\mathbf{h} + W_2 \text{SiLU}(W_1 \mathbf{h})) \cdot \text{target\_rms}$, where $\text{target\_rms}$ is calculated from the statistics of the pre-trained embedding matrix.
    *   **Design Motivation**: Final-layer hidden states and input embeddings reside in different subspaces; direct replacement causes distribution mismatch. The adapter ensures the mapped latent representations are consistent with the embedding space distribution.

3.  **Progressive Alignment**:
    *   **Function**: Provides layer-wise enhanced supervision signals for the iterative latent reasoning process.
    *   **Mechanism**: (a) Within each iteration, align hidden states at `<eol>` (end of latent step) and `<eot>` (end of text step) positions: $\mathcal{L}_{\text{align}} = \frac{1}{L}\sum_{l=1}^{L}\frac{\|\mathbf{H}^{(l)}_{\texttt{<eol>}} - \mathbf{H}^{(l)}_{\texttt{<eot>}}\|_1}{\sigma^{(l)}}$; (b) Use softmax-weighted aggregation across iterations, $\mathbf{v} = \text{softmax}(\alpha[1,...,K])$, where later iterations have higher weights—allowing exploration in early stages and precise alignment later.
    *   **Design Motivation**: Iteration without alignment leads to drift (Ablation: iteration alone drops StrategyQA by 1.31%); progressive weighting avoids over-constraining the early exploration phase.

### Loss & Training

The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{align\_prog}}$. `<latent>` tokens have no explicit text form, so their positions do not contribute to the CE loss. The base model is Llama-3.2-1B, fine-tuned using LoRA on 4×A100 GPUs.

## Key Experimental Results

### Main Results

| Method | GSM8K-Aug (%) | ProsQA (%) | StrategyQA (%) |
| :--- | :--- | :--- | :--- |
| iCoT-KD | 24.11 | 98.00 | 62.88 |
| Coconut | 49.85 | 97.80 | 60.00 |
| CODI | 51.02 | 80.80 | 60.70 |
| Pause Token | 53.37 | 95.80 | 57.64 |
| **Ours** | **56.56** | **99.40** | **63.32** |

### Ablation Study

| Alignment | Iteration | GSM8K-Aug | ProsQA | StrategyQA |
| :--- | :--- | :--- | :--- | :--- |
| ✗ | ✗ | 45.49 | 98.00 | 59.39 |
| ✓ | ✗ | 48.67 (+3.18) | 98.60 (+0.60) | 61.14 (+1.75) |
| ✗ | ✓ | 45.72 (+0.23) | 97.40 (-0.60) | 58.08 (-1.31) |
| ✓ | ✓ | **56.56 (+11.07)** | **99.40 (+1.40)** | **63.32 (+3.93)** |

### Key Findings

*   The combined effect of iteration and alignment far exceeds the sum of their independent contributions—on GSM8K-Aug, they provide gains of 0.23/3.18 individually vs. 11.07 together, indicating strong synergy.
*   **Adding iteration without alignment degrades performance** (ProsQA -0.6%, StrategyQA -1.31%), confirming that unconstrained iteration leads to drift.
*   Optimal numbers of latent tokens and iterations are dataset-specific: GSM8K-Aug performs best with $N=5/K=5$, while StrategyQA peaks at $N=6/K=3$.
*   Qualitative analysis shows latent tokens gradually converge to correct intermediate results—the third token stores intermediate calculations, while the first token encodes operators.

## Highlights & Insights

*   The ablation showing "unconstrained iteration is harmful" strongly proves the necessity of the alignment objective—iteration and alignment are complementary, not redundant.
*   The design of progressive alignment is elegant—allowing exploration early and enforcing convergence later corresponds to the "diverge-then-converge" cognitive process in reasoning.
*   The text-latent interleaving scheme provides a formalization for "when to reason implicitly vs. explicitly."

## Limitations & Future Work

*   Currently uses a fixed number of iterations for all reasoning steps, without dynamic adjustment based on difficulty.
*   The text-latent interleaving pattern (every other step) is fixed and does not learn when to switch to latent mode.
*   Only validated on 1B parameter models; effectiveness on larger models remains unknown.
*   The interpretability of latent reasoning is still limited—while embedding similarity can be analyzed, it is much less intuitive than text CoT.

## Related Work & Insights

*   **vs Coconut**: Coconut reasons in continuous space but via a single forward pass without iterative refinement; Ours introduces iteration + alignment.
*   **vs Pause Token**: Pause Token inserts learnable delay tokens without alignment supervision, leading to limited performance.
*   **vs CODI**: CODI aligns latent and text representations but lacks iteration and performs poorly on ProsQA (80.8% vs 99.4%).
*   **vs Universal Transformer**: UT iterates over text tokens, while Ours iterates over latent representations and interleaves with text reasoning.

## Rating

*   Novelty: ⭐⭐⭐⭐ The combination of iterative latent reasoning, progressive alignment, and text interleaving is novel, though individual components are not entirely new.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers three reasoning types, detailed ablations, hyperparameter analysis, and qualitative analysis, but limited to a 1B model.
*   Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear; ablation designs precisely validate each component's contribution.
*   Value: ⭐⭐⭐⭐ Provides a feasible path for the iteration of latent reasoning; results reveal the critical necessity of alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[ACL 2026\] AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](attnpo_attention-guided_process_supervision_for_efficient_reasoning.md)
- [\[CVPR 2026\] Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](../../CVPR2026/reinforcement_learning/seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)
- [\[ACL 2026\] Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions](controlling_multimodal_conversational_agents_with_coverage-enhanced_latent_actio.md)
- [\[ICLR 2026\] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](../../ICLR2026/reinforcement_learning/thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)

</div>

<!-- RELATED:END -->
