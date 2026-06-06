---
title: >-
  [Paper Note] CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning
description: >-
  [ACL 2026][LLM Reasoning][Chain-of-Thought Compression] The CRISP framework is proposed, discovering that the attention patterns of the `</think>` token reliably distinguish between key and redundant steps in reasoning c…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Chain-of-Thought Compression"
  - "Attention Saliency"
  - "Reasoning Redundancy"
  - "Greedy Search"
  - "Efficient Inference"
date: 2026-05-08
content_hash: 040cc19bbe4055b9
---

# CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning

**Conference**: ACL 2026  
**arXiv**: [2604.17297](https://arxiv.org/abs/2604.17297)  
**Code**: [GitHub](https://github.com/)  
**Area**: LLM Reasoning Efficiency  
**Keywords**: Chain-of-Thought Compression, Attention Saliency, Reasoning Redundancy, Greedy Search, Efficient Inference

## TL;DR

The CRISP framework is proposed, discovering that the attention patterns of the `</think>` token reliably distinguish between key and redundant steps in reasoning chains. Based on this, a greedy search compression pipeline with four atomic operations is designed, reducing token usage by 50-60% while maintaining accuracy.

## Background & Motivation

**Background**: Reasoning LLMs (e.g., DeepSeek-R1, OpenAI o1) achieve powerful reasoning capabilities by generating long chain-of-thought (CoT), but this also incurs significant computational cost and latency. CoT compression is essential for practical deployment.

**Limitations of Prior Work**: Existing CoT compression methods typically rely on external proxy models (such as independent LLMs) to evaluate and prune reasoning steps. However, external compressors are often misaligned with the source model's intrinsic reasoning dynamics—they frequently misjudge critical intermediate steps, such as self-correction, as redundant, thereby breaking the logical coherence of the reasoning chain.

**Key Challenge**: There is a need to find a signal to distinguish "key logical steps" from "redundant steps" without relying on external models (to avoid misalignment), utilizing the model's own internal mechanism instead.

**Goal**: Leverage the model's own internal signals (rather than external proxies) to guide CoT compression.

**Key Insight**: It is observed that the `</think>` token acts as an "information anchor" in deep attention layers—the model primarily focuses on the `</think>` position rather than intermediate reasoning steps when generating the final answer. The attention distribution of `</think>` accurately reflects the contribution of each reasoning step to the final answer.

**Core Idea**: Utilize the attention patterns of the `</think>` token as an intrinsic indicator of step saliency. Construct compressed reasoning paths through greedy search using four atomic operations (Keep, Prune, Rewrite, Fuse), and then use an LLM refiner to restore grammatical coherence.

## Method

### Overall Architecture

CRISP consists of three stages: (1) Original CoT generation—obtaining complete reasoning trajectories from the source model; (2) Key reasoning path search—using `</think>` attention to evaluate step saliency and compressing the reasoning chain through dynamic operators; (3) Refinement and fine-tuning—using an LLM to restore semantic coherence of the compressed path and then fine-tuning the target model with a multi-task objective.

### Key Designs

1.  **Discovery of `</think>` as an Information Anchor**:
    *   **Function**: Provides a step saliency signal without requiring an external model.
    *   **Mechanism**: Attention visualization reveals that in deep layers, the `</think>` token gradually aggregates information from the preceding reasoning chain, and the model primarily attends to the `</think>` position during final answer generation. Step saliency $S_i$ is defined as the normalized sum of attention weights from `</think>` to tokens in step $r_i$ across all layers and heads. High-attention steps encode critical information (pruning them causes PPL to spike), while low-attention steps can be safely removed (PPL increases only slightly).
    *   **Design Motivation**: External proxies are misaligned with the source model's reasoning dynamics, whereas the attention pattern of `</think>` is a direct reflection of what the source model itself "considers important."

2.  **Greedy Search via Four Atomic Operations**:
    *   **Function**: Flexibly compresses the reasoning chain under saliency guidance.
    *   **Mechanism**: Four operations are defined: Keep (retain high-saliency steps), Prune (remove low-saliency steps), Rewrite (simplify steps using an LLM), and Fuse (merge semantically repetitive steps). The dynamic action space allows operations based on saliency scores and semantic similarity constraints. The reward function $R(a) = \log P_\theta(y|x, \mathcal{C} \oplus a(r_i)) - \log P_\theta(y|x, \mathcal{C}) - \beta \cdot \text{Len}(a(r_i))$ balances answer likelihood gain and length penalty.
    *   **Design Motivation**: Simple threshold filtering might break logical dependencies or retain redundancy; the four operations provide a continuous compression granularity from full retention to complete removal.

3.  **Compressed Path Refinement and Multi-task Fine-tuning**:
    *   **Function**: Restores semantic coherence of the compressed path and trains the model.
    *   **Mechanism**: Skeletal paths from greedy search may have grammatical breaks; a high-level LLM refiner restores fluency using the original CoT as a reference. Fine-tuning employs a multi-task strategy with a control token $\kappa$: inputs with $\kappa$ generate compressed reasoning, while those without generate full reasoning, avoiding catastrophic forgetting.
    *   **Design Motivation**: Discrete search operations (especially Prune and Fuse) may introduce logical gaps, requiring a refinement step for restoration.

### Loss & Training

A standard auto-regressive negative log-likelihood loss is used, training on a mixture of original and compressed trajectories. 3 epochs, learning rate $1 \times 10^{-5}$, based on 2,500 samples from the MATH dataset. Attention thresholds $\tau_{\text{high}}$ and $\tau_{\text{low}}$ are set to the top 30% and bottom 20% quantiles, respectively.

## Key Experimental Results

### Main Results

| Method | Model | GSM8K Acc | GSM8K Tok | MATH-500 Acc | MATH-500 TE |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Original | 1.5B | 81.6 | 1669 | 78.2 | 2.22 |
| **Ours** | 1.5B | **80.6** | **587** | **75.0** | **4.14** |
| Original | 7B | 90.8 | 1376 | 87.4 | 2.86 |
| **Ours** | 7B | **90.1** | **374** | **84.2** | **7.35** |

### Ablation Study

| Method | 1.5B Avg TE | 7B Avg TE | Description |
| :--- | :--- | :--- | :--- |
| Original | 2.10 | 2.81 | Baseline |
| CoD (Prompting) | 2.61 | 4.31 | Insufficient granularity control |
| TALE (External) | 2.31 | 3.15 | External misalignment |
| A*-Thought | 2.99 | 4.04 | Search without intrinsic signal |
| **Ours** | **4.31** | **6.80** | Optimal efficiency-accuracy trade-off |

### Key Findings

*   Ours significantly leads all baselines in Token Efficiency (TE) (6.80 vs. 4.31 for the next best on the 7B model).
*   On the 7B model, GSM8K uses only 374 tokens (compared to the original 1376), with an accuracy drop of only 0.7%.
*   The `</think>` attention validation experiment is clear: pruning high-attention steps causes PPL to spike, while pruning low-attention steps leaves PPL almost unchanged.
*   Saliency scores show a non-uniform distribution, with only a few steps contributing highly to the final answer.

## Highlights & Insights

*   **The discovery of `</think>` as an information anchor is highly insightful**: It reveals how the internal attention mechanism of reasoning models "summarizes" the entire reasoning process, a finding with independent value for understanding reasoning models.
*   **The design of four atomic operations provides flexible compression granularity**: It is more refined than simple keep/delete, as Fuse and Rewrite allow for information retention while compressing.
*   **The adoption of the Token Efficiency metric enables a quantifiable comparison of the efficiency-accuracy trade-off.**

## Limitations & Future Work

*   The computational overhead of greedy search (evaluating multiple operations per step) may become a bottleneck for extremely long CoTs.
*   The refinement step depends on external LLMs, introducing additional costs.
*   Validation was only performed on mathematical reasoning datasets; generalization to code and logical reasoning has not been tested.
*   The multi-task training strategy with control tokens is relatively simple; better training schemes may exist.

## Related Work & Insights

*   **vs CoD/TALE (Prompting/External Compression)**: CoD limits length via prompting but lacks fine-grained control; TALE uses external models for compression but introduces misalignment. Ours utilizes the model's own attention signals, avoiding misalignment at the source.
*   **vs RL Methods (e.g., Length Penalty)**: RL methods are computationally expensive and sensitive to reward design. Ours avoids RL instability through post-processing compression.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ The discovery of the `</think>` information anchor is original, and the greedy search design with four operations is sophisticated.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Two model scales, three benchmarks, and multiple baselines, though domain coverage is limited.
*   Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, discovery is engaging, and experiments are well-organized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning](render-of-thought_rendering_textual_chain-of-thought_as_images_for_visual_latent.md)
- [\[NeurIPS 2025\] Inference-Time Chain-of-Thought Pruning with Latent Informativeness Signals](../../NeurIPS2025/llm_reasoning/inference-time_chain-of-thought_pruning_with_latent_informativeness_signals.md)
- [\[ACL 2026\] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization](is_chain-of-thought_really_not_explainability_chain-of-thought_can_be_faithful_w.md)
- [\[ACL 2026\] DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models](drp_distilled_reasoning_pruning_with_skill-aware_step_decomposition_for_efficien.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)

</div>

<!-- RELATED:END -->
