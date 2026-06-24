---
title: >-
  [Paper Note] Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling
description: >-
  [ACL 2025 (Outstanding Paper)][LLM (Other)][Token Recycling] Proposes Token Recycling—a training-free speculative decoding method that stores rejected candidate tokens during the decoding process into a lightweight adjacency matrix, constructs a draft tree via a BFS algorithm, and verifies it utilizing tree attention. It achieves approximately a $2\times$ speedup across all LLM scales with less than 2MB of storage overhead.
tags:
  - "ACL 2025 (Outstanding Paper)"
  - "LLM (Other)"
  - "Token Recycling"
  - "Speculative Decoding"
  - "Adjacency Matrix"
  - "BFS Tree Construction"
  - "Training-Free Acceleration"
date: 2026-05-08
content_hash: 02e49e8625c896b2
---

# Turning Trash into Treasure: Accelerating Inference of Large Language Models with Token Recycling

**Conference**: ACL 2025 (Outstanding Paper)  
**arXiv**: [2504.15754](https://arxiv.org/abs/2504.15754)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Token Recycling, Speculative Decoding, Adjacency Matrix, BFS Tree Construction, Training-Free Acceleration  

## TL;DR

Proposes Token Recycling—a training-free speculative decoding method that stores rejected candidate tokens during the decoding process into a lightweight adjacency matrix, constructs a draft tree via a BFS algorithm, and verifies it utilizing tree attention. It achieves approximately a $2\times$ speedup across all LLM scales with less than 2MB of storage overhead.

## Background & Motivation

**Background**: The rapid growth of LLM parameter scales has made inference latency a core bottleneck. Speculative decoding provides lossless acceleration through a "draft-then-verify" paradigm and has become a mainstream direction for LLM inference acceleration. Existing methods fall into two main categories: draft-model-based methods (such as Medusa and EAGLE) which require additional training, and retrieval-based methods (such as REST) which rely on large n-gram retrieval databases.

**Limitations of Prior Work**: Draft-model-based methods require training small models as drafters, incurring high training costs and needing separate training for each target model. Retrieval-based methods (such as REST) require gigabyte-scale storage to build domain-specific n-gram datastores, which suffer from slow retrieval and poor cross-domain adaptability. Both approaches introduce significant resource overheads, limiting the widespread application of speculative decoding in practical deployments.

**Key Challenge**: The core value of speculative decoding lies in "zero-cost" inference acceleration, but existing methods introduce significant training or storage costs themselves—the overhead of the acceleration method offsets its benefits, making it counterproductive especially in small-scale deployment scenarios.

**Goal**: Design a truly training-free, extremely low-storage speculative decoding method that: (1) requires no additional draft model or training processes; (2) requires no pre-built large retrieval datastores; (3) has minimal storage overhead ($<2\text{MB}$); and (4) adaptively adjusts to the current task and domain.

**Key Insight**: Candidate tokens sampled during decoding but ultimately rejected are "incorrect" for the current step, yet they contain rich next-token probability information—these "discarded" tokens have a high probability of appearing in future sequences. The key insight is to treat these rejected tokens as recyclable "treasure," storing their successor relationships via an adjacency matrix to provide high-quality draft candidates for subsequent steps.

**Core Idea**: Recycle rejected candidate tokens from the decoding process and store them in a lightweight adjacency matrix, and construct a multi-path draft tree via BFS to achieve training-free, adaptive speculative decoding acceleration.

## Method

### Overall Architecture

Token Recycling maintains a lightweight adjacency matrix during inference to record successor relationships between tokens. Each decoding step consists of four stages: (1) **Collection Phase**—stores sampled but unselected candidate tokens (next-tokens with top-$k$ probabilities) from the current step into the adjacency matrix, recording the $k$ most likely successor tokens and their probabilities for each token; (2) **Construction Phase**—starting from the currently accepted tokens, performs a breadth-first search (BFS) on the adjacency matrix to construct a draft tree containing multiple potential paths, expanding the most likely successor nodes at each layer; (3) **Verification Phase**—feeds the entire draft tree into the target LLM in one forward pass via tree attention, using a special tree attention mask to ensure causality, and accepts token paths consistent with the target LLM's distribution (guaranteeing losslessness); (4) **Update Phase**—newly generated candidate tokens after verification update the adjacency matrix again, forming a closed loop of online learning. This entire process starts from scratch, and the adjacency matrix is progressively enriched during decoding.

### Key Designs

1. **Adjacency Matrix Storage**:
    - **Function**: Records global token successor relationships with extremely low storage overhead.
    - **Mechanism**: The matrix size is $vocabulary\_size \times k$ (where $k$ is the maximum number of successors kept per token), storing the most likely successor tokens and their probabilities for each token.
    - **Design Motivation**: Requires only $<2\text{MB}$ of storage (compared to gigabyte-scale retrieval methods), dynamically updates during decoding, and naturally adapts to the current task and domain without pre-building domain-specific databases.

2. **BFS Tree Construction**:
    - **Function**: Generates high-quality, multi-path draft trees starting from the current token.
    - **Mechanism**: Performs a breadth-first search in the adjacency matrix, expanding successor nodes ordered by probability at each layer; the generated draft tree covers multiple parallel speculative paths.
    - **Design Motivation**: The tree depth and width can be flexibly adjusted based on the computational budget—BFS ensures high-probability paths are expanded first, balancing draft coverage with verification computational overhead.

3. **Tree Attention Parallel Verification**:
    - **Function**: Verifies multiple speculative paths in a single forward pass.
    - **Mechanism**: Feeds the entire draft tree into the target LLM using a specialized tree-attention mask, simultaneously evaluating token probabilities for all paths in one forward pass.
    - **Design Motivation**: The natural compatibility with tree attention allows the BFS-constructed multi-path trees to be verified efficiently in parallel, eliminating the need for sequential path-by-path checks.

### Loss & Training

Completely training-free—the entire method self-organizes at inference time. The adjacency matrix starts as empty and is updated online during the decoding process. The adjacency matrix converges to an effective state after decoding approximately 100+ tokens.

## Key Experimental Results

### Main Results

| Baseline Methods | Type | Token Recycling Advantage |
|---------|------|---------------------|
| vs. Training-free Methods (REST, n-gram retrieval) | Training-free | **Speedup improvement +30%** |
| vs. Training-based Methods (Medusa, EAGLE) | Training required | **Speedup improvement +25%** |
| Storage Overhead | - | **$<2\text{MB}$** (REST requires GB-scale) |
| Applicable LLM Scales | - | Achieves **$\sim 2\times$ speedup** across all scales |

### Speedup Across Different Tasks

| Task Type | Speedup |
|----------|--------|
| Dialogue Generation | $\sim 2.0\times$ |
| Code Generation | $\sim 2.0\times$ |
| Translation | $\sim 1.8\times$ |

### Ablation Study

| Ablation Dimension | Findings |
|----------|------|
| Convergence Speed of Adjacency Matrix | Converges to an effective state after decoding 100+ tokens |
| BFS Tree Width/Depth | Optimal balance point exists—too wide wastes verification computation, too deep yields low matching probability |
| Cross-task Adaptability | Online updates of the adjacency matrix allow the method to automatically adapt to different task types |
| Cold Start Phase | Speedup is relatively low during the first ~100 tokens, then rapidly improves |

### Key Findings

- Candidate tokens discarded during decoding exhibit a highly significant recurrence rate, validating the core hypothesis of "rejected token recycling."
- The online learning property of the adjacency matrix naturally endows the method with domain adaptability, eliminating the need for pre-built domain-specific databases.
- Token Recycling is effective across all tested LLM scales ($7\text{B} \sim 70\text{B}+$ ), maintaining a stable speedup of **$1.8\times \sim 2.2\times$**.

## Highlights & Insights

- **The insight of "recycling discarded tokens" is remarkably simple yet elegant**: Candidate tokens naturally produced during decoding contain rich next-token probability information but have historically been discarded. This paper turns this "trash" into "treasure" for inference acceleration.
- **Training-Free + Extremely Low Storage**: A $<2\text{MB}$ adjacency matrix achieves $2\times$ speedup. Compared to methods that require extra training for draft models or maintaining large retrieval stores, the practicality is vastly improved.
- **Strong Adaptability**: The adjacency matrix is updated online during decoding, naturally adapting to the current context, task, and domain.
- **Seamless Integration with Tree Attention**: The multi-path draft tree constructed via BFS can be efficiently verified in parallel using tree attention.

## Limitations & Future Work

- At the start of a conversation, the adjacency matrix is empty, requiring a "cold start" period of roughly 100 tokens.
- The adjacency matrix is based on token-level local successor relationships and cannot capture long-range semantic dependencies.
- In high-temperature sampling (high-stochasticity generation) scenarios, the recurrence rate of tokens decreases, which may reduce the acceleration effect.
- The hybrid use with model-based speculative decoding methods (such as Medusa/EAGLE) has not yet been explored.

## Related Work & Insights

- **vs. Draft-model methods like Medusa/EAGLE**: The latter require training a small model as a drafter, whereas Token Recycling is completely training-free and highly general.
- **vs. Retrieval-augmented speculative decoding like REST**: The latter requires GB-scale retrieval datastores, whereas Token Recycling only requires $<2\text{MB}$ and updates adaptively.
- **vs. N-gram matching methods**: The latter rely on static n-gram tables, whereas the dynamic updates of the adjacency matrix in Token Recycling provide much stronger adaptability.
- **Insights**: The online learning paradigm of adjacency matrices can be extended to other scenarios requiring pattern prediction; it shares a common theme with NSA—making full use of "already computed" byproducts.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](../../ICML2026/llm_nlp/compute_as_teacher_turning_inference_compute_into_reference-free_supervision.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](../../ACL2026/llm_nlp/from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)
- [\[NeurIPS 2025\] StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Model](../../NeurIPS2025/llm_nlp/streambridge_turning_your_offline_video_large_language_model_into_a_proactive_st.md)
- [\[ACL 2025\] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](token_granularity_impact.md)
- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)

</div>

<!-- RELATED:END -->
