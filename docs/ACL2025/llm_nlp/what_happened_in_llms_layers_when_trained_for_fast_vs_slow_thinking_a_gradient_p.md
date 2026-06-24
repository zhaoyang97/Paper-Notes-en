---
title: >-
  [Paper Note] What Happened in LLM Layers when Trained for Fast vs. Slow Thinking: A Gradient Perspective
description: >-
  [ACL 2025][LLM (Other)][Gradient Analysis] This paper systematically investigates the behavioral differences of LLM layers when trained on fast thinking (no/short CoT) vs. slow thinking (detailed CoT) data from a gradient perspective. It reveals that slow thinking training leads to more uniform and stable gradients across layers, whereas fast thinking leads to larger gradients and more severe layer-wise fluctuations. Furthermore, the gradient patterns of slow thinking can dis…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Gradient Analysis"
  - "Fast and Slow Thinking"
  - "Layer-wise Analysis"
  - "Chain-of-Thought"
  - "LLM Training Dynamics"
date: 2026-05-08
content_hash: 56656ca7115ab0d0
---

# What Happened in LLM Layers when Trained for Fast vs. Slow Thinking: A Gradient Perspective

**Conference**: ACL 2025  
**arXiv**: [2410.23743](https://arxiv.org/abs/2410.23743)  
**Code**: [https://github.com/MingLiiii/Layer_Gradient](https://github.com/MingLiiii/Layer_Gradient)  
**Area**: LLM NLP / Interpretability  
**Keywords**: Gradient Analysis, Fast and Slow Thinking, Layer-wise Analysis, Chain-of-Thought, LLM Training Dynamics

## TL;DR

This paper systematically investigates the behavioral differences of LLM layers when trained on fast thinking (no/short CoT) vs. slow thinking (detailed CoT) data from a gradient perspective. It reveals that slow thinking training leads to more uniform and stable gradients across layers, whereas fast thinking leads to larger gradients and more severe layer-wise fluctuations. Furthermore, the gradient patterns of slow thinking can distinguish correct from irrelevant reasoning paths.

## Background & Motivation

**Background**: The interpretability research of LLMs is flourishing, including attention visualization, probing methods, layer pruning analysis, etc. Concurrently, slow thinking training methods based on Chain-of-Thought (CoT) have achieved immense success in reasoning tasks, with models such as DeepSeek-R1 and o1 demonstrating the powerful effects of detailed reasoning paths.

**Limitations of Prior Work**: Although existing studies have illustrated how different layers capture various types of information, direct analysis of layer-wise gradients during training remains insufficient—these gradients directly reflect how LLMs perceive and align with training data. In particular, the underlying mechanism of why slow thinking (detailed CoT) outperforms fast thinking (no CoT) remains unclear.

**Key Challenge**: While the effectiveness of CoT training is well-recognized, "what it does" inside LLMs remains unknown. Specifically: (1) what distinct training dynamics do different layers exhibit when learning fast/slow thinking data? (2) Can gradient patterns reflect the correctness of responses? (3) Can findings on reasoning tasks generalize to knowledge retrieval tasks?

**Goal**: By directly analyzing the gradient characteristics of LLM layers, this paper aims to reveal the divergent dynamics induced by fast vs. slow thinking training in model interiors, offering a new perspective for understanding the efficacy of CoT training.

**Key Insight**: The authors leverage Singular Value Decomposition (SVD) to characterize the properties of layer-wise gradient matrices, utilizing the nuclear norm as a metric for gradient magnitude, and design metrics such as Mean Absolute Difference (MAD) and Relative Difference (RD) between layers to quantify gradient stability and discriminability.

**Core Idea**: Through SVD nuclear norm analysis on layer-wise gradients, this study reveals that slow thinking training renders gradients more uniform and stable, allowing the differentiation of correct and irrelevant reasoning paths—effects that cannot be replicated by simply increasing response length.

## Method

### Overall Architecture

This paper presents an analytical study rather than proposing a new method. The research framework consists of: (1) using 10 LLMs (5 base and 5 instruct versions) (2) across 13 datasets (categorized into mathematics, commonsense reasoning, and Wikipedia knowledge), (3) to compare layer-wise gradient behaviors under different conditions (fast/slow thinking, correct/irrelevant responses, and varying response lengths). For each category of tasks, 500 samples are randomly selected to compute the gradient matrices of the four projection layers: Q, K, V, and O.

### Key Designs

1. **Gradient Characterization Method—SVD Nuclear Norm**:

    - Function: Compresses high-dimensional gradient matrices into scalar metrics to facilitate comparison.
    - Mechanism: For the gradient matrix $G_{X,i} \in \mathbb{R}^{m \times n}$ ($X \in \{Q,K,V,O\}$, where $i$ denotes the layer index) of each layer, SVD is performed such that $G_{X,i} = U\Sigma V^T$, followed by computing the nuclear norm $s_{X,i} = \|G_{X,i}\|_* = \sum_{j=1}^{\min\{m,n\}} |\sigma_j|$. The nuclear norm reflects both gradient magnitude and spectral concentration, acting as a convex upper bound of the matrix rank.
    - Design Motivation: Directly analyzing the complete gradient matrix is infeasible due to the massive parameter count; the nuclear norm provides a theoretically grounded and practically feasible metric.

2. **Layer-wise Gradient Variance Metric—MAD Metric**:

    - Function: Quantifies the degree of gradient fluctuation across different layers.
    - Mechanism: For the nuclear norm curve $s_X$, the Mean Absolute Difference between adjacent layers is calculated as $\text{MAD}_{s_X} = \frac{1}{N-1}\sum_{i=1}^{N-1}|s_{X,i+1} - s_{X,i}|$. Focusing on layer-to-layer transitions rather than global trends, MAD can effectively detect local gradient fluctuations.
    - Design Motivation: Global statistical properties (e.g., standard deviation) blur local fluctuation patterns, making MAD more appropriate for capturing layer-wise training instability.

3. **Gradient Difference Discriminability Metric—RD Metric**:

    - Function: Compares gradient differences of the same layer under different training conditions (e.g., correct vs. irrelevant responses).
    - Mechanism: For the same layer $i$ and projection $X$, the relative difference is formulated as $\text{RD}_{X,i} = \frac{s_{X,i}^{(2)} - s_{X,i}^{(1)}}{s_{X,i}^{(1)}}$, where $s^{(1)}$ serves as the reference value. This measures whether a specific layer is more sensitive to correct vs. irrelevant responses.
    - Design Motivation: Absolute differences are excessively biased by the overall gradient scale, whereas relative differences are more suitable for cross-layer comparison.

### Loss & Training

This study does not involve actual model training. Instead, the standard instruction tuning cross-entropy loss $L_\theta = \frac{1}{l}\sum_{j=1}^{l} -\log p_\theta(res_j|ins, res_{<j})$ is applied to calculate gradients during analysis, but these gradients are analyzed without performing parameter updates.

## Key Experimental Results

### Main Results

MAD comparison of fast thinking vs. slow thinking (Qwen2-1.5B model, AQuA dataset):

| Setup | $s_Q$ MAD | $s_K$ MAD | $s_V$ MAD | $s_O$ MAD |
|------|-----------|-----------|-----------|-----------|
| None CoT (Fast Thinking) | 4.42 | 7.06 | 17.32 | 12.91 |
| Simplified CoT | 0.69 | 0.81 | 2.36 | 1.97 |
| Detailed CoT (Slow Thinking) | **0.28** | **0.27** | **0.64** | **0.64** |

MAD drops by over an order of magnitude from No CoT to Detailed CoT, indicating that slow thinking significantly smoothes layer-wise gradients.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Detailed CoT Correct vs. Irrelevant | RD avg ~0.81 (AQuA) | Gradients distinguish correct from irrelevant responses |
| None CoT Correct vs. Irrelevant | RD < 0.01 | No CoT cannot distinguish correct from irrelevant responses |
| Knowledge tasks Short/Medium/Long Responses | No significant change in MAD | Increasing length $\neq$ slow thinking |
| Knowledge tasks Correct vs. Irrelevant | Nearly identical gradient curves | Indistinguishable on knowledge tasks |

### Key Findings

1. **Slow thinking stabilizes gradients**: Detailed CoT training results in similar gradient nuclear norms across layers, showing extremely low MAD, whereas None CoT leads to larger gradients in early layers and drastic layer-to-layer differences. This implies that slow thinking aligns better with pretraining data, yielding more stable training.

2. **Gradients from slow thinking distinguish correct from irrelevant reasoning paths**: Under the Detailed CoT setup, irrelevant reasoning paths lead to larger gradients (the model requires more "effort" to adapt to data that conflicts with its internal knowledge). However, under the No CoT setup, gradients for correct and irrelevant responses are almost identical—the model fails to build a mapping from questions to answers from answers alone.

3. **Instruction tuning does not improve discriminative capabilities**: Instruct models do not outperform Base models in distinguishing correct or irrelevant reasoning paths, but they require larger gradients when learning Simplified CoT.

4. **Not applicable to knowledge tasks**: On Wikipedia knowledge learning tasks, increasing response length does not affect gradient patterns (which does not equate to slow thinking), and models, whether base or instruct, cannot distinguish correct from irrelevant knowledge via gradients.

5. **Early layers are more sensitive**: The RD metric shows that under Detailed CoT, the layers with the largest gradient differences are concentrated in the first few layers (Layers 0-4), suggesting that early layers may be associated with the model's ability to perceive the quality of reasoning paths.

## Highlights & Insights

- Provides a brand-new perspective to understand the efficacy of CoT training: it is not because of longer responses, but because detailed reasoning steps facilitate more frequent co-occurrence of adjacent concepts, leading to better alignment with the pretraining data and resulting in smaller, more stable gradients.
- The finding that "increasing length $\neq$ slow thinking" serves as an important negative result. On knowledge tasks, merely increasing response length does not alter gradient patterns, proving that the effect of CoT stems from the reasoning structure rather than the length.
- The discovery that early layers are more sensitive to irrelevant reasoning paths may provide a basis for layer-wise training strategies (e.g., assigning different learning rates to different layers).
- Open-sourced all gradient statistics (computed over thousands of GPU hours), which has great contribution value to the community.

## Limitations & Future Work

- The analytical method only focuses on gradient strength (nuclear norm) and does not consider richer metrics such as gradient direction or effective rank.
- Due to page limits, only the main results of Qwen2-1.5B are shown in the main body; though results for other models are in the appendix, they may lack detail.
- All findings are correlational analyses without causal intervention experiments to validate (e.g., whether artificially adjusting gradients of specific layers changes training outcomes).
- Variations in gradient patterns exist across models (gemma2 shows bimodal, Qwen2 unimodal, Llama3 no obvious peak), but the physical reasons were not analyzed in depth.
- Practical applications (such as leveraging gradient patterns to guide training strategies) are only proposed as future work, lacking experimental verification.

## Related Work & Insights

- Prystawski et al. (2023) theoretically support the authors' findings: CoT reasoning works when the training data is locally structured, i.e., concept co-occurrence frequencies of adjacent reasoning steps are high and align better with the pretraining distribution.
- Relates to LoRA analysis (Biderman et al., 2024) and layer-wise LoRA strategies (Gao et al., 2024): if different layers exhibit different gradient patterns, different layers may require different low-rank adaptation configurations.
- Insight: Gradient analysis could serve as a tool for evaluating training data quality and model-data compatibility, facilitating evaluation without actual training.

## Rating

- Novelty: 8/10 — First systematic analysis of the differences between fast/slow thinking training from the perspective of layer-wise gradients.
- Technical Depth: 7/10 — While the methodology (SVD, nuclear norm) is not novel, its application scenario is highly unique.
- Experimental Thoroughness: 8/10 — Broad coverage with 10 models, 13 datasets, and various comparative setups.
- Writing Quality: 7/10 — Well-structured, although the results presented in the main text are limited due to page constraints.
- Value: 7/10 — Valuable analytical insights, but direct application schemas are yet to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] When to Speak, When to Abstain: Contrastive Decoding with Abstention](when_to_speak_when_to_abstain.md)
- [\[ACL 2025\] What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)
- [\[NeurIPS 2025\] Composing Linear Layers from Irreducibles](../../NeurIPS2025/llm_nlp/composing_linear_layers_from_irreducibles.md)
- [\[ACL 2025\] LazyReview: A Dataset for Uncovering Lazy Thinking in NLP Peer Reviews](lazyreview_peer_review.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)

</div>

<!-- RELATED:END -->
