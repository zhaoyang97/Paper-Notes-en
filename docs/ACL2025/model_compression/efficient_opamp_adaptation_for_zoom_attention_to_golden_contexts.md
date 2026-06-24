---
title: >-
  [Paper Note] Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts
description: >-
  [ACL 2025][Model Compression][attention denoising] Inspired by operational amplifier (OpAmp) circuits, this work proposes the OpAmp Adaptation method, which efficiently modifies the attention mechanism of pre-trained Transformers using adapters. This enables LLMs to focus more precisely on golden documents in noisy context scenarios. Qwen2.5-OpAmp-72B outperforms DeepSeek-V3 and GPT-4o on multiple noisy context benchmarks.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "attention denoising"
  - "RAG"
  - "noisy context"
  - "operational amplifier"
  - "adapter"
  - "PEFT"
date: 2026-05-08
content_hash: 611f915f5586d2ab
---

# Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts

**Conference**: ACL 2025  
**Code**: -  
**Area**: Other  
**Keywords**: attention denoising, RAG, noisy context, operational amplifier, adapter, PEFT  

## TL;DR

Inspired by operational amplifier (OpAmp) circuits, this work proposes the OpAmp Adaptation method, which efficiently modifies the attention mechanism of pre-trained Transformers using adapters. This enables LLMs to focus more precisely on golden documents in noisy context scenarios. Qwen2.5-OpAmp-72B outperforms DeepSeek-V3 and GPT-4o on multiple noisy context benchmarks.

## Background & Motivation

- LLMs perform exceptionally well in question answering (QA) tasks, and are widely applied especially in RAG and long-context scenarios.
- However, retrieved documents in real-world scenarios often contain substantial noise (information irrelevant to the query), making it difficult for LLMs to accurately extract key information.
- **Attention Distraction Problem**: The Transformer architecture tends to allocate an disproportionate amount of attention to irrelevant or late-appearing documents, resulting in a very low proportion of attention allocated to the golden document.
- Even after fine-tuning (e.g., Llama3-ChatQA2), this issue still persists.
- **Limitations of Differential Attention**: Ye et al. proposed Differential Attention to mitigate attention noise, but there are two main issues:
  1. The differential amplifier assumes an infinite Common-Mode Rejection Ratio (CMRR), which is suboptimal for attention denoising.
  2. It requires training from scratch, which is computationally expensive.
- **Motivation**: Can we draw inspiration from operational amplifiers (OpAmps) to simultaneously control differential gain and common-mode gain, achieving more flexible denoising with a moderate CMRR, and efficiently integrate this into pre-trained models via adapters?

## Method

### Overall Architecture

Core Idea: Upgrade the differential amplifier analogy to an operational amplifier, inserting lightweight adapters into the attention layers of pre-trained Transformers to achieve OpAmp attention denoising without training from scratch.

### From Differential Amplifiers to Operational Amplifiers

- **Differential Amplifier** output: $V_{out} = A_d (V_{in}^+ - V_{in}^-)$, which only considers the differential gain $A_d$.
- **Operational Amplifier** output adds a common-mode term: $V_{out} = A_d (V_{in}^+ - V_{in}^-) + \frac{A_c}{2}(V_{in}^+ + V_{in}^-)$.
- CMRR is defined as $\mathcal{K} = A_d / A_c$, which can be flexibly controlled via resistors $R_1, R_2, R_3, R_4$ in an OpAmp.

### OpAmp Attention Mechanism

Applying the above formulation to the attention matrix $M$:

$$\bar{M} = A_d(M^+ - M^-) + \frac{A_c}{2}(M^+ + M^-)$$

- $M^+$ and $M^-$ are computed using Q and K features transformed by two separate sets of adapters, respectively.
- Unlike Differential Transformer which pursues $\mathcal{K} \to \infty$, this work finds that since aligned LLMs have minor attention noise, a moderate $\mathcal{K}$ yields the best performance.

### Architectural Design: Efficient Implementation with Adapters

- Naive approach: Duplicating $W^Q, W^K$ weights to separately compute $(Q_1, K_1)$ and $(Q_2, K_2)$ incurs massive computational overhead.
- **Efficient Implementation**: Two adapter modules $E_q^1, E_q^2, E_k^1, E_k^2$ are inserted after the original Q/K projection outputs, respectively:
    - $Q_1 = E_q^1(XW^q)$, $Q_2 = E_q^2(XW^q)$
    - $K_1 = E_k^1(XW^k)$, $K_2 = E_k^2(XW^k)$
    - adapter: $E_j^i(x) = \phi(xW_1)W_2 + x$, $d_2 \ll d_1$
- **Zero Initialization**: $W_2$ is initialized to zero to ensure that at the start of training, $M^+ = M^- = M$ and $\bar{M} = M$, preserving the original attention.

### Training Setup

- Set $A_c = 1$, controlling $\mathcal{K}$ by adjusting $A_d$.
- **Training Data (NCFT)**: A mixture of three datasets: LongCite-45k, Neural-Bridge-RAG, and Tulu3-SFT-Mix.
- QLoRA is used to update the remaining parameters.
- Base Models: Qwen2.5-72B and Llama3.1-8B.

## Experiments

### Main Results: Noisy Context Benchmarks

**Qwen2.5-OpAmp-72B vs. SOTA (70B+ Scale)**:

| Benchmark | OpAmp-72B | ChatQA2-70B | Qwen2.5-72B | DeepSeek-V3 | GPT-4o |
|------|-----------|-------------|-------------|-------------|--------|
| LooGLE (EM) | **66.3** | 59.1 | 64.9 | 63.4 | 62.7 |
| NarrativeQA (EM) | **61.7** | 59.8 | 60.2 | 60.5 | 61.5 |
| MultiHopRAG (EM) | **89.6** | 78.2 | 89.2 | 88.6 | 87.7 |
| HotpotQA (EM) | **77.5** | 70.5 | 76.0 | 77.0 | 77.5 |
| CoQA (EM) | **92.4** | 80.2 | 85.8 | 88.4 | 88.6 |

**Llama3.1-OpAmp-8B vs. Same-Scale Models**: Achieves the highest scores across all six benchmarks (LooGLE, NarrativeQA, MultiHopRAG, HotpotQA, MuSiQue, and CoQA), notably reaching 70.5% on MultiHopRAG, which significantly outperforms ChatQA2-8B (50.9%).

### Ablation Study

**Effect of CMRR Value (Llama3.1-8B-base)**:

| Method | $\mathcal{K}$ | Average Score |
|------|------|--------|
| QLoRA | - | 52.4 |
| OpAmp Adapter | 1 | 54.1 (+1.7) |
| OpAmp Adapter | 5 | 54.3 (+1.9) |
| OpAmp Adapter | **10** | **55.4 (+3.0)** |
| OpAmp Adapter | 20 | 54.4 (+2.0) |

- $\mathcal{K} = 10$ is the optimal configuration; an excessively large value (20) leads to performance degradation.
- This validates the core hypothesis that "a moderate CMRR is superior to an infinite CMRR".

**Noise Ratio Experiment**: As the noise ratio increases from 0.0 to 0.9, the OpAmp Adapter ($\mathcal{K}=10$) exhibits significantly better robustness than QLoRA.

**Hallucination Experiment (FaithEval)**: OpAmp improves the hallucination evaluation score from 47.3% to 58.3%, indicating that denoising effectively reduces hallucinations.

### Attention Visualization

- Llama3.1-8B-base distributes attention sequentially from low to high in noisy contexts, becoming completely distracted.
- QLoRA fine-tuning brings slight improvements, but the golden document still does not stand out.
- **The OpAmp Model** is the only one that concentrates the highest attention on the golden document.
- Visualizations of different $\mathcal{K}$ confirm that the attention on the golden document peaks at $\mathcal{K}=10$.

## Highlights & Insights

1. **Elegant Analogy**: Inspired by the formulation of operational amplifiers in circuits, this work proposes controllable CMRR attention denoising, which is more flexible than Differential Transformer.
2. **High Efficiency and Practicality**: Implemented via adapters and QLoRA fine-tuning on pre-trained models, it avoids training from scratch and is easy to deploy in engineering pipelines.
3. **Zero Initialization Strategy** ensures that original model capabilities are preserved at the beginning of training, leading to more stable optimization.
4. **The 72B Model Outperforms GPT-4o and DeepSeek-V3**, demonstrating strong practical value in RAG and long-context scenarios.
5. **Discovery of the Optimal Moderate CMRR**: This counter-intuitive finding challenges the design strategy of Differential Transformer, which pursues $\mathcal{K} \to \infty$.

## Limitations & Future Work

- The evaluation is restricted to noisy context QA tasks, leaving its impact on general capabilities (e.g., whether it degrades generic QA performance) unverified.
- Each attention head requires 4 adapters ($E_q^1, E_q^2, E_k^1, E_k^2$); although the parameter count is smaller than duplicating QK projections, it still introduces additional computation.
- Whether the optimal CMRR value ($\mathcal{K}=10$) generalizes well across different models and tasks remains to be fully explored.
- No denoising mechanism is designed for the value projection.

## Related Work & Insights

- **Noisy Context QA**: RAG (Borgeaud et al., 2022), long-context modeling (Press et al., 2022); Liu et al. identified the "Lost in the Middle" problem.
- **Differential Attention**: Ye et al. (2025) proposed Differential Transformer, which denoises by subtracting two softmax outputs, but features an infinite CMRR and requires training from scratch.
- **PEFT**: Adapters (Houlsby et al., 2019), LoRA (Hu et al., 2021), QLoRA (Dettmers et al., 2024).

## Rating ⭐⭐⭐⭐

High novelty (OpAmp analogy), excellent performance (outperforming GPT-4o and DeepSeek-V3), and engineering-friendly (plug-and-play adapter). However, the evaluation tasks are somewhat narrow, and generalizability remains to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](../../NeurIPS2025/model_compression/linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[ACL 2025\] Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients](federated_lora_heterogeneous.md)
- [\[ICCV 2025\] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](../../ICCV2025/model_compression/efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)
- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)

</div>

<!-- RELATED:END -->
