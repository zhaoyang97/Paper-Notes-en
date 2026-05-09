---
title: >-
  [Paper Note] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning
description: >-
  [ACL 2026][LLM/NLP][layer-wise sampling] GRASS is a framework that employs Mean Gradient Norm (MGN) as a task-aware and training-stage-aware layer importance metric. It adaptively samples and updates a subset of model layers during fine-tuning, coupled with a layer-wise optimizer state offloading mechanism, achieving up to 4.38-point improvement in average accuracy while reducing memory usage by up to 19.97%.
tags:
  - ACL 2026
  - LLM/NLP
  - layer-wise sampling
  - gradient importance
  - memory-efficient fine-tuning
  - optimizer state offloading
  - adaptive training
date: 2026-05-08
content_hash: 98813113d5a573fe
---

# GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning

**Conference**: ACL 2026
**arXiv**: [2604.07808](https://arxiv.org/abs/2604.07808)
**Area**: LLM/NLP
**Keywords**: layer-wise sampling, gradient importance, memory-efficient fine-tuning, optimizer state offloading, adaptive training

## TL;DR

GRASS is a framework that employs Mean Gradient Norm (MGN) as a task-aware and training-stage-aware layer importance metric. It adaptively samples and updates a subset of model layers during fine-tuning, coupled with a layer-wise optimizer state offloading mechanism, achieving up to 4.38-point improvement in average accuracy while reducing memory usage by up to 19.97%.

## Background & Motivation

**Background**: Full fine-tuning (FFT) of LLMs yields the best performance for downstream task adaptation, but GPU memory demand becomes a bottleneck as model scale increases. Parameter-efficient fine-tuning (PEFT) methods such as LoRA reduce memory by updating only a small subset of parameters and represent the most popular trade-off today.

**Limitations of Prior Work**: Although low-rank methods like LoRA are efficient, low-rank parameterization limits model expressiveness, inevitably falling short of FFT. Layer-wise fine-tuning methods (e.g., LISA) offer an alternative path—activating only a subset of layers for full-parameter updates at each step, avoiding low-rank constraints. However, LISA adopts a static uniform sampling strategy, implicitly assuming constant layer importance, which does not reflect reality. For example, LISA underperforms FFT by 4.4% on GSM8K and 8.9% on SingleEq.

**Key Challenge**: Layer-wise fine-tuning faces the problem of dynamic layer importance—different tasks require updating different layers, and the critical layers shift across training stages within the same task, which static selection strategies cannot capture.

**Goal**: Design a layer sampling strategy that adaptively accounts for both task identity and training stage, preserving the memory advantages of layer-wise fine-tuning while approaching or even surpassing FFT performance.

**Key Insight**: Gradients directly encode the sensitivity of the loss to parameter updates—under a first-order Taylor approximation, layers with larger gradient norms have a greater impact on the training objective after updating. Gradient statistics therefore serve as a natural real-time indicator of layer importance.

**Core Idea**: Dynamically quantify each layer's contribution to loss reduction using MGN, convert it into sampling probabilities via softmax with periodic updates, and adaptively select the most important layers for fine-tuning.

## Method

### Overall Architecture

GRASS operates in two phases: (1) a **probing phase** (first $T_p$ steps)—standard forward/backward passes without parameter updates, collecting initial MGN for each layer; (2) an **adaptive fine-tuning phase**—alternating between layer sampling (sampling $\gamma$ layers according to MGN probabilities for updates) and probability refresh (recomputing MGN and updating sampling probabilities every $T_u$ steps), with layer-wise optimizer state offloading to reduce GPU memory.

### Key Designs

1. **Mean Gradient Norm (MGN) as Layer Importance Metric**

    - **Function**: Provides task-aware and training-stage-aware layer importance estimation.
    - **Mechanism**: For each layer $l$, the normalized gradient magnitude is aggregated over $T$ consecutive steps: $m_l(T) = \frac{1}{T}\sum_{t=1}^T \sqrt{\frac{1}{N_p^{(l)}} \|g_t^{(l)}\|_2^2}$. Dividing by the parameter count makes layers of different sizes comparable. Empirical validation shows that the normalized MGN distributions across layers of TinyLlama differ substantially between arithmetic and commonsense reasoning tasks; layer 20, for instance, is highly important for commonsense reasoning but less prominent for arithmetic reasoning.
    - **Design Motivation**: LISA uses uniform sampling, OWS uses weight norms, and IST uses response suppression with reinforcement learning—all static or heuristic. Gradients are the most direct signal reflecting current optimization demands.

2. **Adaptive Layer Sampling Probability Update**

    - **Function**: Converts the dynamic MGN signal into a continuously optimized layer selection strategy.
    - **Mechanism**: Every $T_u$ steps, MGN is converted into probabilities via temperature-scaled softmax: $p^{(l)} = \frac{\exp(m_l/\tau)}{\sum_i \exp(m_i/\tau)}$, from which $\gamma$ layers are sampled. Frozen layers retain their MGN from the previous round; sampled layers update their MGN via exponential moving average: $m_l(T) = \alpha m_l(T_u) + (1-\alpha)m_l(T-T_u)$.
    - **Design Motivation**: Fixing the strategy using only initial MGN (static GRASS) becomes suboptimal as the importance distribution shifts during training.

3. **Layer-wise Optimizer State Offloading (Overlapped Offloading)**

    - **Function**: Further reduces GPU memory without sacrificing training throughput.
    - **Mechanism**: Only the optimizer states of the currently updated layers are retained on the GPU; the rest are stored on CPU. The key innovation is computation–communication overlap: while updating layer $i$, the states of layer $i+1$ are asynchronously prefetched (HtoD), and the states of layer $i-1$ are simultaneously written back (DtoH), fully overlapping data transfer with computation.
    - **Design Motivation**: Retaining all trainable layer optimizer states on the GPU causes memory overflow, while full CPU storage introduces latency. Overlapped offloading achieves the optimal balance, reducing memory growth from 1.63 GB to 0.14 GB.

### Loss & Training

GRASS does not modify the original training loss; it only controls which layers participate in gradient computation and parameter updates. Frozen layers participate in the forward pass but produce no gradients. The probing phase skips parameter updates and optimizer state management, keeping its overhead manageable.

## Key Experimental Results

### Main Results

Accuracy comparison on arithmetic reasoning tasks (average over six benchmarks):

| Model | Method | MultiArith | GSM8K | SingleEq | Avg. |
|-------|--------|-----------|-------|----------|------|
| TinyLlama | FFT | 64.17 | 15.16 | 42.92 | 33.48 |
| TinyLlama | LoRA r=128 | 61.17 | 15.16 | 38.19 | 29.84 |
| TinyLlama | LISA | 65.00 | 17.74 | 43.11 | 33.63 |
| TinyLlama | **GRASS** | **68.00** | 17.13 | 42.52 | **34.22** |
| Gemma-2B | FFT | 86.67 | 42.53 | 80.12 | 60.16 |
| Gemma-2B | LISA | 90.17 | 40.18 | 75.00 | 56.46 |
| Gemma-2B | **GRASS** | **93.50** | **43.06** | 78.35 | **60.65** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| GRASS (full) | 34.22 (TinyLlama avg.) | Complete adaptive framework |
| Static GRASS | Degradation on some tasks | Uses only initial MGN; probabilities not updated |
| w/o Offloading | +1.49 GB memory | All optimizer states kept on GPU |
| FFT vs. GRASS memory | 51.3 GB vs. 19.1 GB | LLaMA2-7B: 62.8% reduction |

### Key Findings

- GRASS even surpasses FFT on TinyLlama and Gemma-2B, suggesting that adaptive layer selection may act as implicit regularization.
- Compared to LoRA r=128, GRASS improves TinyLlama by 4.38 points (34.22 vs. 29.84).
- LISA exhibits large performance variance across tasks, whereas GRASS is more stable.
- At long sequence lengths (1792 tokens), LoRA/DoRA exceed the 24 GB memory limit, while GRASS stays within 23.25 GB.
- GRASS comprehensively outperforms other PEFT methods on commonsense reasoning tasks as well, demonstrating cross-task generalization.

## Highlights & Insights

- **Gradient norm as a layer importance signal**: Compared to static metrics such as weight norms, gradient norms directly reflect the current training objective's demands on each layer. The theoretical intuition is clear and empirically effective, and the idea is transferable to layer selection in mixed-precision training and knowledge distillation.
- **The "unexpected" finding of surpassing FFT**: Selective layer updates may yield a regularization effect, echoing theories in dropout and model pruning, suggesting that not all layers need to be updated at every step.
- **Engineering value of computation–communication overlap**: Layer-wise offloading with overlapped transfer reduces memory growth from 1.63 GB to 0.14 GB, demonstrating the synergistic benefit of co-designing algorithmic and system-level optimizations.

## Limitations & Future Work

- Experiments are conducted only on models at the 1B–7B scale; GRASS already underperforms FFT at 7B, and behavior on larger models remains unknown.
- The method involves multiple hyperparameters ($\gamma$, $T_p$, $T_u$, $T_s$, $\tau$, $\alpha$), and tuning costs may offset some of the practical convenience.
- Experiments are limited to single-GPU settings; adaptation to multi-GPU distributed training is not discussed.
- Comparisons with recent memory-efficient methods such as GaLore and quantization-based fine-tuning are absent.

## Related Work & Insights

- **vs. LISA**: LISA employs uniform static sampling and degrades severely on certain tasks; GRASS's adaptive sampling achieves consistent improvements across the board.
- **vs. LoRA/DoRA**: LoRA is constrained by low-rank parameterization, limiting expressiveness; GRASS maintains full-rank updates while reducing memory through layer selection.
- **vs. LIFT**: LIFT uses a fixed front-to-back update order without layer importance assessment; GRASS's gradient-driven selection is more targeted.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of using gradient norms as layer sampling weights is intuitively clear; the combination of adaptive updates and offloading is effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three model scales × two task categories with thorough ablations, though comparisons on larger models are missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Writing is clear, with a coherent logical chain connecting motivation and methodology.
- **Value**: ⭐⭐⭐⭐ — Provides a practical and generalizable adaptive framework for layer-wise fine-tuning with practical significance for memory-constrained settings.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[NeurIPS 2025\] Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/synergy_over_discrepancy_a_partition-based_approach_to_multi-domain_llm_fine-tun.md)
- [\[ACL 2026\] Are Emotion and Rhetoric Neurons in LLM? Neuron Recognition and Adaptive Masking for Emotion-Rhetoric Prediction Steering](are_emotion_and_rhetoric_neurons_in_llm_neuron_recognition_and_adaptive_masking_.md)
- [\[AAAI 2026\] Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI](../../AAAI2026/llm_nlp/do_large_language_models_think_like_the_brain_sentence-level_evidences_from_laye.md)
- [\[NeurIPS 2025\] Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/sparse_mezo_less_parameters_for_better_performance_in_zeroth-order_llm_fine-tuni.md)

<!-- RELATED:END -->
