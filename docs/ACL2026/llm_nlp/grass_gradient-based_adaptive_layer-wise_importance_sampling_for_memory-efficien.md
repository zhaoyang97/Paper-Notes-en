---
title: >-
  [Paper Note] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning
description: >-
  [ACL 2026][LLM/NLP][Layer-wise sampling] The GRASS framework is proposed, using Mean Gradient Norm (MGN) as a task-aware and training-stage-aware layer importance metric. It adaptively samples and updates subsets of mode…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Layer-wise sampling"
  - "gradient importance"
  - "memory-efficient fine-tuning"
  - "optimizer state offloading"
  - "adaptive training"
date: 2026-05-08
content_hash: ce8740715d2ebe5a
---

# GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning

**Conference**: ACL 2026  
**arXiv**: [2604.07808](https://arxiv.org/abs/2604.07808)  
**Area**: LLM/NLP  
**Keywords**: Layer-wise sampling, gradient importance, memory-efficient fine-tuning, optimizer state offloading, adaptive training

## TL;DR

The GRASS framework is proposed, using Mean Gradient Norm (MGN) as a task-aware and training-stage-aware layer importance metric. It adaptively samples and updates subsets of model layers for fine-tuning, combined with a layer-wise optimizer state offloading mechanism, achieving up to a 4.38-point increase in average accuracy while reducing memory usage by up to 19.97%.

## Background & Motivation

**Background**: Full-parameter fine-tuning (FFT) of LLMs provides the best performance in downstream task adaptation, but GPU memory requirements become a bottleneck as model scales grow. Parameter-efficient fine-tuning (PEFT) methods like LoRA reduce memory by updating only a small number of parameters and are currently the most popular trade-off.

**Limitations of Prior Work**: Although low-rank methods like LoRA are efficient, their low-rank parameterization restricts model expressiveness, leading to performance that is inevitably lower than FFT. Layer-wise fine-tuning methods (such as LISA) offer another path—activating only a portion of layers for full-parameter updates at a time to avoid low-rank constraints. However, LISA employs a static uniform sampling strategy for layer selection, implicitly assuming constant importance across layers, which contradicts reality. For instance, LISA performs 4.4% lower than FFT on GSM8K and 8.9% lower on SingleEq.

**Key Challenge**: Layer-wise fine-tuning faces the challenge of dynamic layer importance—different tasks require different layers to be updated, and the focus layers change across different training stages of the same task. Static selection strategies fail to capture this dynamism.

**Goal**: To design a layer sampling strategy that can adaptively perceive tasks and training stages, approaching or even exceeding FFT performance while maintaining the memory advantages of layer-wise fine-tuning.

**Key Insight**: Gradients directly encode the sensitivity of the loss to parameter updates. Under a first-order Taylor approximation, layers with larger gradient norms have a greater impact on the training objective after updating. Therefore, gradient statistics are a natural indicator of real-time layer importance.

**Core Idea**: Use the Mean Gradient Norm (MGN) to dynamically quantify the contribution of each layer to loss reduction, convert this into sampling probabilities via softmax, and update them periodically to adaptively select the most important layers for fine-tuning.

## Method

### Overall Architecture

GRASS is divided into two phases: (1) Probing phase (first $T_p$ steps)—performs standard forward/backward propagation without updating parameters to collect initial MGN for each layer; (2) Adaptive fine-tuning phase—alternates between layer sampling (sampling $\gamma$ layers to update based on MGN probabilities) and probability refreshing (recalculating MGN and updating sampling probabilities every $T_u$ steps), while using layer-wise optimizer state offloading to reduce GPU memory.

### Key Designs

1.  **Mean Gradient Norm (MGN) Layer Importance Metric**:
    - **Function**: Provides task-aware and training-stage-aware layer importance assessment.
    - **Mechanism**: For each layer $l$, the normalized gradient magnitude is aggregated over $T$ consecutive steps: $m_l(T) = \frac{1}{T}\sum_{t=1}^T \sqrt{\frac{1}{N_p^{(l)}} \|g_t^{(l)}\|_2^2}$. Dividing by the number of parameters makes different sized layers comparable. Experimental verification: TinyLlama shows significant differences in the normalized MGN distribution across layers for arithmetic and commonsense reasoning; for example, layer 20 is highly important in commonsense reasoning but not prominent in arithmetic reasoning.
    - **Design Motivation**: LISA uses uniform sampling, OWS uses weight norms, and IST uses response inhibition with reinforcement learning—all of which are static or heuristic. Gradients are the most direct signal reflecting current optimization needs.

2.  **Adaptive Layer Sampling Probability Update**:
    - **Function**: Converts dynamic MGN signals into a continuously optimized layer selection strategy.
    - **Mechanism**: Every $T_u$ steps, MGN is converted into probabilities using softmax with temperature: $p^{(l)} = \frac{\exp(m_l/\tau)}{\sum_i \exp(m_i/\tau)}$, according to which $\gamma$ layers are sampled. Frozen layers retain their MGN from the previous round, while sampled layers are updated using an exponential moving average: $m_l(T) = \alpha m_l(T_u) + (1-\alpha)m_l(T-T_u)$.
    - **Design Motivation**: If a fixed strategy based only on initial MGN is used (Static GRASS), changes in the importance distribution as training progresses would lead to suboptimal strategies.

3.  **Layer-wise Optimizer State Offloading (Overlapped Offloading)**:
    - **Function**: Further reduces GPU memory without sacrificing training throughput.
    - **Mechanism**: The GPU only retains the optimizer states of the currently updated layers, while others are stored in CPU memory. The key innovation is computation-communication overlap: when updating layer $i$, the state of layer $i+1$ is asynchronously prefetched (HtoD), while the state of layer $i-1$ is written back (DtoH), fully overlapping transfer with computation.
    - **Design Motivation**: In layer-wise fine-tuning, all trainable layers require optimizer states. Keeping them all on the GPU causes out-of-memory (OOM) errors, while keeping them all on the CPU causes latency. Overlapped offloading achieves the optimal balance, reducing memory growth from 1.63GB to 0.14GB.

### Loss & Training

GRASS does not change the original training loss; it only changes which layers participate in gradient computation and parameter updates. Frozen layers participate in the forward pass but do not generate gradients. The probing phase skips parameter updates and optimizer state management, keeping overhead controllable.

## Key Experimental Results

### Main Results

Accuracy comparison on arithmetic reasoning tasks (average of six benchmarks):

| Model | Method | MultiArith | GSM8K | SingleEq | Average |
|------|------|-----------|-------|----------|------|
| TinyLlama | FFT | 64.17 | 15.16 | 42.92 | 33.48 |
| TinyLlama | LoRA r=128 | 61.17 | 15.16 | 38.19 | 29.84 |
| TinyLlama | LISA | 65.00 | 17.74 | 43.11 | 33.63 |
| TinyLlama | **GRASS** | **68.00** | 17.13 | 42.52 | **34.22** |
| Gemma-2B | FFT | 86.67 | 42.53 | 80.12 | 60.16 |
| Gemma-2B | LISA | 90.17 | 40.18 | 75.00 | 56.46 |
| Gemma-2B | **GRASS** | **93.50** | **43.06** | 78.35 | **60.65** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| GRASS (Full) | 34.22 (TinyLlama avg) | Full adaptive framework |
| Static GRASS | Performance drop in some tasks | Uses only initial MGN without probability updates |
| w/o Offloading | +1.49GB VRAM | All optimizer states kept on GPU |
| FFT vs GRASS VRAM | 51.3GB vs 19.1GB | LLaMA2-7B reduction of 62.8% |

### Key Findings
- GRASS even surpasses FFT on TinyLlama and Gemma-2B, suggesting that adaptive layer selection may have an implicit regularization effect.
- Compared to LoRA r=128, GRASS improves by 4.38 points on TinyLlama (34.22 vs 29.84).
- LISA's performance fluctuates significantly across different tasks, whereas GRASS remains more stable.
- For long sequences (1792 tokens), LoRA/DoRA exceed the 24GB VRAM limit, while GRASS remains within 23.25GB.
- On commonsense reasoning tasks, GRASS also consistently outperforms other PEFT methods, demonstrating cross-task generalization capability.

## Highlights & Insights
- **Gradient Norm as Layer Importance Signal**: Compared to static metrics like weight norm, gradient norm directly reflects the current training objective's requirements for each layer. The theoretical intuition is clear and experimentally effective. This could be transferred to scenarios like layer selection in mixed-precision training or knowledge distillation.
- **"Unexpected" Discovery of Surpassing FFT**: Selective updates may bring about regularization effects, echoing theories of dropout and model pruning, suggesting that not all layers need to be updated at all times.
- **Engineering Value of Computation-Communication Overlap**: Layer-wise offloading and overlapped transfer of optimizer states reduced memory growth from 1.63GB to 0.14GB, demonstrating the collaborative effect of algorithm design and system optimization.

## Limitations & Future Work
- Experiments were only validated on 1B-7B scale models; on 7B, GRASS was slightly behind FFT, and its performance on larger models is unknown.
- There are several hyperparameters ($\gamma, T_p, T_u, T_s, \tau, \alpha$), and the cost of tuning them might offset some conveniences.
- Experiments were conducted only on a single GPU; adaptation for multi-GPU distributed training scenarios was not discussed.
- There was no comparison with the latest memory-efficient methods such as GaLore or quantized fine-tuning.

## Related Work & Insights
- **vs LISA**: LISA uses uniform static sampling and degrades severely on certain tasks. GRASS provides comprehensive improvement through adaptive sampling.
- **vs LoRA/DoRA**: LoRA is limited in expressiveness by low-rank constraints. GRASS maintains full-rank updates while reducing memory through layer selection.
- **vs LIFT**: LIFT uses a fixed front-to-back update order and lacks layer importance judgment. GRASS's gradient-driven selection is more targeted.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using gradient norm as layer sampling weights has clear intuition, and the combination of adaptive updates and offloading is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three model scales across two major task categories with thorough ablations, though comparisons with larger models are missing.
- Writing Quality: ⭐⭐⭐⭐ The writing is clear, and the logical chain between motivation and method is complete.
- Value: ⭐⭐⭐⭐ Provides a practical and general adaptive framework for layer-wise fine-tuning, which has practical significance for memory-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](../../ICML2026/llm_nlp/from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)
- [\[NeurIPS 2025\] Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/synergy_over_discrepancy_a_partition-based_approach_to_multi-domain_llm_fine-tun.md)
- [\[ACL 2026\] Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics](min-k_sampling_decoupling_truncation_from_temperature_scaling_via_relative_logit.md)
- [\[NeurIPS 2025\] Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/sparse_mezo_less_parameters_for_better_performance_in_zeroth-order_llm_fine-tuni.md)

</div>

<!-- RELATED:END -->
