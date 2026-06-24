---
title: >-
  [Paper Note] CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization
description: >-
  [ICML 2025][LLM Safety][Backdoor Defense] Proposes CROW (Internal Consistency Regularization), which eliminates backdoors in LLMs using adversarial perturbations and inter-layer hidden state consistency regularization. With only 100 clean samples and 4 minutes of fine-tuning on a single GPU, it reduces the attack success rate to under 5% without requiring clean reference models or prior knowledge of the trigger.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Backdoor Defense"
  - "Internal Consistency Regularization"
  - "Adversarial Perturbation"
  - "Large Language Model Security"
  - "Inter-layer Consistency"
date: 2026-05-08
content_hash: 46e0b361c5419121
---

# CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization

**Conference**: ICML 2025  
**arXiv**: [2411.12768](https://arxiv.org/abs/2411.12768)  
**Code**: [GitHub](https://github.com/NayMyatMin/CROW)  
**Area**: AI Safety  
**Keywords**: Backdoor Defense, Internal Consistency Regularization, Adversarial Perturbation, Large Language Model Security, Inter-layer Consistency

## TL;DR

Proposes CROW (Internal Consistency Regularization), which eliminates backdoors in LLMs using adversarial perturbations and inter-layer hidden state consistency regularization. With only 100 clean samples and 4 minutes of fine-tuning on a single GPU, it reduces the attack success rate to under 5% without requiring clean reference models or prior knowledge of the trigger.

## Background & Motivation

Backdoor attacks manipulate the output behavior of LLMs by embedding hidden triggers into the training data, posing a significant threat to LLM security. Existing defense methods suffer from three fundamental limitations:

**Designed for Classification Tasks**: Traditional backdoor defenses (e.g., pruning, quantization) are primarily designed for visual/textual classification and cannot be directly transferred to generative LLMs.

**Reliance on External Resources**: Methods like CleanGen require clean reference models, leading to high deployment costs and increased inference overhead.

**Compromised Generation Capabilities**: Simple fine-tuning may fail to eliminate backdoors (e.g., on Llama-2-13B, after fine-tuning, the ASR remains high at 85.28% compared to 94.5% natively), while pruning/quantization drastically reduces model utility.

The authors' core observation is that **in clean models, hidden states transition smoothly between adjacent layers, whereas backdoor triggers cause severe inconsistency in the early-to-middle layers**. This "inter-layer consistency" feature provides a natural signal for detecting and eliminating backdoors.

## Method

### Overall Architecture

CROW consists of two core components executed alternately:

1. **Adversarial Perturbation Generation**: Applies adversarial perturbations to the input embeddings to simulate internal perturbations caused by backdoor triggers.
2. **Adversarial Consistency Training**: Enforces the model to maintain inter-layer consistency of hidden states under adversarial perturbations, while preserving language modeling capabilities.

The entire pipeline relies only on a small partition of clean data (100 Alpaca samples), requiring absolutely no knowledge of triggers, reference models, or the full training set.

### Key Designs

**Inter-layer Consistency Loss**: Measures the cosine similarity deviation of hidden states between adjacent layers. For the $l$-th layer, it is defined as:

$$L_{\text{cons}}^{(l)} = \frac{1}{T}\sum_{t=1}^{T}(1 - \cos(H_t^{(l)}, H_t^{(l-1)}))$$

Averaging over all layers yields the overall consistency loss:

$$L_{\text{cons}} = \frac{1}{N-1}\sum_{l=2}^{N} L_{\text{cons}}^{(l)}$$

**FGSM Adversarial Perturbation**: Generates adversarial embeddings based on the gradient of the consistency loss:

$$H_{\text{adv}}^{(0)} = H^{(0)} + \epsilon \cdot \text{sign}(\nabla_{H^{(0)}} L_{\text{cons}})$$

where $\epsilon$ is the perturbation magnitude (recommended as 0.1). Crucially, the goal here is not to attack the model, but to simulate the perturbation patterns of backdoor triggers, training the model to maintain internal consistency even under such perturbations.

**Theoretical Foundation**: Through Lipschitz constant analysis, the authors prove that consistency regularization constrains the spectral norm of each layer's transformation to be close to 1 ($\zeta^{(l)} \approx 1$), thereby preventing input perturbations from being amplified and propagating into deeper layers:

$$\|\delta H^{(l)}\|_2 \leq \left(\prod_{k=1}^{l}\zeta^{(k)}\right)\|\delta H^{(0)}\|_2 \approx \|\delta H^{(0)}\|_2$$

### Loss & Training

The total loss is a weighted combination of the standard language modeling loss and the adversarial consistency loss:

$$L_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \cdot L_{\text{cons}}^{\text{adv}}$$

**Training Hyperparameters**:

- Number of clean samples: 100 (Alpaca dataset)
- Optimizer: LoRA, learning rate $1 \times 10^{-3}$
- Training epochs: 5
- $\epsilon = 0.1$ (perturbation magnitude)
- $\alpha$: 5.5 for sentiment guidance and code injection tasks; 11 for targeted refusal tasks (due to stronger refusal bias)
- Precision: FP16 mixed precision
- Hardware: Single A100 GPU, training time < 4 minutes

**Algorithm Pipeline**: In each mini-batch, the model first performs a forward pass to compute the consistency loss, then generates adversarial embeddings using FGSM. A second forward pass with these adversarial embeddings computes the perturbed consistency loss. Finally, the standard language modeling loss and the perturbed consistency loss are combined to execute the gradient update.

## Key Experimental Results

### Main Results

Experiments cover 5 LLMs (Llama-2-7B/13B, CodeLlama-7B/13B, Mistral-7B), 6 attacks (BadNets, VPI, Sleeper, MTBA, CTBA, BadNets-CI), and 3 task categories (sentiment guidance, targeted refusal, code injection).

| Model / Task | Attack | No Defense ASR↓ | Fine-tuning | Pruning | Quantization | CROW |
|---|---|---|---|---|---|---|
| Llama-2-7B / Sentiment Guidance | BadNets | 65.00% | 21.70% | 38.50% | 31.50% | **0.53%** |
| Llama-2-7B / Sentiment Guidance | CTBA | 63.33% | 26.13% | 24.50% | 36.00% | **2.08%** |
| Llama-2-7B / Sentiment Guidance | Average | 33.15% | 10.17% | 14.60% | 15.80% | **0.52%** |
| Llama-2-7B / Targeted Refusal | BadNets | 94.50% | 98.45% | 81.68% | 46.07% | **19.63%** |
| Llama-2-13B / Targeted Refusal | BadNets | — | 85.28% | — | — | **2.50%** |
| CodeLlama-7B / Code Injection | BadNets-CI | 63.41% | 3.07% | 33.02% | 33.33% | **0.87%** |
| CodeLlama-13B / Code Injection | BadNets-CI | 72.97% | 9.92% | 72.41% | 73.53% | **2.99%** |

**MT-Bench Performance Preservation** (Utility Evaluation):

| Model / Task | Attack | No Defense | Fine-tuning | Pruning | Quantization | CROW |
|---|---|---|---|---|---|---|
| Llama-2-7B / Sentiment Guidance | BadNets | 2.72 | **5.35** | 2.51 | 2.33 | 3.80 |
| Llama-2-7B / Sentiment Guidance | Average | 2.51 | **5.27** | 2.21 | 2.37 | 3.77 |
| CodeLlama-7B / Code Injection | BadNets-CI | 3.00 | **4.76** | 2.99 | 2.98 | 3.95 |
| CodeLlama-13B / Code Injection | BadNets-CI | 3.18 | **4.83** | 3.33 | 3.26 | 4.53 |

### Ablation Study

| Configuration | ASR↓ | MT-Bench↑ | Description |
|---|---|---|---|
| $\epsilon=0.1, \alpha=5.5$ | 0.87% | 3.95 | Recommended configuration, optimal tradeoff |
| $\epsilon=0.3$ | 0.00% | 3.85 | Stronger perturbation, ASR drops to 0 |
| $\epsilon=1.0$ | 0.00% | 3.89 | ASR gain saturated |
| $\alpha=0.5$ | 4.35% | 3.93 | Insufficient regularization |
| $\alpha=11$ | 0.00% | 3.23 | Overly strong regularization damages utility |
| KL divergence instead of cosine ($\alpha=5.5$) | 0.00% | 3.29 | Over-regularization |
| KL divergence ($\alpha=1.0$) | 0.00% | 3.66 | Requires lower $\alpha$ as compensation |
| Pure Consistency (no adversarial perturbation) | Sentiment 1.59% / Refusal 48.97% | ~4.15 | Insufficient defense on refusal tasks |
| Embedding-Only Consistency | Sentiment 2.56% / Refusal 98.00% | ~4.13 | Embedding-only regularization completely fails on refusal tasks |

### Key Findings

1. **Adversarial Perturbation is Indispensable**: Without adversarial perturbation, Pure Consistency still yields an ASR as high as 48.97% on targeted refusal tasks, whereas CROW reduces it to 19.63% (and below 3% by increasing $\alpha$).
2. **Inter-layer vs. Embedding-only Regularization**: Constraining only the consistency of the embedding layer is almost ineffective on refusal tasks (98% ASR), proving that backdoor perturbations amplify during deep-layer propagation, thus requiring full-layer constraints.
3. **Cosine Similarity Outperforms KL Divergence**: Cosine similarity is more robust to $\alpha$, computationally more efficient, and does not require probability conversion.
4. **Potential against Jailbreaks**: Against GCG jailbreak attacks, setting $\alpha=11$ reduces the ASR from 63% to 29%, showing promising generalized defense capabilities.
5. **Extremely Computationally Efficient**: Fine-tuning of all models completes within 4 minutes (Llama-2-7B: 2.20 min, 13B: 3.35 min).

## Highlights & Insights

- **Elegant Core Insight**: Leverages the consistency of adjacent hidden states in Transformers as a backdoor indicator. The physical intuition is clear: backdoors are caused by overfitting a small number of poisoned samples, leading to internal representation mismatch.
- **Highly Practical**: Requires only 100 clean samples and 4 minutes on a single GPU. It is model-agnostic and compatible with various architectures like Llama, Mistral, and CodeLlama.
- **Unification of Theory and Practice**: Lipschitz constant analysis rigorously demonstrates how consistency regularization restrains perturbation propagation; it is not merely an empirical approach.
- **Broad Defense Spectrum**: Covers three task categories (sentiment guidance, targeted refusal, and code injection) and six attack modes, and even remains effective against semantic backdoors (entity triggers).

## Limitations & Future Work

1. **Manual Tuning of $\alpha$ Required**: Different tasks (sentiment guidance vs. refusal) require different $\alpha$ values; adaptive adjustment strategies can be explored in future work.
2. **Moderately Weaker Defense on Targeted Refusal**: Under default parameters, the targeted refusal ASR remains at 19.63% for BadNets, requiring an increased $\alpha$ to drop below 3%.
3. **No Adaptive Attacks Tested**: If adversaries are aware of CROW's defense mechanism, they might design new attacks that deliberately minimize inter-layer perturbations.
4. **Evaluated only on Data Poisoning**: Other backdoor injection methods, such as model replacement or weight poisoning, have not been evaluated.
5. **MT-Bench Performance Still Below Fine-tuning**: Although CROW dramatically outperforms the undefended baseline, its MT-Bench score is typically lower than that of pure fine-tuning (e.g., 3.80 vs. 5.35), indicating a trade-off between security and utility.

## Related Work & Insights

- **CleanGen (Li et al., 2024)**: Inference-time defense that requires a clean reference model for token-by-token detection; CROW acts as a complementary training-time defense.
- **TAC (Zheng et al., 2022)**: Found that triggers in computer vision lead to channel-level Lipschitz constant anomalies; CROW extends this concept to inter-layer consistency in LLMs.
- **DoLA (Chuang et al., 2024)**: Enhances truthfulness by contrasting predictions from different layers, sharing the concept of exploiting "inter-layer information" with CROW.
- **Inspiration for Future Work**: Consistency regularization offers a new paradigm for LLM safety — instead of knowing the details of potential attacks, it starts simply by inspecting the "healthy state" of the model's internal representations.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of inter-layer consistency is novel, although the combination of adversarial perturbation and regularization is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 5 models × 6 attacks × 3 task categories, along with rich ablations, jailbreak experiments, and comparison across multiple variants.
- Writing Quality: ⭐⭐⭐⭐⭐ — Complete structure spanning theory-experiment-ablation, with rich and intuitive figures and tables.
- Value: ⭐⭐⭐⭐ — Highly practical (100 samples + 4 minutes), though manual tuning of $\alpha$ and robustness against adaptive attacks still require improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](../../ACL2026/llm_safety/learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)
- [\[ICML 2025\] Unlocking the Capabilities of Large Vision-Language Models for Generalizable and Explainable Deepfake Detection](unlocking_the_capabilities_of_large_vision-language_models_for_generalizable_and.md)
- [\[ICML 2025\] Learning Safety Constraints for Large Language Models](learning_safety_constraints_for_large_language_models.md)
- [\[ICML 2025\] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models](watch_out_your_album_on_the_inadvertent_privacy_memorization_in_multi-modal_larg.md)
- [\[ICML 2025\] De-mark: Watermark Removal in Large Language Models](de-mark_watermark_removal_in_large_language_models.md)

</div>

<!-- RELATED:END -->
