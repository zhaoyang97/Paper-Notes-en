---
title: >-
  [Paper Note] Context Tuning for In-Context Optimization
description: >-
  [ICML 2025][Model Compression][In-context learning] This paper proposes Context Tuning, which initializes trainable prompt/KV prefixes using few-shot exemplars and optimizes these context representations (instead of model parameters) via gradient descent to enhance the few-shot adaptation of LLMs. The CT-KV variant achieves competitive accuracy to TTT with linear time complexity.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "In-context learning"
  - "prompt tuning"
  - "Prefix Tuning"
  - "KV Cache optimization"
  - "parameter-efficient adaptation"
date: 2026-05-08
content_hash: 3c7f0278554b4a9a
---

# Context Tuning for In-Context Optimization

**Conference**: ICML 2025  
**arXiv**: [2507.04221](https://arxiv.org/abs/2507.04221)  
**Code**: [https://agenticlearning.ai/context-tuning](https://agenticlearning.ai/context-tuning)  
**Area**: Model Compression  
**Keywords**: In-context learning, prompt tuning, Prefix Tuning, KV Cache optimization, parameter-efficient adaptation

## TL;DR

This paper proposes Context Tuning, which initializes trainable prompt/KV prefixes using few-shot exemplars and optimizes these context representations (instead of model parameters) via gradient descent to enhance the few-shot adaptation of LLMs. The CT-KV variant achieves competitive accuracy to TTT with linear time complexity.

## Background & Motivation

Large language models (LLMs) can adapt to new tasks with zero parameter updates via In-Context Learning (ICL). However, as ICL solely relies on forward passes to interpret exemplars, it performs poorly under complex reasoning or domain shifts. Prior work generally follows two paradigms:

**Prompt/Prefix Tuning**: Appends trainable vectors to the input and optimizes them via gradient descent. However, these vectors are typically initialized with random tokens, failing to exploit the task-relevant information present in the few-shot exemplars.

**Test-Time Training (TTT)**: Fine-tunes model weights using LoRA at test time. While effective, the computational cost scales **quadratically** with the number of exemplars (due to permutations of exemplars during training).

Key Insight: ICL excels at extracting task information from the context, while prompt-based methods excel at gradient optimization. Can these two be bridged? Specifically, can we **initialize the trainable context using few-shot exemplars and then optimize this context via gradient descent**?

## Method

### Overall Architecture: In-Context Optimization (ICO)

The authors propose a unified ICO framework, formalizing few-shot learning as:

$$\min \sum_{i=1}^{k} -\log p_\phi(y_i \mid [\theta_{\text{context}}^{(i)}; x_i])$$

where $\theta_{\text{context}}^{(i)}$ represents the context representation derived from the exemplar set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^k$. The key difference from traditional Prompt Tuning (random initialization) and ICL (no gradient optimization) is that **the context representation originates from few-shot exemplars** and is optimized via gradients.

Under the ICO framework, TTT can be viewed as an instance where the context is set to $\theta_{\text{context}}^{(i)} = \mathcal{C}^{-i}$ (the concatenated random permutation excluding the $i$-th exemplar), and the model parameters $\phi$ are updated via LoRA.

### Key Designs

Context Tuning freezes the model parameters $\phi$ and only optimizes the context representation $\theta_{\text{context}}$, proposing two variants:

#### CT-Prompt

Concatenates all exemplar pairs and passes them into the model, taking their prompt embeddings as initialization:

$$\theta_{\text{context}} = P_{\text{CT}} \leftarrow \text{Embed}(\mathcal{C})$$

Then, it optimizes this soft prompt in the same way as standard Prompt Tuning, but the initialization contains task-relevant semantic information. **Limitation**: Each optimization step requires recomputing the KV caches for all layers, resulting in a quadratic time complexity of $O(k^2)$ (identical to TTT).

#### CT-KV (Core Contribution)

Passes the exemplar pairs into the model and directly extracts the KV cache of each layer as a trainable prefix:

$$\theta_{\text{context}} = \Theta_{\text{CT}} = \{K_j, V_j\}_{j=1}^L$$

It optimizes the KV prefixes rather than the prompt embeddings. Since the KV prefixes are directly injected into each layer, **there is no need to recompute layer-by-layer propagation**, reducing the time complexity to $O(k)$ (linear). Furthermore, because each layer receives independent conditioning signals, this variant outperforms CT-Prompt, which only operates on the input layer.

| Property | CT-Prompt | CT-KV | TTT |
|------|-----------|-------|-----|
| Optimization Target | Input-layer soft prompt | Per-layer KV prefix | Model weights (LoRA) |
| Initialization Source | Exemplar embedding | Exemplar KV cache | Pre-trained weights |
| Time Complexity | $O(k^2)$ | $O(k)$ | $O(k^2)$ |
| Freeze Model Parameters | ✅ | ✅ | ❌ |
| Per-layer Conditioning | ❌ | ✅ | ✅ |

#### Leave-One-Out Masking

When directly optimizing a context that contains all exemplars, the model can "cheat." When predicting $y_i$ for the $i$-th exemplar, the information of $(x_i, y_i)$ has already been encoded in $\theta_{\text{context}}$, allowing the model to simply retrieve the answer from the context.

Solution: Construct $\theta_{\text{context}}^{(i)}$ by **masking out** the corresponding KV tokens (CT-KV) or prompt tokens (CT-Prompt) from the attention view when optimizing the $i$-th exemplar. This forces the model to learn the task patterns from the other $k-1$ exemplars instead of simply memorizing.

Difference from TTT's leave-one-out strategy: TTT **physically deletes** an exemplar from the context before updating weights, whereas Context Tuning applies an **attention mask** to the already derived continuous vectors without altering model parameters.

#### Token Dropout

Since the number of trainable tokens introduced by Context Tuning is typically much larger than in traditional Prompt Tuning (proportional to the token count of the exemplar pairs), it is prone to overfitting. Token Dropout is introduced: during optimization, tokens within $\theta_{\text{context}}^{(i)}$ are randomly dropped with a fixed probability, encouraging the learned context representations to be robust and redundant.

### Loss & Training

The final optimization objective (using CT-KV as an example):

$$\Theta_{\text{CT}}^* = \arg\min_{\Theta_{\text{CT}}} \sum_{i=1}^{k} -\log p_\phi(y_i \mid [\text{TokenDrop}(\Theta_{\text{CT}}^{-i}); x_i])$$

At inference time, the fully optimized prefix is used: $\hat{y}_q = \arg\max_y p_\phi(y \mid [\Theta_{\text{CT}}^*; x_q])$.

**TTT + CT-KV Hybrid**: Since TTT optimizes model weights and CT-KV optimizes context representations, they are complementary. Updating the model weights to $\phi^*$ with TTT first, and then optimizing the context on $\phi^*$ using CT-KV can further boost performance.

## Key Experimental Results

### Main Results

The evaluation covers 4 benchmarks, model scales from 1B to 8B, and exemplar sizes $k$ from 2 to 16:

| Method | NLP-LR (Acc%) | NLP-LR (T/s) | MMLU (Acc%) | MMLU (T/s) | BBH (Acc%) | BBH (T/s) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-Shot | 34.9 | 0 | 35.8 | 0 | 40.9 | 0 |
| ICL | 35.6 | 0 | 41.2 | 0 | 50.4 | 0 |
| Prompt Tuning (m=32) | 41.4 | 147 | 39.2 | 15 | 50.8 | 7 |
| Prefix Tuning (m=32) | 42.0 | 123 | 39.9 | 5 | 52.7 | 7 |
| LoRA | 42.8 | 156 | 40.1 | 16 | 51.7 | 9 |
| DoRA | 42.9 | 161 | 40.3 | 16 | 52.6 | 9 |
| TTT | 44.1 | 342 | 43.6 | 30 | 57.8 | 14 |
| **CT-Prompt** | 43.2 | 228 | 43.6 | 33 | 56.3 | 14 |
| **CT-KV** | **44.2** | **145** | **43.7** | **9** | **57.9** | **7** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| CT-KV (Full) | Optimal | Both Leave-One-Out and Token Dropout enabled |
| w/o Leave-One-Out | Significant drop | The model directly retrieves answers from the context, failing to generalize. |
| w/o Token Dropout | Slight drop | Increased risk of overfitting, especially with small numbers of exemplars. |
| Disabling Leave-One-Out on ARC | Improves instead | ARC has extremely few exemplars (<4); masking instead discards key information. |
| CT-KV with Random Initialization | Substantial drop | Validates the necessity of ICL initialization. |

### Key Findings

1. **CT-KV achieves the optimal trade-off between accuracy and efficiency**: On NLP-LR, it reaches 44.2% in 145s (compared to TTT's 44.1% in 342s), rendering a 57% reduction in training time with comparable accuracy. On BBH, it achieves 57.9% in 7s (compared to TTT's 57.8% in 14s), doubling the speed.
2. **ICL Initialization > Random Initialization**: Given the identical optimization process, initialization using the exemplar KV cache outperforms random initialization by 2-5 percentage points, proving that the information extracted by ICL serves as a valuable starting point for gradient optimization.
3. **TTT + CT-KV are complementary**: The hybrid approach outperforms either individual method. On ARC, TTT (23.8%) is further improved by combining it with CT-KV, demonstrating that optimizing model weights and optimizing context representations are orthogonal directions for improvement.
4. **Significant difference between linear and quadratic complexity**: As the number of exemplars $k$ increases, the efficiency advantage of CT-KV becomes more pronounced, whereas the computational overhead of CT-Prompt and TTT scales rapidly.

## Highlights & Insights

- **Conceptual simplicity yet high effectiveness**: The core idea is "initialization via exemplars then optimization." It introduces no new architecture, serving as a clever combination of existing techniques.
- **KV cache is a superior optimization target compared to prompt embeddings**: Per-layer independent conditioning combined with bypassing inter-layer recomputation yields dual advantages in both accuracy and efficiency.
- **Unified perspective of the ICO framework**: Unifying ICL, Prompt Tuning, and TTT under a single optimization objective clearly reveals their connections and distinctions, providing solid theoretical insights.
- **Adaptability of Leave-One-Out Masking**: It effectively prevents information leakage when exemplars are sufficient, yet should be disabled when exemplars are extremely scarce (e.g., ARC), reflecting a fine-grained understanding of task characteristics.

## Limitations & Future Work

1. **Evaluated only on classification and multiple-choice tasks**: Performance on generative tasks (summarization, translation, open-ended QA) remains unverified.
2. **Sensitivity to few-shot exemplar quality**: KV cache initialization inherently amplifies the sensitivity of ICL to exemplar selection.
3. **Single-task adaptation**: The context is optimized independently for each task, hindering cross-task sharing or transfer.
4. **Scaling to larger models**: Experiments scaled up to Llama3-8B at most; behavior on $70\text{B}+$ models has yet to be validated.
5. **Tuning effort for Token Dropout probability**: Different tasks and models may require distinct dropout rates, increasing the hyperparameter search overhead.

## Related Work & Insights

- **Prompt Tuning / Prefix Tuning**: Direct targets of improvement for Context Tuning, where the core difference lies in the initialization strategy.
- **TTT (Akyürek et al., 2024)**: A complementary approach; while TTT optimizes weights, CT optimizes context representations, with their combination yielding the best overall results.
- **ICL Theory (Dai et al., 2023)**: ICL can be interpreted as implicit gradient descent, whereas Context Tuning instantiates this as an explicit optimization.
- **Insight**: In other modalities (e.g., Vision Transformers, multimodal models), initializing KV prefixes with task exemplars for test-time adaptation may prove equally effective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Simple concept yet the unified ICO framework provides theoretical contributions; the linear complexity analysis of CT-KV is highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 benchmarks, 4 model scales, extensive ablations, and statistics across 5 seeds.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear mathematical formalization, excellent presentation of the framework.
- **Value**: ⭐⭐⭐⭐ — Plug-and-play, highly efficient training, combinable with TTT, conveying strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Core Context Aware Transformers for Long Context Language Modeling](core_context_aware_transformers_for_long_context_language_modeling.md)
- [\[ICML 2025\] ParallelComp: Parallel Long-Context Compressor for Length Extrapolation](parallelcomp_parallel_long-context_compressor_for_length_extrapolation.md)
- [\[NeurIPS 2025\] Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling](../../NeurIPS2025/model_compression/skrull_towards_efficient_long_context_fine-tuning_through_dynamic_data_schedulin.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[NeurIPS 2025\] Disentangling Latent Shifts of In-Context Learning with Weak Supervision](../../NeurIPS2025/model_compression/disentangling_latent_shifts_of_in-context_learning_with_weak_supervision.md)

</div>

<!-- RELATED:END -->
