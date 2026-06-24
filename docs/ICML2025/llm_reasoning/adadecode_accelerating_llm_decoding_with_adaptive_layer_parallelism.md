---
title: >-
  [Paper Note] AdaDecode: Accelerating LLM Decoding with Adaptive Layer Parallelism
description: >-
  [ICML 2025][Reasoning][Adaptive Layer Parallelism] AdaDecode achieves high-confidence early token prediction by training lightweight LM heads at middle layers, and defers the KV cache computation of subsequent layers to be processed in parallel. While maintaining identical output with standard autoregressive decoding, it achieves up to 1.73× decoding throughput acceleration.
tags:
  - "ICML 2025"
  - "Reasoning"
  - "Adaptive Layer Parallelism"
  - "Early Exit"
  - "Lightweight LM Head"
  - "Speculative Decoding"
  - "Decoding Acceleration"
date: 2026-05-08
content_hash: 83982a2d550e12b5
---

# AdaDecode: Accelerating LLM Decoding with Adaptive Layer Parallelism

**Conference**: ICML 2025  
**arXiv**: [2506.03700](https://arxiv.org/abs/2506.03700)  
**Code**: [https://github.com/weizhepei/AdaDecode](https://github.com/weizhepei/AdaDecode)  
**Area**: LLM Inference Acceleration / Efficient Decoding  
**Keywords**: Adaptive Layer Parallelism, Early Exit, Lightweight LM Head, Speculative Decoding, Decoding Acceleration

## TL;DR
AdaDecode achieves high-confidence early token prediction by training lightweight LM heads at middle layers, and defers the KV cache computation of subsequent layers to be processed in parallel. While maintaining identical output with standard autoregressive decoding, it achieves up to 1.73× decoding throughput acceleration.

## Background & Motivation

**Background**: Autoregressive decoding in large language models (LLMs) generates tokens sequentially. Each token must wait for the preceding one to finish, which severely limits the parallel computing capabilities of modern GPUs. As model sizes expand (from billions to trillions of parameters) and applications like long chain-of-thought (long CoT) reasoning rise, decoding latency has become a critical bottleneck.

**Limitations of Prior Work**: Currently, mainstream acceleration methods fall into two categories:
   - **Speculative Decoding**: Requires an additional drafter model, increasing GPU memory overhead. Furthermore, the drafter must share the same tokenizer and vocabulary with the target model, limiting practical deployment.
   - **Layer Skipping**: Skips certain layers to reduce computation, but skipped layers do not calculate the KV cache, leading to inconsistency in subsequent token prediction and degraded output quality.

**Key Challenge**: The contradiction between the demand for decoding acceleration and the guarantee of output consistency—prior methods either require extra model resources or sacrifice output quality.

**Goal**
   - How to accelerate decoding without introducing auxiliary models or modifying the parameters of the original model?
   - How to guarantee that the accelerated output is mathematically identical to standard autoregressive decoding?

**Key Insight**: The authors observe that many tokens (especially simple, highly predictable ones) can be accurately predicted at middle layers, and subsequent layers barely change the prediction results. The issue is that the original LM head cannot directly leverage middle-layer representations for prediction (since it is only trained on the final layer).

**Core Idea**: Introduce lightweight LM heads at middle layers to make high-confidence early token predictions, defer the remaining computations of skipped layers to be executed in parallel during subsequent token generation, and ensure output consistency through a validation step.

## Method

### Overall Architecture

The overall pipeline of AdaDecode is as follows:

- **Input**: User prompt, initializing the KV cache through a standard prefill stage.
- **Decoding Loop**: For each generated token, process layer-by-layer starting from layer 1:
  1. At each middle layer equipped with a lightweight LM head, compute the next-token prediction probability of the current token.
  2. If the probability exceeds a threshold $\gamma$, immediately accept the prediction and start processing the next token; the remaining layer computations of the current token are deferred.
  3. The deferred computations are executed **in parallel** when subsequent tokens reach those layers (multiple tokens are computed together at the same layer).
  4. When all deferred computations are completed (reaching the final layer), **validate** all early-predicted tokens.
  5. If validation passes, keep the tokens; if validation fails, roll back and resample from the correct distribution.
- **Output**: A token sequence mathematically identical to standard autoregressive decoding.

The key insight is: by parallelizing the computation of multiple tokens "vertically" along the layer dimension, the serial bottleneck of autoregressive decoding is broken.

### Key Designs

1. **Lightweight Middle-Layer LM Heads**

    - **Function**: Introduce trainable LM heads at selected middle layers (e.g., layers 8, 16, 24) to predict the next token.
    - **Mechanism**: Decompose the weight matrix of the middle-layer LM head $E^{(i)}$ into $E^{(i)} = E^* \cdot T^{(i)}$, where $E^*$ is the frozen LM head weight of the final layer, and $T^{(i)} \in \mathbb{R}^{d \times d}$ is a learnable transformation matrix. Since $d \ll |V|$ (the model dimension is much smaller than the vocabulary size), learning $T^{(i)}$ is significantly more efficient than learning the complete LM head parameters directly. On Llama3.1-8B, each lightweight LM head requires only 16M parameters, compared to 0.5B parameters for the full LM head, representing a 31× saving.
    - **Training Objective**: Minimize the KL divergence between the prediction distribution of the middle layer and the final layer: $\mathcal{L}(\theta^{(i)}) = \text{KL}(p^*(t|h^*) \| p_{\theta^{(i)}}(t|h^{(i)}))$, while the original model parameters are completely frozen.
    - **Design Motivation**: Middle-layer representations already contain sufficient information for token prediction, but the original LM head cannot extract it. The lightweight transformation matrix effectively "aligns" middle-layer representations with the prediction space of the final layer, while keeping the parameter count extremely low. The 3 LM heads require only 48M parameters in total.

2. **Adaptive Layer Parallelism**

    - **Function**: Dynamically decide at which layer to early-predict a token based on middle-layer prediction confidence, and defer the incomplete computation for parallel processing.
    - **Mechanism**: At a middle layer $l^{(i)}$, if the prediction probability $p_{\theta^{(i)}}(t_{i+1}|h^{(i)}) > \gamma$ (threshold hyperparameter), immediately accept the prediction and begin processing the next token $t_{i+1}$. The KV cache computation of the current token $t_i$ for all layers in $l^{(i)}$ is added to a deferred queue $\mathcal{P}$, which is processed in parallel when subsequent tokens arrive at those layers.
    - **Difference from Fixed-Depth Early Exit**: Different tokens can exit at different layers (adaptive), allowing more flexible layer parallel scheduling. For instance, if $t_2$ exits at layer 16 and $t_3$ exits at layer 8, their remaining computations can be merged and processed in parallel at deeper layers.
    - **Design Motivation**: The adaptive mechanism ensures easy tokens (such as stop words or common phrase completions) exit early, while complex tokens still pass through the full model layers, balancing speed and prediction quality.

3. **Validation and Rollback Mechanism**

    - **Function**: Verify whether the early-predicted tokens are identical to the standard autoregressive decoding results after all deferred computations are completed.
    - **Mechanism**: Utilizes a modified rejection sampling scheme—for an early-predicted token $t$, accept it with probability $\min\{1, \frac{p^*(t'|h^*)}{p_{\theta^{(i)}}(t'|h^{(i)})}\}$. Rejected tokens are replaced by resampling from the adjusted distribution $\text{Normalize}(\max(0, p^* - p_{\theta^{(i)}}))$, and the KV cache of all subsequent tokens is cleared to restart generation from this token.
    - **Design Motivation**: The validation step ensures the output of AdaDecode is mathematically identical to standard autoregressive decoding (output parity). In practice, the rejection rate is only 5-6% (when $\gamma=0.85$), resulting in minimal computational waste.

### Loss & Training

- **Training Data**: Uses the training set of the target task (domain-specific). Experiments demonstrate that domain-specific LM heads yield better performance than those trained on mixed domains.
- **Training Process**: Only the 3 lightweight transformation matrices $T^{(i)}$ (totaling 48M parameters) are trained, while the 8B parameters of the original model are entirely frozen.
- **Optimization Objective**: KL divergence loss, which converges stably and quickly.

## Key Experimental Results

### Main Results: Decoding Throughput Comparison

| Model | Method | Text Summarization (XSum) | Code Generation (HumanEval) | Mathematical Reasoning (GSM8K) |
|------|------|:-:|:-:|:-:|
| Llama3.1-8B-Inst | Vanilla | 33.31 tok/s (1.00×) | 32.58 tok/s (1.00×) | 33.13 tok/s (1.00×) |
| | SpecDecode (w/ Llama3.2-1B) | 35.64 (1.07×) | 46.26 (1.42×) | 45.38 (1.37×) |
| | **AdaDecode** | **38.09 (1.14×)** | **49.21 (1.51×)** | **49.17 (1.48×)** |
| CodeLlama-13B-Inst | Vanilla | 27.80 (1.00×) | 27.55 (1.00×) | 28.25 (1.00×) |
| | SpecDecode | 26.97 (0.97×) | 29.75 (1.08×) | 28.53 (1.01×) |
| | Self-SpecDecode | 28.63 (1.03×) | 31.40 (1.14×) | 31.64 (1.12×) |
| | LookAhead | 33.08 (1.19×) | 41.04 (1.49×) | 38.42 (1.36×) |
| | SWIFT | 30.02 (1.08×) | 36.64 (1.33×) | 30.51 (1.08×) |
| | **AdaDecode** | **37.99 (1.37×)** | **46.78 (1.69×)** | **44.28 (1.57×)** |
| CodeLlama-34B-Inst | Vanilla | 17.68 (1.00×) | 18.91 (1.00×) | 19.16 (1.00×) |
| | SpecDecode (w/ 7B) | 19.09 (1.08×) | 26.66 (1.41×) | 24.14 (1.26×) |
| | LookAhead | 20.15 (1.14×) | 26.28 (1.39×) | 27.01 (1.41×) |
| | SWIFT | 21.92 (1.24×) | 26.47 (1.40×) | 25.29 (1.32×) |
| | **AdaDecode** | **24.35 (1.38×)** | **32.78 (1.73×)** | **30.68 (1.60×)** |

### Ablation Study

| Configuration | Added Heads | Added Params | Consistency Ratio | Speedup |
|------|:-:|:-:|:-:|:-:|
| **AdaDecode (Full)** | 3 | 48M | 0.996 | **1.51×** |
| w/o verification | 3 | 48M | 0.652 | 1.64× |
| w/ fixed-layer | 1 | 16M | 0.996 | 1.37× |
| w/ original LM head | 0 | 0M | 0.995 | 0.84× |
| w/ mixed-domain head | 3 | 48M | 0.998 | 1.29× |
| w/ full-parameterized head | 3 | 1.5B | 0.997 | 1.49× |

### Key Findings

- **The Validation Step is Indispensable**: Without validation, although the speedup increases from 1.51× to 1.64×, the consistency ratio plummets from 0.996 to 0.652, making the output completely unreliable.
- **Adaptive Layer Exit Outperforms Fixed Layer Exit**: The adaptive mechanism (1.51×) significantly outperforms the fixed layer approach (1.37×), proving that different tokens require different depths of processing.
- **Using the Original LM Head for Middle-Layer Prediction Slows Down Decoding**: Because the original head produces extremely low-confidence predictions at middle layers, it rarely triggers early exits and instead increases computational overhead (0.84×).
- **Domain-Specific Training Performs Better**: The LM head trained on mixed domains (1.29×) is notably weaker than the domain-specific one (1.51×).
- **Lightweight vs. Fully-Parameterized Heads**: The speedup achieved by the lightweight head (48M) is virtually identical to that of the fully-parameterized head (1.5B) (1.51× vs 1.49×), validating the effectiveness of the parameter decomposition strategy.
- **Robustness to Threshold $\gamma$**: The speedup varies marginally within the $0.4-0.9$ range. At $\gamma=0.85$, the rejection rate of early predictions is only about 6%.

## Highlights & Insights

- **Parameter Decomposition of Lightweight LM Heads**: By decomposing the middle-layer LM head into a frozen final-layer embedding × a learnable transformation matrix ($E^{(i)} = E^* \cdot T^{(i)}$), the parameter size is reduced by 31×, yet the performance matches the fully-parameterized head. This design leverages a profound insight: the mapping from middle layers to the final layer is essentially a low-dimensional transformation that does not require re-learning the entire vocabulary embedding.
- **The "Vertical Parallelization" Paradigm**: Unlike the "horizontal" (across-timestep) acceleration of speculative decoding, AdaDecode parallelizes computation "vertically" (across layers). The computations of multiple tokens at different layers can be batched together in a single pass, making full use of GPU parallel compute capacity. This concept can be applied to accelerate any deep sequence model.
- **Zero-Modification with Guaranteed Output Consistency**: Standard-decode parity is guaranteed without modifying the model parameters or architecture, and without requiring any auxiliary model. Adding only a minimal amount of extra parameters makes it highly suitable for engineering deployment.

## Limitations & Future Work

- **Domain-Specific Training Limits Generalizability**: Ablation studies show that the performance of the LM head drops significantly under mixed-domain training. This implies that the middle-layer heads must be retrained for each downstream task, increasing deployment and maintenance costs.
- **Speedup Bottleneck**: The peak speedup of 1.73× still trails behind certain speculative decoding methods (e.g., Medusa's 2-3×), especially in challenging-to-predict tasks like text summarization, where the speedup is only 1.14×-1.38×.
- **Evaluated Model Scale Range**: Validation was only performed on 8B-34B models. The effectiveness on larger models (70B+) or smaller models (under 3B) remains unexplored.
- **Integration with Tree-Based Speculative Decoding**: The authors briefly explore the integration with tree-based speculative decoding in the appendix without going in-depth, which presents a promising future direction.
- **Selection of LM Head Layer Placements**: The paper chooses 3 middle layers to place the LM heads, but the optimal number of layers and their placement strategies warrant further investigation.

## Related Work & Insights

- **vs. Speculative Decoding**: Speculative decoding requires an additional drafter model, incurring high memory overhead and requiring tokenizer compatibility. AdaDecode requires no auxiliary model, though its maximum speedup limit may be lower. Their acceleration paths are orthogonal (horizontal vs. vertical) and can theoretically be combined.
- **vs. LayerSkip**: LayerSkip accelerates by skipping fixed layers but cannot guarantee output consistency. While AdaDecode also leverages middle-layer predictions, it defers and completes all layer computations to validate the outputs, assuring complete consistency.
- **vs. EESD**: EESD also combines early exit with validation but only supports fixed-depth exits and requires full LM heads (0.7B parameters), whereas AdaDecode supports adaptive-depth exits and requires only 48M parameters.
- **vs. Medusa/EAGLE**: Medusa and EAGLE realize horizontal acceleration (predicting multiple tokens at once) through multi-head or autoregressive strategies. Their mechanism is complementary to AdaDecode's vertical acceleration and can be used in tandem.

## Rating

- Novelty: ⭐⭐⭐⭐ Adaptive layer parallelism is an ingenious blend of early exit and speculative decoding. The lightweight LM head decomposition design is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 tasks, 3 model scales, along with detailed ablation and hyperparameter analyses, though validation on larger models and more diverse tasks is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear methodology motivation, intuitive comparison tables with related works, and informative illustrations.
- Value: ⭐⭐⭐⭐ Highly practical and deployment-friendly, but the speedup upper bound and the constraints of domain-specific training somewhat lower its general utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning](../../ICLR2026/llm_reasoning/fastgrpo_accelerating_policy_optimization_via_concurrency-aware_speculative_deco.md)
- [\[NeurIPS 2025\] ARM: Adaptive Reasoning Model](../../NeurIPS2025/llm_reasoning/arm_adaptive_reasoning_model.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](../../ICLR2026/llm_reasoning/temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ACL 2026\] DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning](../../ACL2026/llm_reasoning/delta_dynamic_layer-aware_token_attention_for_efficient_long-context_reasoning.md)
- [\[ICLR 2026\] Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning](../../ICLR2026/llm_reasoning/beyond_markovian_reflective_exploration_via_bayes-adaptive_rl_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
