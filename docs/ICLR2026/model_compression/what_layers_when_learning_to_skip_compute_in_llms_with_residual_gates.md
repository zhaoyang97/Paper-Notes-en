---
title: >-
  [Paper Note] What Layers When: Learning to Skip Compute in LLMs with Residual Gates
description: >-
  [ICLR 2026][Model Compression][residual gating] This paper proposes GateSkip—a method that inserts a sigmoid-linear gate at the output of each Attention/MLP branch in a decoder-only Transformer…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "residual gating"
  - "token-level layer skipping"
  - "adaptive depth"
  - "GateSkip"
  - "inference acceleration"
date: 2026-05-08
content_hash: 2ce54c71e3e80bc3
---

# What Layers When: Learning to Skip Compute in LLMs with Residual Gates

**Conference**: ICLR 2026
**arXiv**: [2510.13876](https://arxiv.org/abs/2510.13876)  
**Area**: Model Compression
**Keywords**: residual gating, token-level layer skipping, adaptive depth, GateSkip, inference acceleration

## TL;DR

This paper proposes GateSkip—a method that inserts a sigmoid-linear gate at the output of each Attention/MLP branch in a decoder-only Transformer, jointly optimizes gate sparsity and language modeling objectives during fine-tuning, and at inference time deterministically skips low-importance tokens layer-by-layer using a quantile threshold over gate values, thereby achieving token-level adaptive depth. On Llama 8B, GateSkip saves 15% compute while retaining >90% accuracy; on instruction-tuned models, the full-compute variant actually improves accuracy over the baseline, and ~50% savings still matches the baseline. The method is orthogonal and composable with INT4 quantization, structured pruning, and self-speculative decoding.

## Background & Motivation

**Background**: Current LLMs allocate the same amount of compute to every token at every layer, without distinguishing token difficulty or layer importance. This uniform allocation leads to significant waste in latency-sensitive and resource-constrained settings. Adaptive computation aims to dynamically adjust the computational depth for each token; the dominant approaches fall into two categories: routing-based methods (e.g., Mixture-of-Depths, MoD) and early exit methods.

**Limitations of Prior Work**: Routing-based methods such as MoD place a router at each layer to make discrete top-k decisions about which tokens to skip. This hard routing is training-unstable, requires complex load-balancing losses, and mostly demands integration from the pretraining stage, making post-hoc addition infeasible. Early exit methods attach auxiliary LM heads at intermediate layers and halt when a confidence threshold is reached. However, auxiliary heads alter the hidden state distribution established during pretraining, are difficult to calibrate, and degrade sharply on long-sequence generation tasks.

**Key Challenge**: Existing adaptive depth methods either rely on discrete decisions that destabilize training or require modifications to the pretraining pipeline, making lightweight post-hoc integration into existing models impractical. What is needed is a mechanism that is differentiable, can be added post-training, and enables deterministic skipping at inference time.

**Key Insight**: The authors observe that the residual stream in a Transformer is itself the control channel for information propagation—each layer's output is added back via $h_{\ell+1} = h_\ell + o_\ell$. Inserting a learnable gate that scales $o_\ell$ before it is added to the residual stream provides continuous control over each layer's contribution. The gate remains differentiable during training for gradient stability, and can be thresholded at inference time to yield deterministic skip/no-skip decisions.

**Core Idea**: Placing a sigmoid-linear gate at the residual stream exit reframes the discrete routing problem as a continuously differentiable gating learning problem. During training, the model learns sparse gates while maintaining language modeling quality; at inference, gates are rank-sorted to enable token-level, layer-wise skipping.

## Method

### Overall Architecture

GateSkip adds a lightweight gating module at the output of each Attention and MLP branch in a standard decoder-only Transformer. During training, the gates and the backbone are jointly optimized to maximize sparsity while preserving language modeling performance. At inference, the gate values for all tokens at each layer are aggregated into scalar importance scores, and a quantile threshold determines which tokens skip that layer; skipped tokens have their hidden states and KV cache copied directly to the next layer.

### Key Designs

1. **Residual Gating Mechanism**:

    - Function: Before each layer's Attention/MLP output is added back to the residual stream, a learnable gate modulates the information flow.
    - Mechanism: The standard residual connection $h_{\ell+1} = h_\ell + o_\ell$ is replaced by $h_{\ell+1} = h_\ell + o_\ell \odot g_\ell(h_\ell)$, where $g_\ell(h_\ell) = \sigma(W_G h_\ell + b)$ is a sigmoid-activated linear projection. $W_G \in \mathbb{R}^{H \times H}$ produces an output vector matching the hidden dimension, enabling fine-grained per-dimension gating. The bias is initialized to 5 so that $\sigma(5) \approx 1$, ensuring that early in training the model behaves close to the original pretrained model.
    - Design Motivation: The sigmoid output lies continuously in $[0, 1]$ and is differentiable, avoiding the discrete routing instability of MoD. Placing the gate at the module exit rather than the entrance provides more direct gradient signal—the gate learns "how much of this module's output is worth adding to the residual stream," rather than "whether to enter this module."

2. **Quantile-based Token Selection**:

    - Function: At inference time, converts continuous gate values into deterministic binary skip/no-skip decisions.
    - Mechanism: For each layer $\ell$, the gate vector of each token is collapsed into a scalar importance score $\bar{g}_{\ell,i} = \frac{1}{H}\sum_k g_\ell(h_\ell)_{i,k}$. A quantile threshold $\tau = \text{Quantile}(\{\bar{g}_{\ell,i}\}, 1-b)$ is then computed over all token scores in the current batch, where $b$ is the retention budget. Tokens whose scores fall below $\tau$ skip the layer, with their hidden states and KV cache copied upward.
    - Design Motivation: Quantile thresholding naturally accommodates the distribution differences across layers (the paper finds that gate distributions within a layer are narrow but vary substantially across layers), avoiding the calibration difficulties of a single global threshold.

3. **Progressive Budget Decay Training**:

    - Function: Gradually reduces the token retention budget during training, allowing the model to learn to maintain performance as an increasing fraction of tokens are skipped.
    - Mechanism: The training budget decays linearly from $b_1 = 1.0$ to $b_2 = 0.8$: $b_t = b_1 - (b_1 - b_2)\frac{t}{T_{\text{total}}}$. Token skipping via the quantile threshold is applied during training, but gradients are back-propagated normally through retained tokens.
    - Design Motivation: Starting training with a high skip rate causes instability. Progressive decay allows the model to first learn reliable gate importance estimates before adapting to increasingly aggressive skipping.

### Loss & Training

The total training loss is $\mathcal{L} = \mathcal{L}_{CE} + \lambda_S \mathcal{L}_S$. The first term is the standard next-token prediction cross-entropy. The second term is a gate sparsity penalty using L2 distance: $\mathcal{L}_S = \frac{1}{N_L H}\sum_\ell \sum_k \|g_\ell(h_\ell)_k\|_2$, encouraging gate values toward zero. The paper sets $\lambda_S = 0.1$. Ablations show that L2 outperforms L1 and KL divergence variants—L1 achieves stronger log-likelihood at zero skip rate but degrades rapidly with skipping, while KL divergence yields the best generation quality at zero skip rate but collapses at even marginal skip rates. All parameters (backbone and gates) are jointly updated with AdamW.

## Key Experimental Results

### Main Results: Comparison with Prior Adaptive Computation Methods (Llama-3.2-1B)

| Method | Generation @0% Skip | Generation @15% Skip | Generation @25% Skip | Log-Likelihood @0% | Log-Likelihood @30% |
|------|:---:|:---:|:---:|:---:|:---:|
| Llama-1B (original) | **30.97** | - | - | 49.12 | - |
| Random Skipping | - | 1.67 | 0.67 | - | 23.62 |
| CALM (saturation) | 3.43 | 3.43 | 3.43 | 30.73 | 30.73 |
| FREE (saturation) | 11.57 | 11.57 | 11.57 | 36.02 | 36.02 |
| LayerSkip | 10.65 | 10.65 | 10.65 | 38.25 | **38.25** |
| MoD | 20.83 | 3.96 | 2.91 | 44.18 | 29.33 |
| **GateSkip (Ours)** | 23.53 | **22.14** | **17.67** | 47.35 | 31.74 |

GateSkip substantially outperforms all baselines on generation tasks: at 15% skip rate, accuracy is 22.14%—5.6× higher than MoD and more than 6× higher than CALM/FREE. CALM, FREE, and LayerSkip exhibit generation accuracy that does not change with skip rate (their adaptive mechanisms effectively fail in long-sequence generation), whereas GateSkip exhibits a smooth accuracy-efficiency tradeoff curve.

### Ablation Study

| Design Choice | Generation @15% Skip | Log-Likelihood @15% | Notes |
|----------|:---:|:---:|------|
| **Vector gating (default)** | **23.2** | **37.8** | Per-dimension gating, full model |
| Scalar gating | 20.4 | 36.8 | −2.8 accuracy; granularity insufficient |
| Shared gating | 20.7 | 38.4 | Inter-layer variation suppressed |
| Attention-only skip | 14.9 | 37.5 | MLP layers also contain redundancy |
| MLP-only skip | 7.8 | 32.0 | Skipping MLP has larger impact |
| MLP gate (nonlinear) | 18.5 | 33.9 | Extra parameters cause overfitting |
| Gate at module entrance | 1.0 | 35.7 | **Catastrophic failure**—entrance cannot receive gradient signal from module output |
| Frozen backbone | 12.7 | 37.5 | Backbone adaptation is critical for skipping |

The most critical finding: gate placement at the module exit vs. entrance yields a dramatic difference (23.2 vs. 1.0), validating the core design assumption that the gate must receive gradient signal from downstream of the module output to learn which computations can be skipped.

### Scalability Across Model Sizes

| Model | Generation @0% | Generation @15% | Generation @25% | Accuracy Retention @15% |
|------|:---:|:---:|:---:|:---:|
| Llama-3.2-1B | 26.8 | 23.2 | 19.8 | 86.6% |
| Llama-3.2-3B | 45.0 | 43.3 | 42.1 | 96.2% |
| Llama-3.1-8B | 57.3 | 55.0 | 53.6 | 96.0% |
| Gemma-2-2B | 38.0 | 36.1 | 34.8 | 95.0% |

Larger models retain accuracy better at the same skip rate. Llama 3B retains 96.2% accuracy at 15% skip, suggesting that the proportion of redundant computation is higher in larger models. Consistent results across architectures (Llama vs. Gemma) validate the generality of the method.

### Instruction-Tuned Model (Llama-3B-Instruct)

| Setting | Generation Task | Log-Likelihood |
|------|:---:|:---:|
| Llama-3B-Instruct (original) | 36.5 | 46.3 |
| + Random Skipping @20% | 0.5 | 34.7 |
| + **GateSkip @0% (full compute)** | **49.0 (+12.5)** | 36.7 |
| + GateSkip @20% | 49.0 | 38.8 |
| + GateSkip @30% | 45.6 | 32.9 |
| + GateSkip @45% | 35.0 | 31.0 |

A counterintuitive finding: on instruction-tuned models, GateSkip in full-compute mode improves over the baseline by 12.5 points. This suggests the gate acts as an adaptive regularizer, suppressing unnecessary computation noise. Even at 20% skip (49.0), the gated model exceeds the ungated baseline (36.5).

### Composition with Orthogonal Efficiency Techniques

| Combination | Generation @0% | Generation @25% | LL @15% |
|------|:---:|:---:|:---:|
| GateSkip (32-bit) | 45.0 | 42.1 | 35.6 |
| GateSkip + INT4 Quantization | 42.5 | 41.0 | 35.6 |
| GateSkip + ShortGPT Pruning | - | - | 31.1 |
| GateSkip + Self-Speculative Decoding | - | - | 39.4 |

After INT4 quantization, generation accuracy is retained at 94.4% and log-likelihood is unchanged. Combined with self-speculative decoding, LL reaches 39.4 at 15–30% savings (outperforming GateSkip alone at 37.8), confirming that GateSkip is composable with multiple orthogonal efficiency techniques.

### Key Findings

- Gate placement is the most critical design choice: exit vs. entrance gating differs by 20 accuracy points at 5% skip rate (25.5 vs. 5.5); entrance gating nearly completely fails.
- Vector gating significantly outperforms scalar gating (+2.8), demonstrating that per-dimension control of information flow is more effective than a global switch.
- Joint fine-tuning of the backbone is essential (unfrozen vs. frozen: 10.5-point gap)—the backbone must adaptively encode importance cues into hidden states for the gate to read.
- The learned sparsity patterns are interpretable: BOS tokens consistently receive the highest gate values in early layers (acting as information anchors), punctuation maintains high gate values across layers (information aggregation), and deep-layer gates increasingly focus selectively on content words.
- End-to-end latency measurements on vLLM show that 50% token skipping yields a 16.3% throughput improvement (2698→3141 tokens/s) and 70% skipping yields a 35% improvement.

## Highlights & Insights

- **The residual stream transforms from a passive channel to an active control mechanism**: Conventionally, residual connections are viewed merely as gradient propagation aids. GateSkip demonstrates that inserting gates at the residual stream exit enables fine-grained adaptive depth control, opening a low-invasiveness direction for Transformer efficiency optimization.
- **Instruction tuning + gating = unexpected accuracy gain**: This is the paper's most counterintuitive finding. The gate does not merely reduce computation—it simultaneously acts as an adaptive regularizer, suppressing layer computations that make a negative contribution to the final output. This implies that Transformers contain substantial "harmful computation," not merely "redundant computation."
- **Elegant decoupling of differentiable training and deterministic inference**: During training, gates are continuous sigmoid values ensuring smooth gradients; at inference, a quantile threshold converts them to hard skip decisions. This "continuous training, discrete inference" paradigm is more stable than MoD's straight-through estimator and simpler than the confidence calibration required by early exit methods.
- **Gate values as a Transformer interpretability tool**: Gate values directly reflect "which tokens matter at which layers," revealing that BOS tokens serve as information anchors in early layers (echoing recent "attention sink" research) and can serve as a free by-product for analyzing information flow in Transformers.

## Limitations & Future Work

- **Scale limitations**: Validation is limited to 8B parameters; experiments on 70B+ models are absent. Larger models may exhibit higher redundancy, potentially yielding greater benefits from GateSkip.
- **Limited task coverage**: Evaluation is restricted to English reasoning and language modeling, without validation on multimodal (VLM), code generation, or long-context (>128K) settings.
- **Limited end-to-end speedup**: The paper primarily reports theoretical FLOP savings; measured throughput improvements (16–35%) are notably lower than theoretical values (50–70%), indicating that the engineering overhead of token masking and KV cache copying is non-trivial.
- **Insufficient exploration of gating granularity**: The current method operates at token×layer granularity. Token×head granularity (differentially skipping individual attention heads) could potentially yield finer-grained compute allocation.
- **Lack of dynamic budget scheduling**: Inference currently uses a fixed budget per layer, despite ablation results revealing that different layers have different degrees of redundancy. Dynamically adjusting per-layer budgets based on input content may further improve efficiency.

## Related Work & Insights

- **vs. Mixture-of-Depths (MoD)**: MoD uses a discrete top-k router to select tokens and requires load-balancing losses, resulting in training instability. GateSkip uses a continuous sigmoid gate for stable differentiable training and discretizes only at inference. GateSkip outperforms MoD on generation tasks by a factor of 5–10×.
- **vs. LayerSkip/CALM/FREE**: The adaptive mechanisms of these three methods completely fail in long-sequence generation (accuracy does not vary with skip rate), fundamentally because their exit/skip decisions cannot be correctly triggered in generation mode. GateSkip's token-level gates are computed independently for each generated token, avoiding this problem.
- **vs. Early Exit methods**: Early exit requires additional LM heads and alters the hidden state distribution. GateSkip adds only lightweight linear layers (0.004%–4% parameter overhead) without modifying the original representation space.

## Rating

- Novelty: ⭐⭐⭐⭐ The residual gating idea is concise and elegant, and the decoupled "continuous training, discrete inference" design is clever, though gating itself is not an entirely new concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four model scales × two architectures × dual evaluation (generation + LL) × comprehensive ablations × three orthogonal technique compositions × end-to-end latency measurements × gate interpretability analysis—extremely thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ Method description is concise and clear; experiments are well-organized; each ablation choice is accompanied by a clear comparison and explanation.
- Value: ⭐⭐⭐⭐⭐ Directly practical for efficient LLM inference; post-training compatibility and composability with quantization and pruning lower the barrier to engineering deployment considerably.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](compute-optimal_quantization-aware_training.md)
- [\[NeurIPS 2025\] Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs](../../NeurIPS2025/model_compression/matryoshka_pilot_learning_to_drive_black-box_llms_with_llms.md)
- [\[ICLR 2026\] Draft-based Approximate Inference for LLMs](draft-based_approximate_inference_for_llms.md)
- [\[CVPR 2026\] ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation](../../CVPR2026/model_compression/arche_autoregressive_residual_compression_with_hyp.md)
- [\[ICML 2026\] Demystifying When Pruning Works via Representation Hierarchies](../../ICML2026/model_compression/demystifying_when_pruning_works_via_representation_hierarchies.md)

</div>

<!-- RELATED:END -->
