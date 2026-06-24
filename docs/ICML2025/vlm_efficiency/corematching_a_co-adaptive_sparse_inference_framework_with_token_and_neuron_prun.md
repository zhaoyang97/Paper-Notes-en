---
title: >-
  [Paper Note] CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models
description: >-
  [ICML 2025][Multimodal Efficiency][Inference Acceleration] This work is the first to reveal the intrinsic correlation between token sparsity and neuron sparsity in VLMs—core neurons and core tokens mutually determine and reinforce each other. Based on this correlation, the authors propose the CoreMatching co-adaptive sparse inference framework, achieving simultaneous acceleration in both pre-filling and decoding stages, which leads to a 5× FLOPs reduction and 10× overall spee…
tags:
  - "ICML 2025"
  - "Multimodal Efficiency"
  - "Inference Acceleration"
  - "Token Pruning"
  - "Neuron Sparsity"
  - "Vision-Language Models"
  - "Co-adaptive Sparsity"
date: 2026-05-08
content_hash: e309bb9b788d5dc9
---

# CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models

**Conference**: ICML 2025  
**arXiv**: [2505.19235](https://arxiv.org/abs/2505.19235)  
**Code**: [GitHub](https://github.com/wangqinsi1/2025-ICML-CoreMatching/tree/main)  
**Area**: Multimodal VLM  
**Keywords**: Inference Acceleration, Token Pruning, Neuron Sparsity, Vision-Language Models, Co-adaptive Sparsity

## TL;DR

This work is the first to reveal the intrinsic correlation between token sparsity and neuron sparsity in VLMs—core neurons and core tokens mutually determine and reinforce each other. Based on this correlation, the authors propose the CoreMatching co-adaptive sparse inference framework, achieving simultaneous acceleration in both pre-filling and decoding stages, which leads to a 5× FLOPs reduction and 10× overall speedup.

## Background & Motivation

VLMs (such as LLaVA, BLIP, LLaMA) perform excellently on tasks like image question answering. However, because they must process a large number of image tokens, the inference time and memory overhead far exceed those of text-only LLMs, limiting practical deployment.

The two existing acceleration paradigms each have limitations:

**Token Sparsity**: Exploits the high redundancy of image tokens to keep only a few key tokens. Representative methods like PruMerge (retains 20% tokens) and FastV (discards >50% tokens starting from the second layer) primarily accelerate the **pre-filling stage**. The decoding stage only gains limited benefits from reduced KV cache.

**Neuron Sparsity**: Exploits the fact that a large number of neurons in FFN layers are inactive, skipping the computation of inactive neurons. Representative methods like DejaVu, PowerInfer, and CoreInfer primarily accelerate the **decoding stage**. In the pre-filling stage, the acceleration is limited due to the large number of tokens and low sparsity.

**Core Problem**: Does a deep correlation exist between these two sparsity spaces? Can they be unified to achieve comprehensive acceleration?

This work is the first to systematically study this problem. The authors discover that the tokens whose activation patterns match the core neurons most closely are precisely the most important tokens for the output. This "matching mechanism" organically links the two sparsity paradigms.

## Method

### Overall Architecture

CoreMatching simultaneously computes **Core Neurons** and **Core Tokens** during a single pre-filling forward pass. Subsequently, in the pre-filling stage, only Core Tokens are processed (token-dimension sparsity). In the decoding stage, only Core Neurons are used for FFN computation, and only Core Tokens are retained in the KV cache (neuron-dimension sparsity). The entire pipeline consists of three steps:

1. **Compute Core Neurons**: Count the activation distribution of all tokens in the FFN block, and select a set of most frequently activated neurons.
2. **Match Core Tokens**: Compute the size of the intersection between the activated neurons of each token and the Core Neurons, and select the token subset with the largest intersection as the Core Tokens.
3. **Sparse Inference**: Subsequent layers only transmit Core Tokens; the decoding stage only uses Core Neurons for FFN computation.

### Key Designs

#### 1. Definition and Verification of Core Neurons

For a single token $x$, its token-wise core neurons are defined as the top-$\rho$ highest activated neurons:

$$\mathcal{C}_\rho(x) = \{n \mid a_n \geq \text{Percentile}(A^+, \rho)\}$$

where $A^+ = \{a_n \mid a_n > 0\}$ is the set of positive activations.

For an entire sentence $\mathbf{s} = [x_1, \ldots, x_M]$, sentence-wise core neurons are the top-$\beta$ most frequently appearing neurons in the core neurons of all tokens:

$$\mathcal{C}_\rho^\beta(\mathbf{s}) = \{n \mid f_\rho(n; \mathbf{s}) \geq \text{Percentile}(f_\rho(\mathbf{s}), \beta)\}$$

Experimental verification (LLaVA-1.5-7B, TextVQA): Retaining only 60% of core neurons achieves 55.8% accuracy (against 57.8% for the full model), indicating that a tiny fraction of core neurons can sustain performance.

Furthermore, visualization demonstrates that core neurons are **predictable**: when input semantics are sufficient, core neurons remain almost unchanged.

#### 2. Definition of Core Tokens—Matching from Neurons to Tokens

Key Insight: The size of the intersection between the active neuron set $\Gamma(x)$ of token $x$ and the sentence-wise core neurons $\mathcal{C}_\rho^\beta(\mathbf{s})$ reflects the importance of the token. Tokens with larger intersections contribute more to the model output.

Core Tokens are defined as the set of tokens whose intersection size exceeds a threshold $\tau$:

$$\mathcal{T}_{\text{core}} = \{x \mid |\Gamma(x) \cap \mathcal{C}_\rho^\beta(\mathbf{s})| \geq \tau\}$$

The threshold $\tau$ is adaptively determined using the **Maximum Geometric Distance** method: after sorting the intersection sizes of all tokens, the point on the sorted curve furthest from the diagonal is found as the boundary. This avoids the need to manually set a retention ratio, allowing different samples to retain different numbers of tokens.

#### 3. Projection-guided Criterion—Theoretical Explanation

Why are Core Tokens superior to traditional attention score-based methods? The authors propose the **Projection-guided Criterion** for theoretical analysis.

Traditional methods only use attention scores (i.e., attention weights after softmax) to measure token importance. However, the authors point out that the real contribution of a token to the final output hidden state $h$ depends not only on the attention weight but also on the **directional alignment** (angle information) of the value vector.

Specifically, the contribution of token $x_i$ to the final output hidden state $h$ can be decomposed as:

$$\text{Contribution}(x_i) \propto \alpha_i \cdot \|v_i\| \cdot \cos\theta_i$$

where $\alpha_i$ is the attention weight, $v_i$ is the value vector, and $\theta_i$ is the angle between $v_i$ and the output direction.

Core Tokens naturally tend to select tokens with large $\cos\theta_i$ (aligned directions) and high matching with core neurons' activation patterns, making it theoretically superior to methods relying solely on $\alpha_i$.

#### 4. Two-Stage Acceleration Mechanism

| Stage | Sparsity Dimension | Acceleration Principle |
|------|----------|----------|
| Pre-filling | Token Sparsity | Processes only Core Tokens, reducing attention and FFN computational complexity |
| Decoding | Neuron Sparsity + KV Cache Sparsity | FFN computes using only Core Neurons; KV cache stores only Core Tokens |

The two sparsities are **co-adaptively reinforced**: Core Neurons guide the selection of Core Tokens, while Core Tokens determine the computation range of subsequent Core Neurons, forming a positive feedback loop.

### Loss & Training

CoreMatching is a **training-free** inference acceleration framework requiring no additional training or fine-tuning. Both core neurons and core tokens are computed online using statistics from the forward pass. This is a major advantage, allowing direct application to existing VLMs.

## Key Experimental Results

### Main Results

Evaluated on 10 image understanding benchmarks, with LLaVA-1.5-7B as the base model:

| Dataset | Metric | CoreMatching | FastV | PruMerge | Full Model |
|--------|------|-------------|-------|----------|----------|
| TextVQA | Acc | 55.8% | 54.2% | 53.5% | 57.8% |
| VQAv2 | Acc | Competitive | Significant drop | Moderate drop | Baseline |
| GQA | Acc | Outperforms baseline | Baseline | Baseline | Baseline |
| ScienceQA | Acc | Outperforms baseline | Baseline | Baseline | Baseline |

Hardware acceleration performance (NVIDIA Titan Xp):

| Metric | CoreMatching | Description |
|------|-------------|------|
| FLOPs Reduction | 5× | Combined token + neuron sparsity |
| Overall Speedup | 10× | End-to-end inference time |
| Pre-filling Speedup | 2.1× | Dominated by token sparsity |
| Decoding Speedup | 9.2× | Dominated by neuron sparsity |

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Token Sparsity Only | Pre-filling speedup, limited decoding | Upper bound of traditional methods |
| Neuron Sparsity Only | Decoding speedup, limited pre-filling | CoreInfer paradigm |
| CoreMatching (Co-adaptive) | Simultaneous speedup in both stages | 1+1 > 2 effect |
| Attention-based Token Selection | Worse than CoreMatching | Missing angle information |
| Core Neuron Matching Selection | Optimal | Activation pattern alignment |
| Fixed Threshold vs. Adaptive Threshold | Adaptive is more stable | Maximum Geometric Distance generalizes well |

Core Neurons retention ratio verification (TextVQA / LLaVA-1.5-7B):

| Retention Ratio | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|----------|-----|-----|-----|-----|-----|
| Accuracy | 45.1% | 53.2% | 55.8% | 56.3% | 57.8% |

### Key Findings

1. **5× FLOPs reduction + 10× overall speedup** (on NVIDIA Titan Xp).
2. **2.1× pre-filling speedup + 9.2× decoding speedup**, achieving significant acceleration in both stages for the first time.
3. Outperforms SOTA baselines across 3 hardware devices.
4. Core neurons remain almost unchanged after input semantics are sufficient (predictability); retaining 40% achieves 53.2% accuracy.
5. Core tokens visualization precisely covers the semantic regions of the image most relevant to the textual query.

## Highlights & Insights

- **Fills an important gap**: This is the first work to systematically study the connection between token sparsity and neuron sparsity, finding they are not independent but can guide each other.
- **Elegant unified framework**: Adds only a single matching step in the pre-filling stage to obtain sparsity in both dimensions simultaneously, with a clean design.
- **Training-free**: No need to train predictors or perform extra fine-tuning, directly using forward pass statistics, leading to extremely low deployment barriers.
- **Theoretical contribution**: The Projection-guided Criterion theoretically explains why pure attention scores are insufficient—the value direction information must be considered simultaneously.
- **Adaptive threshold**: The Maximum Geometric Distance method avoids manual ratio adjustment, automatically determining the number of retained tokens for different samples.

## Limitations & Future Work

1. **Model Scale**: Primarily validated on LLaVA-1.5-7B. Generalizability to larger models (13B/70B) and newer architectures remains to be verified.
2. **Video/Multi-image Scenarios**: Currently targets only single-image understanding tasks; applicability to video VLMs needs exploration.
3. **Integration with Compression Methods**: Combining CoreMatching with quantization could further reduce deployment costs.
4. **Layer-wise Adaptive Strategy**: Sparsity characteristics may differ across layers, which warrants layer-wise strategy research.
5. **Long-context Multi-turn Dialogue**: Stability of core neurons under long-text scenarios requires more validation.

## Related Work & Insights

- **CoreInfer** (Wang et al., 2024): The origin of the core neuron concept, focusing only on LLM neuron sparsity. This work extends it to VLMs and combines it with token sparsity.
- **FastV** (Chen et al., 2025): A token pruning baseline based on attention scores, where this work demonstrates that attention scores are not sufficiently accurate.
- **PruMerge** (Shang et al., 2024): Uses image-text attention average scores for token selection and merging.
- **DejaVu / PowerInfer**: MLP predictor-driven neuron sparsity methods.
- **Insights**: There is a deep connection between two seemingly independent efficiency optimization paradigms, which can be generalized to attention head sparsity + token sparsity, MoE expert selection + token routing, etc.

## Rating

| Dimension | Score (1-10) | Description |
|------|-------------|------|
| Novelty | 8 | First to reveal the intrinsic connection of two sparsity paradigms, with a novel perspective. |
| Theoretical Depth | 7 | Projection-guided Criterion provides a meaningful theoretical explanation. |
| Experimental Thoroughness | 7 | 10 benchmarks + 3 hardwares, but model scale is relatively single. |
| Value | 8 | Training-free, 10× speedup, low deployment barrier. |
| Writing Quality | 7 | Clear structure, intuitive diagrams. |
| **Overall** | **7.5** | Solid work in inference acceleration, with inspiring core insights. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EffiVLM-Bench: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models](../../ACL2025/vlm_efficiency/effivlm_bench_vlm_acceleration.md)
- [\[ICML 2025\] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](sparsevlm_visual_token_sparsification_for_efficient_vision-language_model_infere.md)
- [\[ICLR 2026\] LearnPruner: Rethinking Attention-based Token Pruning in Vision Language Models](../../ICLR2026/vlm_efficiency/learnpruner_rethinking_attention-based_token_pruning_in_vision_language_models.md)
- [\[ACL 2025\] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](../../ACL2025/vlm_efficiency/token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)
- [\[ICCV 2025\] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](../../ICCV2025/vlm_efficiency/feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)

</div>

<!-- RELATED:END -->
