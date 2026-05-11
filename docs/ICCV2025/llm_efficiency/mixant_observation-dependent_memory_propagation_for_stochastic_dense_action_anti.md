---
title: >-
  [Paper Note] MixANT: Observation-dependent Memory Propagation for Stochastic Dense Action Anticipation
description: >-
  [ICCV 2025][LLM Efficiency][Action Anticipation] This paper proposes MixANT, which introduces input-dependence into the forgetting gate (A matrix) of Mamba via a Mixture-of-Experts approach. A lightweight router dynamica…
tags:
  - "ICCV 2025"
  - "LLM Efficiency"
  - "Action Anticipation"
  - "Mamba"
  - "Mixture of Experts"
  - "State Space Models"
  - "Dense Prediction"
date: 2026-05-08
content_hash: 583ff146cb37c7cc
---

# MixANT: Observation-dependent Memory Propagation for Stochastic Dense Action Anticipation

**Conference**: ICCV 2025
**arXiv**: [2509.11394](https://arxiv.org/abs/2509.11394)
**Code**: [talalwasim.github.io/MixANT](https://talalwasim.github.io/MixANT/)
**Area**: LLM Efficiency (Sequence Modeling / Mamba Architecture Improvement)
**Keywords**: Action Anticipation, Mamba, Mixture of Experts, State Space Models, Dense Prediction

## TL;DR

This paper proposes MixANT, which introduces input-dependence into the forgetting gate (A matrix) of Mamba via a Mixture-of-Experts approach. A lightweight router dynamically selects context-aware A matrices to control temporal memory propagation, achieving state-of-the-art performance across all three dense action anticipation benchmarks: 50Salads, Breakfast, and Assembly101.

## Background & Motivation

### Problem Definition

Stochastic Long-term Dense Action Anticipation: given an observed video frame sequence covering $\alpha$ proportion of the total video length, the task requires frame-wise prediction of action labels for the future $\beta$ proportion, along with multiple plausible future prediction samples (25 samples) to account for behavioral uncertainty. The prediction horizon typically spans several minutes.

### Limitations of Prior Work

**Quadratic complexity of Transformers**: DiffAnt employs Transformers for dense prediction, but sequence lengths can reach thousands of frames, making quadratic complexity a bottleneck.

**Input-independent A matrix in Mamba**: MANTA achieves strong results on long sequences using standard Mamba, but Mamba only renders three parameters (B, C, Δ) input-dependent, while **the A matrix — the core parameter governing temporal memory propagation — remains static**:

$$h_t = \bar{\mathbf{A}} h_{t-1} + \bar{\mathbf{B}} x_t$$

The A matrix determines how much past information is retained or forgotten. In action anticipation, different contexts (e.g., preparing a salad vs. brewing coffee) require different memory strategies, which a static A matrix cannot accommodate.

**Zero-padded input sequences**: Future frames are zero-padded in anticipation tasks; ideally the A matrix should selectively ignore these zero segments, but a static A cannot do so.

**Technical challenges of making A input-dependent directly**: QK multiplication would destroy Mamba's sub-quadratic complexity advantage; large MLPs introduce excessive parameters.

### Core Motivation

The A matrix governs hidden state evolution, analogous to the forget gate in RNNs, and is critical for sequence modeling. Different semantic contexts demand different forgetting strategies. The key question is: how can A be made input-dependent without sacrificing computational efficiency?

**Core Idea**: Maintain multiple A matrices via a Mixture-of-Experts approach, and use a lightweight router to select the most relevant A matrix based on input features, achieving input-dependence while preserving computational efficiency.

## Method

### Overall Architecture

MixANT consists of $K=15$ sequential processing blocks. The first $K_0=3$ blocks use standard bidirectional Mamba, while the remaining $K_E=12$ blocks use the proposed MixMamba blocks. The overall model is embedded within a diffusion framework: starting from Gaussian noise $\hat{Y}_T$, the model iteratively denoises over $T$ steps to produce dense predictions $\hat{Y}_0$, with DDIM sampling used at inference to generate 25 samples.

### Key Designs

#### 1. **S6+ Algorithm in the MixMamba Layer**

- **Function**: Maintains $E=5$ expert A matrices $\{\mathbf{A}_1, \mathbf{A}_2, ..., \mathbf{A}_E\} \in \mathbb{R}^{E \times D \times N}$ and dynamically selects among them based on input.

- **Routing Mechanism**:
$$\gamma(x) = \text{softmax}(W_g \cdot \text{mean}(x))$$

$$\mathbf{A}(x) = \mathbf{A}_{\hat{e}}, \quad \hat{e} = \arg\max_e \gamma_e(x)$$

where $W_g \in \mathbb{R}^{D \times E}$ is a learnable projection matrix and $\gamma(x) \in \mathbb{R}^{B \times E}$ is the routing vector. Notably, routing is computed solely from observed frame features $F_{t,1:P}^{k-1}$.

- **Design Motivation**: Computational overhead is minimal (one mean pooling + matrix multiplication + softmax), preserving Mamba's sub-quadratic complexity. Hard selection via argmax ensures only one A matrix is used per forward pass, avoiding the cost of mixing multiple A matrices.

#### 2. **Hybrid Architecture Design (Static-then-Mixed)**

- **Function**: The first 3 blocks use standard Mamba; the remaining 12 use MixMamba.
- **Mechanism**: Early layers extract generic low-level features suited to uniform processing; later layers require semantically differentiated decisions, making expert routing appropriate.
- **Design Motivation**: Introducing routing too early forces specialization before meaningful features have been extracted, degrading performance. Ablation experiments confirm $K_0 = 3$ as optimal; both more and fewer static blocks hurt performance.

#### 3. **Unified Router Configuration**

- **Function**: The forward and backward MixSSM units within a MixMamba layer share a single routing vector $\gamma$.
- **Mechanism**: A single routing vector is computed; the forward pass selects $\mathbf{A}_{\hat{e}}$, and the backward pass automatically uses the corresponding $\overleftarrow{\mathbf{A}}_{\hat{e}}$.
- **Design Motivation**: Independent routing would undermine bidirectionality — the forward and backward passes should learn two directions of the same A matrix rather than two entirely different A matrices.

### Loss & Training

$$\mathcal{L}_{total} = (1 - \lambda_{lb}) \mathcal{L}_{rec} + \lambda_{lb} \cdot \mathcal{L}_{lb}$$

- **Reconstruction Loss**: $\mathcal{L}_{rec} = \|Y - \hat{Y}_0\|^2$ (L2 loss between predictions and one-hot ground-truth labels)
- **Load Balancing Loss**:

$$\mathcal{L}_{lb} = \sum_{k=K_0+1}^{K} \text{KL}\left(\frac{C^k}{\sum_e C^k_e} \Big\| \mathcal{U}(E)\right)$$

This encourages uniform utilization of all experts, where $C^k_e = \sum_{b=1}^B \gamma^k_e(F_{t,1:P}^{k-1}(b))$ records the cumulative routing weight assigned to each expert within a batch. During training, noise steps $t$ are sampled via the diffusion process; at inference, DDIM sampling is used to generate 25 prediction samples.

## Key Experimental Results

### Main Results

Breakfast dataset ($\alpha = 0.2$):

| Method | Mean MoC ($\beta$=0.1) | Mean MoC ($\beta$=0.5) | Top-1 ($\beta$=0.1) | Top-1 ($\beta$=0.5) |
|--------|:---:|:---:|:---:|:---:|
| UAAA | 15.7 | 13.0 | 28.9 | 28.0 |
| DiffAnt | 24.7 | 22.3 | 31.3 | 30.1 |
| GTDA | 24.0 | 20.6 | 51.2 | 45.0 |
| MANTA | 27.7 | 23.8 | 55.5 | 46.9 |
| **MixANT** | **29.6** | **25.0** | **57.1** | **48.4** |

Assembly101 dataset ($\alpha = 0.2$, 202 action classes):

| Method | Mean MoC ($\beta$=0.1) | Top-1 ($\beta$=0.1) |
|--------|:---:|:---:|
| GTDA | 6.4 | 18.0 |
| MANTA | 6.7 | 16.9 |
| **MixANT** | **8.0** | **20.3** |

50Salads dataset ($\alpha = 0.2$):

| Method | Mean MoC ($\beta$=0.1) | Top-1 ($\beta$=0.1) |
|--------|:---:|:---:|
| MANTA | 28.6 | 68.3 |
| **MixANT** | **30.3** | **71.5** |

MixANT outperforms all prior methods across all three datasets, all observation ratios, and all prediction horizon settings in nearly every configuration.

### Ablation Study

| Configuration | Mean MoC | Top-1 MoC | Note |
|---------------|:---:|:---:|------|
| E=1 (= standard Mamba) | 27.7 | 55.5 | Baseline |
| E=3 | 28.8 | 56.4 | Clear gain from adding experts |
| **E=5** | **29.6** | **57.1** | **Optimal** |
| E=8 | 28.9 | 55.8 | Too many experts hurts performance |
| $K_0=0$ (all MixMamba) | 28.4 | 55.9 | Premature routing is harmful |
| **$K_0=3$** | **29.6** | **57.1** | **Optimal** |
| $K_0=6$ | 28.7 | 56.2 | Too many static blocks limits capacity |
| Independent routing | 28.5 | 55.7 | Breaks bidirectionality |
| **Unified routing** | **29.6** | **57.1** | **Preserves bidirectional consistency** |
| Without load balancing | 28.6 | 56.2 | Some experts undertrained |
| **With load balancing** | **29.6** | **57.1** | **Uniform expert utilization** |

### Key Findings

- **Input-dependence of the A matrix is critical for anticipation**: Moving from E=1 to E=5 yields +1.9% Mean MoC and +1.6% Top-1, demonstrating that significant gains can be achieved solely by changing how A matrices are selected.
- **Optimal expert count exists**: E=5 is best; too many experts (E=8) leads to sparse training signals, with each expert receiving insufficient supervision.
- **Load balancing loss is essential**: Without this constraint, Expert 2 is selected nearly 50% of the time while E1 and E4 are rarely used, resulting in wasted model capacity.
- **Expert selection patterns reveal semantic structure**: t-SNE visualizations show that, despite being trained only with atomic action supervision, expert selection patterns spontaneously cluster by high-level activity category (e.g., "making a salad" and "brewing tea" consistently engage different expert combinations), demonstrating that the A matrices have learned semantically-aware memory strategies.
- **Largest relative gains on Assembly101**: On this challenging 202-class dataset, Mean MoC improves relatively by approximately 20–32%, indicating that more complex tasks benefit most from input-dependent memory control.

## Highlights & Insights

1. **Precisely identifies a key weakness of Mamba**: Through theoretical analysis and empirical evidence, the paper demonstrates that the input-independence of the A matrix is a bottleneck for Mamba in long-term anticipation tasks — a finding with broad implications for Mamba architecture design.
2. **MoE applied to SSM parameters rather than MLPs**: Unlike prior work that applies MoE to MLPs external to the Mamba block, this paper is the first to introduce mixture-of-experts directly into the core A matrix parameter inside the Mamba block.
3. **Interpretability of expert selection**: The t-SNE clustering results provide strong evidence that A matrices learn semantically-aware memory strategies, enhancing the credibility of the proposed method.
4. **Zero additional inference overhead**: Router computation is negligible (mean pooling + matmul + softmax), and hard selection via argmax means each layer still uses only one A matrix at inference time, matching the complexity of standard Mamba.

## Limitations & Future Work

1. **Validated only on action anticipation**: Whether the MoE-based A matrix approach generalizes to other Mamba applications such as video understanding and language modeling remains to be verified.
2. **Hard-selection routing**: argmax introduces non-differentiability; training relies on gradient flow through the softmax probabilities of the routing vector. Soft MoE variants may yield different behavior.
3. **Fixed number of experts**: All MixMamba layers use the same number of experts; adaptive expert counts may be more effective.
4. **Simple router design**: Only mean pooling and linear projection are used; more sophisticated routing strategies (e.g., attention-based routing) may further improve performance.
5. **Overhead from the diffusion framework**: While the primary contribution lies in the MixMamba layer, the overall method relies on 25-step DDIM sampling and diffusion training, which remains computationally expensive.

## Related Work & Insights

- MANTA demonstrated that Mamba outperforms Transformers and convolutional architectures for dense action anticipation; this work builds on that foundation by identifying and addressing the limitation of the A matrix.
- BlackMamba and MoE-Mamba apply MoE to the MLP layers of Mamba; this paper is the first to introduce MoE into the A matrix itself.
- The SSD framework in Mamba-2, which employs structured matrix constraints, offers an alternative perspective worth comparing against the proposed approach.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to propose a mixture-of-experts approach for the A matrix; problem identification is precise and the solution is concise and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation on three datasets with multi-dimensional ablations (expert count, static block count, routing configuration, load balancing) and expert selection visualization.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; the entry point via Mamba's parameter input-dependence is natural and well-structured.
- **Value**: ⭐⭐⭐⭐ — The proposed improvement to the Mamba architecture has broad applicability beyond action anticipation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[CVPR 2026\] StoryTailor: A Zero-Shot Pipeline for Action-Rich Multi-Subject Visual Narratives](../../CVPR2026/llm_efficiency/storytailora_zero-shot_pipeline_for_action-rich_multi-subject_visual_narratives.md)
- [\[ICLR 2026\] TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection](../../ICLR2026/llm_efficiency/tokenseek_memory_efficient_fine_tuning_via_instance-aware_token_selection.md)
- [\[ICLR 2026\] Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents](../../ICLR2026/llm_efficiency/did_you_check_the_right_pocket_cost-sensitive_store_routing_for_memory-augmented.md)
- [\[NeurIPS 2025\] Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads](../../NeurIPS2025/llm_efficiency/let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e.md)

</div>

<!-- RELATED:END -->
