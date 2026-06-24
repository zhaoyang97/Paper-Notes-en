---
title: >-
  [Paper Note] Parameter-Efficient Fine-Tuning of State Space Models
description: >-
  [ICML 2025][Model Compression][State Space Models] This work presents the first systematic benchmarking of six PEFT methods on State Space Models (SSMs/Mamba). It reveals that LoRA should be applied to linear projection layers rather than SSM modules, and proposes Sparse Dimension Tuning (SDT) to selectively update key state dimensions for more efficient SSM fine-tuning.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "State Space Models"
  - "Parameter-Efficient Fine-Tuning"
  - "LoRA"
  - "Mamba"
  - "Sparse Dimension Tuning"
date: 2026-05-08
content_hash: 57a8c5bd47720d9a
---

# Parameter-Efficient Fine-Tuning of State Space Models

**Conference**: ICML 2025  
**arXiv**: [2410.09016](https://arxiv.org/abs/2410.09016)  
**Code**: [GitHub](https://github.com/furiosa-ai/ssm-peft)  
**Area**: Model Compression/Parameter-Efficient Fine-Tuning  
**Keywords**: State Space Models, Parameter-Efficient Fine-Tuning, LoRA, Mamba, Sparse Dimension Tuning

## TL;DR

This work presents the first systematic benchmarking of six PEFT methods on State Space Models (SSMs/Mamba). It reveals that LoRA should be applied to linear projection layers rather than SSM modules, and proposes Sparse Dimension Tuning (SDT) to selectively update key state dimensions for more efficient SSM fine-tuning.

## Background & Motivation

**Background**: Parameter-Efficient Fine-Tuning (PEFT) methods, such as LoRA and Prompt Tuning, have been widely validated on Transformers. However, State Space Models (SSMs) like Mamba are rapidly gaining traction as emerging architectures, and their unique recurrent structures (matrices $A$, $B$, $C$) are fundamentally distinct from the attention mechanisms in Transformers.

**Limitations of Prior Work**: Directly transferring existing PEFT methods to SSMs yields unstable performance. There is a lack of systematic comparative studies to guide practitioners in selecting appropriate methods and target modules. In particular, the parameters inside SSM modules (such as discretized matrices $\bar{A}$ and $\bar{B}$) have specific mathematical structures, making general low-rank approximations not necessarily applicable.

**Key Challenge**: The state space dimensions ($H$ states $\times$ $D$ channels) of SSMs form a two-dimensional parameter structure. Existing PEFT methods treat all parameters uniformly and compress them equally, ignoring the differences in how much individual dimensions contribute to the final model outputs.

**Goal**: This work aims to answer three key questions: (1) Which PEFT methods are suitable for SSMs? (2) Which modules should be fine-tuned? (3) Can the unique parameter structures of SSMs be leveraged to design more efficient fine-tuning strategies?

**Key Insight**: Establish empirical understanding through large-scale benchmarking, and then propose targeted sparse fine-tuning schemes based on the discrepancies in the channel norms of the SSM state transition matrix $\bar{A}$.

**Core Idea**: Different channels of SSMs have vastly different impacts on the output. Sorting by $\|\bar{A}^{(d)}\|$ and updating only the most critical subset of channels can achieve or even surpass the performance of full fine-tuning.

## Method

### Overall Architecture

The work is divided into two phases. The first phase is a systematic benchmarking that evaluates six PEFT methods—Prompt Tuning, Prefix-tuning, Additional-scan, LoRA, DoRA, and BitFit—across architectures like S4, S6 (Mamba), and Jamba, covering multiple tasks such as GLUE, CelebA, and ImageNet. The second phase, built on findings from the benchmark, introduces SDT, which selectively updates critical state dimensions within the SSM modules.

### Key Designs

1. **Module Selection Strategy (Where to Apply PEFT)**:

    - **Function**: Determine which components of the SSM architecture the PEFT methods should be applied to.
    - **Mechanism**: Separate the Mamba block into the SSM module (matrices $A$, $B$, $C$) and linear projection layers (e.g., `in_proj`, `out_proj`, `x_proj`), and separately evaluate the effect of applying LoRA to these different components.
    - **Design Motivation**: Experiments reveal that applying LoRA directly to the SSM module ($A, B, C$) performs significantly worse than applying it to linear projection layers. This is because SSM matrices have specific mathematical constraints (e.g., $A$ must be negative-definite to guarantee stability), and low-rank perturbations may disrupt these structural properties.

2. **Sparse Dimension Tuning (SDT)**:

    - **Function**: Selectively update the most critical subset of parameters within the SSM module.
    - **Mechanism**: Apply a one-epoch warmup with full SSM updates to evaluate parameter importance, then sort by channel norm $\|\bar{A}^{(d)}\|$ and freeze the $\beta \cdot |D|$ least important channels. Among the remaining active channels, freeze the $\alpha \cdot |H|$ least important states based on state norm. For S4, $\bar{B}$ remains frozen and only $\bar{A}$ and $C$ are updated; for S6, $\bar{A}$, $W_B$, and $W_C$ are updated using only channel-level selection.
    - **Design Motivation**: The norms of different channels in the SSM state transition matrix $\bar{A}$ can differ by several orders of magnitude. Channels with small norms decay extremely fast in response to input signals, contributing almost no useful information. Selectively freezing these "dead channels" saves parameters without losing expressiveness.

3. **Theoretical Guarantee (Theorem 1)**:

    - **Function**: Provide theoretical support for the effectiveness of SDT.
    - **Mechanism**: Prove that SDT-P (the parameterized version of SDT) needs to update only $\lceil D \cdot L^*/L \rceil$ channels to adapt the pretrained model to the target model, where $L^*$ is the effective dimension of the target model and $L$ is the total number of channels in the pretrained model.
    - **Design Motivation**: Empirical findings alone lack rigor. The theorem demonstrates that the sparsity of SDT does not come at the expense of expressiveness but rather exploits the inherent low-rank structure within SSMs.

### Loss & Training

Standard loss functions of downstream tasks (cross-entropy for classification, MSE for regression) are used. For the training strategy, SDT requires one warmup epoch to compute the importance ranking of channels/states, after which selected parameters are frozen for normal fine-tuning. SDT can be combined with LoRA/DoRA: SDT focuses on SSM modules, while LoRA/DoRA is applied to the linear projection layers.

## Key Experimental Results

### Main Results

| Method | Target Module | GLUE Average | Trainable Params |
|:---:|:---:|:---:|:---:|
| Full Fine-tuning | All | 80.5 | 100% |
| Prompt Tuning | Input Embedding | 63.8 | <1% |
| LoRA | SSM Module | 76.9 | ~0.5% |
| LoRA | Linear Projection Layers | **81.2** | ~0.5% |
| LoRA | SSM + LinProj | 80.3-89.8 | ~1% |
| Additional-Scan | SSM Extension | 73.2 | ~2% |

| Method | Architecture | CelebA | ImageNet |
|:---:|:---:|:---:|:---:|
| LoRA (LinProj) | Mamba | **61.0%** | — |
| Additional-Scan | Mamba | 26.9% | — |
| SDT + DoRA | Jamba (GLUE) | **69.2** | — |
| DoRA alone | Jamba (GLUE) | 67.9 | — |

### Ablation Study

| Configuration | GLUE Score | Training Time | Note |
|:---:|:---:|:---:|:---:|
| LoRA (LinProj only) | 81.2 | 410s | Baseline |
| LoRA (SSM only) | 76.9 | — | SSM module not suitable for LoRA |
| SDT + LoRA | ~81.5 | 330s | 19.5% faster |
| Input Injection | 63.8-85.6 | — | Unstable |
| SDT (same budget) vs LoRA (SSM) | ~10x lower MSE | — | SDT performs far better than LoRA on the SSM module |

### Key Findings

- LoRA performs best when applied to linear projection layers (81.2), whereas applying it directly to the SSM module leads to significant degradation (76.9). This is contrary to general practice on Transformers.
- Prompt Tuning and input injection methods perform poorly on SSMs (63.8) because the recurrent nature of SSMs rapidly "forgets" prefix information.
- Under the same parameter budget, SDT achieves ~10x lower reconstruction error on the SSM module compared to LoRA, demonstrating that dimension selection is more suitable for SSMs than low-rank decomposition.
- The combination of SDT and DoRA consistently yields a +1.3 improvement on Jamba, while being faster during training.

## Highlights & Insights

- Precise problem definition: PEFT in SSMs is indeed an unexplored area, and the systematic benchmark provides a valuable reference for the community.
- The counter-intuitive finding that "LoRA is not suitable for the SSM module" is highly valuable, revealing fundamental differences between SSM parameter structures and Transformer attention weights.
- The core intuition of SDT originates from close observations of the norm distribution in matrix $\bar{A}$. The method is simple yet mathematically grounded.
- The actual speedup (19.5%) is achieved through savings in gradient computation enabled by frozen weight sparsity.

## Limitations & Future Work

- The overhead of the warmup epoch is not fully discussed, which could turn into a non-trivial cost for large models (like Jamba-52B).
- The selection of hyperparameters $\alpha$ and $\beta$ in SDT is heuristic, lacking an automated strategy for assessing channel importance.
- Only S4 and Mamba (S6) were evaluated. More recent SSM variants (like Mamba-2 or Griffin) remain to be tested.
- Absolute performance on CelebA (61.0%) remains low, indicating room for improvement in PEFT schemes targeting SSMs on dense visual prediction tasks.

## Related Work & Insights

- **vs LoRA (Transformer)**: On Transformers, LoRA typically performs best when applied to QKV projections, but the equivalent locations in SSMs (matrices $A$/$B$/$C$) turn out to be the worst choices. This warns against blindly transferring PEFT methods across different architectures.
- **vs BitFit**: BitFit (tuning biases only) shows decent performance on SSMs, suggesting that bias terms in SSMs might carry more adaptation information than their counterparts in Transformers.
- **vs Adapter**: The effect of inserting traditional adapters between sequential modeling layers in SSMs warrants further exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic study of SSM PEFT; SDT is mathematically driven and cleanly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 PEFT methods, 3 SSM architectures, and various downstream tasks, with thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured; the benchmarking portion offers direct, actionable guidelines for practitioners.
- **Value**: ⭐⭐⭐⭐ Highly valuable to the SSM community; the ideas of SDT can be extended to other models with structured parameters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [\[ICML 2025\] MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles](moragent_parameter_efficient_agent_tuning_with_mixture-of-roles.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](../../ACL2025/model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](../../CVPR2025/model_compression/expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[ICML 2025\] LoRA Fine-Tuning without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs](lora_fine-tuning_without_gpus_a_cpu-efficient_meta-generation_framework_for_llms.md)

</div>

<!-- RELATED:END -->
