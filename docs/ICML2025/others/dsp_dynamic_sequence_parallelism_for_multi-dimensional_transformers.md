---
title: >-
  [Paper Note] DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers
description: >-
  [ICML2025][Sequence Parallelism] To address the issue where existing sequence parallelism methods in multi-dimensional Transformers (e.g., spatio-temporal attention models in video generation) can only shard along a single dimension, leading to massive redundant communication, this paper proposes Dynamic Sequence Parallelism (DSP). By dynamically switching the parallel dimension between computation stages (instead of communicating inside modules) using efficient all-to-all op…
tags:
  - "ICML2025"
  - "Sequence Parallelism"
  - "Multi-Dimensional Transformer"
  - "Dynamic Switching"
  - "All-to-All Communication"
  - "Video Generation"
date: 2026-05-08
content_hash: 28322759ddae6b69
---

# DSP: Dynamic Sequence Parallelism for Multi-Dimensional Transformers

**Conference**: ICML2025  
**arXiv**: [2403.10266](https://arxiv.org/abs/2403.10266)  
**Code**: [https://github.com/NUS-HPC-AI-Lab/VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys)  
**Area**: Others  
**Keywords**: Sequence Parallelism, Multi-Dimensional Transformer, Dynamic Switching, All-to-All Communication, Video Generation

## TL;DR

To address the issue where existing sequence parallelism methods in multi-dimensional Transformers (e.g., spatio-temporal attention models in video generation) can only shard along a single dimension, leading to massive redundant communication, this paper proposes Dynamic Sequence Parallelism (DSP). By dynamically switching the parallel dimension between computation stages (instead of communicating inside modules) using efficient all-to-all operations for resharding, DSP achieves a 32.2% to 10× end-to-end throughput improvement and reduces communication overhead by at least 50%.

## Background & Motivation

**Background**: Scaling multi-dimensional Transformers to long sequences is critical in fields such as video generation (OpenSora, Latte), image generation, protein structure prediction, and spatio-temporal information processing. Long sequences lead to massive activation memory overhead and degraded computational speed, necessitating sequence parallelism to distribute the load.

**Limitations of Prior Work**: Current mainstream sequence parallelism methods—Ring Attention, Megatron-SP, and DeepSpeed-Ulysses—all belong to **Embedded Sequence Parallelism**, which can only shard along a single sequence dimension. For multi-dimensional Transformers, this strategy suffers from severe communication redundancy:
   - Ring Attention using P2P ring communication is inefficient in high-latency environments
   - Megatron-SP is restricted by the number of attention heads, requiring additional all-gather and reduce-scatter operations
   - DeepSpeed-Ulysses is likewise restricted by the number of attention heads

**Key Challenge**: A key characteristic of multi-dimensional Transformers is that **computations along different sequence dimensions are independent** (e.g., temporal attention and spatial attention are calculated separately in video models). However, embedded methods overlook this characteristic, introducing a significant amount of unnecessary communication operations inside the modules to switch parallel dimensions.

**Goal**: How to exploit the independent computation characteristics of each dimension in multi-dimensional Transformers to design a sequence parallelism method that dynamically switches parallel dimensions between computation stages, thereby minimizing communication overhead.

**Key Insight**: The authors observe that since a multi-dimensional Transformer only needs to perform attention along one dimension at each computation stage, the parallel sharding dimension can be set orthogonal to the current computation dimension—i.e., sharding the "dimension not involved in the calculation". This completely eliminates the need for communication within the module.

**Core Idea**: Dynamically switch sharding dimensions using all-to-all between computation stages so that the parallel dimension is always orthogonal to the computation dimension, fundamentally eliminating redundant communication inside modules.

## Method

### Overall Architecture

The workflow of DSP is as follows:
- **Input**: A multi-dimensional sequence tensor $\mathbf{X} \in \mathbb{R}^{[B, S_1, S_2, \ldots, S_K, C]}$, distributed across $N$ GPUs
- **Model Start**: Shards the full sequence along a certain dimension to each GPU via the **Split** operation
- **Intermediate Computation**: Each computation stage processes attention for one sequence dimension; between two stages, a **Dynamic Switch** (all-to-all) is used to switch the sharding dimension, ensuring that the computation dimension of the next stage is not sharded
- **Model End**: Collects shards from all GPUs via the **Gather** operation to reconstruct the full sequence
- **Output**: An output tensor with the same shape as the input

The core mechanism is: **resharding only occurs between stages, not within stages**, meaning DSP is completely decoupled from the computation logic within modules.

### Key Designs

1. **Dynamic Switch**:

    - **Function**: Switches the sharding dimension of the sequence from $S_i$ to $S_j$ between two computation stages
    - **Mechanism**: Re-distributes the tensor from $\mathbb{R}^{[B, S_1, \ldots, S_i/N, \ldots, S_j, \ldots, S_K, C]}$ to $\mathbb{R}^{[B, S_1, \ldots, S_i, \ldots, S_j/N, \ldots, S_K, C]}$ via a single all-to-all collective communication, i.e., $\mathbf{Y} = \text{DynamicSwitch}(\mathbf{X}, i, j)$
    - **Design Motivation**: The all-to-all communication volume is only $M/N$ (where $M$ is the total sequence size), which is significantly smaller than the repetitive all-reduce/all-gather operations required inside modules in embedded methods. Moreover, Dynamic Switch is only executed between stages, with a lower frequency than intra-module communications in embedded methods
    - **Difference from Prior Methods**: Embedded methods (such as Ring Attention) require continuous communication (P2P transmission of KV) during attention computation, whereas DSP does not interfere with attention computation at all and only performs a dimension switch before and after

2. **Split**:

    - **Function**: Shards an unsharded full sequence along a specific dimension across GPUs
    - **Mechanism**: A purely local operation that transitions from state $\hat{s}$ to state $s_i$, with a communication volume of 0
    - **Design Motivation**: Used at the model's front end to distribute the input sequence to each device for the first time. Since it is only a local reshape + slice, no communication is required

3. **Gather**:

    - **Function**: Restores a sharded sequence back to the full sequence
    - **Mechanism**: Collects shards via an all-gather operation, with a communication volume of $M$
    - **Design Motivation**: Used at the end of the model or for rare global operations that require accessing all dimensions. Since it is only used at the very beginning or end of the model, its overhead is negligible

4. **State Transition System**:

    - **Function**: Uniformly describes all parallel states using state markers $s_i$ (sharded along dimension $i$) and $\hat{s}$ (unsharded)
    - **Mechanism**: Three primitives (Switch, Split, Gather) cover all possible state transitions
    - **Design Motivation**: Provides a unified formal abstraction, enabling DSP to adapt to any multi-dimensional Transformer architecture. It can automatically generate a communication plan by simply labeling which dimension needs to be complete for each module in the computation graph

### Adaptability & Flexibility

DSP is decoupled from module computation logic, yielding excellent generality:
- **No Module Constraints**: No modifications are needed for internal implementations of modules like attention or FFN
- **No Head Count Limits**: Unlike Megatron-SP and DeepSpeed-Ulysses, it is not restricted by the number of attention heads
- **Easy to Integrate**: Provides high-level APIs, allowing integration into PyTorch-based distributed frameworks with only a few lines of code

### Theoretical Communication Analysis

Assuming a multi-dimensional Transformer has $K$ sequence dimensions, DSP requires $K-1$ Dynamic Switches in a complete forward pass, each with a communication volume of $M/N$, resulting in a total communication volume of $(K-1) \cdot M/N$. In contrast, embedded methods (which perform all-gather + reduce-scatter inside each attention module) require a communication volume of $O(K \cdot M)$. The advantage of DSP becomes even more prominent when $N$ is large.

## Key Experimental Results

### Main Results: End-to-End Throughput Comparison

The paper validates the effectiveness of DSP on video generation models (OpenSora/STDiT, Latte) and protein structure prediction (Evoformer of AlphaFold2):

| Model | Method | Sequence Parallelism Degree | Throughput Gain | Communication Reduction |
|------|------|-----------|----------|-----------|
| OpenSora (STDiT) | DSP vs Ring-Attention | N=2 | +32.2% | ≥50% |
| OpenSora (STDiT) | DSP vs Ring-Attention | N=4 | +75% | ≥50% |
| OpenSora (STDiT) | DSP vs Ring-Attention | N=8 | ~3× | ≥75% |
| Latte | DSP vs DeepSpeed-Ulysses | N=4 | +50%~2× | ≥50% |
| Latte | DSP vs Megatron-SP | N=8 | up to 10× | ≥75% |

### Ablation Study on Communication Overhead

| Method | Communication Primitive | Communication Frequency per Layer | Message Size per Comm. | Total Communication Volume (Relative) |
|------|---------|------------|-----------|---------------|
| Ring Attention | P2P (ring) | $O(N)$ per head | KV blocks | 1.0× (baseline) |
| DeepSpeed-Ulysses | all-to-all × 2 | 2 per layer | $M/N$ | ~0.8× |
| Megatron-SP | all-gather + reduce-scatter | 2 per layer | $M$ | ~1.2× |
| **DSP** | all-to-all × 1 (Switch) | 1 per stage transition | $M/N$ | **≤0.5×** |

### Key Findings

- **Communication Volume is the Decisive Factor**: The core advantage of DSP lies in the dramatic reduction of communication volume. At $N=8$, the communication volume of DSP is less than 25% of Ring Attention, which directly translates into a significant throughput gain
- **Higher Parallelism Leads to Greater Advantage**: As the number of GPUs $N$ increases, DSP’s relative advantage expands from 32.2% to 10×. This occurs because the communication overhead of embedded methods grows linearly with $N$, whereas the message size per Switch in DSP ($M/N$) actually decreases
- **Universally Effective Across Different Architectures**: DSP significantly improves throughput for both STDiT (separated spatio-temporal attention) and Latte (alternating spatio-temporal attention), demonstrating excellent generality
- **Equally Effective on Protein Structure Prediction (Evoformer)**: This shows that DSP is not limited to video generation, but is applicable to all multi-dimensional Transformer scenarios

## Highlights & Insights

- **Decoupling Communication with Computational Structure**: The most ingenious design is identifying the computational independence of each dimension in multi-dimensional Transformers and "pushing" the resharding operations between computation stages. This is a system optimization strategy of "changing data layout without changing computation logic," which can be transferred to other models with phase-based computational characteristics
- **Elegant State Transition Abstraction**: Describing all parallel strategies in a unified manner using the $s_i / \hat{s}$ state system and three primitives (Switch, Split, Gather) is not only concise but also makes automated scheduling possible. This abstraction can inspire the design of other parallel strategies
- **Clever Application of All-to-All Communication**: All-to-all is typically utilized for distributing the head dimension in traditional SP, but DSP applies it to switch sequence dimensions. This yields a communication volume of only $M/N$, which is far lower than the $M$ of all-gather. This insight serves as a reference for future designs of parallel strategies
- **Zero Intrusiveness**: DSP does not modify any code inside the modules, only inserting resharding operations between modules, which achieves extremely low code intrusiveness. This design philosophy ("performing the right communication at the right place") is highly valuable for systems developers to learn

## Limitations & Future Work

- **Only Applicable to Multi-Dimensional Transformers**: The core assumption of DSP is that the model contains stages computed independently along different sequence dimensions. For standard one-dimensional LLMs (such as GPT), DSP cannot be applied directly. However, with the rise of multimodal large models, multi-dimensional structures are becoming increasingly common
- **Automation Level of Dynamic Scheduling**: The paper provides high-level APIs, but the insertion points of Switch still need to be manually specified by users. Future work could incorporate compiler techniques (e.g., XLA, Triton) to automatically analyze the computation graph and insert optimal Switch points
- **Combination with Tensor Parallelism**: The paper primarily compares DSP with other SP methods, but in practice, large-scale training often employs a hybrid of TP + SP + DP. How to optimally integrate DSP into 3D/4D parallel strategies remains an open question
- **Impact of Heterogeneous Dimension Lengths**: When $S_1 \gg S_2$ (e.g., temporal frames in a video are far fewer than spatial pixels), is the efficiency of Switch affected? The paper does not analyze this scenario in depth
- **Cross-Node Communication Latency**: DSP's all-to-all is highly efficient intra-node (NVLink/NVSwitch), but when crossing nodes (InfiniBand), the latency increases. The paper lacks a detailed scalability analysis across multiple nodes

## Related Work & Insights

- **vs Ring Attention (Li et al., 2021; Liu et al., 2023)**: Ring Attention uses ring-style P2P communication to pass KV during calculation, resulting in a large communication volume proportional to $N$. DSP completely avoids communication during computation, only performing a single all-to-all between stages. The communication volume is only $M/N$, and the advantage expands as $N$ increases
- **vs DeepSpeed-Ulysses (Jacobs et al., 2023)**: Ulysses uses two all-to-all operations to convert between the head dimension and sequence dimension, but is constrained by the head count ($N \leq H$). DSP's Switch is independent of the head count, offering better scalability
- **vs Megatron-SP (Korthikanti et al., 2022)**: Megatron-SP is extended based on tensor parallelism, requiring all-gather + reduce-scatter, resulting in a communication volume of $O(M)$ and similarly limited by the head count. DSP achieves a lower order of magnitude with a communication volume of $O(M/N)$
- **Insights**: DSP's concept of "communicating between stages rather than within stages" can be extended to other systems with multi-phase computations, such as expert parallelism in Mixture of Experts, multi-scale feature pyramid networks, etc.

## Rating

- Novelty: ⭐⭐⭐⭐ The abstraction of dynamically switching parallel dimensions is a brand-new sequence parallelism paradigm, although the core communication primitive (all-to-all) is not a new technology
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple scenarios of video generation and protein prediction, with significant throughput improvements and comparisons covering various baselines
- Writing Quality: ⭐⭐⭐⭐⭐ The formal definition is clear, the state transition system is elegant, illustrations are intuitive, and the global logic is smooth
- Value: ⭐⭐⭐⭐ Highly practical value for distributed training/inference of multi-dimensional Transformers, especially in the context of the rapid development of video generation and multimodal large models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training](../../ICML2026/others/amdp_asynchronous_multi-directional_pipeline_parallelism_for_large-scale_models_.md)
- [\[ECCV 2024\] Superpixel-Informed Implicit Neural Representation for Multi-Dimensional Data](../../ECCV2024/others/superpixel-informed_implicit_neural_representation_for_multi-dimensional_data.md)
- [\[ICML 2025\] Residual Matrix Transformers: Scaling the Size of the Residual Stream](residual_matrix_transformers_scaling_the_size_of_the_residual_stream.md)
- [\[ICML 2025\] How Do Transformers Learn Variable Binding in Symbolic Programs?](how_do_transformers_learn_variable_binding_in_symbolic_programs.md)
- [\[ACL 2025\] Bregman Conditional Random Fields: Sequence Labeling with Parallelizable Inference](../../ACL2025/others/bregman_conditional_random_fields_sequence_labeling_with_parallelizable_inferenc.md)

</div>

<!-- RELATED:END -->
