---
title: >-
  [Paper Note] msf-CNN: Patch-based Multi-Stage Fusion with Convolutional Neural Networks for TinyML
description: >-
  [NeurIPS 2025][LLM/NLP][TinyML] This paper proposes msf-CNN, a multi-stage patch-based fusion optimization technique based on a directed acyclic graph (DAG) shortest-path algorithm. By efficiently searching for the optimal fusion configuration of CNNs, msf-CNN achieves 50%–87% reduction in peak RAM usage compared to existing methods (MCUNetV2, StreamNet) across various microcontrollers (ARM Cortex-M, RISC-V, ESP32), while maintaining controllable computational overhead.
tags:
  - NeurIPS 2025
  - LLM/NLP
  - TinyML
  - CNN fusion
  - microcontroller
  - patch-based inference
  - memory optimization
  - DAG shortest path
  - IoT
date: 2026-05-08
content_hash: f6200143dd384911
---

# msf-CNN: Patch-based Multi-Stage Fusion with Convolutional Neural Networks for TinyML

**Conference**: NeurIPS 2025
**arXiv**: [2505.11483](https://arxiv.org/abs/2505.11483)
**Code**: [GitHub](https://github.com/TinyPART/msf-CNN)
**Area**: LLM/NLP
**Keywords**: TinyML, CNN fusion, microcontroller, patch-based inference, memory optimization, DAG shortest path, IoT

## TL;DR

This paper proposes msf-CNN, a multi-stage patch-based fusion optimization technique based on a directed acyclic graph (DAG) shortest-path algorithm. By efficiently searching for the optimal fusion configuration of CNNs, msf-CNN achieves 50%–87% reduction in peak RAM usage compared to existing methods (MCUNetV2, StreamNet) across various microcontrollers (ARM Cortex-M, RISC-V, ESP32), while maintaining controllable computational overhead.

## Background & Motivation

The AI of Things (AIoT) paradigm drives the deployment of deep neural networks onto microcontrollers (MCUs), yet MCUs are extremely resource-constrained:

**Severe memory bottleneck**: Typical IoT devices have RAM < 50 KiB and Flash < 250 KiB (RFC7228), whereas a single convolutional layer of quantized ResNet-34 alone requires approximately 414.72 KiB of RAM.

**Potential of patch-based fusion**: Fusion techniques can save up to 95% of RAM and decouple input size from memory usage. However, existing approaches suffer from the following issues:
   - High recomputation overhead within fused blocks
   - Input size constraints that affect applications such as medical imaging and audio processing
   - Implementations tightly coupled to specific hardware (e.g., ARM Cortex-M7) and specific models (e.g., MobileNetV2 inverted blocks)

**Underexplored search space**: MCUNetV2 fuses only head layers heuristically, while StreamNet uses brute-force search over a limited space; neither explores the potential of multiple fusion blocks.

## Method

### Overall Architecture

The core mechanism of msf-CNN is to model a CNN as a DAG of data nodes, reformulate fusion optimization as a shortest-path problem on the graph, and solve for the optimal fusion configuration using classical graph algorithms.

Pipeline: CNN → DAG modeling (nodes = tensors, edges = operators/fused blocks) → encode RAM and MAC into edge weights → graph algorithm searches for the optimal path → code generation and deployment.

### Key Designs

**DAG representation**: The CNN is modeled as $G = (V, E)$, where:
- Nodes $v_0, \ldots, v_n$ represent input/output tensors of consecutive layers
- An edge $e = v_n \to v_{n+1}$ corresponds to a single-layer operator; $e = v_n \to v_{n+m}$ ($m > 1$) corresponds to a fused block
- A complete computation path $S$ from $v_0$ to $v_n$ corresponds to a fusion configuration

**RAM encoding**: The RAM usage of each edge is:

$$P_{e_i} = I + O + Buf$$

where $I, O$ are the sizes of the input/output tensors and $Buf$ is the fusion buffer size (with $Buf = 0$ for non-fused layers). The peak RAM of the full path is:

$$P_S = \max_{j=1\ldots n} P_{e_{i_j}}$$

**Computational overhead encoding**: The total computation cost of a path is the sum of MACs:

$$C_S = \sum_{j=1}^{n} C_{e_{i_j}}$$

The computational overhead factor is defined as $F = C_S / C_{vanilla}$.

**Dual optimization problems**:
- **P1**: Minimize peak RAM subject to $F < F_{max}$
- **P2**: Minimize computation cost subject to $P < P_{max}$

**Graph search strategy for P1**: The method iteratively removes the edge with the highest RAM in subgraphs, solves the shortest-path problem on each subgraph, constructs a candidate solution set $\{S_0, S_1, \ldots\}$, and selects the optimal solution satisfying the constraint. Complexity is reduced from $O(2^{V-2})$ to $O(V^3)$.

**Pruning for P2**: Edges exceeding the RAM limit are directly removed, and the shortest path is computed on the remaining graph.

**Iterative operator optimization**:
- **Iterative global average pooling**: Processes inputs element-by-element; RAM for a $7\times7$ pooling operation is compressed to 2% of the original, with no additional computational overhead.
- **Iterative fully connected layers**: Multiplies each input element with the corresponding weight column and accumulates; RAM for a $1024 \to 256$ layer is compressed to 20%.

### Loss & Training

msf-CNN is a compiler optimization tool and does not involve model training. It is implemented on top of microTVM v0.16.0, converting models to an intermediate representation, rewriting the computation graph, and generating C code for deployment on various MCU boards via RIOT OS.

## Key Experimental Results

### Main Results

**Minimum peak RAM usage (kB)**:

| Method | MBV2-w0.35 | MN2-vww5 | MN2-320K |
|--------|-----------|----------|----------|
| Vanilla (no fusion) | 194.44 | 96 | 309.76 |
| MCUNetV2 | 63 | 45 | 215 |
| StreamNet | 66 | 44 | 208 |
| **msf-CNN** | **8.56** | **15.37** | **51.16** |

RAM reduction: msf-CNN reduces RAM by 87%–96% compared to Vanilla, and by 65%–87% compared to MCUNetV2/StreamNet.

**Inference latency (ms) under msf-CNN minimum-RAM configuration**:

| MCU | MBV2-w0.35 | MN2-vww5 | MN2-320K |
|-----|-----------|----------|----------|
| STM32F767 (Cortex-M7 216MHz) | 1996.8 (2.5×) | 1723.0 (3.4×) | 19329.9 (4.4×) |
| ESP32-S3 (Xtensa 240MHz) | 6748.2 | 5974.1 | 76763.6 |
| ESP32-C3 (RISC-V 160MHz) | 6792.7 | 6248.9 | 73713.8 |
| SiFive (RISC-V 320MHz) | 10000.0 | OOM | OOM |

### Ablation Study

**Optimization results under different constraints (MBV2-w0.35)**:

| Constraint Type | Constraint Value | Peak RAM (kB) | Overhead F |
|----------------|-----------------|--------------|-----------|
| P1: $F_{max}$ | 1.1 | 67.91 | 1.1 |
| P1: $F_{max}$ | 1.3 | 21.29 | 1.3 |
| P1: $F_{max}$ | 1.5 | same as above | same as above |
| P1: $F_{max}$ | ∞ | 7.89 | 1.68 |
| P2: $P_{max}$ | 16 kB | 15.34 | 1.38 |
| P2: $P_{max}$ | 32 kB | 25.67 | 1.25 |
| P2: $P_{max}$ | 128 kB | 83.07 | 1.02 |
| P2: $P_{max}$ | 256 kB | 181.44 | 1.0 |

The heuristic method (MCUNetV2, fusing only head layers) achieves 32.08 kB RAM / F=1.59, whereas msf-CNN can identify solutions with lower overhead at a comparable RAM budget.

### Key Findings

1. msf-CNN can deploy MBV2-w0.35 even onto a SiFive board with only 16 kB of RAM.
2. MCU architecture has a significant impact on latency: the ESP32-S3 (240MHz Xtensa) is actually slower than the ESP32-C3 (160MHz RISC-V) on larger models.
3. Under reasonable constraints ($F_{max} = 1.3$), msf-CNN achieves substantial memory savings with minimal increase in RAM overhead.
4. After adding the CMSIS backend, msf-CNN's Pareto frontier approaches that of StreamNet, which applies hardware-specific optimizations for ARM.
5. Iterative global pooling and fully connected layers further compress RAM without any additional computational overhead.

## Highlights & Insights

1. **Elegance of graph-theoretic modeling**: Reformulating the fusion search as a classical shortest-path/minimax-path problem enables efficient solutions via well-established graph algorithms.
2. **Exploration of multiple fusion blocks**: This work is the first to systematically explore combinations of multiple fusion blocks within a CNN, rather than fusing only head layers.
3. **Hardware generality**: The open-source implementation supports three mainstream MCU architectures: ARM Cortex-M, RISC-V, and ESP32.
4. **Search space pruning**: Complexity is reduced from exponential $O(2^{V-2})$ to polynomial $O(V^3)$, enabling optimization of large networks in seconds on a PC.
5. **Iterative operator design**: The iterative implementations of global pooling and fully connected layers are simple, effective, and can be seamlessly integrated into fused blocks.

## Limitations & Future Work

1. **CNN-only support**: The current framework handles only convolutional neural networks and does not support attention mechanisms or RNNs.
2. **Single caching strategy**: Only the H-Cache scheme is implemented; more flexible strategies such as 2D Cache (used by StreamNet) are not integrated.
3. **Fixed iteration granularity**: Each iteration produces only one output element; this parameter has a significant impact on memory and computation but is not optimized.
4. **Higher latency than StreamNet**: The lack of hardware-specific optimizations (e.g., CMSIS instruction sets) results in higher latency in the general-purpose implementation.
5. **Non-MCU platforms unvalidated**: Although theoretically applicable to CPU/GPU/FPGA, experiments are limited to MCUs.

## Related Work & Insights

- **MCUNetV2**: Its heuristic strategy of fusing only head layers is simple but suboptimal; msf-CNN demonstrates that systematic search yields better solutions.
- **StreamNet**: Its 2D caching strategy effectively reduces recomputation; integration into msf-CNN is planned for future work.
- **TinyNAS / Once-for-All**: NAS-based methods require retraining, whereas msf-CNN requires no modification to the model itself.
- **Insight**: The DAG + shortest-path modeling paradigm can be extended to other optimization dimensions, such as attention layers and mixed-precision quantization.

## Rating

- ⭐⭐⭐⭐ **Novelty**: The graph-theoretic modeling combined with multi-stage fusion search is original, elegantly transforming a combinatorial optimization problem into a classical graph algorithm.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Validated across 6 hardware platforms, 3 models, and multiple constraint settings, providing comprehensive coverage.
- ⭐⭐⭐⭐ **Value**: The open-source, multi-platform tool offers direct practical utility for embedded AI engineers.
- ⭐⭐⭐ **Limitations**: Restricted to CNNs, limited caching strategies, and lacking systematic comparison with NAS-based methods.

**Overall**: ⭐⭐⭐⭐ (3.5/5) — A solid systems optimization work with an elegant graph-theoretic formulation and convincing multi-platform validation. The primary weaknesses are its restriction to CNN architectures and the limited caching strategies. The work delivers tangible engineering value to the TinyML community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](cat_circular-convolutional_attention_for_sub-quadratic_transformers.md)
- [\[NeurIPS 2025\] Opinion Maximization in Social Networks by Modifying Internal Opinions](opinion_maximization_in_social_networks_by_modifying_internal_opinions.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [\[AAAI 2026\] ProFuser: Progressive Fusion of Large Language Models](../../AAAI2026/llm_nlp/profuser_progressive_fusion_of_large_language_models.md)
- [\[NeurIPS 2025\] PluralisticBehaviorSuite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies](pluralistic_behavior_suite_stress-testing_multi-turn_adherence_to_custom_behavio.md)

<!-- RELATED:END -->
