---
title: >-
  [Paper Note] Mixture of Lookup Experts
description: >-
  [ICML 2025][LLM Efficiency][Mixture-of-Experts] MoLE (Mixture of Lookup Experts) is proposed, which modifies the input of routing experts in MoE from intermediate features to embedding tokens. This allows experts to be reparameterized into lookup tables (LUTs) and offloaded to storage devices before inference, thereby achieving inference speeds and memory footprints comparable to dense models while maintaining MoE-level performance.
tags:
  - "ICML 2025"
  - "LLM Efficiency"
  - "Mixture-of-Experts"
  - "Lookup Table"
  - "Expert Offloading"
  - "Inference Acceleration"
  - "Model Deployment"
date: 2026-05-08
content_hash: 4ab4354791784232
---

# Mixture of Lookup Experts

**Conference**: ICML 2025  
**arXiv**: [2503.15798](https://arxiv.org/abs/2503.15798)  
**Code**: [Available](https://github.com/JieShibo/MoLE)  
**Area**: LLM Efficiency  
**Keywords**: Mixture-of-Experts, Lookup Table, Expert Offloading, Inference Acceleration, Model Deployment

## TL;DR

MoLE (Mixture of Lookup Experts) is proposed, which modifies the input of routing experts in MoE from intermediate features to embedding tokens. This allows experts to be reparameterized into lookup tables (LUTs) and offloaded to storage devices before inference, thereby achieving inference speeds and memory footprints comparable to dense models while maintaining MoE-level performance.

## Background & Motivation

**Background**: MoE architectures reduce inference FLOPs by activating only a subset of experts and have been widely adopted by large language models like Mixtral and DeepSeek-MoE.

**Limitations of Prior Work**: Although MoE reduces computation, all expert parameters must still be loaded into VRAM. For instance, with Mixtral-8x7B, each token activates only 13B parameters, but the total parameter count is up to 46B, requiring at least 92GB of VRAM (FP16).

**Limitations of Existing Solutions**: Expert offloading methods place experts in CPU memory or disk and load them on demand. However, they suffer from two major flaws:

**High Communication Latency**: Different experts must be loaded at each decoding step, and the single-step loading latency under PCIe 4.0 can reach up to 0.7s.

**Unfriendly to Batch Inference**: Different samples may select different experts. When the batch size increases, more (or even all) experts must be loaded.

**Key Challenge**: MoE experts need to be loaded onto the GPU for computation, which is the root cause of communication overhead. If experts can be made "computation-free," there would be no need to load bulky parameters.

**Ours**: The core idea of MoLE is to restrict the input of experts to embedding tokens (a finite discrete set), precompute the experts into lookup tables (LUTs) after training, and simply query the tables based on the input IDs during inference instead of performing matrix operations.

**Key Insight**: Leveraging the property that embedding layer outputs correspond one-to-one with input IDs, the method transforms FFN computation into precomputation plus table-lookup operations, fundamentally eliminating communication and computation bottlenecks for experts.

## Method

### Overall Architecture

MoLE exhibits different structures in the training and inference phases:

- **Training Phase**: Similar to MoE, it contains $N$ routing experts and one shared expert, with two key differences: (1) the input to the routing experts is the embedding token instead of intermediate features; (2) all experts are activated instead of top-k.
- **Inference Phase**: Routing experts are reparameterized as lookup tables (LUTs) and offloaded to storage devices. During inference, results are retrieved directly by looking up tables using the input ID, eliminating the need for any expert computation.

### Key Designs

1. **Embedding Token as Expert Input**:

   In traditional MoEs, experts receive intermediate features $\boldsymbol{h}$ as input. MoLE changes the input of routing experts to the embedding token $\boldsymbol{e} = \text{Embedding}(i)$. Since the output of the embedding layer is solely determined by the input ID, the possible inputs to the experts are restricted to $|\mathcal{V}|$ discrete values, which is the vocabulary size.

   The computation of the MoLE layer during the training phase is:

    $\boldsymbol{h}' = \sum_{j=1}^{N} \big(g_j \cdot \text{FFN}_j(\boldsymbol{e})\big) + \text{FFN}_{shared}(\boldsymbol{h}) + \boldsymbol{h}$

   where $g_j$ is computed by the router based on the intermediate feature $\boldsymbol{h}$.

   **Design Motivation**: Although the experts no longer receive contextual information, the router and the shared expert still use intermediate features, and the expert outputs affect the behavior of subsequent attention layers. Consequently, the model retains its context modeling capability.

2. **All-Expert Activation Strategy**:

   Traditional MoEs employ top-k sparse activation to reduce FLOPs. Since MoLE's routing experts do not require computation during inference (only table lookups), all $N$ experts can be activated:

    $\{g_j\}_{j=1}^{N} = \text{SoftMax}(\{\boldsymbol{h} \cdot \boldsymbol{r}_j\}_{j=1}^{N})$

   **Design Motivation**: Full activation eliminates the top-k selection mechanism of MoEs, making the model fully differentiable without requiring auxiliary losses such as load balance loss or z-loss to prevent routing collapse. Ablation experiments demonstrate that adding auxiliary losses actually degrades performance.

3. **Lookup Table (LUT) Reparameterization**:

   After training, the outputs of all experts are precomputed for each possible input ID $i$:

    $\boldsymbol{v}_j^i = \text{FFN}_j(\text{Embedding}(i)) \in \mathbb{R}^d$

   The lookup table $\text{LUT}_l = \{\{\boldsymbol{v}_j^i\}_{j=1}^{N}\}_{i=1}^{|\mathcal{V}|}$ is constructed, simplifying the inference computation to:

    $\boldsymbol{h}' = \sum_{j=1}^{N} (g_j \cdot \boldsymbol{v}_j^i) + \text{FFN}_{shared}(\boldsymbol{h}) + \boldsymbol{h}$

   **Design Motivation**: The LUT can be fully offloaded to storage devices. Each inference step only needs to load $dN$ parameters (i.e., the $N$ expert output vectors corresponding to the current token), whereas MoE needs to load $2dkD_r$ parameters. For a 410M model, MoLE's per-token loading size is only 1/2000 of MoE's.

### Loss & Training

- **Training Loss**: Uses only the standard language modeling cross-entropy loss, entirely consistent with dense models.
- **No Auxiliary Loss**: Due to all-expert activation, there is no routing collapse issue, eliminating the need for load balance loss or z-loss.
- **Training Data**: A subset of the Pile dataset with 100B tokens.
- **LUT Quantization**: Post-quantization (NF4/NF3) can be performed on the LUT after training, reducing the storage overhead to 25% of the original with almost zero performance loss.

## Key Experimental Results

### Main Results

| Size | Model | Offloaded Params | Per-Token Load | ARC-C | ARC-E | BoolQ | HellaSwag | PIQA | AVG |
|------|------|---------|-------------|-------|-------|-------|-----------|------|-----|
| 160M | Dense | 0B | 0M | 20.3 | 45.9 | 57.1 | 29.7 | 64.0 | 38.8 |
| 160M | MoE-10E | 0.3B | 57M | 21.7 | 49.5 | 51.6 | 32.0 | 66.8 | 40.3 |
| 160M | MoLE-4E | 1.8B | 0.037M | 21.9 | 48.5 | 60.7 | 31.2 | 65.1 | 40.8 |
| 160M | MoLE-16E | 7.4B | 0.15M | 22.4 | 48.6 | 60.3 | 32.7 | 68.3 | 41.9 |
| 410M | Dense | 0B | 0M | 21.8 | 50.8 | 56.8 | 33.8 | 66.5 | 41.8 |
| 410M | MoE-34E | 3.4B | 201M | 25.0 | 57.0 | 59.7 | 39.9 | 71.5 | 46.6 |
| 410M | MoLE-16E | 19.7B | 0.39M | 23.6 | 57.0 | 60.9 | 37.6 | 70.8 | 45.7 |
| 1B | Dense | 0B | 0M | 24.1 | 56.9 | 52.8 | 37.6 | 69.5 | 44.3 |
| 1B | MoE-10E | 2.7B | 537M | 25.9 | 57.8 | 53.8 | 40.7 | 72.0 | 46.6 |
| 1B | MoLE-4E | 6.6B | 0.26M | 25.5 | 58.8 | 61.7 | 39.8 | 71.7 | 47.4 |

### Ablation Study

| Configuration | AVG | Description |
|------|-----|------|
| LM loss only (Default) | 41.9 | Optimal setting for MoLE |
| LM loss + load balance loss | 41.7 | Auxiliary loss leads to performance degradation |
| LM loss + load balance + z-loss | 40.6 | Further degradation, misaligned optimization objectives |
| Expert hidden dimension $D_r = d$ | 40.8 | Smaller expert capacity |
| Expert hidden dimension $D_r = 4d$ | 41.9 | Best performance-to-cost ratio |
| Expert hidden dimension $D_r = 16d$ | 41.7 | Capacity saturation, no extra gain |
| Number of experts N=2 | 39.7 | Too few, insufficient capacity |
| Number of experts N=16 | 41.9 | Better balance point |
| Number of experts N=32 | 42.3 | Continuous improvement, showing scalability |
| LUT FP16 | 40.8 | 3.5GB storage |
| LUT NF4 Quantization | 40.9 | 0.9GB storage, almost lossless |
| LUT NF3 Quantization | 40.5 | 0.7GB storage, slight degradation |

### Key Findings

1. **Substantial Communication Reduction**: The parameter size loaded per token in MoLE is only 1/1500 to 1/2000 of MoE, enabling offloading to low-bandwidth storage (such as disks and network storage).
2. **Inference Speed Comparable to Dense Models**: In V100 tests, MoLE's decoding latency is essentially identical to dense models, and significantly lower than MoE + offloading.
3. **Friendly to Batch Inference**: Unlike MoE, where latency rises sharply as the batch size increases (due to more experts needing to be loaded), MoLE's latency remains almost unchanged with varying batch sizes.
4. **Full Activation Outperforms Sparse Activation**: The performance gain from all-expert activation (+1.5) is sufficient to offset the performance loss from using embedding inputs (-0.7).
5. **Increasing the number of experts shows scalability**, but increasing the expert hidden dimension saturates beyond a certain threshold, indicating an upper limit to the LUT capacity.
6. **Huge LUT Quantization Potential**: NF4 quantization reduces storage by 75% with almost no performance loss, pointing to significant redundancy in LUTs.

## Highlights & Insights

1. **Exquisite Core Insight**: Utilizing the fixed correspondence between embedding outputs and input IDs, the continuous FFN computation is transformed into a precomputed table lookup over a finite discrete set, which is an extremely elegant engineering transformation.
2. **Decoupled Training-Inference Structure**: Using the full FFN during training ensures proper gradient flow and model capacity, while using the LUT during inference achieves zero computation. This reparameterization approach is highly inspiring.
3. **Simplicity**: No auxiliary loss, top-k selection, or complex caching/prefetching strategies are required; the entire approach is remarkably clean.
4. **Stunning LUT Quantization Results**: NF4 quantization has almost no effect on performance, implying that the token-level representations learned by the experts possess excellent low-rank or low-precision characteristics.

## Limitations & Future Work

1. **LUT Storage remains relatively large**: Although the communication volume is extremely low, the total storage capacity of the LUT can be 2.4 to 7.4 times that of offloaded experts (e.g., 7.4GB vs 1.0GB in the 160M model), which becomes more severe for models with large vocabularies (e.g., 100k+).
2. **Small Scale of Experiments**: The evaluation is conducted only up to 1B activated parameters and has not been validated on 7B+ scales, leaving its scaling behavior questionable.
3. **Experts do not receive contextual information**: This is an inevitable compromise for adopting LUTs, which limits the expressiveness of the experts. Although the paper argues that the router and shared expert can compensate for this, its impact on more complex tasks remains unexplored.
4. **Limited to MLP-style experts**: The work does not explore other expert designs (such as attention-based experts) or more diverse discrete input spaces.
5. **Lack of comparison with MoE compression methods**: Alternative methods such as Expert Pruning or Expert Merging can also reduce the storage and communication overheads of MoE.
6. **Insufficient Analysis of the Prefill Phase**: The paper primarily focuses on decoding latency, leaving the efficiency of batch LUT queries during the prefill phase undiscussed.

## Related Work & Insights

- **MoE Series**: Mixtral, DeepSeek-MoE, and OLMoE serve as dominant baselines. MoE++ (Jin et al., 2025) also explores zero-computation experts but takes a different route.
- **Expert Offloading**: Eliseev & Mazur (2023), Pre-gated MoE (Hwang et al., 2024), and others accelerate inference by optimizing prefetching and caching strategies, but they remain limited by the communication bottleneck.
- **Reparameterization Concept**: Similar to the decoupled training-inference philosophy of RepVGG, translating a complex training architecture into a simplified inference structure.
- **Insights**: The approach of "pre-storing computation results as lookup tables" can be generalized to other modules with finite discrete inputs, such as positional encodings or specific types of adapters.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ [The core idea—limiting MoE expert inputs to embeddings to achieve LUT reparameterization—is highly novel and elegant]
- Experimental Thoroughness: ⭐⭐⭐⭐ [Includes multiple scales, extensive ablations, efficiency analysis, and quantization experiments, but only up to 1B parameters and lacks validation on larger LLMs]
- Writing Quality: ⭐⭐⭐⭐⭐ [Clear logic, closely linked from motivation to method and ablations, with well-designed figures and tables]
- Value: ⭐⭐⭐⭐ [Provides a brand new direction for deploying MoEs in VRAM-constrained scenarios, though large-scale validation is still needed to confirm its practical utility]

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] MoH: Multi-Head Attention as Mixture-of-Head Attention](moh_multi-head_attention_as_mixture-of-head_attention.md)
- [\[ICML 2025\] Autonomy-of-Experts Models (AoE)](autonomy-of-experts_models.md)
- [\[NeurIPS 2025\] On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](../../NeurIPS2025/llm_efficiency/on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)
- [\[ICLR 2026\] DirMoE: Dirichlet-Routed Mixture of Experts](../../ICLR2026/llm_efficiency/dirmoe_dirichlet-routed_mixture_of_experts.md)
- [\[ICML 2026\] L$^3$: Large Lookup Layers](../../ICML2026/llm_efficiency/l3_large_lookup_layers.md)

</div>

<!-- RELATED:END -->
