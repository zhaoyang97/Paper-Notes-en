---
title: >-
  [Paper Note] MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure
description: >-
  [ICLR 2026][LLM Efficiency][LoRA] MiSS replaces the dual-matrix $BA$ update of LoRA with a shard-sharing structure "expanded" from a single zero-initialized small matrix $D$. This approach accelerates convergence while simultaneously excelling in memory and computational efficiency, thereby achieving a superior trade-off in the performance–memory–efficiency triangle.
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "LoRA"
  - "Parameter-Efficient Fine-tuning"
  - "Single Matrix"
  - "Shard-sharing"
  - "Convergence"
  - "Pareto Frontier"
date: 2026-05-08
content_hash: ee4db5097616c2d1
---

# MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=gohmWoUSoS](https://openreview.net/forum?id=gohmWoUSoS)  
**Code**: [https://github.com/Joluck/MiSS](https://github.com/Joluck/MiSS)  
**Area**: Efficient LLM Fine-tuning (PEFT / LoRA)  
**Keywords**: LoRA, Parameter-Efficient Fine-tuning, Single Matrix, Shard-sharing, Convergence, Pareto Frontier  

## TL;DR
MiSS replaces the dual-matrix $BA$ update of LoRA with a shard-sharing structure "expanded" from a single zero-initialized small matrix $D$. This approach accelerates convergence while simultaneously excelling in memory and computational efficiency, thereby achieving a superior trade-off in the performance–memory–efficiency triangle.

## Background & Motivation
- **Background**: LoRA reduces trainable parameters to a minimum using low-rank decomposition $\Delta W \approx BA$ and is currently the most mainstream PEFT method. Two improvement paths have emerged: one focuses on enhancing **adaptability** (using better initializations like PiSSA, LoRA-GA, and DoRA to approach full-parameter fine-tuning convergence), while the other focuses on enhancing **efficiency** (using parameter sharing or compression like VeRA and MoS to reduce memory and computation).
- **Limitations of Prior Work**: Innovations in these two paths often come at the expense of each other. Methods like PiSSA offer good adaptability but suffer from long initialization times and potential incompatibility with certain optimizers. Conversely, efficiency-focused methods like VeRA and MoS are fast to initialize and save memory, but further decomposing LoRA matrices weakens expressivity, leading to performance degradation. More subtly, while AdaLoRA, DoRA, and VeRA reduce the number of trainable parameters (TPs), they still follow the $B(Ax)$ serial matrix multiplication logic. **Spatial complexity and FLOPs remain bottlenecked at $O((d+k)\times r)$**—"fewer parameters" does not equate to "faster computation or lower memory occupancy."
- **Key Challenge**: It is difficult to balance performance, memory, and efficiency simultaneously; existing variants typically benefit in one dimension while suffering in another.
- **Goal**: To identify a PEFT structure that occupies the Pareto frontier by excelling across performance, memory, and computational efficiency dimensions simultaneously.
- **Core Idea** (**Single-Matrix Simplified Optimization**): The authors re-examine the root cause of LoRA's slow convergence and find that the requirement for $B$ and $A$ to be **updated simultaneously** significantly increases optimization complexity (echoing observations from S2FT regarding fixing one matrix and LoRA+ regarding differential learning rates). Thus, the hypothesis is: **training a single matrix can simplify optimization without sacrificing expressivity.** This is supported by initial gradient norm experiments showing that MiSS, like other fast-converging variants, has an initial gradient norm significantly larger than original LoRA.

## Method

### Overall Architecture
MiSS discards the dual-matrix structure of LoRA, retaining only a single zero-initialized small matrix $D$. The weight update is formulated as $\Delta W = \text{expand}(D)$, where an "expansion/replication" operator tiles the small matrix into a large matrix of the same shape as $W_0$. This structure is inherently low-rank (identical rows keep the effective rank small), preserving low-rank characteristics while simplifying the update from "dual-matrix multiplication" to "single-matrix expansion." To avoid the computational and memory overhead of explicitly constructing a large matrix, the authors provide an equivalent efficient form, MiSSe, which reformulates "expanding the output dimension" as "aggregating the input dimension," ensuring both training and initialization are fast and lean.

```mermaid
flowchart LR
    subgraph LoRA
        x1[x] --> A[A r×k] --> B[B d×r] --> o1[BAx]
    end
    subgraph MiSS["MiSS (Conceptual Form)"]
        D0[D N×k\nzero-init] --> EX[expand / Replicate Shards] --> dW[ΔW d×k] --> o2[ΔWx]
    end
    subgraph MiSSe["MiSSe (Efficient Form)"]
        x2[x b×l×k] --> SUM[Block Sum S=Σ x_g] --> DS[D·S, D∈R^{d×r}] --> o3[ΔWx = DS]
    end
```

### Key Designs

**1. Shard-Sharing Single Matrix Expansion (MiSS Conceptual Form): Treating "Repetition" as Low-rankness.** The authors observe that a large matrix constructed by repeating a small matrix is naturally low-rank—if the rows within each shard are identical, the total rank is at most the number of shards $N$. Accordingly, the output dimension $d$ is divided into $N$ shards $\{s_1,\dots,s_N\}$ ($\sum s_i=d$). A small matrix $D\in\mathbb{R}^{N\times k}$ is used, where the update for the $i$-th shard equals the $i$-th row of $D$, $D_i$, repeated vertically $s_i$ times:
$$(\text{expand}(D))^\top = [(\mathbf{1}_{s_1}D_1)^\top\ (\mathbf{1}_{s_2}D_2)^\top\ \dots\ (\mathbf{1}_{s_N}D_N)^\top]$$
The forward pass becomes $y = W_0x + \text{expand}(D)x$, reducing parameter count from $d\times k$ to $N\times k$. Zero initialization of $D$ ensures the training starting point is $\Delta W=0$, preserving pre-trained outputs.

**2. Input Aggregation Equivalent Efficient Form (MiSSe): Replacing Matrix Multiplication with Block Summing.** Directly calculating $\text{expand}(D)x$ in the conceptual form has time complexity $O(bldk)$ and memory overhead $O(dk)$, which is still expensive. The authors flip the perspective from "partitioning the output dimension" to "partitioning the input dimension": re-defining $D\in\mathbb{R}^{d\times r}$ and partitioning the input dimension $k$ into $r$ blocks, each of size $g=\lfloor k/r\rfloor$. Summing along the $k$ dimension within each block yields an aggregated vector:
$$S = \sum_{i=1}^{g} x^{(i)} \in \mathbb{R}^{b\times l\times r}, \qquad \Delta W x = DS,\quad y = W_0x + DS.$$
This implicitly implements $\text{expand}(D)x = DS$ while only requiring the storage of $D\in\mathbb{R}^{d\times r}$ without explicitly constructing the $d\times k$ matrix. The implementation is extremely lightweight (the core of the pseudo-code is a single line: `result + x @ self.D.expand(...)`). Its essence lies in using "input block summing" to leverage local redundancies in the input, compressing the computational dimension while retaining critical information.

**3. Computational Decomposition Perspective: Pinpointing the Efficiency Advantage in Input Transformation.** The authors decompose the update into two stages—CStep1 (Input Transformation) and CStep2 (Output Projection). They point out that $D$ aligns dimensionally with LoRA's $B$ (both correspond to output dimension $d$ and serve as output projection matrices). Thus, CStep2 for both involves a $d\times r$ matrix multiplication with identical overhead. The difference is **entirely concentrated in CStep1**: LoRA requires an expensive $Ax$ ($O(BLkr)$), while MiSSe only performs a cheap block sum $\text{sum}(x)$ ($O(BLk)$). Consequently, the total FLOPs for MiSSe decrease from $O(BL(kr+rd))$ to $O(BL(k+rd))$ with a spatial complexity of $O(d\times r)$. In the comparison in Table 2, this is the only method that truly breaks the $O((d+k)\times r)$ lower bound—addressing the paradox that "fewer parameters $\neq$ faster computation."

## Key Experimental Results

### Main Results

NLU (RoBERTa-base, GLUE subset, trainable ratio 0.236%):

| Method | MNLI | SST-2 | CoLA | QNLI | MRPC | Avg |
|------|------|-------|------|------|------|-----|
| LoRA | 85.63 | **94.03** | 62.40 | 91.37 | 87.98 | 84.28 |
| PiSSA | **85.72** | 93.64 | 67.28 | 91.40 | 88.11 | 85.23 |
| **Ours (MiSS)** | 85.71 | 93.60 | **72.86** | **91.43** | **88.14** | **86.35** |

NLG (Multiple models, Avg of GSM8K/Math/HumanEval/Mbpp, selected):

| Model | Method | Trainable | Math | Avg |
|------|------|-----------|------|-----|
| Llama2-7B | LoRA / PiSSA / **Ours** | 89.9M / 89.9M / **87.0M** | 5.22 / 6.92 / **8.58** | 24.72 / 27.70 / **29.30** |
| Mistral-7B | LoRA / PiSSA / **Ours** | 94.4M / 94.4M / **87.0M** | 15.82 / 18.13 / **18.85** | 40.12 / 44.45 / **47.79** |
| Llama2-13B | LoRA / PiSSA / **Ours** | 250M / 250M / 255M | 12.60 / 13.82 / **15.74** | 34.60 / 39.52 / **42.11** |
| Qwen3-4B | LoRA / PiSSA / **Ours** | 74.3M / 74.3M / **70.1M** | 15.20 / 26.00 / **34.82** | 62.79 / 66.21 / **68.22** |

Across five mainstream LLMs, MiSS generally achieves the best or second-best average scores. The Gain is particularly significant in complex reasoning tasks (e.g., Qwen3-4B Math jumped from PiSSA's 26.00 to 34.82), often while using fewer trainable parameters.

### Ablation Study

Complexity comparison (Table 2, with $d$ as output dim, $k$ as input dim, $r$ as rank):

| Method | Spatial Complexity | FLOPs | Trainable Parameters |
|------|-----------|-------|-----------|
| FT | $O(dk)$ | $O(dk)$ | $d\cdot k$ |
| LoRA | $O((d+k)r)$ | $O((d+k)r)$ | $(d+k)\cdot r$ |
| AdaLoRA | $O((d+k+r)r)$ | $O((d+k+r)r)$ | $(d+k)r+r^2$ |
| LoHA | $O(2r(d+k))$ | $O(2r(d+k))$ | $2(d+k)r$ |
| VeRA | $O((d+k)r+r+d)$ | Same as left | $d+r$ |
| **MiSSe** | $O(d\times r)$ | $O(k+d\times r)$ | $d\cdot r$ |

Vision tasks (VTAB-1K): MiSS achieved an average score of 88.02 for images and 72.96 for video, performing on par with or better than LoRA/DoRA but with a parameter budget of only approximately 0.4 TPs compared to LoRA/DoRA's 0.8 TPs. This demonstrates that the efficiency advantage is transferable to multimodal tasks.

### Key Findings
- **Initial Gradient Norm**: The initial gradient norm of MiSS is significantly larger than that of original LoRA and close to full-parameter fine-tuning, leading to faster early convergence. This validates the hypothesis that "single-matrix optimization is simpler" (Figure 1).
- **No Free Lunch Experiment**: In a controlled single-layer MLP setting, SVD methods like PiSSA show good adaptability but initialization time spikes with parameter count; VeRA/AdaLoRA initialize quickly but show weak adaptability. MiSS occupies favorable positions across initialization time, training time, and minimum validation loss curves.
- **Pareto Frontier**: Considering performance, memory, and efficiency in three dimensions, MiSS is the only method that maintains adaptability while compressing spatial complexity and FLOPs to the $O(d\times r)$ magnitude.

## Highlights & Insights
- **Maximizing "Repetition = Low-Rank"**: Replacing low-rank products with a single replicated small matrix results in a structure simpler than $BA$, while maintaining equivalent low-rank constraints.
- **Pinpointing the Paradox**: Diagnosis of "fewer parameters $\neq$ faster computation"—noting that most LoRA variants only compress parameter count without modifying the serial matrix multiplication logic, which limits FLOPs/memory gains.
- **Dual Equivalence of Conceptual and Efficient Forms**: The relation $\text{expand}(D)x = DS$ unifies "output dimension partitioning" and "input dimension aggregation," facilitating both easy initialization and efficient computation with nearly zero engineering overhead (already integrated into HuggingFace PEFT).
- **Zero Inference Overhead + Scalable Serving**: The single-matrix structure is isomorphic to $W_0$ and can be fused, retaining LoRA’s zero-overhead inference property.

## Limitations & Future Work
- **Orthogonality with Initialization/Optimizer Methods**: The authors did not include LoRA-GA or LoRA+ in their comparisons, assuming MiSS is orthogonal and additive, though combined experiments are missing to verify actual gains.
- **Impact of Shard Granularity**: Quantitative discussions regarding the relationship between shard count $N$/rank $r$, expressivity, and convergence speed are relatively shallow. Whether shard partitioning (equal vs. non-equal) needs to be adaptive for different layers remains an open question.
- **Expressivity Upper Bound**: Shard-sharing forces identical rows within a shard, theoretically limiting the degrees of freedom of $\Delta W$. Whether this hits a ceiling in tasks requiring fine-grained directional adjustments warrants further exploration.
- **Scale and Long-Context Scenarios**: Experiments focused on the 7B–13B scale. Numerical stability and gains of blockwise summation in larger models or with longer sequences remain to be tested.

## Related Work & Insights
- **Adaptability Path**: PiSSA (SVD initialization), LoRA-GA (Gradient alignment), DoRA (Magnitude/Direction decoupling), OLoRA (QR orthogonal initialization)—these improve initial gradients to approach full-parameter fine-tuning but introduce expensive preprocessing.
- **Efficiency Path**: VeRA (Frozen random matrix + sharing), MoS/ProLoRA (Parameter sharing/partitioning)—these save memory but weaken expressivity. The insight from MiSS is: **Instead of patching the $BA$ framework, reconstruct the entire update operator using "Single Matrix + Repetition = Low-Rank"** to capture benefits from both paths.
- **Response to S2FT/LoRA+**: S2FT reduces degrees of freedom by fixing one matrix, and LoRA+ uses differential learning rates; both suggest that "dual-matrix simultaneous updating" is the root of slow convergence. MiSS pushes this insight to the extreme by retaining only one matrix.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The perspective of "repetitive matrix as low-rank + single matrix expansion" is refreshing. The equivalent conversion between output sharding and input aggregation is cleverly designed, representing a substantive reconstruction of the LoRA structure rather than an initialization tweak.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers NLU/NLG/Vision, five mainstream LLMs, theoretical complexity analysis, Pareto frontier, and initial gradient norm verification. Evidence chain is complete; points deducted for lacking combined experiments with LoRA-GA/LoRA+ and shallow shard granularity ablation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation progresses logically (slow convergence → single matrix hypothesis → gradient norm validation). Table 1 clearly illustrates the forward pass/initialization of variants, and the computational breakdown table clarifies the source of efficiency.
- **Value**: ⭐⭐⭐⭐ — Genuinely excels in the performance–memory–efficiency triple and is integrated into the PEFT ecosystem, offering direct practical value for LoRA deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LoRAGen: Structure-Aware Weight Space Learning for LoRA Generation](loragen_structure-aware_weight_space_learning_for_lora_generation.md)
- [\[ICLR 2026\] PLoP: Precise LoRA Placement for Efficient Finetuning of Large Models](plop_precise_lora_placement_for_efficient_finetuning_of_large_models.md)
- [\[ICLR 2026\] LoRA-S: An Efficient Low Rank Adaptation scheme via Sylvester equation](lora-s_an_efficient_low_rank_adaptation_scheme_via_sylvester_equation.md)
- [\[ICLR 2026\] On-the-Fly Adaptation to Quantization: Configuration-Aware LoRA for Efficient Fine-Tuning of Quantized LLMs](on-the-fly_adaptation_to_quantization_configuration-aware_lora_for_efficient_fin.md)
- [\[ICLR 2026\] Meta-UCF: Unified Task-Conditioned LoRA Generation for Continual Learning in Large Language Models](meta-ucf_unified_task-conditioned_lora_generation_for_continual_learning_in_larg.md)

</div>

<!-- RELATED:END -->
