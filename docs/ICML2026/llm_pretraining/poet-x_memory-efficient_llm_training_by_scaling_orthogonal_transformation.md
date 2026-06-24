---
title: >-
  [Paper Note] POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation
description: >-
  [ICML2026][LLM Pretraining][Memory-efficient training] POET-X implements a system-level acceleration and memory optimization for POET (reParameterized Orthogonal Equivalence Training), which is training-stable but slow and memory-intensive. By combining input-centric reconstruction, permutation kernel acceleration, block-diagonal batch parallelism, half-storage CNP, and Triton fusion, it achieves a 3× memory reduction and 8× speedup compared to the original POET. This allows…
tags:
  - "ICML2026"
  - "LLM Pretraining"
  - "Memory-efficient training"
  - "Orthogonal Equivalence Transformation"
  - "Spectral Preservation"
  - "Sparse training"
  - "CUDA kernels"
date: 2026-05-08
content_hash: 3b1b9f9ca267b7cb
---

# POET-X: Memory-efficient LLM Training by Scaling Orthogonal Transformation

**Conference**: ICML2026  
**arXiv**: [2603.05500](https://arxiv.org/abs/2603.05500)  
**Code**: spherelab.ai/poetx (Project Page)  
**Area**: LLM Pre-training / LLM Efficiency  
**Keywords**: Memory-efficient training, Orthogonal Equivalence Transformation, Spectral Preservation, Sparse training, CUDA kernels  

## TL;DR
POET-X implements a system-level acceleration and memory optimization for POET (reParameterized Orthogonal Equivalence Training), which is training-stable but slow and memory-intensive. By combining input-centric reconstruction, permutation kernel acceleration, block-diagonal batch parallelism, half-storage CNP, and Triton fusion, it achieves a 3× memory reduction and 8× speedup compared to the original POET. This allows for pre-training 8B~13B LLMs on a single H100, while AdamW triggers OOM under identical settings.

## Background & Motivation
**Background**: Training large models is computationally expensive and often unstable. POET reparameterizes each weight as $\bm{W}_{RP}=\bm{R}\bm{W}_0\bm{P}$, where $\bm{W}_0$ is fixed random weight, and $\bm{R}, \bm{P}$ are trainable orthogonal matrices. Since orthogonal transformations preserve singular values (spectral preservation) and hyper-sphere energy is provably low under Gaussian initialization, POET provides a highly stable optimization framework.

**Limitations of Prior Work**: Although stable, POET suffers from poor memory efficiency and runs much slower than Adam, primarily due to numerous large-scale matrix multiplications. Paradoxically, while the orthogonal matrices in POET are constrained to be sparse (block-diagonal) and highly **parameter-efficient**, the original implementation fails to translate this sparsity into **memory efficiency**. It actually consumes more memory than AdamW because it must store intermediate activations like the transformed weights $\bm{W}_{RP}$ for backpropagation. Consequently, POET struggles to scale beyond 3B parameters.

**Key Challenge**: Parameter efficiency $\neq$ memory efficiency. POET is hindered by the gap between being "theoretically sparse" and "engineering-wise bloated"—the sparse orthogonal matrices are not efficiently computed or stored.

**Goal**: Make orthogonal equivalence transformation **scalable**. This research analyzes the memory and runtime overhead of every computation step in POET's forward and backward passes to convert "parameter efficiency" into "memory efficiency."

**Core Idea**: The mathematical foundations of POET (spectral preservation, stability) are preserved, while the "algorithm engineering" is redesigned. By shifting from weight-centric computation to input-centric (matrix-free) forms and using custom CUDA/Triton kernels to eliminate redundant matrix construction and activation storage, POET's memory footprint is reduced to LoRA-like levels and its runtime to Adam-like levels.

## Method

### Overall Architecture
POET-X is built on **block-stochastic POET**, which ensures balanced updates across weight matrix dimensions even with few trainable parameters. Each step parameterizes the orthogonal matrix $\bm{R}_i$ as a sandwich structure: "row permutation → block-diagonal orthogonal matrix → column permutation," formulated as $\bm{R}_i=\bm{\Psi}_i^\top\,\mathrm{Diag}(\tilde{\bm{G}}^1_i,\dots)\,\bm{\Psi}_i$. $\bm{P}_i$ follows a similar structure. These are periodically multiplied into the weights $\bm{W}_i=\bm{R}_i\bm{W}_{i-1}\bm{P}_i$. The core challenge is to **perform these multiplications rapidly and with minimal memory while strictly satisfying orthogonal constraints**. POET-X optimizes the calculation chain $\bm{z}=\bm{G}_P^\top\bm{W}\bm{G}_R^\top\bm{x}$ across calculation form, permutations, block-diagonal construction, orthogonal parameterization, and activation storage.

### Key Designs

**1. Input-centric (matrix-free) reconstruction: Replacing weight multiplication with matrix-vector products to eliminate activation storage**

Original POET operates directly on weights $\bm{W}\leftarrow\bm{R}_i\bm{W}\bm{P}_i$ (weight-centric), with complexity $\mathcal{O}(nm^2)$. Calculating gradients for $\bm{R}_i$ and $\bm{P}_i$ requires accessing $\bm{W}$, increasing memory usage. Inspired by matrix-free methods for solving large linear systems, POET-X rewrites the update in input-centric form: $\bm{P}_i^\top(\bm{W}^\top(\bm{R}_i^\top\bm{x}))$. This converts the process into a sequence of linear mappings, avoiding the storage of intermediate activations tied to the weight matrix.

**2. Permutation operator acceleration and reduction: Using CUDA index mapping and reducing 4 permutations to 2**

A full inference pass involves four permutation matrices. For **acceleration**, the authors avoid explicit construction of permutation matrices, implementing custom CUDA operators for index mapping—e.g., $(\bm{W}')_{i,:}=\bm{W}_{\pi_p(i),:}$, which requires only a set of permutation indices and provides up to 20× speedup. For **reduction**, it was discovered that 2 of the 4 permutations in the input-centric forward pass can be pre-merged into the weights. Since $\bm{W}$ is fixed during the inner loop of $\bm{G}_P, \bm{G}_R$ optimization, this eliminates repetitive permutations and yields another 1.1~1.8× speedup.

**3. Block-diagonal batch parallelism + Efficient CNP: Avoiding large sparse matrices and fusing kernels**

This set of optimizations targets the construction and orthogonalization of matrices. **Block-diagonal batch parallelism**: Instead of explicitly constructing large sparse matrices like $\bm{G}_P=\mathrm{Diag}(\tilde{\bm{G}}^1_P,\dots)$, POET-X treats each block as an independent matrix for batch-wise multiplication, saving up to 31% memory and achieving ~2.3× speedup. **Efficient CNP**: POET uses Cayley-Neumann Parameterization to approximate an orthogonal matrix from a skew-symmetric matrix $\bm{Q}=-\bm{Q}^\top$. POET-X only stores the upper triangle of $\bm{Q}$, cutting memory for POET parameters in half. By rearranging the CNP polynomial, it was found that all calculations rely only on $\bm{Q}$ and $\bm{Q}^2$. A Triton kernel was developed to load these tensors once into shared memory and compute high-order terms in-place, yielding 2~3× speedup via kernel fusion.

**4. Gradient checkpointing and quantized training: Recomputing activations to reach PEFT-level memory usage**

The simplified forward pass consists of three matrix multiplications: mm1: $\bm{a}=\bm{G}_R^\top\bm{x}$, mm2: $\bm{b}=\bm{W}\bm{a}$, mm3: $\bm{z}=\bm{G}_P^\top\bm{b}$. Analysis shows that calculating $\nabla_{\bm{G}_P}$ requires storing $\bm{b}\in\mathbb{R}^{N\times m}$, while mm2 and mm1 require minimal storage since $\bm{W}$ has no gradient. Two variants are proposed: $\text{POET-X}_{\text{fast}}$, using standard Autograd, and $\text{POET-X}_{\text{mem}}$, using gradient checkpointing to recompute $\bm{b}$. Combined with custom CUDA kernels, $\text{POET-X}_{\text{mem}}$ extends to $\text{POET-XQ}$, which stores only low-bit quantized weights and dequantizes them on the fly, supporting memory-efficient quantized training.

### Loss & Training
POET-X does not change the training objective; it optimizes the engineering implementation of the POET optimizer. All forward/backward passes (including batch parallelism, CNP, and permutations) are implemented using custom Triton/CUDA kernels for fine-grained control over GPU memory and computation. The block size $b$ (e.g., 256/512) is the primary hyperparameter determining the trade-off between trainable parameters and memory.

## Key Experimental Results

### Main Results
On single-layer profiling, the latency for a single forward+backward pass was reduced from 10.59ms (original POET) to 1.38ms ($\text{POET-X}_{\text{fast}}$) and 1.89ms ($\text{POET-X}_{\text{mem}}$), representing an overall 3× memory reduction and 8× speedup. For Llama-8B single-card pre-training ($L_{\max}=256$, 5B tokens), validation perplexity results are as follows:

| Method | Trainable Params (M) | Memory (G) | Val PPL |
|------|--------------|---------|---------|
| AdamW | 2764.47 | 81.03 | 12.69 |
| Muon (Kimi) | 2764.47 | 70.94 | 11.45 |
| APOLLO | 2764.47 | 80.60 | 12.97 |
| GaLore | 2764.47 | 74.50 | 14.88 |
| POET-X (b=256) | **366.64** | 60.58 | 12.76 |
| POET-X (b=512) | 570.06 | **68.52** | **12.05** |

POET-X achieves a better PPL than AdamW (12.05 vs 12.69) with only 1/5~1/8 of the trainable parameters and a significantly lower memory footprint.

### Quantization Training Comparison
$\text{POET-XQ}$ (8-bit) provides superior perplexity compared to 8-bit versions of APOLLO and GaLore while consuming less memory:

| Method | Params (M) | Memory (G) | Val PPL |
|------|---------|---------|---------|
| 8-bit APOLLO | 2764.47 | 66.37 | 20.49 |
| 8-bit GaLore | 2764.47 | 66.28 | 17.74 |
| POET-XQ (b=256) | 366.64 | **51.66** | 16.21 |
| POET-XQ (b=512) | 570.06 | 60.65 | **14.78** |

### Key Findings
- **Sparsity translates to memory efficiency**: Total memory footprint is reduced to PEFT levels, enabling 8B~13B model training on a single card.
- **Cumulative effect of kernel optimizations**: Custom permutation CUDA operators (up to 20×), CNP Triton fusion (2~3×), and batch parallelism (~2.3×) combine for an overall 8× speedup.
- **Fast vs. Mem trade-off**: $\text{POET-X}_{\text{fast}}$ offers backward latency comparable to standard linear layers, while $\text{POET-X}_{\text{mem}}$ prioritizes memory saving, which is essential for quantized training.

## Highlights & Insights
- **Input-centric "matrix-free" reconstruction is the core unlocker**: Shifting from $\bm{R}\bm{W}\bm{P}$ to a sequence of matrix-vector products avoids storing weight-bound activations—a strategy applicable to any algorithm involving sandwiching large matrices.
- **Permutations without matrices**: Replacing explicit matrix multiplication with bijective index mapping $(\bm{W}')_{i,:}=\bm{W}_{\pi(i),:}$ uses zero extra memory and provides a 20× speedup.
- **CNP rearrangement**: Identifying that high-order terms rely only on $\bm{Q}$ and $\bm{Q}^2$ allows for efficient kernel fusion, aligning mathematical structure with GPU memory hierarchies.

## Limitations & Future Work
- POET-X inherits the inductive biases of POET (spectral preservation, fixed $\bm{W}_0$), and its performance upper bound is constrained by the POET framework (PPL remains slightly behind Muon).
- Significant efficiency gains rely on custom CUDA/Triton kernels, which introduces high coupling with hardware and potential maintenance costs on non-NVIDIA platforms.
- While it succeeds in single-card "memory victory," more investigation is needed for multi-node distributed scaling and performance under extremely long sequence lengths.
- The quantized variant $\text{POET-XQ}$ still shows noticeable perplexity degradation compared to full precision (e.g., 12.05 to 14.78 for b=512).

## Related Work & Insights
- **vs. Original POET**: Same mathematics but a complete engineering rewrite—3× memory reduction and 8× speedup.
- **vs. AdamW**: POET-X achieves better PPL with fewer parameters and lower memory, succeeding where AdamW OOMs on 8B single-card setups.
- **vs. GaLore / APOLLO**: While they focus on low-rank projections to compress optimizer states, POET-X uses orthogonal sparsity and kernel optimization, achieving better PPL and lower memory at both full and 8-bit precision.
- **vs. LoRA (PEFT)**: POET-X reaches LoRA-level memory efficiency but for spectral-preserving **pre-training**, and the orthogonal matrices can be merged into weights after training to avoid inference overhead.

## Rating
- Novelty: ⭐⭐⭐⭐ (Uses known POET math, but the system-level scaling via matrix-free and kernel fusion is solid)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Extensive profiling, 8B/13B pre-training, and quantization analysis; distributed scaling slightly lacking)
- Writing Quality: ⭐⭐⭐⭐ (Clear optimization narrative and technical analysis)
- Value: ⭐⭐⭐⭐⭐ (Highly valuable for researchers with limited compute to train billion-parameter LLMs)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](../../ICLR2026/llm_pretraining/scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](../../ACL2026/llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[ICML 2026\] AC-ODM: Actor–Critic Online Data Mixing for Sample-Efficient LLM Pretraining](ac-odm_actor--critic_online_data_mixing_for_sample-efficient_llm_pretraining.md)
- [\[ICML 2026\] Explaining Data Mixing Scaling Laws](explaining_data_mixing_scaling_laws.md)
- [\[ICML 2026\] Inverse Depth Scaling From Most Layers Being Similar](inverse_depth_scaling_from_most_layers_being_similar.md)

</div>

<!-- RELATED:END -->
