---
title: >-
  [Paper Note] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models
description: >-
  [ICML 2026][Physics & Scientific Computing][Softmax substitution] The authors deconstruct traditional Softmax attention into two independent components: "non-negativity" and "L1 normalization." They prove that L1 normali…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Softmax substitution"
  - "Softplus attention"
  - "length extrapolation"
  - "attention sharpening"
  - "attention sink"
date: 2026-05-08
content_hash: ac4ee3d3a17bfaf1
---

# Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2501.13428](https://arxiv.org/abs/2501.13428)  
**Code**: Not explicitly provided (Experiments based on GPT-2 small + RoPE reproduction)  
**Area**: LLM Efficiency / Attention Mechanism / Length Extrapolation  
**Keywords**: Softmax substitution, Softplus attention, length extrapolation, attention sharpening, attention sink

## TL;DR
The authors deconstruct traditional Softmax attention into two independent components: "non-negativity" and "L1 normalization." They prove that L1 normalization, rather than the exponential function, is the critical factor. Consequently, they replace the exponential with Softplus and a dynamic length scale factor to derive LSSA, followed by an additional power-function-based "re-weighting" to sharpen the attention distribution. The resulting LSSAR maintains a nearly constant validation loss at 16× the training length and enables a GPT-109M model to "re-discover" Newton's law of universal gravitation from trajectory data.

## Background & Motivation

**Background**: The core of the Transformer is the scaled dot-product attention $A = \mathrm{Softmax}(QK^T/\sqrt{d} + M)$. Softmax has become the default LLM component due to its smoothness, differentiability, and "non-negative normalization." however, Softmax-based attention fails significantly in two scenarios: (i) When model scale reaches the trillion-parameter level, the exponential operation $e^x$ prone to numerical instability; (ii) When inference token length far exceeds training length, the attention distribution becomes increasingly flat, failing to focus on critical tokens—leading to the dual problems of "attention smoothing" and "attention sink," which constitute a major architectural bottleneck for length extrapolation.

**Limitations of Prior Work**: Existing Softmax-free attentions (Sigmoid attention, ReLU attention, etc.) resolve numerical stability but either lose length extrapolation capability entirely (loss spikes multiple times at 8K) or cut off gradient paths for distant tokens due to "dead neurons." Post-hoc remedies like position interpolation or ALiBi merely stretch the training length embeddings without addressing the fundamental cause of attention distribution flattening.

**Key Challenge**: All existing solutions default to the assumption that the non-negativity of Softmax is the source of its effectiveness, focusing on "finding a better non-negative activation." However, once "non-negativity" and "L1 normalization" are coupled within the same function, they cannot be independently regulated, making it impossible to replace the numerically unstable exponential while retaining normalization advantages.

**Goal**: (i) Re-analyze which specific part of Softmax determines attention performance; (ii) Design a numerically stable and length-extrapolatable normalization stage; (iii) Architecturally eliminate attention smoothing to ensure the distribution remains naturally sharp.

**Key Insight**: The authors express Softmax as $\mathrm{Softmax}(x) = \phi(x)/\|\phi(x)\|_1$, where $\phi(x) = e^x$ is responsible only for non-negativity, and the $L_1$ norm governs normalization competition. Ablation studies (Appendix Table A4/A6) reveal that replacing $\phi$ with any "globally non-zero" function like Softplus does not degrade performance, whereas removing $L_1$ normalization leads to model collapse—indicating that L1 is the critical component.

**Core Idea**: Decompose attention into two stages: "Normalization" and "Sharpening." The normalization stage uses Softplus + a dynamic length scale factor for stability and extrapolation. The sharpening stage uses a power function $\mathrm{ReLU}^p$ followed by re-normalization to "squeeze" the attention distribution onto a few highly relevant tokens, fundamentally curing attention smoothing from an architectural standpoint.

## Method

### Overall Architecture
LSSAR (Length Scaled Softplus Attention with Re-weighting) consists of two stages in series. The first stage (Normalization Stage, LSSA): Performs $L_2$ normalization on rows of $Q$ and $K$ to bound the dot product within $[-1, 1]$; uses Softplus instead of $e^x$ as the non-negativity function; utilizes $\log d \cdot \log N$ as a position-based dynamic scale factor; and finally applies $L_1$ normalization per row. The second stage (Sharpening Stage): Operates on the LSSA output by "multiplying by $N$ (number of tokens at that position) → subtracting a bias matrix $O$ → applying ReLU to the $p$-th power → re-applying $L_1$ normalization." Both stages are integrated into the original GPT-2 small (124M) + RoPE framework, keeping other parts unchanged.

### Key Designs

1. **Softmax Deconstruction + LSSA: Normalization Stage with Softplus + Dynamic $\log N$ Scale Factor**:

    - **Function**: Replaces $e^x$ with a numerically stable function specifically designed for length extrapolation, while using a row-varying scale factor to ensure attention entropy remains relatively invariant across different token counts.
    - **Mechanism**: First, set $Q_i \leftarrow Q_i/\|Q_i\|_2$ and $K_i \leftarrow K_i/\|K_i\|_2$ so that $QK^T \in [-1,1]$. Then, set $A = \mathrm{Softplus}((\log d \cdot \log \mathbf N)\odot QK^T)\odot M'$, where $\mathbf N$ is an $L\times L$ matrix with each element in row $i$ equal to $i$ (the actual number of attended tokens). Finally, $A_i \leftarrow A_i/\|A_i\|_1$. The $\log N$ factor ensures training distribution entropy remains constant with length and adaptively scales during inference, providing stable dynamics at any length.
    - **Design Motivation**: Ablation showed $L_1$ is core. $e^x$ is just one implementation of a "globally differentiable non-zero" function, but its range explodes. Softplus = $\log(1+e^x)$ is also globally non-zero but stable, and its derivative (sigmoid) has a steep slope in $[-1,1]$, naturally fitting cosine normalization. The $\log N$ factor follows entropy invariance analysis (Chiang & Cholak 2022), adjusting scale by position rather than global length to give appropriate "temperatures" to both early and late tokens.

2. **Re-weighting: Sharpening Stage with $\mathrm{ReLU}^p$ + L1 Normalization**:

    - **Function**: Adds a sharpening operation to the stable dense distribution from the normalization stage, pulling weights significantly toward the maximum values to eliminate attention smoothing in long sequences.
    - **Mechanism**: $A \leftarrow \mathrm{ReLU}^p(A\odot\mathbf N - O)$, followed by $A_i \leftarrow A_i/\|A_i\|_1$. $O$ is an all-ones bias matrix (first 3 rows set to 0 to prevent early training instability), effectively shifting the distribution center toward 0. ReLU filters out elements below the threshold, and the $p$-th power further amplifies peaks. The paper proves that as $p\to\infty$, for any $x_l<x_m$, $\lim_{p\to\infty}(x_m^p - x_l^p)/\sum_k x_k^p = 1$, meaning the maximum value tends toward 1 and others toward 0, equivalent to the limit of a hard argmax.
    - **Design Motivation**: Using ReLU directly as $\phi$ causes "dead neurons"—if a token score falls below 0, it stops receiving gradients. The key insight is separating "non-negativity" from "sharpening": Softplus ensures all tokens participate in L1 competition and maintain gradient paths, while post-sharpening occurs on an already stable distribution. Since the gradient link is established weight-wise, pruning via ReLU in the sharpening stage does not kill learning. $p$ acts as an "inverse temperature coefficient," performing temperature annealing on a softmax-like distribution.

3. **Minimally Invasive Integration with Existing LLMs**:

    - **Function**: Allows the scheme to be integrated into GPT-like architectures with near "zero modification," without altering RoPE/PE/Feed-forward layers.
    - **Mechanism**: The $\mathrm{Softmax}(\cdot)$ in the attention module is replaced by "LSSA → re-weighting." A mask matrix $M'$ (lower triangle is 1) replaces the original $-\infty$ mask (since Softplus output is non-negative, masking changes from adding $-\infty$ to multiplying by 0/1). All other modules retain GPT-2 small default settings. FlashAttention-style kernels can be reused for forward/backward passes by leveraging elementwise properties.
    - **Design Motivation**: The engineering value of length extrapolation requires a plug-and-play solution. By keeping innovations inside the attention module, LSSAR remains orthogonal to and can be stacked with existing techniques like RoPE, positional interpolation, or sliding window attention.

### Loss & Training
All models are based on GPT-2 small (124M) + RoPE, trained on FineWeb-10B for 18,865 steps with a sequence length of 1024, totaling 10.2B training tokens + 0.1B validation tokens, using 8×A100 80GB. $p$ is a critical hyperparameter; the paper reports two settings: $p=3$ (optimal at 1K length) and $p=15$ (optimal at 8K and beyond, balancing sparsity with strong sharpening).

## Key Experimental Results

### Main Results
All methods tested on GPT-2 124M + RoPE, trained at 1K length, extrapolated to 2K/4K/8K to measure validation loss:

| Attention | 1K | 2K | 4K | 8K |
|-----------|-----|-----|-----|-----|
| Softmax baseline | 3.19 | 4.17 | 5.45 | 6.28 |
| Sigmoid (RamapuRam 2024) | 3.19 | 7.46 | 11.84 | 14.50 |
| ReLU (Wortsman 2023) | 3.21 | 6.27 | 8.50 | 10.35 |
| LSSA (Normalization only) | 3.19 | 4.13 | 5.30 | 5.94 |
| LSSAR ($p=3$) | 3.18 | 4.24 | 5.41 | 6.30 |
| **LSSAR ($p=15$)** | **3.19** | **3.19** | **3.23** | **3.32** |

Zero-shot downstream (Softmax 124M vs. LSSAR 124M): ARC-E 39.77→40.57, HellaSwag 32.42→33.03, PIQA 64.09→65.34, SciQ 60.6→62.1, SummScreen 1.68→6.31.

### Ablation Study

| Configuration | 8K Val Loss | Description |
|---------------|-------------|-------------|
| Full LSSAR ($p=15$) | 3.32 | Full model, extrapolation nearly lossless |
| LSSA only (No re-weighting) | 5.94 | Contribution of sharpening; largest gap in long sequences |
| Re-weighting + Softmax ($p=15$) | 7.02 | Softmax as normalization crashes; requires two-stage matching |
| Sigmoid + L1 + Re-weighting ($p=15$) | 3.86 | L1 + Re-weighting is the effective combo, but worse than LSSAR |
| ReLU as $\phi$ | >10 | "Dead neurons," unusable for long sequences |

Passkey retrieval (needle-in-a-haystack):

| Length | Softmax Accuracy | LSSAR ($p=15$) Accuracy |
|--------|------------------|--------------------------|
| 1K | 64% | **86%** |
| 1.5K | 0% | 45% |
| 4K | 0% | 20% |
| 8K | 0% | Non-zero |

### Key Findings
- Validation loss remains almost constant at 8K length (3.19→3.32), representing one of the first results where "length extrapolation is almost free."
- In the Passkey task, the Softmax model drops to 0% immediately beyond the training length, while LSSAR maintains non-zero accuracy at 8× training length; this serves as a clean probe for whether attention is actually sharpening.
- In symbolic regression experiments, GPT-109M + LSSAR successfully recovers Newton's law $F\propto m_1/r^2$ from planetary trajectory sequences, whereas the Softmax version produces meaningless formulas. Even trillion-parameter models like o3, Claude 4 Sonnet, and Gemini 2.5 Pro fail at this task, suggesting the inductive bias of the attention mechanism itself is more critical than scale for learning physical laws.
- Low $p$ (3) is optimal at 1K, but larger $p$ (15) is needed for long sequences; this indicates that the optimal degree of sparsity is length-dependent, suggesting potential for adaptive $p$.

## Highlights & Insights
- The conclusion "Softmax = $\phi$ + L1, where L1 is key" warrants a re-evaluation of the Softmax-free attention field, which has historically focused on replacing the non-negative function.
- Decoupling "sparsity from gradient paths" is a central insight—using Softplus to guarantee gradients for all tokens, then using $\mathrm{ReLU}^p$ for post-sharpening, is equivalent to pruning a fully connected gradient graph rather than building a sparse one, avoiding "dead neurons."
- The $\log d \cdot \log N$ row-varying scale factor can be migrated to any normalization operation needing length extrapolation (e.g., cross-attention, retrieval scoring), as its core is "scaling temperature by the number of tokens actually participating in normalization."
- The symbolic regression experiment provides a rigorous new evaluation: whether a model can learn the $1/r^2$ law from trajectories reflects the attention's inductive bias more accurately than traditional NLP benchmarks.

## Limitations & Future Work
- The LSSA scale factor $\log d \log N$ was validated at $L=1024, d=64$. The authors admit the $\log d$ part might require retuning for larger $d$; no large-scale hyperparameter sweep was provided.
- $p=15$ is empirically optimal without theoretical derivation. Whether learnable or adaptive $p$ is needed for unknown tasks/lengths remains an open question.
- Experiments were conducted at the 124M / 109M scale; loss comparisons for 7B/13B models were not reported. Whether extrapolation advantages hold at massive scale needs verification.
- $\mathrm{ReLU}^p$ risks numerical overflow when $p$ is large; $x^{15}$ has a massive dynamic range, requiring specialized kernels for FP16 training stability.

## Related Work & Insights
- **vs. Sigmoid attention (Ramapuram 2024)**: The authors demonstrate that Sigmoid attention fails because it lacks L1 normalization; adding L1 brings Sigmoid close to Softmax performance—doing "non-negative + L1" correctly is sufficient, but LSSA is more stable.
- **vs. ReLU attention series**: ReLU suffers from catastrophic failure on long sequences due to hard-threshold dead neurons; LSSAR places the hard threshold in the re-weighting stage (post-L1), avoiding the pitfalls of direct backbone sparsification.
- **vs. Positional Interpolation / ALiBi / NTK RoPE**: These are post-hoc PE-level remedies that do not change the fundamental cause of attention smoothing; LSSAR fixes smoothing at the source and is compatible with these encoding schemes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Softmax = $\phi$ + L1, L1 is core" is a disruptive conclusion; two-stage design and symbolic regression are highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation loss, zero-shot, passkey retrieval, and symbolic regression provide comprehensive evidence; lacks billion-parameter scale validation.
- Writing Quality: ⭐⭐⭐⭐ The argument chain (Deconstructing Softmax → LSSA → Re-weighting) is very clear; theoretical analysis ($p\to\infty$) is concise.
- Value: ⭐⭐⭐⭐⭐ Near-free length extrapolation and numerical stability make this one of the few "stable and extrapolatable" Softmax-free solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Quiver: Quantum-Informed Views for Enhanced Representations in Large ML Models](quiver_quantum-informed_views_for_enhanced_representations_in_large_ml_models.md)
- [\[ICML 2026\] Understanding Catastrophic Forgetting In LoRA via Mean-Field Attention Dynamics](understanding_catastrophic_forgetting_in_lora_via_mean-field_attention_dynamics.md)
- [\[ICML 2026\] $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval](mathbbr2k_is_theoretically_large_enough_for_embedding-based_top-k_retrieval.md)
- [\[ICLR 2026\] Sublinear Time Quantum Algorithm for Attention Approximation](../../ICLR2026/physics/sublinear_time_quantum_algorithm_for_attention_approximation.md)
- [\[CVPR 2026\] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](../../CVPR2026/physics/nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)

</div>

<!-- RELATED:END -->
