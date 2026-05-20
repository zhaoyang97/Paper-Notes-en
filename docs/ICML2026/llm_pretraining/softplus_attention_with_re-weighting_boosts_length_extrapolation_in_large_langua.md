---
title: >-
  [Paper Note] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models
description: >-
  [ICML 2026][LLM Pretraining][Softmax alternative] The authors decompose traditional Softmax attention into two independent components: "non-negativization + L1 normalization…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Softmax alternative"
  - "Softplus attention"
  - "length extrapolation"
  - "attention sharpening"
  - "attention sink"
date: 2026-05-08
content_hash: dabd1d6274ff1b69
---

# Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2501.13428](https://arxiv.org/abs/2501.13428)  
**Code**: Not explicitly provided (experiments based on GPT-2 small + RoPE reproduction)  
**Area**: LLM Efficiency / Attention Mechanisms / Length Extrapolation  
**Keywords**: Softmax alternative, Softplus attention, length extrapolation, attention sharpening, attention sink

## TL;DR
The authors decompose traditional Softmax attention into two independent components: "non-negativization + L1 normalization," and demonstrate that the truly critical part is L1 normalization rather than the exponential. They replace the exponential with Softplus plus a dynamic length scaling factor to obtain LSSA, and then apply a power function-based "re-weighting" to sharpen the attention. The resulting LSSAR maintains nearly unchanged validation loss at 16× the training length and enables GPT-109M to "rediscover" Newton's law of universal gravitation from trajectory data.

## Background & Motivation

**Background**: The core of the Transformer is scaled dot-product attention $A = \mathrm{Softmax}(QK^T/\sqrt{d} + M)$. Softmax, due to its smoothness, differentiability, and "non-negative normalization," has become the default component in LLMs. However, attention learned in two scenarios fails badly: (i) At trillion-scale model sizes, the exponential $e^x$ is prone to numerical instability; (ii) When inference token length far exceeds training length, the attention distribution becomes increasingly flat, making it difficult to focus on key tokens—these are the notorious "attention smoothing" and "attention sink" problems, which are among the main architectural bottlenecks limiting LLM length extrapolation.

**Limitations of Prior Work**: Existing Softmax-free attentions (Sigmoid attention, ReLU attention, etc.) address numerical stability but either completely lose length extrapolation ability (loss increases several-fold at 8K) or, due to "dead neurons," sever gradient paths for distant tokens. Post-hoc remedies like positional interpolation and ALiBi merely stretch embeddings for the training length, without addressing the root cause of attention distribution flattening.

**Key Challenge**: All existing approaches assume that the non-negativity of Softmax is its core effective property and focus on finding a "better non-negative activation." However, coupling "non-negativization" and "L1 normalization" within a single function prevents independent control, making it impossible to replace the numerically unstable exponential while retaining the benefits of normalization.

**Goal**: (i) Re-examine which part of Softmax truly determines attention performance; (ii) Design a numerically stable and length-extrapolatable normalization stage; (iii) Further, structurally eliminate attention smoothing so that the attention distribution naturally remains sharp.

**Key Insight**: The authors rewrite Softmax as $\mathrm{Softmax}(x) = \phi(x)/\|\phi(x)\|_1$, where $\phi(x) = e^x$ is responsible only for non-negativization, and the $L_1$ norm handles normalization and competition. Ablation studies (Appendix Table A4/A6) show that replacing $\phi$ with any globally nonzero function like Softplus does not degrade performance, but removing $L_1$ normalization causes model collapse—thus, L1 is the key.

**Core Idea**: Decompose attention into two stages: "normalization + sharpening." The normalization stage uses Softplus plus a dynamic length scaling factor for stability and extrapolation; the sharpening stage applies a power function of ReLU (i.e., $\mathrm{ReLU}^p$) followed by normalization, squeezing the attention distribution onto a few most relevant tokens, structurally curing attention smoothing.

## Method

### Overall Architecture
LSSAR (Length Scaled Softplus Attention with Re-weighting) consists of two sequential stages. The first stage (Normalization, LSSA): $Q, K$ rows are $L_2$ normalized to constrain dot products to $[-1, 1]$; Softplus replaces $e^x$ as the non-negative function; $\log d \cdot \log N$ serves as a position-dependent scaling factor; each row is then $L_1$ normalized. The second stage (Sharpening): On LSSA's output, perform "multiply by $N$ (number of tokens per position) → subtract bias matrix $O$ → apply ReLU and raise to the $p$-th power → $L_1$ normalization." Both stages are integrated into the original GPT-2 small (124M) + RoPE framework, with all other components unchanged.

### Key Designs

1. **Softmax Decomposition + LSSA: Softplus + Dynamic $\log N$ Scaling in Normalization Stage**:

    - **Function**: Replace $e^x$ with a numerically stable, length-extrapolatable function, and use a row-wise scaling factor to maintain relative entropy invariance across different token counts.
    - **Mechanism**: First, $Q_i \leftarrow Q_i/\|Q_i\|_2$, $K_i \leftarrow K_i/\|K_i\|_2$ so $QK^T \in [-1,1]$; then $A = \mathrm{Softplus}((\log d \cdot \log \mathbf N)\odot QK^T)\odot M'$, where $\mathbf N$ is an $L\times L$ matrix with the $i$-th row filled with $i$ (i.e., the number of tokens attended by that row); finally, $A_i \leftarrow A_i/\|A_i\|_1$. The $\log N$ factor ensures entropy invariance with respect to length during training and adaptively rescales for any inference length, providing stable dynamics.
    - **Design Motivation**: Ablations show $L_1$ is the core; $e^x$ is just one globally nonzero, differentiable choice, but its range can explode. Softplus $=\log(1+e^x)$ is also globally nonzero but numerically stable, and its derivative (sigmoid) is steep in $[-1,1]$, naturally matching cosine normalization. The $\log N$ factor, from Chiang & Cholak 2022's entropy invariance analysis, adjusts scaling per row rather than by total length, ensuring both head and tail tokens receive appropriate "temperature."

2. **Re-weighting: $\mathrm{ReLU}^p$ + L1 Normalization in Sharpening Stage**:

    - **Function**: Further sharpen the stable, dense distribution from the normalization stage, pulling weights toward the maximum and structurally eliminating attention smoothing on long sequences.
    - **Mechanism**: $A \leftarrow \mathrm{ReLU}^p(A\odot\mathbf N - O)$, $A_i \leftarrow A_i/\|A_i\|_1$. $O$ is an all-ones bias matrix (first 3 rows set to 0 to avoid early training instability), shifting the distribution center near zero; ReLU masks out elements below threshold; raising to the $p$-th power further amplifies peaks. The paper proves: as $p\to\infty$, for any $x_l<x_m$, $\lim_{p\to\infty}(x_m^p - x_l^p)/\sum_k x_k^p = 1$, i.e., after sharpening, the maximum approaches 1 and all others approach 0, equivalent to the hard argmax limit.
    - **Design Motivation**: Using ReLU directly as $\phi$ leads to "dead neurons"—once a token's score drops below zero, it never receives gradients again. The key insight is to fully separate "non-negativization" and "sharpening": first use Softplus so all tokens participate in L1 competition and retain gradient paths, then apply post-hoc sharpening on the stabilized distribution. Since the gradient chain is already established, even if some tokens are masked by ReLU in the sharpening stage, learning is not interrupted. $p$ acts as an "inverse temperature," equivalent to annealing the existing softmax-like distribution.

3. **Minimal-Intrusive Integration into Existing LLMs**:

    - **Function**: Enable the entire scheme to be plugged into GPT architectures with near-zero modification, without altering RoPE/PE/FFN or any other components.
    - **Mechanism**: Replace the attention module's $\mathrm{Softmax}(\cdot)$ with "LSSA → re-weighting" in one step; the mask matrix $M'$ (lower triangle is 1) replaces the original $-\infty$ mask (since Softplus outputs non-negative values, masking changes from adding $-\infty$ to multiplying by 0/1). All other modules retain GPT-2 small defaults, and FlashAttention-style kernel functions can directly reuse the elementwise nature for forward/backward passes.
    - **Design Motivation**: The engineering value of length extrapolation requires the solution to be plug-and-play. All innovations are deliberately confined within attention, avoiding conflicts with RoPE, positional interpolation, sliding window attention, etc., so LSSAR can be orthogonally combined with existing long-context techniques.

### Loss & Training
All models are based on GPT-2 small (124M) + RoPE, trained on FineWeb-10B with sequence length 1024 for 18,865 steps, totaling 10.2B training tokens + 0.1B validation tokens, on 8×A100 80GB. $p$ is a key hyperparameter; the paper reports $p\in\{3, 15\}$: $p=3$ is optimal at 1K length, $p=15$ is optimal at 8K and above (smaller $p$ balances sparsity and smoothness, larger $p$ strongly sharpens).

## Key Experimental Results

### Main Results
All methods use the same GPT-2 124M + RoPE, trained at length 1K and extrapolated to 2K/4K/8K for validation loss:

| Attention | 1K | 2K | 4K | 8K |
|-----------|-----|-----|-----|-----|
| Softmax baseline | 3.19 | 4.17 | 5.45 | 6.28 |
| Sigmoid (RamapuRam 2024) | 3.19 | 7.46 | 11.84 | 14.50 |
| ReLU (Wortsman 2023) | 3.21 | 6.27 | 8.50 | 10.35 |
| LSSA (Normalization only) | 3.19 | 4.13 | 5.30 | 5.94 |
| LSSAR ($p=3$) | 3.18 | 4.24 | 5.41 | 6.30 |
| **LSSAR ($p=15$)** | **3.19** | **3.19** | **3.23** | **3.32** |

Downstream zero-shot (Softmax 124M vs LSSAR 124M): ARC-E 39.77→40.57, HellaSwag 32.42→33.03, PIQA 64.09→65.34, SciQ 60.6→62.1, SummScreen 1.68→6.31.

### Ablation Study

| Configuration | 8K Validation Loss | Notes |
|---------------|-------------------|-------|
| Full LSSAR ($p=15$) | 3.32 | Complete model, nearly lossless extrapolation |
| LSSA only (no re-weighting) | 5.94 | Contribution of sharpening stage, largest gap on long sequences |
| Re-weighting + Softmax ($p=15$) | 7.02 | Using Softmax for normalization collapses, showing two stages must match |
| Sigmoid + L1 + re-weighting ($p=15$) | 3.86 | Shows L1 + re-weighting is the truly effective combination, but still inferior to LSSAR |
| ReLU as $\phi$ | >10 | "Dead neurons," unusable for long sequences |

Passkey retrieval (needle-in-a-haystack):

| Length | Softmax Accuracy | LSSAR ($p=15$) Accuracy |
|--------|------------------|-------------------------|
| 1K | 64% | **86%** |
| 1.5K | 0% | 45% |
| 4K | 0% | 20% |
| 8K | 0% | Nonzero |

### Key Findings
- Validation loss at 8K length barely increases (3.19→3.32), representing the first "almost free" length extrapolation among all candidate methods.
- On the Passkey task, Softmax models drop to 0% immediately past training length, while LSSAR maintains nonzero accuracy at 8× training length—this is the cleanest probe for whether attention is truly sharpened.
- In symbolic regression, GPT-109M + LSSAR can recover Newton's law $F\propto m_1/r^2$ from planetary trajectory sequences, while Softmax GPT produces physically meaningless formulas; even trillion-parameter LLMs like o3, Claude 4 Sonnet, Gemini 2.5 Pro fail on this task, suggesting that the inductive bias of the attention mechanism may matter more than model scale for learning physical laws.
- $p$ is optimal at 3 for 1K, but needs to be larger (15) for long sequences; this indicates the optimal degree of sparsity is length-dependent, suggesting future work on adaptive $p$.

## Highlights & Insights
- The conclusion "Softmax decomposition → L1 is the key" alone warrants a re-examination of the entire attention replacement research direction: years of Softmax-free work have focused on replacing the non-negative function, which is the wrong direction.
- Decoupling "sparsity and gradient paths" is the key insight—first use Softplus to ensure all tokens receive gradients, then apply ReLU + power function for post-hoc sharpening. This is akin to pruning the fully connected gradient graph of Softmax rather than building a sparse graph directly, avoiding the "dead neuron" problem of ReLU attention.
- The row-wise scaling factor $\log d \cdot \log N$ can be transferred to any normalization operation requiring length extrapolation (e.g., cross attention, retrieval scoring); its essence is "scaling temperature by the actual number of tokens participating in normalization."
- The symbolic regression experiment provides a novel and rigorous evaluation: whether the model can learn the $1/r^2$ law from physical trajectories, which better reflects the inductive bias of attention than traditional NLP benchmarks.

## Limitations & Future Work
- The scaling factor $\log d \log N$ in LSSA is validated at $L=1024$, $d=64$; the authors acknowledge that the $\log d$ part may need retuning for larger $d$, and no hyperparameter sweep for large models is provided.
- $p=15$ is empirically optimal, but lacks theoretical justification; whether a learnable or adaptive $p$ is needed for unknown tasks/lengths remains an open question.
- Experiments are only reported for 124M / 109M scale; no loss comparison for 7B/13B models is given, so whether the long-sequence extrapolation advantage holds at large scale remains to be verified.
- $\mathrm{ReLU}^p$ at large $p$ still risks numerical overflow, especially at $p=15$ where $x^{15}$ has a huge dynamic range, requiring dedicated kernels for FP16 training stability.

## Related Work & Insights
- **vs Sigmoid attention (Ramapuram 2024)**: The authors show that the performance drop is due to missing L1 normalization; adding L1 brings sigmoid close to Softmax—essentially, "non-negative + L1" suffices, and LSSA is even more stable.
- **vs ReLU attention series**: ReLU's hard threshold causes dead neurons and catastrophic failure on long sequences; LSSAR places the hard threshold in the re-weighting stage (after L1 normalization), avoiding the pitfall of direct sparsification in the main attention.
- **vs Positional Interpolation / ALiBi / NTK rope**: These are post-hoc remedies at the PE layer and do not address the root cause of attention smoothing; LSSAR directly solves smoothing at the attention level and can be combined with these positional encoding schemes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Softmax = $\phi$ + L1, L1 is the core" is a disruptive conclusion; the two-stage design and symbolic regression experiment are highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation loss, downstream zero-shot, passkey retrieval, and symbolic regression experiments are mutually reinforcing; lacks large-scale model validation.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from Softmax decomposition to LSSA to re-weighting is very clear, and the theoretical analysis ($p\to\infty$ limit) is concise.
- Value: ⭐⭐⭐⭐⭐ Length extrapolation is nearly free + numerically stable, making this one of the few "both stable and extrapolatable" alternatives for LLM attention.

## Related Papers

- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](../../ICLR2026/llm_efficiency/understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](../../ICLR2026/llm_efficiency/dnd_boosting_large_language_models_with_dynamic_nested_depth.md)
- [\[AAAI 2026\] The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models](../../AAAI2026/llm_efficiency/the_curious_case_of_analogies_investigating_analogical_reasoning_in_large_langua.md)
- [\[ACL 2025\] Giraffe: Design Choices for Extending the Context Length of Visual Language Models](../../ACL2025/llm_efficiency/design_choices_for_extending_the_context_length_of_visual_language_models.md)
- [\[ICLR 2026\] EvoEngineer: Mastering Automated CUDA Kernel Code Evolution with Large Language Models](../../ICLR2026/llm_efficiency/evoengineer_mastering_automated_cuda_kernel_code_evolution_with_large_language_m.md)

</div>

<!-- RELATED:END -->
