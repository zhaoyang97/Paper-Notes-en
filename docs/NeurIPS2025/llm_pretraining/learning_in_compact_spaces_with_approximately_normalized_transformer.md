---
title: >-
  [Paper Note] Learning in Compact Spaces with Approximately Normalized Transformer
description: >-
  [NeurIPS 2025][LLM Pretraining][approximate normalization] This paper proposes anGPT (Approximately Normalized GPT), which exploits the concentration of vector norms in high-dimensional spaces to replace per-sample exact…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "approximate normalization"
  - "concentration of measure"
  - "compact space"
  - "weight-decay-free"
  - "convergence acceleration"
date: 2026-05-08
content_hash: 60def23f9c3c5a3a
---

# Learning in Compact Spaces with Approximately Normalized Transformer

**Conference**: NeurIPS 2025
**arXiv**: [2505.22014](https://arxiv.org/abs/2505.22014)
**Code**: [github.com/automl/anGPT](https://github.com/automl/anGPT)
**Area**: LLM Pretraining
**Keywords**: approximate normalization, concentration of measure, compact space, weight-decay-free, convergence acceleration

## TL;DR

This paper proposes anGPT (Approximately Normalized GPT), which exploits the concentration of vector norms in high-dimensional spaces to replace per-sample exact normalization with simple scalar multiplication. The method achieves 40% convergence speedup over GPT+ (with QK-norm) while eliminating weight decay and learning rate warmup, incurring only 3% runtime overhead.

## Background & Motivation

**Background**: Deep Transformer training relies heavily on normalization and regularization techniques (LayerNorm / RMSNorm / QK-Norm) to ensure stability, requiring the tuning of numerous hyperparameters including weight decay coefficients and learning rate warmup steps.

**Limitations of Prior Work**: nGPT constrains all parameters and representations strictly to a hypersphere, which greatly accelerates convergence but introduces additional normalization layers, increasing training runtime by approximately 9% and inference by approximately 3%. The computational cost of exact normalization is non-negligible for large models.

**Key Challenge**: There is an inherent trade-off between the convergence acceleration brought by normalization and the computational overhead of normalization itself. The ideal scenario would be to retain the benefits of normalization (stabilizing input scale, preventing residual stream norm explosion) while avoiding expensive per-sample normalization operations.

**Goal**: (a) Can a cheaper approach achieve the effect of "approximate normalization"? (b) Can weight decay and learning rate warmup be simultaneously eliminated?

**Key Insight**: The Concentration of Measure phenomenon in high-dimensional spaces — when the dimension $d$ is sufficiently large, the values of Lipschitz functions on a hypersphere concentrate tightly around their expected values. Vector norms can therefore be approximated by an input-independent constant factor.

**Core Idea**: Replace per-sample exact normalization with an input-independent scalar normalization factor $\nu$, combined with a parameter norm constraint ($\|w\|_2 \leq 1$), to construct a Transformer in a "compact space" that obtains the convergence advantages of normalization without additional normalization layers.

## Method

### Overall Architecture

anGPT is built upon the standard GPT architecture (SwiGLU + RoPE + QK-Norm) with three levels of modification: (1) replacing the classical residual connection with LERP-based residual updates; (2) removing RMSNorm layers and replacing them with scalar approximate normalization factors $\nu$; (3) constraining the L2 norm of each row vector of all weight matrices to be $\leq 1$ (rather than $= 1$). After input token embedding, the model passes through $L$ layers of MHA + MLP blocks, and produces logits via a learnable scaling vector $s_Z$ and a head linear layer.

### Key Designs

1. **Approximate Normalization Factor $\nu$**

    - **Function**: Provides an input-independent scalar scaling factor for the output of each linear projection and activation function, such that the output vector norm is approximately 1.
    - **Mechanism**: Based on the concentration theorem for Lipschitz functions on high-dimensional spheres. For output $g(x)$, its norm $\|g(x)\|_2$ concentrates around the expected value $\mathbb{E}[\|g(x)\|_2]$ with exponential probability, with deviation probability $\leq 2\exp(-cdt^2/\|g\|_{\text{Lip}}^2)$. The factor $\nu = 1/\sqrt{\mathbb{E}[\|g(x)\|_2^2]}$ can therefore be precomputed as a fixed constant.
    - **Specific factors**: QKV projection $\nu_{qkv} = \sqrt{d_m/d_h}$; output projection $\nu_p = \sqrt{d_h/d_m}$; upsampling $\nu_{uz} = \sqrt{d_m/(4d_m)} = 0.5$; downsampling $\nu_d = \sqrt{4d_m/d_m} = 2$; SiLU activation $\nu_{acf} \approx 3.74$ (Monte Carlo estimate).
    - **Design Motivation**: Eliminates the overhead of computing $\|x\|_2$ exactly at every step as required by nGPT. These factors are constants and can be absorbed directly into the weight matrices at inference time.

2. **LERP Residual Update + Approximate Normalization**

    - **Function**: Replaces the classical residual $h \leftarrow h + x$ with linear interpolation $h \leftarrow h + \alpha(x - h)$, augmented with a normalization factor.
    - **Mechanism**: The expected norm after the residual update is $\mathbb{E}[\|(1-\alpha)h + \alpha x\|^2] = 1 - 2\alpha + 2\alpha^2$ (assuming $h, x$ are independent, unit-normed, and zero-mean), so the normalization factor is $\nu(\alpha) = 1/\sqrt{\alpha^2 + (1-\alpha)^2}$. This factor is a function of $\alpha$ and must be updated each step, but with negligible computational cost.
    - **Design Motivation**: nGPT applies an exact norm operation after LERP; anGPT replaces this with the scalar multiplication $\nu(\alpha)$, reducing computation.

3. **Parameter Norm Constraint ($\|w\|_2 \leq 1$)**

    - **Function**: Constrains the L2 norm of each row of weight matrices to be no greater than 1 (rather than strictly equal to 1).
    - **Mechanism**: nGPT requires strict weight normalization ($\|w\|_2 = 1$), which excludes the zero vector and limits expressiveness. anGPT relaxes this to an upper-bound constraint, allowing weights to be learned freely while remaining bounded.
    - **Design Motivation**: (a) Eliminates the need for weight decay (parameters are already bounded); (b) ensures the derived normalization factors remain valid (since $\|Wx\|$ is controlled by $\|w\|$); (c) initializing weights as normalized at the start of training removes the need for learning rate warmup.

4. **Parameter Reparameterization**

    - **Function**: Reparameterizes all learnable scaling parameters ($\alpha_A, \alpha_M, s_Z$) to unify optimization dynamics.
    - **Mechanism**: Surrogate parameters $\hat{s}_a$ are optimized directly, mapped back to true parameters via $s_a = (s_{a,\text{init}}/s_{a,\text{scale}}) \cdot \hat{s}_a$, ensuring all stored parameters are of comparable magnitude (approximately $s_{a,\text{scale}}$), allowing Adam's adaptive learning rate mechanism to operate uniformly across different parameter types.

### Loss & Training

- Standard cross-entropy language modeling loss.
- Adam (anGPT / nGPT) or AdamW (GPT+) with cosine annealing schedule.
- anGPT requires no weight decay or learning rate warmup; $s_{a,\text{init}} = 0.01$.
- Dataset: SlimPajama (627B tokens); tokenizer: GPT-NeoX (50K vocabulary); context length: 2048.

## Key Experimental Results

### Main Results: Convergence Speedup on 0.5B Model

| Setting | GPT+ | nGPT | anGPT |
|---------|------|------|-------|
| With QK-norm, avg. convergence speedup | 1.0× | 1.29× | **1.40×** |
| Without QK-norm, avg. convergence speedup | 1.0× | 1.8× | **2.0×** |
| Training step time (A100×4) | 0.1416s | 0.1552s (+9.6%) | 0.1455s (+2.75%) |
| Inference overhead | — | ~3% | **0%** (factors absorbed into weights) |

Consistent ~40% convergence speedup is also observed on 250M and 1.0B models.

### Ablation Study (0.5B, 10B tokens, OpenWebText)

| Configuration | Improvement over GPT+ (no QK) |
|---------------|-------------------------------|
| GPT+ without QK-norm | Baseline |
| GPT+ with QK-norm | +2.1% |
| nGPT | +3.0% |
| nGPT + parameter norm constraint (replacing strict normalization) | +3.4% |
| anGPT (full configuration) | **+3.8%** |

### Key Findings

- **Consistent scaling laws**: The scaling exponents of anGPT and GPT+ are nearly identical (32M→1B), indicating that the architectural improvement represents a "constant-factor speedup" rather than a change in scaling behavior.
- **Larger batch sizes**: anGPT can accommodate larger batch sizes on bigger models, facilitating multi-node parallelism.
- **Residual normalization factor is most critical**: Ablations show that removing the normalization factor from the residual LERP leads to training instability; estimation errors within 2% for other factors do not affect results.
- **Downstream tasks**: The 1B model consistently outperforms GPT+ on 6 benchmarks, with gains of 3%–22%.
- **Comparison with DyT**: DyT (Dynamic Tanh), which replaces normalization layers, degrades performance by 6.5%; LN scaling yields no meaningful gain; anGPT improves by 1.3%.

## Highlights & Insights

- **"The blessing of dimensionality"**: The concentration of norms of high-dimensional vectors makes it possible to approximate per-sample normalization with a constant factor — an elegant mathematical perspective that turns the "curse" of dimensionality into a "blessing."
- **Decoupling stabilization from loss optimization**: In conventional LayerNorm, the learnable parameter $\gamma$ simultaneously serves to stabilize activations and participate in gradient descent. anGPT decouples these two roles — $\nu$ handles stabilization while model parameters focus solely on optimizing the loss.
- **Simplified hyperparameters**: After eliminating weight decay and warmup, only learning rate and batch size require tuning, making scaling law estimation more efficient.
- **Zero inference overhead**: Constant normalization factors can be absorbed into weight matrices before deployment, incurring no additional computation at inference time.

## Limitations & Future Work

- Validation is limited to the GPT architecture; other architectures such as Vision Transformers and diffusion models have not been tested.
- The largest training scale is 7× Chinchilla (0.5B × 70B tokens); effectiveness at larger scales remains unknown.
- The derivation of normalization factors for the attention matrix assumes dense attention, whereas attention is typically sparse in practice. FlashAttention does not expose effective non-zero ratios, which remains an open problem.
- Low-precision (FP8) training adaptation is a promising direction: bounded activations and weights in compact spaces are naturally suited to low-precision representation.

## Related Work & Insights

- **vs. nGPT**: nGPT strictly normalizes all representations onto a hypersphere at the cost of 9.6% training overhead and 3% inference overhead. anGPT replaces exact normalization with approximations, achieving comparable or slightly better performance at only 2.75% overhead (0% at inference).
- **vs. DyT**: DyT replaces LayerNorm with tanh, targeting runtime reduction without modifying the representation space. DyT degrades performance by 6.5% in experiments, suggesting that simply replacing the activation shape is insufficient — systematically maintaining representation norms is essential.
- **vs. LN Scaling (Curse of Depth)**: Sun et al. scale activations by $1/\sqrt{l}$ per layer to mitigate norm explosion in deep networks, which only alleviates rather than eliminates the problem. anGPT addresses it systematically through normalization factors applied across the full computation graph.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The theoretical angle of connecting concentration of measure to approximate normalization is elegant, though the overall architectural modifications are a natural extension of nGPT.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-scale models (32M–1B), ablations, scaling laws, downstream evaluation, and comparisons with DyT and LN-scaling are all thoroughly covered.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, and the three-way architectural comparison in Table 1 is immediately informative.
- **Value**: ⭐⭐⭐⭐ High practical value — eliminates hyperparameters and reduces overhead; larger-scale validation is still needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities](born_a_transformer_--_always_a_transformer_on_the_effect_of_pretraining_on_archi.md)
- [\[ACL 2026\] Compact Example-Based Explanations for Language Models](../../ACL2026/llm_pretraining/compact_example-based_explanations_for_language_models.md)
- [\[NeurIPS 2025\] Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)
- [\[NeurIPS 2025\] Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](ricl_temporal_credit.md)
- [\[NeurIPS 2025\] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models](learning_the_wrong_lessons_syntactic-domain_spurious_correlations_in_language_mo.md)

</div>

<!-- RELATED:END -->
