---
title: >-
  [Paper Note] YAQA: End-to-End KL Minimization LLM Adaptive Weight Quantization
description: >-
  [ICML 2026][LLM/NLP][Quantization] YAQA replaces the proxy objective of LLM weight quantization from "layer-wise activation error" to "end-to-end model output KL divergence." By using a Kronecker-decomposed Hessian sketc…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Quantization"
  - "Adaptive Rounding"
  - "End-to-end KL"
  - "Hessian Sketching"
  - "Kronecker Decomposition"
date: 2026-05-08
content_hash: 0256f8dd1f217952
---

# YAQA: End-to-End KL Minimization LLM Adaptive Weight Quantization

**Conference**: ICML 2026  
**arXiv**: [2505.22988](https://arxiv.org/abs/2505.22988)  
**Code**: Not yet released  
**Area**: Model Compression / LLM Quantization  
**Keywords**: Quantization, Adaptive Rounding, End-to-end KL, Hessian Sketching, Kronecker Decomposition

## TL;DR
YAQA replaces the proxy objective of LLM weight quantization from "layer-wise activation error" to "end-to-end model output KL divergence." By using a Kronecker-decomposed Hessian sketch, it provides the first end-to-end error bound. It reduces KL divergence by approximately 30% relative to GPTQ/LDLQ, achieves higher accuracy than Quantization-Aware Training (QAT), and maintains unchanged inference speed.

## Background & Motivation

**Background**: LLM quantization follows two paths—QAT learns low-precision representations by modifying the training process, offering high quality but at immense cost; PTQ maps full-precision weights to a discrete codebook post-hoc (e.g., GPTQ/LDLQ), which is popular due to its low cost. GPTQ uses the Hessian of "current layer activation error" $H_1 = \mathbb{E}[x^\top x]$ as a proxy for end-to-end error.

**Limitations of Prior Work**: $H_1$ only considers the input distribution of the current layer, completely ignoring how subsequent layers amplify or cancel out rounding errors. Consequently, "layer-wise optimal" does not equal "model-level optimal," often leading to unnecessarily high KL divergence. GuidedQuant/SqueezeLLM use block-diagonal approximations of the Empirical Fisher, but these originate from cross-entropy task loss rather than the true KL Hessian. Their block structures are heuristic without theoretical guarantees—empirically, increasing the number of blocks leads to inconsistent results.

**Key Challenge**: Direct adaptive rounding against $\nabla^2 L(W^*) \in \mathbb{R}^{mn \times mn}$ (the true KL Hessian with respect to a layer's weights) suffers from scale explosion. To maintain a tractable structure, the approximation quality must be provable. Existing structural approximations either lack bounds or approximate poorly.

**Goal**: Find a structured Hessian sketch that allows LDLQ-style iterative rounding within $O(m+n)$ steps while strictly controlling end-to-end KL through "cosine similarity with the true Hessian."

**Key Insight**: The authors introduce "Structural Nilpotence Degree" (SND), a combinatorial quantity, to characterize the convergence steps of LDLQ. They prove that for a Kronecker product $L_O \otimes L_I$, $\mathrm{snd}(L_O \otimes L_I) = \mathrm{snd}(L_O) + \mathrm{snd}(L_I) \le m+n-1$, which maps "tractable computation" directly onto Kronecker decomposition.

**Core Idea**: Use Kronecker decomposition $\tilde{H} = H_O \otimes H_I$ as an approximation of $\nabla^2 L(W^*)$. Obtain "near-optimal" $H_O, H_I$ via power iteration on the true Fisher. The rounding algorithm adds a symmetric output-side feedback component to LDLQ, significantly lowering KL while taking $\approx 2\times$ LDLQ time.

## Method

### Overall Architecture

YAQA consists of two components: (1) **Rounding Algorithm**—generalizing LDLQ to a Kronecker-decomposed Hessian sketch, resulting in $W = Q(W^* + L_O'^{\top} \Delta L_I' + L_O'^{\top} \Delta + \Delta L_I')$, where $L_O', L_I'$ are the LDL triangular factors of $H_O, H_I$ minus the identity matrix, and $\Delta = W^* - W$; (2) **Hessian Sketching**—constructing $H_O, H_I$ via power iteration to maximize the alignment between $\tilde{H}$ and the true Hessian under the Frobenius inner product. This process is performed independently for each linear layer without changing the inference structure; thus, the inference speed is determined by the codebook (e.g., E8P) and is unrelated to YAQA.

### Key Designs

1. **SND + Kronecker Hessian for End-to-End LDLQ**:

    - **Function**: Characterizes the Hessian structures that allow "tractable LDLQ" and selects an optimal one.
    - **Mechanism**: Defines $\mathrm{snd}(L)$ as the nilpotence degree of a binary nilpotent matrix sharing the same support as $L - I$, proving LDLQ converges in $\le \mathrm{snd}(L)$ steps. For the Kronecker product $L_O \otimes L_I$, $\mathrm{snd}$ is the sum of the degrees of both sides. Thus, $\tilde{H} = H_O \otimes H_I$ allows symmetric "input + output" feedback while requiring only $O(m+n)$ small matrix multiplications. Under this framework, GuidedQuant is equivalent to running LDLQ on a block-diagonal approximation without output feedback, explaining its saturation beyond 4 blocks.
    - **Design Motivation**: Previous PTQ algorithms forced a choice between "layer-wise $H_1$ (no output feedback)" or "QAT (expensive but global)." Kronecker is the first structure to achieve "end-to-end" and "tractable" simultaneously.

2. **End-to-End KL Error Bound + Cosine Similarity Objective**:

    - **Function**: Formulates the model KL upper bound based on the relationship between the Hessian sketch and the true Hessian, guiding the selection of $H_O, H_I$.
    - **Mechanism**: Theorem 3.4 proves $\mathrm{vec}(\Delta) H \, \mathrm{vec}(\Delta)^\top \le \|H\|_F (\|\Delta\|_F^2 \sqrt{2 - 2c} + \text{(incoherence/trace term)})$, where $c = \langle H, H_O \otimes H_I \rangle / (\|H\|_F \|H_O\|_F \|H_I\|_F)$ is the cosine similarity. This implies that the closer the sketch is "in direction" to the true Hessian, the tighter the end-to-end KL bound. Simultaneously, $H_O, H_I$ must maintain low incoherence and low rank.
    - **Design Motivation**: This is the first time any quantization algorithm has obtained an end-to-end error bound, upgrading "Hessian sketch selection" from empirical heuristics to an optimizable objective. Cosine similarity suggests using power iteration for approximation.

3. **Two Scalable Power Iteration Hessian Sketches**:

    - **Function**: Computes $H_O, H_I$ at LLM scale without directly manipulating the $mn \times mn$ true Hessian.
    - **Mechanism**: **Sketch A** assumes token independence within sequences, approximating $H \approx \mathbb{E}[x^\top x \otimes (\nabla_y \ell)^\top (\nabla_y \ell)]$. Starting from $(H_I)_0 = H_1, (H_O)_0 = I$, it converges in $\approx 3$ power iteration steps (approx. 20 GPU-hours for a 10B model). **Sketch B** runs one round of power iteration on the true Fisher (starting from $I, I$) using sequence-level gradients, requiring approx. 30 GPU-hours for a 10B model. Both use a modified backward pass for distributed power iteration, similar to Shampoo's preconditioning but using true Fisher.
    - **Design Motivation**: Direct Monte-Carlo estimation of the true Hessian has high variance. Sketch A trades variance for bias, suitable for low-data regimes. Sketch B tolerates higher variance via a single power iteration round, yielding better quality with more data.

### Loss & Training

YAQA is a PTQ method with no explicit loss. The optimization objective is implicitly $\mathrm{tr}(\Delta^\top H_O \Delta H_I)$, the proxy loss under the Kronecker sketch. The rounding algorithm uses fixed-point iteration (Equation 5/6), compatible with scalar or vector quantizers. It is complementary to QuIP#'s randomized Hadamard transform—the latter makes $W$ near-Gaussian and reduces incoherence, while the former handles precise Hessian computation.

## Key Experimental Results

### Main Results: LLM Quantization Quality

| Model / Setting | Method | KL ↓ (vs FP) | Downstream Benchmarks (acc%) ↑ |
|------------|------|-------|-----------------|
| Llama 3.1 8B Inst, W2 | LDLQ (GPTQ) | Baseline | Baseline |
| Llama 3.1 8B Inst, W2 | GuidedQuant | Slightly better | Slightly better |
| Llama 3.1 8B Inst, W2 | **YAQA Sketch A** | $\approx -30\%$ vs LDLQ | Significantly ahead |
| Llama 3.1 8B Inst, W2 | **YAQA Sketch B** | Lowest | Highest |
| Llama 3.1 8B Inst, W2 | QAT | Higher than YAQA | Lower than YAQA |

(Data summarized from abstract and tables; Sketch B establishes a new PTQ SOTA on multiple chat/reasoning tasks.)

### Ablation Study

| Setting | KL ↓ | Description |
|------|------|------|
| LDLQ ($H_O = I, H_I = H_1$) | Baseline | YAQA degenerate case |
| Sketch A, 1 step | Medium | Initialized with $H_1$ |
| Sketch A, 3 steps | Excellent | Empirical convergence |
| Sketch B, 2K sequences | Excellent | SOTA in 1 GPU-hour |
| Sketch B, 64K sequences | Best | 30 GPU-hours |
| GuidedQuant, >4 blocks | No improvement | Lack of output feedback |

### Key Findings

- Empirically, $H_O$ is near low-rank, matching the theoretical condition where the YAQA bound is strictly tighter than the LDLQ bound, explaining the success of the Kronecker sketch.
- Sketch B with one power iteration round outperforms Sketch A, showing that true Fisher variance is manageable with sequence-level estimates—strict convergence of power iteration is not required.
- YAQA achieves SOTA with minimal data (2K sequences, 1 GPU-hour), a major selling point for PTQ practicality.
- The result that KL is lower than QAT's is counter-intuitive but theoretically sound: QAT uses first-order descent and may get stuck in local optima; YAQA performs "one-shot optimal rounding within the Hessian ellipsoid," bypassing QAT's optimization difficulties.

## Highlights & Insights

- **First End-to-End KL Bound**: Converts "Hessian sketch selection" into a clear mathematical problem of maximizing cosine similarity and controlling incoherence/rank, avoiding past empirical structures.
- **SND Framework Unification**: GPTQ, LDLQ, and GuidedQuant are all unified under the SND/Kronecker perspective, clarifying the presence or absence of output feedback.
- **Tractability and Optimality**: Low SND ensures speed, cosine similarity ensures quality, and power iteration is the classic tool for optimal Frobenius Kronecker approximation—a natural fit.
- **True Fisher vs. Empirical Fisher**: YAQA highlights that for the KL objective, true Fisher (Monte-Carlo sampling of logits) must be used instead of Empirical Fisher (task loss) to avoid directional bias.

## Limitations & Future Work

- Focuses only on weight-only PTQ; transferability to activation and KV-cache quantization is not detailed.
- 30 GPU-hours for Sketch B remains heavy for 70B+ models; exploring aggressive sparsification or low-rank approximations to reduce cost is necessary.
- The cosine similarity bound still contains incoherence/trace terms, and rank is not fully controlled; tightening the bound by controlling the effective rank of $H_O, H_I$ is a future direction.
- Global Hessian behavior for non-linear layers (e.g., Attention Softmax) and cross-layer coupling has not been deeply analyzed.

## Related Work & Insights

- **vs GPTQ / LDLQ**: Equivalent to the degenerate case of YAQA where $H_O = I, H_I = H_1$; YAQA's bound is theoretically tighter when $H_O$ is low-rank.
- **vs GuidedQuant / SqueezeLLM**: Both attempt to go beyond $H_1$, but use Empirical Fisher + block-diagonal approximations, lacking output feedback and end-to-end bounds.
- **vs QAT / DiscQuant / PV-Tuning**: QAT requires long training; YAQA proves that a single power iteration can be more accurate than QAT, boosting confidence in the PTQ route.
- **vs Shampoo / KFAC**: Shares the Kronecker + Power Iteration sketching philosophy, but applies it to the rounding direction of PTQ rather than optimizer preconditioning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to map end-to-end KL bounds to quantization algorithms and provide a provable algorithm-theory loop via SND/Kronecker.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Tested across Llama/Gemma scales and multiple bit configurations, outperfoming LDLQ, GuidedQuant, and QAT, including data requirement ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical derivations, but the SND/Kronecker arguments are dense and have a high entry barrier; the appendix is extensive.
- **Value**: ⭐⭐⭐⭐⭐ Directly useful for LLM deployment: pushes quality to or beyond QAT with near-zero inference cost, representing a significant advancement for PTQ.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Margin-Adaptive Confidence Ranking for Reliable LLM Judgement](margin-adaptive_confidence_ranking_for_reliable_llm_judgement.md)
- [\[NeurIPS 2025\] Decoupled Entropy Minimization](../../NeurIPS2025/llm_nlp/decoupled_entropy_minimization.md)
- [\[ICML 2026\] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)
- [\[ICML 2026\] SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models](spa-cache_singular_proxies_for_adaptive_caching_in_diffusion_language_models.md)
- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](../../ACL2026/llm_nlp/grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)

</div>

<!-- RELATED:END -->
