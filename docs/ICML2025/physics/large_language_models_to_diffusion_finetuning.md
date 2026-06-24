---
title: >-
  [Paper Note] L2D: Large Language Models to Diffusion Finetuning
description: >-
  [ICML 2025][Physics & Scientific Computing][LLM Finetuning] This paper proposes the L2D finetuning method, which treats a seed pretrained LLM as a single-step diffusion model and introduces a parallel diffusion path to achieve multi-step inference scaling. Without modifying the original weights, it obtains monotonically increasing accuracy as the number of inference steps increases, achieving consistent improvements across mathematical, coding…
tags:
  - "ICML 2025"
  - "Physics & Scientific Computing"
  - "LLM Finetuning"
  - "Diffusion Framework"
  - "Test-Time Scaling"
  - "LoRA"
  - "Classifier-free Guidance"
date: 2026-05-08
content_hash: b67d6669816c2fb8
---

# L2D: Large Language Models to Diffusion Finetuning

**Conference**: ICML 2025  
**arXiv**: [2501.15781](https://arxiv.org/abs/2501.15781)  
**Code**: [github.com/SakanaAI/L2D](https://github.com/SakanaAI/L2D)  
**Area**: Self-Supervised  
**Keywords**: LLM Finetuning, Diffusion Framework, Test-Time Scaling, LoRA, Classifier-free Guidance

## TL;DR
This paper proposes the L2D finetuning method, which treats a seed pretrained LLM as a single-step diffusion model and introduces a parallel diffusion path to achieve multi-step inference scaling. Without modifying the original weights, it obtains monotonically increasing accuracy as the number of inference steps increases, achieving consistent improvements across mathematical, coding, and reasoning tasks on four LLMs.

## Background & Motivation
**Background**: Autoregressive LLMs have achieved tremendous success in the language domain, but inherently lack the capability to scale inference computation on demand—computation per token is fixed, making it impossible to allocate more computation to critical decisions.

**Limitations of Prior Work**: (1) Existing test-time scaling methods (prompting, token-level search) are limited by the generated token space, offering restricted scalability; (2) Training language diffusion models from scratch lags far behind their autoregressive counterparts, questioning their applicability to the language domain; (3) Parameter-efficient finetuning like LoRA is lightweight but cannot provide inference-time scaling capabilities.

**Key Challenge**: How to endow LLMs with the inference-time scaling characteristics of the diffusion framework while preserving their existing "System 1" understanding capabilities?

**Key Insight**: Viewing the next-token prediction of LLMs (with no prior information, $t=0$) as a special case of single-step diffusion, and introducing multi-step diffusion capability as a natural extension via finetuning.

**Core Idea**: Instead of training a language diffusion model from scratch, a parallel diffusion path is added to a pretrained LLM, reusing its knowledge to achieve multi-step reasoning.

## Method

### Overall Architecture
L2D introduces a parallel "diffusion path" alongside the frozen LLM backbone path. During training, for each target token $y^k$, a timestep $t$ and a noisy token $x_t = t \cdot V_y + (1-t) \cdot x_0$ ($x_0 \sim \mathcal{N}(0, \sigma^2 I)$) are sampled. The diffusion path accesses the KV cache of the main path via cross-attention to predict $y$. During inference, starting from pure noise, denoising is performed step-by-step through Euler integration, updating $x_t$ by sampling token embeddings at each step, and eventually outputting the final prediction.

### Key Designs
1. **Parallel Diffusion Path Architecture**:
    - **Function**: Construct a fully parallel, lightweight Transformer path alongside the frozen LLM.
    - **Mechanism**: The diffusion path $f_{\theta_d}$ has the same number of layers as the main path $f_{\theta_l}$. Each layer contains an MLP (reusing the main path weights + LoRA) and cross-attention (where queries come from diffusion tokens, and keys/values come from the main path's self-attention KV cache). Fusion is performed only at the final layer via a weighted sum $f_{\theta_l} + w_d(t) f_{\theta_d}$, where $w_d(t) = w_{\theta_d}(t) - w_{\theta_d}(0)$ ensures that the original LLM output is unaffected at $t=0$.
    - **Design Motivation**: (1) Freezing the main path preserves original capabilities; (2) Shared KV cache ensures the main path only needs to be computed once during inference; (3) Independent timestep sampling enables training to be parallelized across sequences.

2. **Cross-Entropy Diffusion Training**:
    - **Function**: Train the language diffusion model using standard cross-entropy (CE) loss rather than MSE.
    - **Mechanism**: The loss is $L^{CE}(\theta) = -\mathbb{E}_{x_0, x_1, t}[\log(f_\theta(x_t, t, c)_y)]$, where $x_t = t \cdot V_y + (1-t) \cdot x_0$. The diffusion path still outputs vocabulary logits, but additionally receives $x_t$ containing partial information of the target token ($t=0$ for pure noise, $t=1$ for perfect information). A rectified flow schedule is adopted with $\alpha_t = t, \beta_t = 1-t$.
    - **Design Motivation**: CE loss directly aligns with standard LM training—equivalent to standard next-token prediction at $t=0$, making L2D a natural extension of LMs.

3. **Classifier-Free Guidance + Adaptive ODE Solver**:
    - **Function**: Introduce powerful guidance techniques from the diffusion field and adaptive computation allocation.
    - **Mechanism**: During training, the class embedding $g_j$ is dropped out with a certain probability. During inference, a guided prediction is constructed as $\hat{x}_g = w_g f_\theta(x_t,t,g_j,c) - (1-w_g) f_\theta(x_t,t,g_0,c)$. An adaptive ODE solver (second-order Runge-Kutta) automatically adjusts the number of inference steps for each token based on the diffusion error.
    - **Design Motivation**: Guidance equips the LLM with expert-level generation capability for specific tasks; the adaptive solver allows the model to autonomously decide to spend more computation on complex problems.

### Loss & Training
The model is trained for 1 epoch using the cross-entropy diffusion loss $L^{CE}$, with the AdamW optimizer and a 100-step warm-up followed by linear decay. $\sigma=64$ (a high noise standard deviation concentrates diffusion steps in meaningful intervals), diffusion dimension $\bar{d}=256$, and LoRA rank 8. By default, inference uses a midpoint solver with 8 discrete steps (15 evaluations of $f_{\theta_d}$).

## Key Experimental Results

### Main Results (Across 4 LLMs)

| Model | Method | Math | Code | General Knowledge | Avg | Params |
|------|------|------|------|---------|------|--------|
| Llama 1B | Baseline | 11.93 | 47.63 | 28.54 | 28.54 | - |
| | +LoRA ft. | 18.68 | 44.82 | - | 29.97 | 3M |
| | +Full ft. | 22.94 | 31.04 | - | 27.04 | 1235M |
| | **+L2D** | **28.02** | **49.80** | - | **35.50** | 73M |
| Qwen 2.5 7B | Baseline | 11.98 | 73.01 | - | 46.65 | - |
| | +LoRA ft. | 51.95 | 83.83 | - | 63.34 | 10M |
| | **+L2D** | **63.21** | **84.00** | - | **67.58** | 233M |

### Extended Experiments (Llama 1B)

| Method | Math | Code | All Tasks |
|------|------|------|---------|
| L2D (15 steps) | 28.02 | 49.80 | 35.50 |
| L2D (127 steps) | 28.39 | 51.90 | 36.24 |
| L2D (Adaptive solver) | 30.26 | 49.53 | 36.34 |
| L2D + token search | **35.95** | 49.79 | **38.57** |
| LoRA ft. → L2D | 29.19 | 48.45 | 35.51 |

### Key Findings
- Increasing the number of inference steps in L2D leads to a monotonic increase in accuracy, replicating the scaling characteristics of diffusion models.
- The adaptive solver automatically allocates more steps on difficult tasks like MATH/MMLU (averaging 118 steps vs. a fixed 15 steps).
- L2D is orthogonal to traditional finetuning and search—the three can be combined (L2D + token search reaches 38.57).
- Full fine-tuning severely degrades performance on coding tasks (31.04 vs. 47.63 baseline), whereas L2D improves it (49.80).

## Highlights & Insights
- The observation that "LLMs are single-step diffusion models" establishes a unified perspective between autoregressive and diffusion frameworks.
- The design of $w_d(0)=0$ ensures that L2D never compromises the original LLM's single-step capability—a true "strictly additive" property.
- The adaptive ODE solver enables LLMs to autonomously allocate computation on a per-token basis, analogous to "System 2 thinking" without relying on CoT.
- L2D is orthogonally compatible with LoRA, full fine-tuning, and token search, opening up a new dimension for scaling.

## Limitations & Future Work
- Inference overhead scales linearly (requires 15 evaluations of $f_{\theta_d}$), which remains challenging for real-time applications.
- Although the 73M–281M new parameters are far fewer than in full fine-tuning, they are still significantly higher than LoRA's 3M–13M.
- Finetuning is only performed on instruction-following data, leading to limited improvements on tasks requiring new world knowledge.
- Classifier-Free Guidance requires predefined task classes, which limits its generalizability.

## Related Work & Insights
- **vs MDLM/Plaid (NeurIPS24)**: Training language diffusion models from scratch lags far behind autoregressive LMs. L2D bypasses this bottleneck by finetuning a pretrained LLM.
- **vs LoRA (ICLR22)**: Highly lightweight with ~3M parameters but lacks inference-time scaling capabilities. L2D trading 73M parameters achieves a qualitative leap (28.54 → 35.50).
- **vs Chain-of-Thought**: CoT "thinks" by generating more tokens, but its computation allocation is inflexible. L2D's adaptive solver allows for per-token computation allocation.
- **Insight**: The "iterative refinement" paradigm of the diffusion framework may be an overlooked orthogonal direction for LLM inference scaling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The approach of introducing the scaling characteristics of the diffusion framework to autoregressive LLMs is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive evaluation across 4 models, 6 tasks, various scaling methods, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative arc from single-step to multi-step diffusion is exceptionally clear and elegant.
- Value: ⭐⭐⭐⭐⭐ Opens up a new dimension for LLM inference scaling, orthogonally compatible with traditional methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Finetuning Stellar Spectra Foundation Models with LoRA](finetuning_stellar_spectra_foundation_models_with_lora.md)
- [\[ICML 2025\] Liger: Linearizing Large Language Models to Gated Recurrent Structures](liger_linearizing_large_language_models_to_gated_recurrent_structures.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](../../ICML2026/physics/softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](../../NeurIPS2025/physics/encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[ICML 2025\] Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)

</div>

<!-- RELATED:END -->
