---
title: >-
  [Paper Note] The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?
description: >-
  [ICML 2026][LLM Pretraining][GLU] Under the NTK perspective, it is proven that GLU reformulates the kernel matrix of a two-layer network as the "Hadamard product of the original NTK and the data Gram matrix." This signif…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "GLU"
  - "SwiGLU"
  - "Neural Tangent Kernel"
  - "Condition Number"
  - "Training Dynamics"
date: 2026-05-08
content_hash: 5acaf3208ea9ccf1
---

# The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?

**Conference**: ICML 2026  
**arXiv**: [2605.20749](https://arxiv.org/abs/2605.20749)  
**Code**: https://github.com/Zemdalk/GLU-NTK (Available)  
**Area**: Optimization Theory / Neural Tangent Kernel / LLM Architecture  
**Keywords**: GLU, SwiGLU, Neural Tangent Kernel, Condition Number, Training Dynamics  

## TL;DR
Under the NTK perspective, it is proven that GLU reformulates the kernel matrix of a two-layer network as the "Hadamard product of the original NTK and the data Gram matrix." This significantly compresses the condition number and accelerates convergence. Empirical results further show that GLU does not improve the generalization gap; its benefits stem entirely from superior optimization.

## Background & Motivation

**Background**: From LLaMA, Qwen, and DeepSeek to almost all modern open-source large models, FFN layers have defaulted to GLU variants such as SwiGLU/GEGLU. Mathematically, $\mathrm{GLU}_\phi(\mathbf{x}) = (\mathbf{P}\mathbf{x}) \odot \phi(\mathbf{W}\mathbf{x})$ simply adds a linear gate to the original non-gated feed-forward block. However, both literature and engineering experience consistently report that GLU converges faster and performs better than pure ReLU/GELU.

**Limitations of Prior Work**: The advantages of GLU remain largely empirical, with almost no interpretable theory. Existing explanations often resort to vague terms like "gating provides second-order nonlinearity and enhances expressivity," which neither explains why GLU is effective even in small models like two-layer MLPs nor accounts for a counter-intuitive phenomenon in training curves: ReLU initially converges faster than ReGLU before being overtaken (loss-crossing).

**Key Challenge**: Conflating "fast training error reduction" with "small generalization gap" leads to incorrect attribution. A clean decomposition is $\mathcal{L}_\mathcal{D}(f_\theta) = \mathcal{L}_S(f_\theta) + (\mathcal{L}_\mathcal{D}(f_\theta) - \mathcal{L}_S(f_\theta))$, where the former is optimization and the latter is generalization. These must be discussed separately to locate the true contribution of GLU.

**Goal**: (1) Use a theoretically analyzable framework to derive the GLU kernel matrix and characterize its spectral properties relative to non-gated counterparts; (2) Translate spectral differences into observable training curve phenomena (including loss-crossing); (3) Empirically compare the generalization gaps of GLU and non-GLU models to determine whether GLU improves optimization or generalization.

**Key Insight**: The authors choose the NTK framework—on one hand, NTK encodes training dynamics entirely into the spectrum of a kernel matrix; on the other, existing conclusions show that the steps required for gradient descent to reach $\epsilon$ error is $\mathcal{O}(\kappa\log(1/\epsilon))$, where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number of the NTK. Convergence speeds can be directly compared by calculating the extreme eigenvalues of GLU and non-GLU NTKs.

**Core Idea**: Under LeCun initialization, the NTK of a two-layer ReGLU model approximately satisfies $\tilde{\mathbf{K}} \approx \mathbf{K} \odot (\mathbf{X}\mathbf{X}^\top/d)$, which is the "Hadamard product of the original ReLU NTK and the data Gram matrix." This Hadamard reweighting significantly compresses the NTK spectrum ($\lambda_{\max}$ drops by an order of magnitude, while $\lambda_{\min}$ increases), improving the condition number from $\mathcal{O}(n/d)$ to $\mathcal{O}(n/d^2)$. Thus, the advantage of GLU collapses to a "better-conditioned NTK matrix."

## Method

This paper does not propose a new method but provides a "mechanism-level explanation" for GLU's superiority. The argument proceeds in five steps: deriving the analytical form of the NTK $\rightarrow$ estimating extreme eigenvalues using random matrix theory $\rightarrow$ translating spectral differences into convergence orders $\rightarrow$ explaining loss-crossing via spectral decomposition $\rightarrow$ empirically ruling out generalization gap explanations.

### Overall Architecture

Consider a two-layer network with input $\mathbf{x}\in\mathbb{R}^d$ and hidden width $m$. The non-gated model is $z(\mathbf{x}) = \mathbf{V}\phi(\mathbf{W}\mathbf{x})$, and the gated model is $z(\mathbf{x}) = \mathbf{V}[(\mathbf{P}\mathbf{x}) \odot \phi(\mathbf{W}\mathbf{x})]$. Weights are independently Gaussian initialized as $W_{ij}\sim\mathcal{N}(0,\sigma_w^2)$, $P_{ij}\sim\mathcal{N}(0,\sigma_p^2)$, and $V_{ij}\sim\mathcal{N}(0,\sigma_v^2)$, following the LeCun setting $\sigma_w^2 = \sigma_p^2 = 1/d, \sigma_v^2 = 1/m$. Analytical forms for NTK matrices $\mathbf{K}$ and $\tilde{\mathbf{K}}$ are derived by taking expectations over parameters, and extreme eigenvalues of various components are estimated using the Marchenko–Pastur distribution, El Karoui kernel matrix expansion, and Weyl’s inequality.

### Key Designs

1.  **Hadamard-Product Structure of GLU NTK**:
    - **Function**: Reformulates the gated model's NTK as a multiplicative relationship with the non-gated NTK to allow direct comparison.
    - **Mechanism**: After taking expectations and substituting LeCun initialization (utilizing $\sigma_v^2 + \sigma_p^2 \approx \sigma_p^2$ for large $m$), the relation $\tilde{K}_{ij} \approx K_{ij}\cdot(\mathbf{x}_i^\top\mathbf{x}_j/d)$ is obtained, or $\tilde{\mathbf{K}} \approx \mathbf{K}\odot(\mathbf{X}\mathbf{X}^\top/d)$ in matrix form. This indicates that GLU is equivalent to element-wise reweighting of the original NTK with a "normalized version of the data Gram matrix."
    - **Design Motivation**: By establishing an explicit algebraic link between the two kernel matrices, all differences are compressed into the $\mathbf{X}\mathbf{X}^\top/d$ term. Spectral analysis then only requires studying the Wishart matrix and its self-Hadamard product, avoiding direct asymptotic analysis of the gated model.

2.  **Order Reduction in Condition Number (Core Theorem 3.1)**:
    - **Function**: Translates the Hadamard structure into quantitative condition number bounds.
    - **Mechanism**: The authors use the arc-cosine kernel formula to Taylor expand the ReLU NTK into three parts: $\mathbf{K} = \alpha\mathbf{X}\mathbf{X}^\top + \beta\mathbf{rr}^\top + \gamma\mathbf{D}$ (Gram matrix, rank-1 update of $\mathbf{r}_i = \|\mathbf{x}_i\|$, and diagonal correction). The gated version is rewritten via Hadamard as $\tilde{\mathbf{K}} = (\alpha/d)(\mathbf{X}\mathbf{X}^\top)\odot(\mathbf{X}\mathbf{X}^\top) + (\beta/d)(\mathbf{rr}^\top)\odot(\mathbf{X}\mathbf{X}^\top) + (\gamma/d)\mathbf{D}^2$. Estimating scales of $\lambda_{\max}$ and $\lambda_{\min}$ for each block and applying Weyl’s inequality yields $\lambda_{\max}(\mathbf{K}) = \mathcal{O}(mn/d)$ and $\lambda_{\max}(\tilde{\mathbf{K}}) = \mathcal{O}(mn/d^2)$. While minimum eigenvalues remain $\mathcal{O}(m)$, $\lambda_{\min}(\tilde{\mathbf{K}}) \geq \lambda_{\min}(\mathbf{K})$, reducing the condition number from $\kappa(\mathbf{K}) = \mathcal{O}(n/d)$ to $\kappa(\tilde{\mathbf{K}}) = \mathcal{O}(n/d^2)$.
    - **Design Motivation**: Turns the abstract conclusion that "GLU is better" into a verifiable statement that "the condition number differs by a factor of $d$ for $d$-dimensional inputs." This provides a geometric image where the GLU NTK is significantly more "diagonally dominant."

3.  **Spectral Decomposition Explanation for Loss-Crossing**:
    - **Function**: Uses the same spectral map to explain the counter-intuitive training curve where ReLU is faster early on while ReGLU overtakes later.
    - **Mechanism**: In the NTK regime, MSE loss decays independently along each eigen-direction; error in the $i$-th direction shrinks at $(1 - \eta\lambda_i)^t$. Early convergence is dominated by $\lambda_{\max}$, and later stages by $\lambda_{\min}$. Since ReLU has a larger $\lambda_{\max}$, it decays faster along principal directions initially. However, ReGLU has a larger $\lambda_{\min}$, leading to higher convergence rates for remaining components, eventually overtaking after some steps. This is formalized in Proposition 4.1 (closed-form loss $\mathbb{E}_\theta[L_k]$) and Corollary 4.2.
    - **Design Motivation**: Incorporates "training loss crossing"—often mistaken for random noise—into a spectral perspective, providing a monotonic verifiable criterion consistent with the condition number theorem.

### Loss & Training
This is a theoretical analysis and does not introduce new losses or strategies. Experiments follow standard MSE/Cross-Entropy and SGD/AdamW settings across two-layer MLPs, MLP-Mixer, ViT, and GPT-2, comparing ReLU/ReGLU, GELU/GEGLU, and SiLU/SwiGLU activation pairs.

## Key Experimental Results

### Main Results

| Target | Main Phenomena | Theoretical Alignment |
|--------|----------------|-----------------------|
| Synthetic NTK Matrix (Var. $d$) | ReGLU $\lambda_{\max}$ is significantly smaller than ReLU, $\lambda_{\min}$ is slightly larger; lower condition number by one order. | Matches analytical estimates in Propositions B.6/B.9 across dimensions. |
| ViT FFN with GLU variants (CIFAR/ImageNet) | Condition number trend: Gated < Non-gated. | Consistent with Theorem 3.1: GLU compresses the spectrum in real architectures. |
| GPT-2 FFN with SwiGLU/GEGLU | Pre- and post-training NTK condition numbers are smaller than SiLU/GELU controls. | Suggests condition number improvement persists in LLMs. |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| 2-layer MLP, ReLU vs ReGLU, Gaussian input, lr=0.005 | Early ReLU loss lower $\rightarrow$ late ReGLU overtakes. | Reproduces loss-crossing; consistent with Corollary 4.2 two-stage criterion. |
| GELU/GEGLU, SiLU/SwiGLU comparisons | Similar crossing observed. | Phenomenon is independent of specific activation; validates mechanism explanation. |
| Increased lr (e.g., 0.008) | Crossing compressed, early advantage disappears. | Large lr accelerates all directions uniformly, consistent with spectral explanation. |
| Scatter plots $(L_S, L_\mathcal{D}-L_S)$ for MLP-Mixer, ViT, GPT-2 | GLU and non-GLU distributions overlap almost entirely; Energy Distance permutation test $p \geq 0.05$. | Falsifies the hypothesis that "GLU reduces the generalization gap"; optimizers (SGD $\rightarrow$ AdamW) cause much larger shifts. |

### Key Findings

- Under the NTK perspective of two-layer networks, the advantage of GLU is fully captured by a clean Hadamard-product formula, largely independent of the specific activation function. This explains why ReGLU, GEGLU, and SwiGLU are all effective.
- "GLU learns better" is empirically equivalent to "GLU has higher optimization efficiency for the same training loss." **It is not because its generalization is stronger**—a common misattribution the paper refutes via statistical energy distance tests.
- Loss-crossing is not noise but a byproduct of the $\lambda_{\max}/\lambda_{\min}$ spectral difference; this criterion can diagnose whether new architectures truly improve optimization.

## Highlights & Insights

- The Hadamard-product structure is the most elegant result: it links a structural design (gating) with a statistical object (data Gram matrix), providing a mechanism that is clear and extendable to other scenarios like attention.
- The philosophy of explicitly decoupling training error and generalization gap is valuable: many "performance gains" in modern LLMs confuse the two. Plotting scatters by $\mathcal{L}_\mathcal{D} = \mathcal{L}_S + \text{gap}$ is a cheap but effective diagnostic tool.
- The geometric image of "diagonal dominance + increased gradient angles" ($\cos\tilde{\phi}_{ij} = \cos\phi_{ij}\cdot\cos\alpha_{ij}$) suggests gating is equivalent to better separating samples in the gradient feature space, aligning with recent gradient angle theories.

## Limitations & Future Work

- The entire theory is built within the NTK regime of two-layer networks. Explanations for real LLMs rely on numerical condition numbers and empirical extensions of loss-crossing; rigorous spectral analysis for deep networks and attention mechanisms remains an open problem.
- Corollary 4.2 requires strong conditions (e.g., $d\geq 5, n\geq 300$) and may not apply to low-dimensional or small-data scenarios.
- The paper explains "why GLU is fast" but not "how much computational budget should be moved from other modules to the gate"—a resource allocation problem more pressing in LLM engineering.
- Generalization conclusions are based on energy distance tests; dependence on hyperparameters and data scales requires further cross-scenario validation.

## Related Work & Insights

- **vs De Ryck et al. 2024 / Liu et al. 2025 (NTK Convergence)**: This work utilizes the $\mathcal{O}(\kappa\log(1/\epsilon))$ backbone but is the first to apply it to GLU specifically, providing explicit condition number improvement magnitudes via Hadamard products.
- **vs Shazeer 2020 (GLU Variants Empirical Study)**: Shazeer provided empirical rankings; this work provides first-principles explanations. Both corroborate each other.
- **vs El Karoui 2010 (Kernel Random Matrix Theory)**: This work uses El Karoui’s tools to handle self-Hadamard products of Wishart matrices, bridging architecture design and random matrix theory.
- **vs Wang 2025 et al. (Gated Attention)**: The Hadamard explanation can likely be transferred to GLU-attention, predicting similar condition number compression effects—a promising direction for future verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](../../AAAI2026/llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)
- [\[NeurIPS 2025\] Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](../../NeurIPS2025/llm_pretraining/efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)
- [\[NeurIPS 2025\] Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](../../NeurIPS2025/llm_pretraining/broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)
- [\[ICML 2026\] Edit-Based Refinement for Parallel Masked Diffusion Language Models](edit-based_refinement_for_parallel_masked_diffusion_language_models.md)
- [\[ICML 2026\] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition](infolaw_information_scaling_laws_for_large_language_models_with_quality-weighted.md)

</div>

<!-- RELATED:END -->
