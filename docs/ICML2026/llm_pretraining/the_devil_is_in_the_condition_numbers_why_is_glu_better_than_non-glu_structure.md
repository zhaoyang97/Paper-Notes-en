---
title: >-
  [Paper Note] The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?
description: >-
  [ICML 2026][LLM Pretraining][GLU] From an NTK perspective, this work proves that GLU rewrites the kernel matrix of a two-layer network as the "Hadamard product of the original NTK and the data Gram matrix," which significantly compresses the condition number and accelerates convergence. Empirical results demonstrate that GLU does not improve the generalization gap; its benefits derive entirely from superior optimization.
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "GLU"
  - "SwiGLU"
  - "Neural Tangent Kernel"
  - "Condition Number"
  - "Training Dynamics"
date: 2026-05-08
content_hash: 7bb1fee6eeb6d547
---

# The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?

**Conference**: ICML 2026  
**arXiv**: [2605.20749](https://arxiv.org/abs/2605.20749)  
**Code**: https://github.com/Zemdalk/GLU-NTK (Available)  
**Area**: Optimization Theory / Neural Tangent Kernel / LLM Architecture  
**Keywords**: GLU, SwiGLU, Neural Tangent Kernel, Condition Number, Training Dynamics  

## TL;DR
From an NTK perspective, this work proves that GLU rewrites the kernel matrix of a two-layer network as the "Hadamard product of the original NTK and the data Gram matrix," which significantly compresses the condition number and accelerates convergence. Empirical results demonstrate that GLU does not improve the generalization gap; its benefits derive entirely from superior optimization.

## Background & Motivation

**Background**: From LLaMA, Qwen, and DeepSeek to almost all modern open-source LLMs, the FFN layers have defaulted to GLU variants such as SwiGLU/GEGLU. Mathematically, $\mathrm{GLU}_\phi(\mathbf{x}) = (\mathbf{P}\mathbf{x}) \odot \phi(\mathbf{W}\mathbf{x})$ simply multiplies the original non-gated feed-forward block by a linear gate. However, both literature and engineering experience consistently report that GLU converges faster and performs better than pure ReLU/GELU.

**Limitations of Prior Work**: The advantages of GLU remain largely empirical, with almost no interpretable theory. Existing explanations often rely on vague terms like "gating provides second-order nonlinearity to enhance expressivity," which neither explains why GLU is effective even in small models like two-layer MLPs nor accounts for the counter-intuitive "loss-crossing" phenomenon in training curves: ReLU converges faster initially, only to be overtaken by ReGLU later.

**Key Challenge**: Conflating "fast training error reduction" with "small generalization gap" leads to incorrect attribution. A rigorous decomposition is $\mathcal{L}_\mathcal{D}(f_\theta) = \mathcal{L}_S(f_\theta) + (\mathcal{L}_\mathcal{D}(f_\theta) - \mathcal{L}_S(f_\theta))$, where the former represents optimization and the latter represents generalization. These must be discussed separately to identify the true contribution of GLU.

**Goal**: (1) Utilize a theoretically analyzable framework to derive the GLU kernel matrix and characterize its spectral properties relative to its non-gated counterparts; (2) Translate spectral differences into observable training curve phenomena (including loss-crossing); (3) Empirically compare the generalization gaps of GLU and non-GLU models to determine whether GLU improves optimization or generalization.

**Key Insight**: The authors select the NTK framework. On one hand, NTK encodes training dynamics into the spectrum of a kernel matrix; on the other hand, established results show that the steps required for gradient descent to reach $\epsilon$ error is $\mathcal{O}(\kappa\log(1/\epsilon))$, where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number of the NTK. By calculating the extreme eigenvalues of GLU and non-GLU NTKs, convergence speeds can be directly compared.

**Core Idea**: Under LeCun initialization, the NTK of a two-layer ReGLU model approximately satisfies $\tilde{\mathbf{K}} \approx \mathbf{K} \odot (\mathbf{X}\mathbf{X}^\top/d)$, i.e., the "Hadamard product of the original ReLU NTK and the data Gram matrix." This Hadamard reweighting significantly compresses the NTK spectrum ($\lambda_{\max}$ decreases by an order of magnitude while $\lambda_{\min}$ remains stable or increases), improving the condition number from $\mathcal{O}(n/d)$ to $\mathcal{O}(n/d^2)$. Thus, the advantage of GLU can be reduced to a "better-conditioned NTK matrix."

## Method

### Overall Architecture

This paper does not propose a new method but instead provides a mechanistic proof of "why GLU is effective" using the Neural Tangent Kernel (NTK) framework. The logical chain is as follows: first, derive the analytical form of the gated and non-gated two-layer network NTKs under LeCun initialization, identifying a Hadamard product relationship; next, use random matrix theory to estimate extreme eigenvalues, translating this multiplication structure into a quantitative bound of a "lower-order condition number"; finally, relate the spectral differences back to training dynamics to explain both faster convergence and the loss-crossing phenomenon, while empirically ruling out the "GLU improves generalization" hypothesis.

Specifically, the setup involves: input $\mathbf{x}\in\mathbb{R}^d$, hidden width $m$, non-gated model $z(\mathbf{x}) = \mathbf{V}\phi(\mathbf{W}\mathbf{x})$, and gated model $z(\mathbf{x}) = \mathbf{V}[(\mathbf{P}\mathbf{x}) \odot \phi(\mathbf{W}\mathbf{x})]$. Weights are independently Gaussian initialized as $W_{ij}\sim\mathcal{N}(0,\sigma_w^2)$, $P_{ij}\sim\mathcal{N}(0,\sigma_p^2)$, and $V_{ij}\sim\mathcal{N}(0,\sigma_v^2)$, following LeCun initialization: $\sigma_w^2 = \sigma_p^2 = 1/d$ and $\sigma_v^2 = 1/m$. Spectral estimations are built upon the Marchenko–Pastur distribution, El Karoui kernel matrix expansion, and Weyl’s inequality.

### Key Designs

**1. GLU NTK Equals "Original NTK ⊙ Data Gram": Collapsing Architectural Differences into a Multiplicative Term**

Past explanations of GLU's superiority were vague because there was no established relationship between the gated and non-gated NTKs. The authors address this by taking expectations over parameters and applying LeCun initialization. By utilizing $\sigma_v^2 + \sigma_p^2 \approx \sigma_p^2$ for large $m$, they derive the element-wise relationship $\tilde{K}_{ij} \approx K_{ij}\cdot(\mathbf{x}_i^\top\mathbf{x}_j/d)$, or in matrix form $\tilde{\mathbf{K}} \approx \mathbf{K}\odot(\mathbf{X}\mathbf{X}^\top/d)$. Gating is equivalent to element-wise reweighting of the original NTK by a "normalized version of the data Gram matrix." This represents a purely architectural design (multiplying by a linear gate) being equated to a purely statistical object (the data Gram matrix). Consequently, all differences focus on the $\mathbf{X}\mathbf{X}^\top/d$ term.

**2. Order-of-Magnitude Decrease in Condition Number (Core Theorem 3.1): Verifiable Statements on the Factor of $d$**

The authors translate the Hadamard structure into calculable convergence metrics. Using the arc-cosine kernel formula, the ReLU NTK is expanded into three components: $\mathbf{K} = \alpha\mathbf{X}\mathbf{X}^\top + \beta\mathbf{rr}^\top + \gamma\mathbf{D}$ (data Gram, rank-1 update of $\mathbf{r}_i = \|\mathbf{x}_i\|$, and diagonal correction). The gated version becomes:

$$\tilde{\mathbf{K}} = \frac{\alpha}{d}(\mathbf{X}\mathbf{X}^\top)\odot(\mathbf{X}\mathbf{X}^\top) + \frac{\beta}{d}(\mathbf{rr}^\top)\odot(\mathbf{X}\mathbf{X}^\top) + \frac{\gamma}{d}\mathbf{D}^2.$$

By estimating $\lambda_{\max}$ and $\lambda_{\min}$ via Marchenko–Pastur and El Karoui expansions and applying Weyl’s inequality, they find $\lambda_{\max}(\mathbf{K}) = \mathcal{O}(mn/d)$ while $\lambda_{\max}(\tilde{\mathbf{K}}) = \mathcal{O}(mn/d^2)$. Since $\lambda_{\min}$ remains $\mathcal{O}(m)$ such that $\lambda_{\min}(\tilde{\mathbf{K}}) \geq \lambda_{\min}(\mathbf{K})$, the condition number drops from $\kappa(\mathbf{K}) = \mathcal{O}(n/d)$ to $\kappa(\tilde{\mathbf{K}}) = \mathcal{O}(n/d^2)$—a full factor of $d$ improvement. This characterizes GLU NTK with a "diagonal dominant" geometric image: diagonal terms are amplified while off-diagonal terms are suppressed.

**3. Explaining Loss-crossing via Spectral Decomposition: Not Noise, but a Spectral Consequence**

The loss-crossing phenomenon—where ReLU is faster initially but ReGLU overtakes it later—is often dismissed as random noise. The authors explain this through the NTK spectrum: in the NTK regime, MSE along each eigen-direction decays as $(1 - \eta\lambda_i)^t$. Early training is dominated by $\lambda_{\max}$, where ReLU’s larger $\lambda_{\max}$ provides a faster initial drop. Late training is dominated by $\lambda_{\min}$, where ReGLU’s larger $\lambda_{\min}$ ensures faster convergence of the remaining error components. This is formalized in Proposition 4.1 for infinite-width loss $\mathbb{E}_\theta[L_k] \propto \mathrm{Tr}[(\mathbf{I}-\eta\mathbf{K})^{2k}\mathbf{K}] + \mathbf{Y}^\top(\mathbf{I}-\eta\mathbf{K})^{2k}\mathbf{Y}$ and Corollary 4.2, which identifies a temporal boundary separating the two phases.

### Loss & Training

This is a theoretical analysis and does not introduce new losses or training strategies. Experiments follow standard MSE/Cross-entropy with SGD/AdamW defaults across two-layer MLPs, MLP-Mixer, ViT, and GPT-2, comparing ReLU/ReGLU, GELU/GEGLU, and SiLU/SwiGLU.

## Key Experimental Results

### Main Results

| Target | Primary Phenomenon | Theoretical Alignment |
|--------|------|------|
| Numerical Synthetic NTK ($d$ variation) | ReGLU $\lambda_{\max}$ is significantly smaller than ReLU, $\lambda_{\min}$ is slightly larger; condition number is an order lower | Matches analytical estimates in Prop B.6/B.9 across dimensions |
| ViT FFN replacement (CIFAR/ImageNet) | Condition number trend: Gated < Non-gated | Consistent with Theorem 3.1: GLU still compresses the spectrum in real architectures |
| GPT-2 FFN variations (SwiGLU/GEGLU) | NTK condition number remains smaller than SiLU/GELU before and after training | Indicates condition number improvement persists in LLMs |

### Ablation Study

| Configuration | Key Metrics | Explanation |
|------|---------|------|
| Two-layer MLP, ReLU vs ReGLU, Gaussian input, lr=0.005 | Early ReLU loss lower $\rightarrow$ Late ReGLU overtakes | Reproduces loss-crossing; consistent with Corollary 4.2 |
| GELU/GEGLU, SiLU/SwiGLU comparisons | Crossing still observed | Phenomenon is independent of specific activation; validates mechanism |
| Increased lr (e.g., 0.008) | Crossing compressed; early advantage disappears | High lr accelerates all eigen-directions, consistent with spectral view |
| Scalar plots of $(L_S, L_\mathcal{D}-L_S)$ for various models | GLU and non-GLU distributions almost overlap; $p \geq 0.05$ | Refutes "GLU reduces generalization gap" hypothesis; optimizer choice (AdamW) moves points more significantly |

### Key Findings

- From the NTK perspective of two-layer networks, the advantage of GLU is captured by a clean Hadamard product formula, largely independent of the specific activation function. This explains why ReGLU, GEGLU, and SwiGLU are all effective.
- Experimentally, "GLU learns better" is equivalent to "GLU has higher optimization efficiency at the same training loss," **not because it generalizes better**. This work uses energy distance statistical tests to provide counter-evidence to the common misattribution of generalization.
- Loss-crossing is not noise but a byproduct of the $\lambda_{\max}/\lambda_{\min}$ spectral difference; this can serve as a diagnostic criterion to determine if a new architecture truly improves optimization.

## Highlights & Insights

- The Hadamard-product structure is the most elegant result of the paper: it links an architectural design (adding a gate) to a statistical object (the data Gram matrix), providing a clear mechanism extensible to other modules like attention.
- The philosophy of explicitly decoupling training error and generalization gap is valuable: many "gains" in modern LLMs are products of confounding these two. Scatter plots of $\mathcal{L}_\mathcal{D} = \mathcal{L}_S + \text{gap}$ are simple yet effective diagnostics.
- The geometric image of "diagonal dominance + increased gradient angles" ($\cos\tilde{\phi}_{ij} = \cos\phi_{ij}\cdot\cos\alpha_{ij}$) suggests that gating helps separate samples in the gradient feature space, aligning with recent gradient angle theories and serving as a guide for designing new activations.

## Limitations & Future Work

- The theory is established within the NTK regime for two-layer networks; explanations for real-world LLMs rely on numerical condition numbers and empirical extensions of loss-crossing. Rigorous spectral analysis for deep architectures with attention remains an open problem.
- Corollary 4.2 requires conditions such as $d \geq 5$ and $n \geq 300$, which may not hold in low-dimensional or small-data scenarios.
- The paper explains "why GLU is faster" but does not address "how much compute budget should be allocated to gates"—a more pressing engineering question. Condition numbers could serve as a proxy for architecture search.
- The generalization conclusions depend on the energy distance test; further validation across hyperparameter scales and data regimes is needed.

## Related Work & Insights

- **vs De Ryck et al. 2024 / Liu et al. 2025 (NTK Convergence Theory)**: This work utilizes the $\mathcal{O}(\kappa\log(1/\epsilon))$ framework but is the first to apply it specifically to GLU architectures, providing explicit bounds for condition number improvement via Hadamard products.
- **vs Shazeer 2020 (Empirical Study of GLU Variants)**: While Shazeer provides empirical rankings, this work provides a first-principles explanation. They are mutually supportive.
- **vs El Karoui 2010 (Kernel Random Matrix Theory)**: This work uses these technical tools to handle the self-Hadamard products of Wishart matrices, bridging architectural questions and random matrix problems.
- **vs Wang 2025 (Gated Attention)**: The Hadamard explanation can be migrated to GLU-attention, predicting similar condition number compression effects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Byte Latent Transformer: Patches Scale Better Than Tokens](../../ACL2025/llm_pretraining/byte_latent_transformer.md)
- [\[ICLR 2026\] Rethinking Data Curation in LLM Training: Online Reweighting Offers Better Generalization than Offline Methods](../../ICLR2026/llm_pretraining/rethinking_data_curation_in_llm_training_online_reweighting_offers_better_genera.md)
- [\[AAAI 2026\] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](../../AAAI2026/llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)
- [\[CVPR 2025\] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](../../CVPR2025/llm_pretraining/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](../../ACL2025/llm_pretraining/splintering_nonconcatenative_languages_for_better_tokenization.md)

</div>

<!-- RELATED:END -->
