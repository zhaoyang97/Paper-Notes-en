---
title: >-
  [Paper Note] CoFrGeNet: Continued Fraction Architectures for Language Generation
description: >-
  [ICML 2026][Image Generation][Continued Fractions] This paper introduces "continued fractions," a class of functions with optimal rational approximation properties…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Continued Fractions"
  - "CoFrNet"
  - "Attention Alternatives"
  - "FFN Alternatives"
  - "Parameter Efficiency"
date: 2026-05-08
content_hash: 00ab6d280476d329
---

# CoFrGeNet: Continued Fraction Architectures for Language Generation

**Conference**: ICML 2026  
**arXiv**: [2601.21766](https://arxiv.org/abs/2601.21766)  
**Code**: Not publicly released (IBM Research)  
**Area**: Efficient LLM Architectures / Language Generation / Transformer Alternatives  
**Keywords**: Continued Fractions, CoFrNet, Attention Alternatives, FFN Alternatives, Parameter Efficiency

## TL;DR
This paper introduces "continued fractions," a class of functions with optimal rational approximation properties, into generative Transformers. By designing CoFrNet replacement modules (CAttnU/CAttnM/Cffn) for multi-head attention and FFNs, it reduces $d$ divisions to 1 through a closed-form "continuants" expression. On GPT2-xl and Llama-3.2B, it achieves comparable or superior downstream performance using only $\frac{2}{3}\sim\frac{1}{2}$ of the original parameters.

## Background & Motivation
**Background**: Transformers are the mainstream for language models, but the quadratic complexity of attention and the 4-8× parameter expansion of FFNs lead to rapid growth in model scale. Significant improvements have focused on these components: Linformer/Synthesizer linearize attention, Multi-Query/GQA reduce KV heads, Slim Attention removes value matrices, and Sparse Attention limits token spans. However, **almost all involve subtractions within the "attention" or "standard MLP" function classes**. SSM/Mamba takes a different path but remains a linear recurrence of hidden states.

**Limitations of Prior Work**: (1) Most methods aim for acceleration while maintaining similar parameter counts, either sacrificing expressiveness (linear attention performance drops) or requiring complex tuning (MoE/sparse). (2) **No systematic change of the function class has been explored**—all variants are essentially combinations of matrix multiplication + softmax + ReLU/GELU, which are "polynomials + element-wise non-linearities."

**Key Challenge**: To drastically compress parameters without quality loss, relying purely on existing function classes (polynomials + activations) is difficult—**polynomial approximations have a clear upper bound on expressiveness for a fixed order**. In contrast, **continued fractions (rational functions) can more tightly approximate arbitrary functions at the same order** (a classic number theory result: truncated fractions are closer to the true value than any rational number with the same denominator).

**Goal**: Extend CoFrNet (Puri et al. 2021) from **supervised learning to generative modeling**. Specifically, three problems must be solved: (a) multi-dimensional outputs, (b) sequence causality constraints, and (c) reducing the overhead of $d$ divisions in $1/x$ non-linearities (division is significantly slower than multiplication on hardware).

**Key Insight**: Express the continued fraction using "continuant polynomials" as $\tilde f(a) = K_{d-1}(a_2,\dots,a_d) / K_d(a_1,\dots,a_d)$, compressing the entire ladder into a "ratio of two polynomials." The gradients can also be derived in a closed form as ratios of continuants—thus, regardless of depth $d$, only a single $1/K_d$ calculation is required.

**Core Idea**: **Replace attention QKV matrices and FFN expanded layers with "continued fraction ladder ensembles + continuant closed-form computation."** This changes the function class mathematically and reduces hardware division to $O(1)$ engineering-wise.

## Method

### Overall Architecture
CoFrGeNet replaces the two main components of a Transformer block with CoFrNet ladder ensembles:
- **CAttnU / CAttnM** (replaces causal multi-head attention, offering two implementations);
- **Cffn** (replaces FFN, removing the traditional $\alpha\sim 4$ expansion).

The unified implementation of ladder ensembles follows Equation (8) $y = Ux + Vz, \ z_j = \tilde f(W^{(j)} x)$: each ladder $j$ uses a set of parameters $W^{(j)}$ to project the input into $d$-dimensional partial denominators. The CF layer uses continuant recurrence to calculate $K_0,\dots,K_d$ and $1/K_d$, outputs $z_j = K_{d-1}/K_d$, and finalizes the output via linear combination. A custom PyTorch autograd.Function computes both forward continuants and backward gradients simultaneously.

### Key Designs

1. **Continuant-Based Implementation**:

    - **Function**: Compresses a $d$-depth continued fraction ladder (which normally requires $d$ divisions) into a single division, accelerating both forward and backward passes.
    - **Mechanism**: Calculates all continuants using the recurrence $K_k(a_{d-k+1},\dots,a_d) = a_{d-k+1} K_{k-1} + K_{k-2}$ in $O(d)$ multiplications/additions. The value $\tilde f(a) = K_{d-1}/K_d$ (one division), and gradients follow Proposition 1 as $\partial \tilde f / \partial a_k = (-1)^k (K_{d-k}(a_{k+1},\dots,a_d) / K_d(a_1,\dots,a_d))^2$. Since all gradients share the same denominator $K_d$, only one $1/K_d$ calculation is needed. This is wrapped in `torch.autograd.Function`, caching $K_*$ and $1/K_d$.
    - **Design Motivation**: A standard implementation requires $d$ divisions for $d$ layers; modern GPUs are 5-20× slower at division than multiplication. This rewrite allows for deeper ladders without division penalties. It also performs $\text{sgn}(K_d)\max(|K_d|, \epsilon)$ clipping once to avoid singularities—retaining more expressiveness than clipping $d$ times (Puri et al. 2021). Inference time dropped from 5898 μs to 628 μs, a **~10× speedup**.

2. **Causal Token-Token Mixing (CAttnU / CAttnM)**:

    - **Function**: Implements token-token information mixing using CoFrNet ladders while maintaining causality, reducing parameters from $4p^2$ to $l(2d+l+1)$ or $L(p+l)+p^2$.
    - **Mechanism**:
        - **CAttnU**: Transposes the embedding vs. sequence dimension $l$ (similar to MLP-Mixer) and uses **univariate ladders** so $x_i$ only receives one dimension of token $i$. Two ensembles output $y_1$ and $y_2$, followed by **upper triangular linear layers** $U_1, U_2$ to ensure causality. Element-wise multiplication $O = U_1 y_1 \odot U_2 y_2$ generates cross-terms.
        - **CAttnM**: Uses $L$ $p$-variate ladders to output $y_1, y_2$, concatenated into a fully connected layer $F$. A **causal softmax** calculates attention weights $A = \text{Csoftmax}([y_1, y_2] F)$, followed by the standard $O = AV$ where $V = X W^v$.
    - **Design Motivation**: Direct $p$-variate ladders on transposed dimensions break causality, necessitating univariate ladders + triangular constraints. The element-wise product $U_1 y_1 \odot U_2 y_2$ is crucial—it generates **cross-dimensional terms** to enhance the weak expressiveness of single univariate ladders. CAttnM is closer to "lightweight standard attention," making it more stable with slightly more parameters.

3. **FFN Replacement (Cffn) + Training Schedule**:

    - **Function**: Replaces FFN with $L$ $p$-variate ladders, **eliminating the $\alpha\sim4$ expansion** and reducing parameters from $2\alpha p^2$ to $Lp(d+1) + 2p^2$.
    - **Mechanism**: Cffn uses $p$-variate ladders with gated **non-expanded** ($\alpha=1$) representations. It employs a dyadic schedule: only linear parts are updated for $t/2$ steps; depth-1 ladder parameters are released at $t/2$; depth-2 at $3t/4$, and so on.
    - **Design Motivation**: FFN expansion layers are considered redundant. Continued fractions provide enough expressiveness to omit expansion. The dyadic schedule addresses training instability—rational function gradients near $K_d \to 0$ are sharp. Stabilizing the linear backbone before introducing high-order rational corrections acts as a "curriculum from linear to rational approximation." Table 5 shows significant PPL degradation without it (Wikitext2 PPL 17.12 to 26.71).

### Loss & Training
Next-token cross-entropy is maintained. Optimizer: Adam. Learning rates for GPT2-xl: $6\times 10^{-4}$ (pre-training), $0.25\times 10^{-4}$ (baseline fine-tune), and $0.125\times 10^{-4}$ (Ours). Weight decay: 0.1. Singularity protection: $\epsilon = 0.01$. Ladder depth $d \in \{1,3,5,7\}$, width $L \in \{1,3,5,7\}$.

## Key Experimental Results

### Main Results
GPT2-xl (1.5B) vs. three CoFrGeNet variants on OWT and GneissWeb (GW) pre-training, fine-tuned on GLUE:

| Data | Model | Params | MNLI | QQP | QNLI | SST2 | COLA | MRPC | RTE |
|------|------|------|------|-----|------|------|------|------|-----|
| OWT | GPT2-xl | 1.5B | 86.89 | 88.93 | 91.35 | 93.56 | 81.78 | 79.83 | 60.27 |
| OWT | **CoFrGeNet-F** | **985M** | **87.26** | **89.95** | **91.89** | **94.16** | **82.59** | **80.21** | **61.35** |
| OWT | CoFrGeNet (Dual) | **798M** | 87.11 | 89.36 | 91.79 | 93.91 | 81.97 | 79.93 | 61.25 |
| OWT | Synthesizer-D | 1.2B | 84.93 | 86.82 | 90.13 | 91.34 | 80.15 | 77.95 | 59.83 |
| OWT | Sparse Attn | 1.21B | 85.27 | 86.38 | 90.93 | 92.72 | 80.76 | 77.42 | 59.36 |
| GW | GPT2-xl | 1.5B | 78.28 | 86.83 | **82.93** | 91.82 | 74.18 | 77.72 | 60.19 |
| GW | **CoFrGeNet-F** | 985M | **79.62** | **87.26** | 82.73 | **92.36** | **74.83** | **78.01** | **61.35** |
| GW | CoFrGeNet | **798M** | 79.05 | 86.98 | 82.12 | 92.13 | 74.38 | 77.95 | 61.11 |

Downstream Perplexity (OWT pre-trained):

| Model | Params | PTB | Wikitxt2 | Lbda | AgNews | LM1B | Wikitxt103 |
|------|------|-----|---------|------|--------|------|-----------|
| GPT2-xl | 1.5B | 30.12 | 18.30 | 8.66 | 37.13 | 41.20 | 17.50 |
| **CoFrGeNet-F** | **985M** | **29.89** | **17.12** | **8.12** | **35.72** | **40.14** | **16.14** |
| CoFrGeNet | 798M | 30.03 | 17.96 | 8.55 | 36.47 | 40.86 | 17.17 |

### Ablation Study

| Config | Metric | Description |
|------|---------|------|
| Continuants vs. Naive (CoFrGeNetB) | Inference 628 vs 5898 μs | **10× inference speedup**, proves continuant necessity |
| Continuant Training Time | 178 hr vs 203 hr (CoFrGeNetB) | 12-13% faster training |
| w/o dyadic schedule | Wikitext2 PPL 17.12 → 26.71 | Crucial for stability, avoids 50% degradation |
| FFN vs. Attention replacement | F remains superior | FFN replacement provides more gain than attention |

### Key Findings
- **FFN are more replaceable than attention**: CoFrGeNet-F (FFN change only) performs best despite having fewer parameters (985M). This supports the consensus that "FFN serves as a memory bank" and doesn't require $4\times$ expansion if the function class is expressive enough.
- **Continuant implementation is the engineering key**: Without it, continued fractions are impractical due to division costs.
- **Effective for small models**: On Llama-3.2B, CoFrGeNet is competitive or superior on 8 open-domain QA and reasoning tasks.
- **Singularity protection + Output clipping**: Clipping ladder outputs during testing is essential for robust deployment.

## Highlights & Insights
- **Changing the function class is a rare innovation**: This is one of the first systematic proofs that non-polynomial / non-element-wise activation classes (rational functions) can be competitive in generative models.
- **Translating number theory to engineering efficiency**: Using continuants to reduce $d$ divisions to 1 based on the property that "gradients are also continuant ratios" is elegant.
- **Plug-in compatibility**: Since it replaces specific modules, the training/inference pipeline (tokenizer, KV cache, LoRA) remains unchanged, reducing adoption costs nearly to zero.
- **Dyadic training as a general trick**: Gradually releasing higher-order rational corrections is a valuable method for controlling the instability of rational networks.

## Limitations & Future Work
- **Closed source**: Limits independent reproduction.
- **Scaling evidence**: Missing verification on 7B+ scales.
- **Incompatible with modern attention stacks**: CAttnM still requires $l \times l$ softmax, making it hard to integrate directly with FlashAttention.
- **Numerical stability**: While $\epsilon$-clipping prevents poles, it is unclear how the model behaves near these boundaries in the long term.

## Related Work & Insights
- **vs. Synthesizer/Sparse Attention**: Superiority at lower parameter counts suggests changing function classes is more effective than structural pruning.
- **vs. GQA**: GQA compresses parameters via head sharing; CoFrNet replaces the whole QKV calculation. These are orthogonal and can be combined (as verified on Llama).
- **vs. CoFrNets (2021)**: This work scales the concept from simple supervised learning to complex engineering challenges like causality and multi-dimensional outputs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bringing continued fractions to language generation is a high-level innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across GPT2/Llama and multiple benchmarks; missing 7B+ and direct wall-clock against Mamba.
- Writing Quality: ⭐⭐⭐⭐ Rigorous math; somewhat dense notation.
- Value: ⭐⭐⭐⭐ Highly practical for industry parameter reduction while opening new academic research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Esoteric Language Models: A Family of Any-Order Diffusion LLMs](esoteric_language_models_a_family_of_any-order_diffusion_llms.md)
- [\[ACL 2026\] Multimodal Large Language Models for Multi-Subject In-Context Image Generation](../../ACL2026/image_generation/multimodal_large_language_models_for_multi-subject_in-context_image_generation.md)
- [\[CVPR 2026\] Organizing Unstructured Image Collections using Natural Language](../../CVPR2026/image_generation/organizing_unstructured_image_collections_using_natural_language.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](../../CVPR2026/image_generation/language-free_generative_editing_from_one_visual_example.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](../../NeurIPS2025/image_generation/continuous_diffusion_model_for_language_modeling.md)

</div>

<!-- RELATED:END -->
