---
title: >-
  [Paper Note] CoFrGeNet: Continued Fraction Architectures for Language Generation
description: >-
  [ICML 2026][Image Generation][Continued fractions] This work introduces the function class of "continued fractions," known for optimal rational approximation…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Continued fractions"
  - "CoFrNet"
  - "Attention replacement"
  - "FFN replacement"
  - "Parameter efficiency"
date: 2026-05-08
content_hash: c14db59a70035233
---

# CoFrGeNet: Continued Fraction Architectures for Language Generation

**Conference**: ICML 2026  
**arXiv**: [2601.21766](https://arxiv.org/abs/2601.21766)  
**Code**: Not released (IBM Research)  
**Area**: Efficient LLM Architectures / Language Generation / Transformer Alternatives  
**Keywords**: Continued fractions, CoFrNet, Attention replacement, FFN replacement, Parameter efficiency

## TL;DR
This work introduces the function class of "continued fractions," known for optimal rational approximation, into language generation Transformers. CoFrNet replacement modules (CAttnU/CAttnM/Cffn) are designed for multi-head attention and FFN, respectively. By leveraging the closed-form "continuants," $d$ divisions are reduced to a single division. On GPT2-xl and Llama-3.2B, downstream performance is matched or even improved with only $\frac{2}{3}\sim\frac{1}{2}$ of the parameters.

## Background & Motivation
**Background**: Transformer is the mainstream architecture for language models, but the quadratic complexity of attention and the 4-8× parameter expansion of FFN have led to rapid model size growth. Many improvements have targeted these two components: Linformer/Synthesizer linearize attention, Multi-Query/GQA reduce KV heads, Slim Attention removes the value matrix, Sparse Attention restricts token span. However, **almost all remain within the "attention" or "standard MLP" function class**. SSM/Mamba take a different route but still rely on linear hidden state recurrence.

**Limitations of Prior Work**: (1) Most methods accelerate computation while keeping parameter count similar, either sacrificing expressiveness (linear attention loses performance) or requiring complex tuning (MoE/sparse); (2) **No one has systematically changed the function class**—all variants essentially combine matrix multiplication + softmax + ReLU/GELU, i.e., "polynomial + elementwise nonlinearity."

**Key Challenge**: To significantly reduce parameters without sacrificing model quality, it is difficult to break through by relying solely on the existing function class (polynomial + activation), since **the expressiveness of polynomial approximation at fixed degree is clearly bounded**; whereas **continued fractions (rational functions) can more tightly approximate any function at the same degree** (a classical result in number theory: truncated fractions are closer to the true value than any rational number with the same denominator).

**Goal**: Extend CoFrNet from (Puri et al. 2021) from **supervised learning to generative modeling**, addressing three new challenges: (a) outputs change from scalar to multi-dimensional, (b) sequence causality constraints, (c) reducing the $d$ divisions of $1/x$ nonlinearity (division is an order of magnitude slower than multiplication on hardware).

**Key Insight**: Use the "continuant polynomial" of continued fractions to express $\tilde f(a) = K_{d-1}(a_2,\dots,a_d) / K_d(a_1,\dots,a_d)$, compressing the entire ladder into a "ratio of two polynomials," and derive the gradient in closed form as a ratio of continuants—thus, regardless of $d$, only one $1/K_d$ computation is needed.

**Core Idea**: **Replace attention's QKV matrix multiplication and FFN's expanded hidden layer with a "continued fraction ladder set + continuant closed-form computation"**—mathematically changing the function class, and engineering-wise reducing division to $O(1)$ times.

## Method

### Overall Architecture
CoFrGeNet replaces the two main components of the Transformer block with CoFrNet ladder sets:
- **CAttnU / CAttnM** (replace causal multi-head attention, two implementations provided);
- **Cffn** (replace FFN, removing the traditional $\alpha\sim 4$ expansion).

The unified implementation of the ladder set follows equation (8): $y = Ux + Vz, \ z_j = \tilde f(W^{(j)} x)$: each ladder $j$ uses a parameter set $W^{(j)}$ to project the input to $d$-dimensional partial denominators, the CF layer recursively computes $K_0,\dots,K_d$ and $1/K_d$ via continuants, outputs $z_j = K_{d-1}/K_d$, and linearly combines them for the final output. The entire process is encapsulated in a custom PyTorch autograd.Function, computing both forward continuants and backward gradients.

### Key Designs

1. **Continuant-Based "Single Division" Implementation**:

    - **Function**: Compresses the "standard $d$-deep continued fraction ladder" (which requires $d$ divisions) into a single division, accelerating both forward and backward passes.
    - **Mechanism**: Uses the recurrence in (4)(5), $K_k(a_{d-k+1},\dots,a_d) = a_{d-k+1} K_{k-1} + K_{k-2}$, to compute all continuants with only $O(d)$ additions and multiplications; the continued fraction value is $\tilde f(a) = K_{d-1}/K_d$ (single division), and the gradient, per Proposition 1, is $\partial \tilde f / \partial a_k = (-1)^k (K_{d-k}(a_{k+1},\dots,a_d) / K_d(a_1,\dots,a_d))^2$—all gradients share the same denominator $K_d$, so only one $1/K_d$ computation is needed for all partial derivatives. Packaged as `torch.autograd.Function`, with forward saved-for-backward caching of $K_*$ and $1/K_d$.
    - **Design Motivation**: The standard ladder implementation (1) requires a division at each layer, totaling $d$ divisions; on modern GPU/hardware, division is 5-20× slower than multiplication, making deep ladders impractical. The continuant reformulation allows for deeper ladders without division overhead, and only a single $\text{sgn}(K_d)\max(|K_d|, \epsilon)$ clipping is needed to avoid pole divergence—retaining more expressiveness than (Puri et al. 2021), which clips $d$ times before each division. Experiments show CoFrGeNetB (without continuants) has inference time 5898 μs, reduced to 628 μs with continuants, **nearly 10× speedup**.

2. **Causal Attention Replacement: CAttnU / CAttnM (Causal Token-Token Mixing)**:

    - **Function**: Implements token-token information mixing with CoFrNet ladders under causality constraints, reducing parameter count from $4p^2$ to $l(2d+l+1)$ or $L(p+l)+p^2$.
    - **Mechanism**:
        - **CAttnU** (left): Transposes the input tensor along embedding vs sequence length $l$ (similar to MLP-Mixer), uses a **univariate ladder** so $x_i$ only receives a certain dimension of the $i$-th token. Two ensembles output $y_1 = w_0^{(1)} \odot x + (w_1^{(1)} \odot x)^{\circ-1}$ and $y_2$ (for depth 2), each with an **upper-triangular linear layer** $U_1, U_2$ to ensure token $i$ is only influenced by tokens $\le i$. Finally, $O = U_1 y_1 \odot U_2 y_2$, producing cross terms via elementwise multiplication;
        - **CAttnM** (right): No transposition, uses $L$ $p$-variate ladders to output $y_1, y_2$, concatenates and passes through a fully connected layer $F$, then applies **causal softmax** to obtain attention weights $A = \text{Csoftmax}([y_1, y_2] F)$ (token $i$ only attends to tokens $\le i-1$), and proceeds as in standard attention $O = AV$, where $V = X W^v$.
    - **Design Motivation**: Directly applying $p$-variate ladders on the transposed token dimension breaks causality (one output depends on all tokens), so univariate ladders + upper-triangular constraints are necessary. The elementwise product $U_1 y_1 \odot U_2 y_2$ is a key design—single univariate ladders are weak, but elementwise multiplication produces **cross-dimensional terms** that significantly enhance expressiveness. CAttnM is closer to "lightweight standard attention"—retaining the value matrix, only replacing QK logit computation with CoFrNet, more stable but with slightly more parameters. Table 1 compares parameters: when $l \sim p$ (as in GPT/Llama), both replacements significantly reduce parameters.

3. **Non-Expanded Cffn and Dyadic Progressive Training (FFN Replacement + Training Schedule)**:

    - **Function**: Replaces FFN with $L$ $p$-variate ladders, **removing the $\alpha\sim4$ expansion**, reducing parameters from $2\alpha p^2$ to $Lp(d+1) + 2p^2$.
    - **Mechanism**: Cffn directly uses $p$-variate ladders (no transposition, so cross-feature mixing does not affect causality), with gated **non-expanded** ($\alpha=1$) input. Training adopts a dyadic schedule: only the linear part is updated initially; after $t/2$ steps, depth-1 ladder parameters are released; after $3t/4$ steps, depth-2 is released; and so on, with depth-$i$ trained only in the last $t/2^i$ steps.
    - **Design Motivation**: The FFN expansion layer contributes most parameters but is considered redundant; the expressiveness of continued fractions suffices to eliminate expansion. The dyadic schedule addresses the practical issue of "full parameter training from scratch diverges"—CoFrNet's rational functions have sharp gradients near $K_d \to 0$, and early training of deep parameters is unstable; fixing the linear backbone first allows the model to converge in polynomial space, then gradually releasing higher-order rational correction terms acts as a curriculum from linear to rational approximation. Table 5 shows that without dyadic schedule, PPL degrades significantly (on OWT, PTB drops from 29.89 to 33.72, Wikitext2 from 17.12 to 26.71), confirming its necessity.

### Loss & Training
The original next-token cross-entropy of GPT2-xl/Llama is retained; optimizer is Adam, learning rate for GPT2-xl pretraining is $6\times 10^{-4}$, finetuning $0.25\times 10^{-4}$ (baseline) and $0.125\times 10^{-4}$ (CoFrGeNet), weight decay 0.1, no dropout. $\epsilon = 0.01$ is used for pole protection; ladder depth $d \in \{1,3,5,7\}$, width $L$ also $\{1,3,5,7\}$. GPT2-xl is trained on 16 H100 GPUs with DDP, Llama on 128 H100 GPUs with FSDP for 2M steps.

## Key Experimental Results

### Main Results
GPT2-xl (1.5B) vs three CoFrGeNet variants, OWT and GneissWeb (GW) pretraining, downstream GLUE finetuning:

| Data | Model | Params | MNLI | QQP | QNLI | SST2 | COLA | MRPC | RTE |
|------|-------|--------|------|-----|------|------|------|------|-----|
| OWT | GPT2-xl | 1.5B | 86.89 | 88.93 | 91.35 | 93.56 | 81.78 | 79.83 | 60.27 |
| OWT | **CoFrGeNet-F** | **985M** | **87.26** | **89.95** | **91.89** | **94.16** | **82.59** | **80.21** | **61.35** |
| OWT | CoFrGeNet (dual replacement) | **798M** | 87.11 | 89.36 | 91.79 | 93.91 | 81.97 | 79.93 | 61.25 |
| OWT | Synthesizer-D | 1.2B | 84.93 | 86.82 | 90.13 | 91.34 | 80.15 | 77.95 | 59.83 |
| OWT | Sparse Attn | 1.21B | 85.27 | 86.38 | 90.93 | 92.72 | 80.76 | 77.42 | 59.36 |
| GW | GPT2-xl | 1.5B | 78.28 | 86.83 | **82.93** | 91.82 | 74.18 | 77.72 | 60.19 |
| GW | **CoFrGeNet-F** | 985M | **79.62** | **87.26** | 82.73 | **92.36** | **74.83** | **78.01** | **61.35** |
| GW | CoFrGeNet | **798M** | 79.05 | 86.98 | 82.12 | 92.13 | 74.38 | 77.95 | 61.11 |

Downstream perplexity (OWT pretraining):

| Model | Params | PTB | Wikitxt2 | Lbda | AgNews | LM1B | Wikitxt103 |
|-------|--------|-----|----------|------|--------|------|------------|
| GPT2-xl | 1.5B | 30.12 | 18.30 | 8.66 | 37.13 | 41.20 | 17.50 |
| **CoFrGeNet-F** | **985M** | **29.89** | **17.12** | **8.12** | **35.72** | **40.14** | **16.14** |
| CoFrGeNet | 798M | 30.03 | 17.96 | 8.55 | 36.47 | 40.86 | 17.17 |
| Synthesizer-D | 1.2B | 31.47 | 19.35 | 9.92 | 39.84 | 41.94 | 18.91 |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Continuants vs naive implementation (CoFrGeNetB) | Inference 628 vs 5898 μs | **10× inference speedup**, validating the necessity of continuant form |
| Continuants training time | 178 hr vs 203 hr (CoFrGeNetB) | 12-13% faster training (CoFrGeNet dual replacement is even 6% faster than GPT2-xl) |
| w/o dyadic schedule (CoFrGeNet-F OWT) | Wikitext2 PPL 17.12 → 26.71 | Progressive training is crucial; without it, performance degrades by over 50% |
| FFN-only replacement (CoFrGeNet-F 985M) vs attention-only replacement (CoFrGeNet-A 1.21B) | F generally best | FFN replacement contributes more than attention replacement |
| CAttnM vs CAttnU | M slightly better | CAttnM is the default in the report |

### Key Findings
- **FFN is more worth replacing than attention**: CoFrGeNet-F (FFN-only) achieves the best results with the fewest parameters (985M), indicating that most "parameter redundancy" in transformers lies in the expanded FFN rather than attention; this aligns with the interpretability community's consensus that "FFN is a memory bank"—which does not require $4\times$ expansion, and CoFrNet's rational function expressiveness suffices.
- **Continuant implementation is the engineering key**: Without it, continued fractions are impractical due to division cost; with it, deep ladders incur almost no overhead, making this function family a viable attention/FFN replacement.
- **Effective for small models**: On Llama-3.2B (already using GQA as an efficient attention baseline), CoFrGeNet is competitive or better on 8 open-domain QA and reasoning tasks (see Appendix Table 6), demonstrating robustness across architectures.
- **Pole handling + output clipping retained**: At test time, ladder outputs are clipped to the training interval to prevent pole explosion, which is necessary for engineering robustness.

## Highlights & Insights
- **"Changing function class" is a rare direction for transformer architecture innovation**: This work is among the first in recent years to systematically demonstrate that "non-polynomial + non-elementwise activation" function classes (continued fractions/rational functions) can be competitive in generative models, opening an empirical window for alternatives beyond attention/MLP.
- **Number theory properties directly translate to engineering efficiency**: Continuants, an 18th-century tool, are cleverly used here—since "the gradient is also a ratio of continuants," $d$ divisions are reduced to one. This approach of "using classical mathematical structures for hardware friendliness" is elegant and transferable to any scenario requiring deep rational approximation (ODE networks, neural rational diffusion).
- **"Plug-in replacement" is highly industry-friendly**: Since ladders only replace MHA and FFN, the entire training/inference pipeline (data, tokenizer, KV cache, LoRA, etc.) requires no modification, making integration cost nearly zero—especially suitable for large industry workflows.
- **Dyadic progressive training** is a general trick for rational/fractional networks—staging "linear backbone → higher-order correction" is a simple way to control rational function training instability, analogous to progressive growing in GANs.

## Limitations & Future Work
- **No open-source code**: Limits independent reproduction and follow-up; IBM's internal work seems focused on GPT2-xl and Llama-3.2B scales, lacking empirical results for 7B/13B models.
- **Does not touch modern attention optimization stacks**: CAttnM still requires $l \times l$ causal softmax, and cannot directly leverage FlashAttention/PagedAttention; engineering wall-clock advantages may be offset by these low-level optimizations.
- **Numerical stability boundaries unclear**: Although $\epsilon$ clamping avoids poles, the authors do not analyze whether the model "plays gradients near poles" in late training; more research is needed on stable intervals across domains/datasets.
- **Missing comparison with SOTA large models**: No direct comparison with "non-transformer" efficient routes like Mamba/RWKV/Linear Attention, and the baseline is the GPT2 series (already dated), so the relative position to Llama-3-8B, Mistral, etc., is unclear.
- **Transferability to other architectures**: The authors explicitly list "replacing Mamba hidden state functions with CoFrNet" as future work; similarly, replacing the MLP block in ViT/Diffusion Transformers is worth considering.

## Related Work & Insights
- **vs Synthesizer-D / Sparse Attention**: This work significantly outperforms these classic efficient attention methods with equal or fewer parameters, indicating that "changing function class" is more effective than "sparsifying" or "changing QK form."
- **vs Multi-Query / GQA**: MQA/GQA reduce parameters by sharing K/V heads; CoFrNet replaces the entire QKV with ladders; the two are orthogonal and can be combined (CoFrGeNet-F has been validated to coexist with Llama's GQA).
- **vs Linformer / Linear Attention**: Linear attention trades expressiveness for $O(n)$ complexity; CoFrNet keeps complexity unchanged but changes the function class, mainly addressing parameter and quality issues—different objectives.
- **vs CoFrNets (Puri et al. 2021)**: The original CoFrNet only demonstrated universal approximation in supervised learning; this work is its first successful application to generative modeling, solving three new challenges: multi-dimensional output, causality, and division efficiency—truly engineering an academic idea.
- **vs Mamba / RWKV**: The authors acknowledge no direct comparison with SSMs and list "redesigning SSM hidden state functions with CoFrNet" as future work, leaving ample room for exploration.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically introducing continued fractions into language generation is a rare "function class change" innovation in recent years; the engineering of continuants is also a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two backbones (GPT2-xl + Llama-3.2B), three pretraining sets (OWT + GneissWeb + docling), GLUE + 6 PPL datasets + 8 QA/reasoning tasks; but lacks 7B+ large models and wall-clock comparisons.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations (Proposition 1 + Appendix Lemma 2), clear architecture diagrams; but notation is dense ($y_1, U_1, F, A$ require repeated reference), posing a high barrier for new readers.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play solution for "reducing parameters by 1/3-1/2 without performance drop," with high industrial value; academically, opens a new direction for "non-polynomial function class LMs."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VecGlypher: Unified Vector Glyph Generation with Language Models](../../CVPR2026/image_generation/vecglypher_unified_vector_glyph_generation_with_language_models.md)
- [\[ICML 2026\] Esoteric Language Models: A Family of Any-Order Diffusion LLMs](esoteric_language_models_a_family_of_any-order_diffusion_llms.md)
- [\[ACL 2026\] Multimodal Large Language Models for Multi-Subject In-Context Image Generation](../../ACL2026/image_generation/multimodal_large_language_models_for_multi-subject_in-context_image_generation.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](../../AAAI2026/image_generation/unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[CVPR 2026\] Organizing Unstructured Image Collections using Natural Language](../../CVPR2026/image_generation/organizing_unstructured_image_collections_using_natural_language.md)

</div>

<!-- RELATED:END -->
