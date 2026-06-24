---
title: >-
  [Paper Note] UMoE: Unifying Attention and FFN with Shared Experts
description: >-
  [NeurIPS 2025 Spotlight][LLM Efficiency][Unified MoE Architecture] By reformulating the multi-head attention mechanism, this work reveals that attention shares the same "two-layer matrix multiplication" structure as FFN layers. Based on this insight, UMoE is proposed as a unified architecture that employs identically designed experts for both attention and FFN layers with parameter sharing, outperforming existing FFN-MoE and Attention-MoE baselines on both Base (134M) and Lar…
tags:
  - "NeurIPS 2025 Spotlight"
  - "LLM Efficiency"
  - "Unified MoE Architecture"
  - "Pre-Mixing Attention"
  - "Expert Sharing"
  - "Attention-FFN Fusion"
  - "Parameter Efficiency"
date: 2026-05-08
content_hash: 2c8c0246270cc181
---

# UMoE: Unifying Attention and FFN with Shared Experts

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.07260](https://arxiv.org/abs/2505.07260)  
**Code**: [github.com/ysngki/UMoE](https://github.com/ysngki/UMoE)  
**Area**: LLM Efficiency
**Keywords**: Unified MoE Architecture, Pre-Mixing Attention, Expert Sharing, Attention-FFN Fusion, Parameter Efficiency

## TL;DR

By reformulating the multi-head attention mechanism, this work reveals that attention shares the same "two-layer matrix multiplication" structure as FFN layers. Based on this insight, UMoE is proposed as a unified architecture that employs identically designed experts for both attention and FFN layers with parameter sharing, outperforming existing FFN-MoE and Attention-MoE baselines on both Base (134M) and Large (1.1B) models.

## Background & Motivation

**Background**: Sparse MoE is the dominant paradigm for scaling LLM capacity. Two main lines of research exist: FFN-MoE (Switch Transformer, DeepSeek-MoE), which replaces FFN layers with MoE, and Attention-MoE (MoA, SwitchHead), which replaces attention layers with MoE. These two lines have developed independently, employing different expert designs.

**Limitations of Prior Work**: Attention-MoE consistently underperforms FFN-MoE — under matched parameter and compute budgets, both MoA and SwitchHead fall short of fine-grained FFN-MoE. The performance gap stems from two factors: (a) the structural difference between attention and FFN layers leads to divergent expert designs; (b) introducing sparsity into attention-MoE requires sacrificing the expressiveness of standard attention (e.g., MoA must share K/V projections, limiting per-head independence).

**Key Challenge**: Attention appears to involve multiple projections and softmax nonlinearity, making it structurally distinct from the "two-layer matrix multiplication" of FFN. The question is whether an equivalent reformulation exists that unifies the two.

**Goal**: (a) Can attention be restructured to expose an FFN-like internal structure? (b) Can the same expert design serve both attention and FFN layers, enabling parameter sharing?

**Key Insight**: After decomposing $W_o$ per head in multi-head attention, the order of matrix multiplication is rearranged — token mixing (weighted aggregation) is performed first, followed by the $W_v W_o$ projection. This transforms the "projection component" of attention into the same two-layer structure as FFN.

**Core Idea**: Attention = token mixing + FFN-like expert processing. FFN = self-attention (identity attention matrix) + FFN-like expert processing. The two differ only in the token mixing operation, and experts can therefore be shared.

## Method

### Overall Architecture

UMoE abstracts each Transformer layer into three fundamental components: **experts** (two-layer FFN), **token mixing** (weighted summation), and **routers** (top-k selection). Each layer contains one attention MoE module and one FFN MoE module, both sharing the same set of expert parameters. The sole distinction between attention and FFN is that the attention module performs weighted token aggregation before feeding into the experts, while the FFN module feeds tokens directly into the experts.

### Key Designs

1. **Pre-Mixing Attention**

    - **Function**: Equivalently rewrites standard multi-head attention as $y = \sum_{i=1}^{h}(a_i X)(W_v^i W_o^i)$
    - **Mechanism**: The conventional formulation computes $o_i = a_i X W_v^i$ (project, then aggregate, then output projection). By exploiting the associativity of matrix multiplication, the order is rearranged to "aggregate first, then project": $(a_i X)$ produces a contextualized representation, after which the two-layer projection $(W_v^i W_o^i)$ is equivalent to an activation-free FFN.
    - **Design Motivation**: This reformulation exposes the hidden FFN structure within attention. Once a nonlinear activation is inserted between the two projection layers, the result is a standard FFN expert, naturally unifying with the expert design of FFN-MoE.
    - **Distinction from the Conventional Form**: Mathematically equivalent, but offering a fundamentally new perspective — each attention head is an expert that applies FFN processing to contextualized inputs.

2. **Unified MoE Architecture**

    - **Function**: Unifies experts across attention and FFN layers as identically structured two-layer MLPs (with intermediate dimension $d_v$), with top-k routers for expert selection.
    - **Mechanism**: The attention MoE output is $y = \sum_{i \in \mathcal{T}} p_i E_i(a_i X)$, and the FFN MoE output is $y = \sum_{i \in \mathcal{T}} p_i E_i(x)$. Both use the identical experts $E_i$; the only difference lies in the input — attention uses the contextualized input $a_i X$, while FFN uses the raw token $x$.
    - **Design Motivation**: FFN-MoE can be viewed as a special case of attention MoE where the attention matrix degenerates to the identity, with each token attending only to itself. Unification allows expert parameters to be shared across both modules, improving parameter efficiency.

3. **Low-Rank Expert Query Projection**

    - **Function**: Generates an independent query vector for each expert.
    - **Mechanism**: $q_i = x W_q + x W_a^i W_b^i$, where the first term is shared across experts and the second term provides expert-specific queries via low-rank matrices $W_a^i \in \mathbb{R}^{d \times r}$ and $W_b^i \in \mathbb{R}^{r \times d_k}$.
    - **Design Motivation**: Each expert requires a distinct attention pattern, but full-rank per-expert query projections are parameter-prohibitive. Low-rank decomposition maintains expert specialization while controlling parameter count, keeping total parameters comparable to existing MoE models.

4. **Parameter Sharing Strategy**

    - **Function**: Attention and FFN modules share the same set of fixed experts while retaining independent routers.
    - **Mechanism**: Experiments show that sharing fixed experts with independent routers is the optimal configuration (PPL 22.82 vs. 23.02 for no sharing).
    - **Design Motivation**: Shared experts allow attention layers to benefit from MoE scaling without increasing total parameters; independent routers enable each module to select different expert subsets according to its own requirements.

### Loss & Training

- Language modeling cross-entropy loss combined with the load-balancing auxiliary loss from Switch Transformer.
- Decoder-only Transformer with RoPE; experts implemented as two-layer MLPs with nonlinear activations.
- Datasets: FineWeb-Edu 100B (primary) and Wikitext-103 (comparison); LLaMA tokenizer (32K vocabulary).
- Base model: 12 layers / 768 hidden dim / 134M parameters; Large model: 24 layers / 2048 hidden dim / 1.1B parameters.

## Key Experimental Results

### Main Results: PPL Comparison (FineWeb-Edu 50B tokens)

| Model | Total Params | FineWeb PPL↓ | Wikitext PPL↓ | MACs |
|-------|-------------|-------------|--------------|------|
| Dense (Base) | 134M | 25.79 | 30.41 | 525G |
| FFN-MoE | 535M | 21.19 | 27.94 | 530G |
| MoA | 525M | 22.28 | 27.57 | 486G |
| SwitchHead | 533M | 22.91 | 29.47 | 542G |
| UMoE-Att | 547M | 20.81 | 27.45 | 611G |
| **UMoE** | **540M** | **20.44** | **26.67** | 616G |

On the Large model (1.1B dense → 3.6B MoE): UMoE achieves PPL 15.95 vs. FFN-MoE 16.09 vs. MoA 16.72, again achieving the best performance.

### Ablation Study

| Experiment | Key Finding |
|-----------|------------|
| Parameter sharing strategy | Shared fixed experts + independent routers is optimal (22.82); no sharing yields 23.02 |
| Expert allocation (20 total) | All to attention: PPL 21.75 > 16:4 split: 22.50 > 4:16 split: 22.82 |
| Activation function | Removing nonlinearity degrades PPL by 1.2–1.6, but does not collapse (token mixing preserves nonlinearity) |
| Pre-mixing vs. Post-mixing | Pre-mixing significantly outperforms post-mixing (contextualized inputs facilitate more accurate retrieval) |

### Key Findings

- **Attention experts are more valuable than FFN experts**: Assigning all experts to the attention layer achieves the best PPL (21.75), indicating that attention layers have greater expressive power and that FFN is indeed a special case of attention.
- **Negligible computational overhead**: Approximately 1.17× compute on the Base model and only 1.03× on the Large model — because expert computation scales quadratically with dimension while token mixing scales only linearly.
- **Experts exhibit dual specialization**: Shared experts develop distinct specialization patterns in attention and FFN layers (e.g., Expert 64 handles punctuation in attention and degree adverbs in FFN), demonstrating that shared parameters can efficiently support multiple functions.
- **Consistently leading zero-shot performance**: Base model UMoE 40.06% vs. FFN-MoE 39.55%; Large model UMoE 47.58% vs. FFN-MoE 47.12%.

## Highlights & Insights

- **Theoretical elegance of the unified perspective**: Equivalently rewriting multi-head attention as "token mixing + two-layer FFN" reveals the essential unity of attention and FFN. This is not merely an engineering simplification but a deeper understanding of the internal mechanisms of the Transformer.
- **FFN as a degenerate form of attention**: This insight is compelling — when the attention matrix is the identity, the attention layer degenerates into an FFN layer. Ablation experiments corroborate this, confirming that attention experts are more valuable than FFN experts.
- **KV-cache friendly**: Pre-mixing attention requires caching only a single key + hidden state pair per token (rather than separate K/V pairs per head), making it naturally well-suited for memory-efficient inference.

## Limitations & Future Work

- Pre-mixing attention is incompatible with GQA (which already has a single K/V pair), though further compression via MLA (Multi-head Latent Attention) remains a potential direction.
- Token mixing introduces approximately 1.17× computational overhead on small models; although this amortizes to 1.03× on large models, it warrants consideration in resource-sensitive settings.
- Experts are implemented as two-layer MLPs without gating, and stronger variants such as SwiGLU are not explored; the authors acknowledge that adopting SwiGLU may yield further improvements.
- The largest experimental scale is 1.1B dense / 3.8B MoE; validation at the 7B+ scale is absent.
- More efficient alternatives to token mixing (e.g., linear attention) remain unexplored, which the paper explicitly identifies as a future direction.

## Related Work & Insights

- **vs. MoA**: MoA treats entire attention heads as experts, achieving sparsification via shared K/V and independent Q/O projections. This constrains attention expressiveness, as all heads must attend over the same K/V. UMoE performs token mixing directly on hidden states, with each expert having its own low-rank query, preserving greater flexibility. On the Large model: UMoE PPL 15.95 vs. MoA 16.72.
- **vs. SwitchHead**: SwitchHead treats individual projection matrices within attention (Q/K/V/O) as experts, requiring separate MoE construction for each. UMoE merges $W_v W_o$ and directly treats it as an FFN expert, yielding a simpler design with superior performance.
- **vs. DeepSeek-MoE**: UMoE's fine-grained expert design is inspired by DeepSeek-MoE but extends it from FFN layers to attention layers, additionally enabling cross-module expert sharing.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The unified attention-FFN perspective is highly insightful; the finding that "FFN is a degenerate form of attention" is compelling and well-supported.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-scale comparisons, ablations, expert analysis, and zero-shot evaluations are all provided, though validation at the 7B+ scale is missing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The argumentation chain from reformulation to unification is exceptionally clear, with intuitive pseudocode and illustrations.
- **Value**: ⭐⭐⭐⭐ Significant implications for MoE architecture design — separate design of Attention-MoE and FFN-MoE may no longer be necessary in future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](../../ICML2026/llm_efficiency/theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)
- [\[NeurIPS 2025\] Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads](let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e.md)
- [\[NeurIPS 2025\] On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)
- [\[NeurIPS 2025\] Tensor Product Attention Is All You Need](tensor_product_attention_is_all_you_need.md)
- [\[ICLR 2026\] TyphoonMLA: A Mixed Naive-Absorb MLA Kernel For Shared Prefix](../../ICLR2026/llm_efficiency/typhoonmla_a_mixed_naive-absorb_mla_kernel_for_shared_prefix.md)

</div>

<!-- RELATED:END -->
