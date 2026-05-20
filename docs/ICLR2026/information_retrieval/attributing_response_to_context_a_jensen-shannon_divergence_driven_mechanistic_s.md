---
title: >-
  [Paper Note] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation
description: >-
  [ICLR 2026][Information Retrieval & RAG][Context Attribution] This paper proposes ARC-JSD, a method that computes the Jensen-Shannon Divergence (JSD) between response distributions under full context and sentence-ablated…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "Context Attribution"
  - "Jensen-Shannon Divergence"
  - "RAG"
  - "Mechanistic Interpretability"
  - "Attention Heads"
  - "MLP Layers"
  - "Hallucination Mitigation"
date: 2026-05-08
content_hash: c44dc0791972be82
---

# Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation

**Conference**: ICLR 2026
**arXiv**: [2505.16415](https://arxiv.org/abs/2505.16415)  
**Code**: [https://github.com/ruizheliUOA/ARC_JSD](https://github.com/ruizheliUOA/ARC_JSD)  
**Area**: RAG / Interpretability / Mechanistic Analysis
**Keywords**: Context Attribution, Jensen-Shannon Divergence, RAG, Mechanistic Interpretability, Attention Heads, MLP Layers, Hallucination Mitigation

## TL;DR
This paper proposes ARC-JSD, a method that computes the Jensen-Shannon Divergence (JSD) between response distributions under full context and sentence-ablated context, enabling efficient and accurate RAG context attribution without fine-tuning, gradient computation, or surrogate models. Combined with Logit Lens for mechanistic analysis, ARC-JSD identifies the attention heads and MLP layers responsible for context attribution, and reduces hallucination rates by approximately 39% via a gating mechanism.

## Background & Motivation

**Background**: RAG improves LLM generation accuracy by incorporating external context, yet reliably attributing generated content to specific context passages (context attribution) remains an open challenge.

**Limitations of Prior Work**:
   - Manual annotation is prohibitively expensive (Zeng et al., 2021; Slobodkin et al., 2024)
   - Gradient-based methods (MIRAGE) require backpropagation, incurring high computational cost
   - ContextCite requires hundreds of forward passes to train a linear surrogate model
   - DPO-based fine-tuning approaches (SelfCite) require additional training

**Key Challenge**: Existing methods struggle to balance attribution accuracy and computational efficiency—they are either accurate but expensive, or fast but imprecise.

**Key Insight**: Leveraging the mathematical properties of JSD—symmetry, boundedness ($[0, \log 2]$), and scale-invariance—to directly measure the change in response distribution upon ablating individual context sentences, bypassing surrogate model training.

**Core Idea**: If removing a context sentence induces the largest shift in the model's output distribution (highest JSD), that sentence is most critical to the generated response.

## Method

### Overall Architecture
ARC-JSD consists of two modules: (1) JSD-based context attribution (identifying key sentences); and (2) JSD + Logit Lens mechanistic analysis (identifying key attention heads and MLP layers).

### Key Designs

1. **JSD-Driven Context Attribution (§4.1)**

    - Function: For each sentence $c_i$ in the context, compute the JSD between the response distribution under full context and that under the ablated context.
    - Core formula: $\text{JSD}(c_i) = \sum_{j=1}^{|\mathcal{R}|} \text{JSD}(\mathcal{P}_{\text{LM}}(r_j|\mathcal{C},\mathcal{Q}) \| \mathcal{P}_{\text{LM}}(r_j|\mathcal{C}_{\text{ABLATE}}(c_i),\mathcal{Q}))$
    - The sentence with the highest JSD is identified as the most relevant: $c_{\text{Top-1}} = \arg\max_{c_i} \text{JSD}(c_i)$
    - Design Motivation: Accumulating JSD over response tokens captures locally sensitive tokens (e.g., named entities) without being dominated by high-entropy tokens.

2. **JSD + Logit Lens Mechanistic Analysis (§5)**

    - Function: Extends JSD analysis from the model level down to individual attention heads and MLP layers.
    - Mechanism: For each attention head $(\ell,h)$ and each MLP layer $\ell$, intermediate representations are projected into vocabulary space via Logit Lens, and JSD is computed between full-context and ablated-context distributions.
    - Key Findings: Attention heads responsible for context attribution are concentrated primarily in **higher layers**, while MLP layers contribute most in the mid-to-upper layers, partially consistent with findings in Wu et al. (2025a) under the NIAH setting.

3. **Semantic Gain Validation (§6)**

    - Function: Validates the components identified by JSD from an independent angle by measuring the cosine similarity gain toward the correct answer.
    - Mechanism: $\Delta^{\ell,\text{Attn}}$ and $\Delta^{\ell,\text{MLP}}$ are defined to quantify the semantic gain of each attention and MLP layer. Spearman $\rho$ is computed between JSD rankings and semantic gain rankings; Table 3 shows significant positive correlation, providing mutual validation.

4. **JSD Gating for Hallucination Reduction (§7)**

    - Function: Uses JSD scores as a confidence gate to suppress high-JSD attention heads and MLP layers with negative semantic gain.
    - Gating formula: $\text{Mask} = 0.7 + 0.3 \times \text{sigmoid}(G)$; when $G < 0$, the mask approaches 0.7, reducing the contribution of the corresponding component.
    - Effect: On HotpotQA, Qwen2-7B-IT hallucination rate drops from 13.4% to 8.2% (↓39%), with negligible change in Factual F1 (76.1→75.9).

### Computational Efficiency
- ARC-JSD FLOPs: $2PT|\mathcal{C}|^2$ ($P$: parameter count, $T$: tokens per sentence, $|\mathcal{C}|$: number of sentences)
- ContextCite (256 calls) FLOPs: $2PT \times 256^2$; ARC-JSD is cheaper when $|\mathcal{C}| < 256$
- MIRAGE requires gradient computation with FLOPs of $4PT|\mathcal{C}|(2|\mathcal{C}|+1)$
- Achieves approximately 3× practical speedup

## Key Experimental Results

### Datasets & Models
- Three QA datasets: TyDi QA (440, single-hop), HotpotQA (1000, multi-hop), MuSiQue (1000, multi-hop, avg. 93.6 context sentences)
- Four instruction-tuned models: Qwen2-1.5B/7B-IT, Gemma2-2B/9B-IT
- Additional generalization evaluation: LLaMA-3.1-8B-IT, Qwen3-Next-80B-A3B-IT

### Main Results (Context Attribution Top-1 Accuracy)
- ARC-JSD consistently dominates all baselines on the compute-accuracy trade-off on MuSiQue (Fig. 2a)
- Average attribution accuracy improves by approximately **10.7%**
- ContextCite-32 is faster than ARC-JSD when $|\mathcal{C}|>32$, but consistently achieves lower attribution accuracy
- ARC-JSD lies on the Pareto optimal front, balancing accuracy and efficiency

### Metric Comparison Ablation (§8, Fig. 6)

| Metric | Relative Performance |
|--------|----------------------|
| JSD | **Best**; symmetric, bounded, scale-invariant |
| KL | Diverges when ablated distribution has zero probability; incomparable across layers |
| TV | Bounded but too coarse; cannot distinguish high-entropy tail shifts from key-token probability transfers |
| Wasserstein | Requires defining distances over a 152K vocabulary; $O(V^3)$ complexity |
| MMD | Requires kernel function and token distance definition |

### Mechanistic Analysis Validation
- Table 3: Spearman $\rho$ between JSD rankings and semantic gain rankings is significant ($p<0.05$ or $p<0.01$) across all datasets and models
- Table 5: JSD change when ablating top-10 JSD attention heads (2.23±0.12) is significantly larger than when ablating random 10 heads (1.53±0.76)

### Hallucination Reduction (Table 4)

| Setting | Hallucination Rate | Factual F1 |
|---------|-------------------|------------|
| Base RAG | 13.4% | 76.1 |
| Gate Top-5 Attn & MLP | **8.2%** | 75.9 |
| Gate Random 5 | 12.7% | 69.4 |

### Generalizability
- The compute-accuracy advantage is maintained on LLaMA-3.1-8B-IT and Qwen3-Next-80B-A3B-IT (MoE) (Fig. 7)

## Highlights & Insights
- **Simplicity and Efficiency**: The method is conceptually straightforward—sentence-level ablation + JSD comparison—requiring no auxiliary model training, and can be integrated into any RAG system in a plug-and-play manner.
- **Theoretically Grounded Choice of JSD**: Symmetry avoids directionality issues; boundedness enables meaningful cross-layer comparison; the ablation comparisons against KL/TV/Wasserstein are convincing.
- **Closed-Loop Mechanistic Analysis**: JSD localization → semantic gain validation → causal ablation verification → gating application constitutes a complete validation and application pipeline.
- **Visualization of MLP-Layer Semantic Evolution**: Logit Lens visualizations reveal how Qwen2 progressively transitions from Chinese tokens to English in upper layers (e.g., "一只→A", "翅膀→wings"), consistent with the language anchoring phenomenon.
- **Practical Value**: The gating mechanism reduces hallucination rates by 39% without any retraining.

## Limitations & Future Work
- **Quadratic Complexity in Context Length**: The $O(|\mathcal{C}|^2)$ complexity remains expensive for very long contexts (e.g., hundreds of sentences); the paper does not discuss how to scale the approach.
- **Only Top-1 Attribution is Evaluated**: Existing QA datasets provide only sentence-level gold labels; finer-grained attribution (phrase-level or clause-level) remains underexplored.
- **Limited Scale of Gating Experiments**: Hallucination reduction is validated on only 200 HotpotQA samples; large-scale and multi-dataset validation is absent.
- **No Direct Accuracy Comparison with Fine-Tuning Methods such as SelfCite**: The comparison is limited to the compute-accuracy trade-off plot.
- **Threshold Selection for "All JSD Scores Small"** (0.02 bits) lacks systematic analysis.

## Related Work & Insights
- **vs. ContextCite (Cohen-Wang et al., 2024)**: ContextCite requires hundreds of forward passes to train a surrogate model, and the linearity assumption may miss non-linear dependencies; ARC-JSD directly quantifies true distributional change via JSD.
- **vs. MIRAGE (Qi et al., 2024)**: Gradient-based attribution is computationally expensive, and aggregating token-level signals to the sentence level introduces information loss.
- **vs. Wu et al. (2025a)**: Their NIAH setting evaluates copy-paste behavior, whereas this paper targets the more realistic scenario of paraphrasing and information integration.
- **vs. Sun et al. (2025)**: Sun et al. focus on localizing sources of hallucination, while this paper localizes sources responsible for correct generation; the two are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying JSD to RAG context attribution is novel and well-motivated, with strong integration of theory and practice.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, four plus two models, multi-angle ablation and validation with consistent results.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear, mathematical derivations are coherent, and case studies are intuitive.
- Value: ⭐⭐⭐⭐ The plug-and-play attribution method combined with mechanistic insights meaningfully advances RAG transparency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](../../ACL2026/information_retrieval/context_attribution_with_multi-armed_bandit_optimization.md)
- [\[ICLR 2026\] Embedding-Based Context-Aware Reranker](embedding-based_context-aware_reranker.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](q_rag_long_context_multi_step_retrieval.md)
- [\[ICLR 2026\] Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation](bayesian_attention_mechanism_a_probabilistic_framework_for_positional_encoding_a.md)

</div>

<!-- RELATED:END -->
