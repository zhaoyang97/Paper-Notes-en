---
title: >-
  [Paper Note] ARC-JSD: Attributing Response to Context via Jensen-Shannon Divergence Driven Mechanistic Study
description: >-
  [NeurIPS 2025][Context Attribution] ARC-JSD proposes a RAG context attribution method based on Jensen-Shannon Divergence — by comparing the JSD between model output distributions with and without specific context sentences, it localizes the context that a response depends on without fine-tuning or gradient computation. The method achieves 3× faster computation than baselines, improves Top-1 attribution accuracy by 10.7% on average, and reveals via Logit Lens that attribution-relevant attention heads are concentrated in higher layers.
tags:
  - NeurIPS 2025
  - Context Attribution
  - Jensen-Shannon Divergence
  - Mechanistic Interpretability
  - RAG Hallucination
  - Logit Lens
date: 2026-05-08
content_hash: 2e8236759afb32b0
---

# ARC-JSD: Attributing Response to Context via Jensen-Shannon Divergence Driven Mechanistic Study

**Conference**: NeurIPS 2025
**arXiv**: [2505.16415](https://arxiv.org/abs/2505.16415)
**Code**: [https://github.com/ruizheliUOA/ARC_JSD](https://github.com/ruizheliUOA/ARC_JSD)
**Area**: RAG / Interpretability
**Keywords**: Context Attribution, Jensen-Shannon Divergence, Mechanistic Interpretability, RAG Hallucination, Logit Lens

## TL;DR
ARC-JSD proposes a RAG context attribution method based on Jensen-Shannon Divergence — by comparing the JSD between model output distributions with and without specific context sentences, it localizes the context that a response depends on without fine-tuning or gradient computation. The method achieves 3× faster computation than baselines, improves Top-1 attribution accuracy by 10.7% on average, and reveals via Logit Lens that attribution-relevant attention heads are concentrated in higher layers.

## Background & Motivation

**Background**: RAG systems inject retrieved context into LLMs to reduce hallucinations, but reliable attribution methods for identifying which part of the context a generated response depends on remain lacking.

**Limitations of Prior Work**: ContextCite requires multiple perturbation samples; ALTI-Logit requires layer-wise gradient computation; MIRAGE requires a surrogate model — all are computationally expensive. More critically, mechanistic understanding of how RAG models internally utilize context is absent.

**Key Challenge**: Attribution requires evaluating the influence of each context sentence, yet per-sentence full re-inference is prohibitively slow; a training-free, gradient-free, and efficient attribution method is needed.

**Goal**: (a) Provide a computationally efficient and accurate context attribution method; (b) reveal which internal components (attention heads / MLP layers) are responsible for context attribution.

**Key Insight**: JSD is a bounded, symmetric distribution distance measure computable directly from logits. The contribution of each sentence is quantified by comparing output distributions between the full context and the context with that sentence removed.

**Core Idea**: Ablate context sentence-by-sentence → compute JSD change in output distributions → the sentence with the highest JSD is the most relied-upon context + apply Logit Lens to reveal the attribution mechanism.

## Method

### Overall Architecture
Input: query $Q$ + context $C = \{c_1, ..., c_n\}$ + model response $R$ → for each $c_i$, construct the ablated context $C_{\text{ABLATE}}(c_i) = C \setminus \{c_i\}$ → compute $\text{JSD}(c_i) = \sum_{j=1}^{|R|} \text{JSD}(P_{LM}(r_j|C,Q) \| P_{LM}(r_j|C_{\text{ABLATE}}(c_i),Q))$ → the $c_i$ with the highest JSD is the attribution result.

### Key Designs

1. **JSD Attribution Computation**:

    - Function: Quantify the contribution of each context sentence to the response.
    - Mechanism: For each generated token $r_j$, compare output probability distributions under the full context and the ablated context. JSD is bounded in $[0, \log 2]$, symmetric, and directly computable from softmax logits. The cumulative JSD across all tokens serves as the contribution score for a given sentence.
    - Design Motivation: Compared to KL divergence (unbounded, asymmetric), total variation distance (too coarse), and Wasserstein distance ($O(V^3)$ complexity), JSD offers the most balanced trade-off.

2. **Logit Lens Mechanistic Analysis**:

    - Function: Localize internal components responsible for context attribution.
    - Mechanism: JSD is computed separately for each attention head and MLP layer (measuring change after ablating the highest-contributing sentence). $\text{JSD}_{\text{Attn}}^{\ell,h}$ and $\text{JSD}_{\text{MLP}}^{\ell}$ quantify attribution sensitivity per component. Cross-validation via semantic gain metric (Spearman correlation between cosine similarity change and JSD ranking).
    - Design Motivation: Attribution-relevant attention heads are found to concentrate in middle-to-high layers — consistent with the "retrieval → decision" processing pipeline.

3. **Hallucination Mitigation Application**:

    - Function: Use JSD scores as a confidence gate to reduce hallucinations.
    - Mechanism: When the highest JSD score falls below a threshold (indicating weak dependence on any context sentence), the response is flagged as high hallucination risk and withheld.
    - Design Motivation: Hallucination rate decreases from 13.4% to 8.2% (a 39% reduction) with negligible change in Factual F1 (76.1 → 75.9).

### Loss & Training
- Training-free method; all computation occurs at inference time.
- Computational complexity $O(2PT|C|^2)$, achieving 3× speedup over ContextCite-32 on long contexts.

## Key Experimental Results

### Main Results (Top-1 Attribution Accuracy)

| Method | TyDi QA | Hotpot QA | MuSiQue | Avg. Gain |
|--------|---------|-----------|---------|-----------|
| ContextCite | baseline | baseline | baseline | — |
| ALTI-Logit | — | — | — | — |
| **ARC-JSD** | — | — | — | **+10.7%** |

Evaluated on Qwen2-1.5B/7B-IT and Gemma2-2B/9B-IT.

### Hallucination Mitigation

| Setting | Hallucination Rate | Factual F1 |
|---------|--------------------|-----------|
| No Gate | 13.4% | 76.1% |
| JSD Gate | **8.2%** | 75.9% |
| Random Gate | 12.7% | 69.4% |

### Ablation Study

| Experiment | Finding |
|------------|---------|
| Ablate Top-10 JSD attention heads | Accuracy drops significantly (avg. JSD change 2.23±0.12) |
| Ablate random 10 heads | Minimal change (1.53±0.76) |
| JSD vs. semantic gain Spearman correlation | 0.67–0.79 (p<0.01), validating consistency between metrics |
| Extend to LLaMA-3.1-8B / Qwen3-80B | Advantage maintained |

### Key Findings
- Attribution-relevant attention heads concentrate in middle-to-high layers, not the bottom or top — consistent with the intuition of "retrieval first, decision second."
- MLP layers in middle layers also exhibit significant attribution contribution — indicating that context information propagates not only through attention.
- Token-level JSD analysis reveals that key entity tokens (e.g., named entities, numbers) exhibit the highest JSD — validating the method's soundness.
- JSD gating is far more effective than random gating (8.2% vs. 12.7% hallucination rate), confirming that JSD genuinely captures attribution quality.

## Highlights & Insights
- **JSD is an ideal choice**: bounded + symmetric + efficient + differentiable — among distribution comparison metrics, very few simultaneously satisfy all these properties.
- **Integration of mechanistic analysis and practical methodology**: the work yields both a deep understanding of the model's internal workings and directly applicable attribution and hallucination mitigation tools.
- **Token-level JSD visualization is highly intuitive**: it directly reveals "which tokens depend most on the context," offering practical value for debugging RAG systems.

## Limitations & Future Work
- Layer-level analysis does not reach neuron-level granularity — SAE probes could enable finer analysis.
- Applications beyond RAG (e.g., membership inference attacks) remain unexplored.
- Experiments are primarily conducted on small-to-medium models; scaling behavior on very large models is unknown.

## Related Work & Insights
- **vs. ContextCite**: Requires multiple perturbation samples with additional $O(|C|^2/32)$ overhead; ARC-JSD is more direct.
- **vs. ALTI-Logit**: Requires layer-wise gradients, slower by a factor of $|R| \cdot L / |C|$.
- **vs. Attention Weight Attribution**: Attention weights do not equate to causal contribution; JSD is more reliable.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of JSD attribution and Logit Lens mechanistic analysis is novel and natural.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 benchmarks + 4 models + mechanistic analysis + hallucination mitigation + causal ablation + token-level analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Method motivation is clear; metric selection is well-justified.
- Value: ⭐⭐⭐⭐⭐ Provides an efficient and practical tool for RAG interpretability and hallucination mitigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Better Estimation of the Kullback-Leibler Divergence Between Language Models](better_estimation_of_the_kullback--leibler_divergence_between_language_models.md)
- [\[ACL 2026\] Context-Value-Action Architecture for Value-Driven Large Language Model Agents](../../ACL2026/interpretability/context-value-action_architecture_for_value-driven_large_language_model_agents.md)
- [\[NeurIPS 2025\] ValuePilot: A Two-Phase Framework for Value-Driven Decision-Making](valuepilot_a_two-phase_framework_for_value-driven_decision-making.md)
- [\[NeurIPS 2025\] nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [\[NeurIPS 2025\] Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals](knowing_when_to_stop_efficient_context_processing_via_latent_sufficiency_signals.md)

</div>

<!-- RELATED:END -->
