---
title: >-
  [Paper Note] Copy-Paste to Mitigate Large Language Model Hallucinations
description: >-
  [ICLR 2026][Hallucination Detection][Hallucination Mitigation] This paper proposes a Copy-Paste generation paradigm that trains LLMs to preferentially copy spans directly from retrieved context rather than paraphrasing t…
tags:
  - "ICLR 2026"
  - "Hallucination Detection"
  - "Hallucination Mitigation"
  - "RAG"
  - "Copy-Paste"
  - "DPO"
  - "Faithfulness"
date: 2026-05-08
content_hash: 858e00665a1d15f5
---

# Copy-Paste to Mitigate Large Language Model Hallucinations

**Conference**: ICLR 2026
**arXiv**: [2510.00508](https://arxiv.org/abs/2510.00508)  
**Code**: [https://github.com/longyongchao/CopyPasteLLM](https://github.com/longyongchao/CopyPasteLLM)  
**Area**: Hallucination Detection
**Keywords**: Hallucination Mitigation, RAG, Copy-Paste, DPO, Faithfulness

## TL;DR
This paper proposes a Copy-Paste generation paradigm that trains LLMs to preferentially copy spans directly from retrieved context rather than paraphrasing them freely. Combined with high-copy-preference DPO training, the approach improves faithfulness on counterfactual RAG benchmarks from 80.2% to 92.8%.

## Background & Motivation

**Background**: RAG (Retrieval-Augmented Generation) reduces hallucinations by supplying LLMs with external context; however, LLMs tend to paraphrase rather than directly cite the context during generation, leading to information distortion and hallucinations.

**Limitations of Prior Work**: The paraphrasing process introduces two types of hallucinations—"Twist" (distorting facts present in the context) and "Causal" (upstream errors in the causal chain propagating downstream). Attribution methods mark sources but do not alter the generation process itself.

**Key Challenge**: There is a fundamental trade-off between fluent paraphrasing and faithful copying—while paraphrasing reads more naturally, each reformulation introduces a risk of hallucination.

**Goal**: Can LLMs be trained to copy context spans as directly as possible while maintaining readability?

**Key Insight**: The analysis is grounded in an attention-anchoring perspective—when the previously generated token is copied from the context, the query vector of the next token is strongly correlated with the context key vectors, naturally promoting continued copying.

**Core Idea**: Train LLMs to develop a "high-copy preference" by using DPO to steer the model toward responses that directly embed context spans.

## Method

### Overall Architecture
The approach consists of two stages: (1) Copy-Paste Prompting to generate high-copy-rate candidate responses via three strategies (CP-Order / CP-Link / CP-Refine), and (2) multi-dimensional filtering + Elo ranking + DPO training.

### Key Designs

1. **Three Copy Strategies**:

    - CP-Order: Strict extraction—reordering relevant sentences from the context.
    - CP-Link: Allows transition phrases of no more than 15 words to connect copied spans.
    - CP-Refine: An iterative writer–reviewer loop of up to 5 rounds to improve readability while maintaining a high copy rate.

2. **Quantitative Metrics**:

    - Copy Coverage $\kappa$ = proportion of response tokens originating from the context.
    - Copy Density $\delta$ = emphasizes long contiguous spans (weighted by the square of span length).

3. **DPO Training**:

    - Requires only 365 high-quality preference pairs.
    - Multi-dimensional filtering: AlignScore / MiniCheck (faithfulness), $\kappa$ / $\delta$ (copy intensity), query relevance, and fluency.
    - Answer Stamping: the correct answer is appended at the end of the response to prevent answer omission caused by excessive copying.

## Key Experimental Results

### Main Results

| Dataset | Model | Method | Accuracy |
|--------|------|------|--------|
| FaithEval (Counterfactual) | Llama-3-8B | Context-DPO | 80.2% |
| FaithEval (Counterfactual) | Llama-3-8B | **CopyPasteLLM** | **92.8%** |
| ConFiQA-MC | Llama-3-8B | Attributed | 37.3% |
| ConFiQA-MC | Llama-3-8B | **CopyPasteLLM** | **82.5%** |

### Ablation Study

| Variant | FaithEval | Note |
|------|----------|------|
| w/o Copy Preference | 71.2% | No high-copy training data |
| w/o Answer Stamping | 45.1% | Excessive copying causes answer omission |
| **CopyPasteLLM** | **92.8%** | Full method |

### Key Findings
- Answer Stamping is critical—its removal causes accuracy to drop sharply from 92.8% to 45.1%.
- Effective training requires only 365 preference pairs, demonstrating extremely high data efficiency.
- Copy Density is a better predictor of faithfulness than Coverage; long contiguous spans are more reliable than short fragments.

## Highlights & Insights
- **Attention-Anchoring Theory**: Copy operations enjoy a natural advantage at the attention mechanism level—when the previous token is copied from context, key-value vectors naturally guide continued copying, creating a "copy momentum."
- **High-Efficiency Training with Minimal Data**: DPO on just 365 samples is sufficient to substantially shift generation style, suggesting that "copy vs. paraphrase" is primarily a preference issue rather than a capability issue.
- **Necessity of Answer Stamping**: Explicitly prompting the model to state the answer at the end balances copy faithfulness with response completeness.

## Limitations & Future Work
- High copy rates may reduce the naturalness and readability of responses.
- Validation is limited to English RAG tasks; cross-lingual generalization remains unknown.
- The Copy-Paste strategy may be ill-suited for questions requiring inferential synthesis rather than direct span retrieval.

## Related Work & Insights
- **vs. Context-DPO**: Both employ DPO, but Context-DPO does not explicitly optimize for copy preference, whereas this work directly optimizes the copying behavior.
- **vs. Attributed LLM**: Attribution methods only annotate source references without modifying the generation process; this work intervenes at the level of the generation paradigm itself.

## Rating
- Novelty: ⭐⭐⭐⭐ The "copy-over-paraphrase" paradigm is novel and counterintuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-model validation with clear ablations.
- Writing Quality: ⭐⭐⭐⭐ The attention-anchoring analysis is insightful.
- Value: ⭐⭐⭐⭐⭐ High practical value; directly applicable to RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Revis: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](../../ICML2026/hallucination/revis_sparse_latent_steering_to_mitigate_object_hallucination_in_large_vision-la.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[CVPR 2026\] Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](../../CVPR2026/hallucination/tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)
- [\[NeurIPS 2025\] Teaming LLMs to Detect and Mitigate Hallucinations](../../NeurIPS2025/hallucination/teaming_llms_to_detect_and_mitigate_hallucinations.md)
- [\[ICLR 2026\] Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)

</div>

<!-- RELATED:END -->
