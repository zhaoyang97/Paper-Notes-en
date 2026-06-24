---
title: >-
  [Paper Note] CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter
description: >-
  [ACL 2025][speculative decoding] CORAL improves the feature consistency of the draft model in multi-step training via Cross-Step Representation Alignment (CSRA), and compresses the inference latency of the large-vocabulary LM head via a weight grouping mechanism, achieving a $2.50\text{--}4.07\times$ speedup on LLaMA3/Qwen2.5, outperforming EAGLE-2 and HASS.
tags:
  - "ACL 2025"
  - "speculative decoding"
  - "draft model"
  - "representation alignment"
  - "LM head compression"
  - "vocabulary"
date: 2026-05-08
content_hash: b51729d0d5b27980
---

# CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter

**Conference**: ACL 2025  
**arXiv**: [2502.16880](https://arxiv.org/abs/2502.16880)  
**Code**: None  
**Area**: Others  
**Keywords**: speculative decoding, draft model, representation alignment, LM head compression, vocabulary

## TL;DR
CORAL improves the feature consistency of the draft model in multi-step training via Cross-Step Representation Alignment (CSRA), and compresses the inference latency of the large-vocabulary LM head via a weight grouping mechanism, achieving a $2.50\text{--}4.07\times$ speedup on LLaMA3/Qwen2.5, outperforming EAGLE-2 and HASS.

## Background & Motivation

**Background**: Speculative decoding utilizes a lightweight draft model to pre-generate tokens, which are verified in parallel by a target model. EAGLE uses the hidden states of the target model to train the draft model, but suffers from training-inference misalignment (training uses target states, whereas inference uses its own states). HASS introduces multi-step training to let the draft model adapt to its own features, but the significant input discrepancy across multiple steps makes training convergence difficult.

**Limitations of Prior Work**: (1) The input features at different steps during multi-step training vary significantly, making it difficult for the lightweight draft model to adapt; (2) potential gradient conflicts may occur between different steps; (3) modern LLMs employ increasingly larger vocabularies (LLaMA3 128K, Qwen2.5 152K), rendering the LM head a latency bottleneck for the draft model—for example, the LM head accounts for the majority of the total latency in the Qwen2.5-7B draft model.

**Key Challenge**: Multi-step training improves draft accuracy but introduces feature inconsistency, while large vocabularies enhance LLM capability but slow down the draft model.

**Goal**: Simultaneously address the feature consistency issue in multi-step training and the latency bottleneck of large vocabularies.

**Key Insight**: Enforcing feature consistency across multiple training steps using contrastive learning + selectively activating LM head parameters using routing groups.

**Core Idea**: Cross-step contrastive alignment to keep draft features consistent across different training steps + group-activation of LM head parameters to alleviate the latency of large vocabularies.

## Method

### Overall Architecture
Multi-step training: the draft model outputs features at each step $\rightarrow$ CSRA constrains features at the same position across different steps to remain consistent using contrastive learning $\rightarrow$ standard CE loss is simultaneously used for classification training. Inference: the draft model generates tokens $\rightarrow$ LM head router selectively activates a subset of parameters $\rightarrow$ target model verifies.

### Key Designs

1. **Cross-Step Representation Alignment (CSRA)**:

    - **Function**: Forces the draft model to output consistent features at the same position across different steps in multi-step training.
    - **Mechanism**: Employs contrastive learning—features at the same position across different steps serve as positive pairs, while features at different positions serve as negative pairs, minimizing the distance between positive pairs.
    - **Design Motivation**: In multi-step training, step 1 uses target states whereas steps 2/3 use self-generated states, leading to potentially large discrepancies in feature distributions. CSRA enforces consistency to stabilize the draft model.
    - **Difference from HASS**: HASS only performs multi-step training without constraining feature consistency.

2. **LM Head Weight Grouping**:

    - **Function**: Groups the weights of the large-vocabulary LM head based on token embedding similarity and selectively activates only the relevant groups during inference.
    - **Key Observation**: The LM head of large vocabularies ($128\text{K}\text{--}152\text{K}$ tokens) accounts for over $50\%$ of the draft model's latency. For instance, the parameter count of the LM head in Qwen2.5's draft model exceeds that of the transformer layers.
    - **Mechanism**: Uses a router to predict which weight group the current token belongs to $\rightarrow$ only activates the matrix multiplication of parameters in that group.
    - **Effectiveness**: Drastically reduces draft model latency, especially for large-vocabulary models.

## Key Experimental Results

### Main Results: Speedup Ratio (Temperature=0)

| Method | LLaMA3-8B | Qwen2.5-7B | LLaMA2-7B |
|------|----------|-----------|----------|
| Vanilla Decoding | $1.0\times$ | $1.0\times$ | $1.0\times$ |
| EAGLE-2 | $\sim 2.5\times$ | $\sim 2.3\times$ | $\sim 2.8\times$ |
| HASS | $\sim 2.7\times$ | $\sim 2.5\times$ | $\sim 3.0\times$ |
| **CORAL** | **$\sim 3.2\times$** | **$\sim 3.5\times$** | **$\sim 4.07\times$** |

### Ablation Study

| Configuration | Speedup | Description |
|------|------|------|
| Full CORAL | Best | CSRA + LM head grouping |
| w/o CSRA | Significant drop | Low accuracy due to feature inconsistency in multi-step training |
| w/o LM head grouping | Decline (latency) | Increased latency in large-vocabulary models |
| HASS + LM head grouping | Moderate | Indicates both contributions are independently effective |

### Key Findings
- **Large vocabulary is the new bottleneck for draft models**: The latency proportion of the LM head in the draft models of LLaMA3 (128K vocabulary) and Qwen2.5 (152K vocabulary) increases from $\sim 20\%$ in LLaMA2 (32K vocabulary) to over $\sim 50\%$.
- **CSRA significantly improves acceptance rate**: Cross-step alignment increases the average acceptance length $\tau$ of the draft model.
- **CORAL's advantages are more pronounced on large-vocabulary models**: Because the gains of LM head compression are more substantial on larger vocabularies.

## Highlights & Insights
- **Identifying the LM head as a new bottleneck for draft models**: As LLM vocabularies expand to 100K+, this overlooked issue becomes increasingly severe. Identifying and solving this problem is a significant contribution.
- **Application of contrastive learning in speculative decoding training**: Utilizing contrastive learning to constrain consistency in multi-step training is both natural and effective.
- **Exceptional practical acceleration performance**: Achieving a $2.50\text{--}4.07\times$ speedup is a highly competitive result in the speculative decoding domain.

## Limitations & Future Work
- LM head grouping requires an extra router, introducing a minor latency overhead.
- The grouping strategy is based on token embedding similarity, which might not be robust to OOD tokens.
- The hyperparameters of contrastive learning in CSRA need tuning.
- Not yet verified on MoE models.

## Related Work & Insights
- **vs EAGLE-2 (Li et al., 2024)**: EAGLE utilizes target states training + tree attention. CORAL improves multi-step training accuracy through CSRA and enhances speed via LM head compression, thoroughly outperforming it.
- **vs HASS (Zhang et al., 2024)**: HASS proposes multi-step training to tackle training-inference misalignment. CORAL builds on this by adding CSRA consistency constraints and LM head acceleration.
- **vs CLaSp**: CLaSp is a training-free layer-skipping strategy (limited speedup but plug-and-play). CORAL requires training the draft model but yields much higher speedups.

## Rating
- Novelty: ⭐⭐⭐⭐ Discovered the LM head bottleneck + CSRA consistency constraint; the two contributions are independent and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three LLM families across three benchmarks, complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Analysis-driven method design with clear logic.
- Value: ⭐⭐⭐⭐⭐ High practical deployment value with up to $4.07\times$ speedup.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?](are_any-to-any_models_more_consistent_across_modality_transfers_than_specialists.md)
- [\[ACL 2025\] LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning](legalreasoner_step-wised_verification-correction_for_legal_judgment_reasoning.md)
- [\[ACL 2025\] Consistent Client Simulation for Motivational Interviewing-based Counseling](consistent_client_simulation_for_motivational_interviewing-based_counseling.md)
- [\[ACL 2025\] Research Borderlands: Analysing Writing Across Research Cultures](research_borderlands_analysing_writing_across_research_cultures.md)
- [\[ACL 2025\] MEXMA: Token-level Objectives Improve Sentence Representations](mexma_token-level_objectives_improve_sentence_representations.md)

</div>

<!-- RELATED:END -->
