---
title: >-
  [Paper Note] Precise Information Control in Long-Form Text Generation
description: >-
  [NeurIPS 2025][Social Computing][precise information control] This paper proposes the Precise Information Control (PIC) task, which requires LLMs to generate long-form text that strictly adheres to a given set of claims…
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "precise information control"
  - "faithfulness hallucination"
  - "verifiable claims"
  - "preference learning"
  - "long-form generation"
date: 2026-05-08
content_hash: cb370869d23a7a24
---

# Precise Information Control in Long-Form Text Generation

**Conference**: NeurIPS 2025
**arXiv**: [2506.06589](https://arxiv.org/abs/2506.06589)
**Code**: None
**Area**: Social Computing
**Keywords**: precise information control, faithfulness hallucination, verifiable claims, preference learning, long-form generation

## TL;DR
This paper proposes the Precise Information Control (PIC) task, which requires LLMs to generate long-form text that strictly adheres to a given set of claims (neither omitting nor adding information). The authors construct PIC-Bench to evaluate 8 tasks, finding that over 70% of outputs from state-of-the-art models contain faithfulness hallucinations. Through weakly supervised preference data construction combined with DPO training, the proposed PIC-LM improves the F1 of an 8B model from 69.1% to 91.0%.

## Background & Motivation

**Background**: Hallucinations in LLM long-form generation are categorized as factual hallucinations (contradicting real-world knowledge) and faithfulness hallucinations (contradicting the provided input context). While substantial work addresses factual hallucinations, faithfulness hallucinations are equally critical — even when correct contextual information is provided, models may still add unsupported content or omit key information.

**Limitations of Prior Work**: (a) Faithfulness evaluation is typically binary ("faithful/unfaithful"), offering insufficient granularity; (b) no standardized task exists to quantify the precision of information control when LLMs are given explicit claims; (c) existing models still exhibit severe hallucination even when users explicitly specify what information should be included.

**Key Challenge**: Faithfulness hallucinations are theoretically eliminable (since the correct answer is present in the input), yet in practice the rate of perfect faithfulness among SOTA models does not exceed 30%.

**Goal**: (a) Formally define and evaluate information control precision in long-form generation; (b) train models capable of precisely controlling output information.

**Key Insight**: Using verifiable claims as the unit of granularity, faithfulness is decomposed into precision (not saying too much) and recall (not saying too little).

**Core Idea**: PIC = given a claim set $C$, every claim in the generated text is supported by $C$ (precision), and every claim in $C$ appears in the generation (recall).

## Method

### Overall Architecture
Two stages: (1) PIC-Bench evaluation — converting 8 long-form generation tasks into PIC format and evaluating model performance via claim extraction and verification; (2) PIC-LM training — SFT followed by weakly supervised preference data DPO post-training on Llama 3.1 8B Instruct.

### Key Designs

1. **PIC Task Formalization**

    - Function: Reframes long-form generation as a claim-level precise control problem.
    - Mechanism: Input = instruction $\mathcal{I}$ + claim set $C = \{c_1, \ldots, c_n\}$. The model generates response $\theta(\mathcal{I}, C)$, from which claims $C' = \{c'_1, \ldots, c'_m\}$ are extracted. **Full PIC**: $C' = C$ (neither omission nor addition), evaluated with F1. **Partial PIC**: $C' \subseteq C$ (a subset may be selected but nothing added), evaluated with precision.
    - Design Motivation: Full PIC suits scenarios requiring complete coverage (biographies, paraphrase), while Partial PIC suits scenarios allowing selective citation (summarization, RAG QA). Together, the two modes cover the core requirements of practical applications.

2. **Weakly Supervised Preference Data Construction**

    - Function: Automatically generates PIC-oriented preference pairs for DPO training without human annotation.
    - Mechanism: For each sample $(\mathcal{I}, C_{orig}, y_{orig})$, a subset of claims is randomly removed to obtain $C_{perturb} \subset C_{orig}$, and the SFT model generates $y_{perturb}$ conditioned on $C_{perturb}$. Two preference pairs are derived: when $C_{orig}$ is the context, $y_{orig}$ is preferred over $y_{perturb}$ (the correct response should contain complete information), and vice versa. Normalized log-probability difference serves as a proxy signal for instruction following to adaptively select the construction type: if $\sigma(\frac{\log p_\theta(y_{orig})}{L} - \frac{\log p_\theta(y_{perturb})}{L})$ exceeds threshold $\tau$, the first construction is chosen.
    - Design Motivation: Sampling the two constructions with equal probability may impair instruction-following ability (a response conditioned on too few claims may fail to adequately address the instruction). Adaptive selection achieves a balance between PIC performance and instruction following.

3. **Claim Extraction and Verification Pipeline**

    - Function: Automatically decomposes long-form outputs into verifiable claims and checks support relationships.
    - Mechanism: An LLM-based claim extractor decomposes outputs into independent verifiable claims at a granularity between sentence-level and atomic-level; a claim verifier checks semantic equivalence. Human consistency validation confirms the reliability of the pipeline.
    - Design Motivation: The choice of verifiable claim granularity is critical — sentence level may conflate supported and unsupported information, while atomic level lacks sufficient context for independent verification.

### Loss & Training
- Base model: Llama 3.1 8B Instruct
- SFT stage: Fine-tuned on high-quality PIC-formatted data (No Robots + FLAN + CNN + EntityBios + long-form QA)
- DPO stage: Length-normalized DPO on weakly supervised preference data

## Key Experimental Results

### Main Results (PIC-Bench Full Setting, F1)

| Model | Avg. F1 | Perfect F1 Rate | Hardest Task (PopBios-CF) |
|-------|---------|----------------|--------------------------|
| Llama 3.1 8B Inst. | 69.1 | 3.7% | 23.7 |
| Tulu 3 8B | 76.9 | 4.9% | 51.3 |
| Llama 3.3 70B Inst. | 78.9 | 11.3% | 72.5 |
| QwQ 32B | 84.5 | 18.0% | 67.5 |
| GPT-4o | 83.1 | 17.0% | 71.3 |
| Claude 3.5 Sonnet | 87.1 | 24.7% | 82.6 |
| **PIC-LM (8B, Ours)** | **91.0** | **43.9%** | **84.2** |

### Downstream Applications

| Task | Baseline (Llama 8B) | PIC-LM | Gain |
|------|---------------------|--------|------|
| ASQA (RAG QA) EM Recall | 52.5% | 61.5% | +17.1% |
| Birthplace Factual Precision | 65.9% | 86.0% | +30.5% |
| QAMParI F1@5 | 13.5% | 22.6% | +67.4% |

### Key Findings
- **Over 70% of SOTA model outputs contain faithfulness hallucinations**: Even Claude 3.5 Sonnet achieves a perfect F1 rate of only 24.7%, demonstrating that precise information control remains far from solved.
- **Counterfactual scenarios are the most challenging**: All models perform worst on PopBios-CF (where an entity is replaced with another well-known person), indicating that models tend to follow parametric memory when it conflicts with contextual information.
- **The 8B PIC-LM outperforms all open-source and closed-source models**: 91.0% vs. Claude 3.5 Sonnet's 87.1%, demonstrating the effectiveness of targeted post-training.
- **Faithfulness gains transfer to factual accuracy gains**: PIC-LM substantially improves accuracy in RAG and fact-checking pipelines, suggesting that precise context-following ability is critical infrastructure for reducing factual hallucinations.

## Highlights & Insights
- **Elegant simplicity of the PIC formulation**: The complex problem of "hallucination detection" is reduced to "precision/recall over a claim set," making the problem well-defined and quantifiable. The two settings (full/partial) cover the core application scenarios.
- **Ingenuity of weakly supervised preference data construction**: Randomly removing claims to create $C_{perturb}$ automatically yields preference pairs without human annotation, yet achieves significant improvements. Using log-probability difference as a proxy signal for instruction following is also an elegant design choice.
- **Transfer effect from faithfulness to factuality**: The paper demonstrates that precise context-following not only reduces faithfulness hallucinations but also indirectly improves factual accuracy — providing both theoretical and empirical support for "retrieve/verify first, then faithfully generate" pipelines.

## Limitations & Future Work
- Claim extraction and verification rely on LLMs, which may themselves introduce errors.
- Perfect PIC (zero hallucination) remains at only 43.9% even for PIC-LM, indicating the problem is far from fully resolved.
- High precision in Partial PIC may come at the cost of usefulness (an overly conservative generation style).
- Training data includes in-domain tasks; out-of-distribution generalization remains an avenue for improvement.

## Related Work & Insights
- **vs. FActScore**: FActScore evaluates factuality of generated facts against Wikipedia, while PIC evaluates faithfulness of outputs against inputs — the two approaches are complementary.
- **vs. Conformal Importance Summarization**: That work provides recall guarantees for retaining important sentences; PIC simultaneously requires both precision and recall at the claim level, imposing stricter constraints.
- **vs. RLHF for hallucination**: General RLHF learns preferences via a reward model, whereas PIC-LM uses automatically constructed PIC-specific preference data, yielding more targeted supervision.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both the PIC task formalization and the weakly supervised preference construction represent significant contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 tasks, 13+ models, downstream application validation, extensive ablation studies
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is precise, experiments are comprehensive, and the paper is clearly structured
- Value: ⭐⭐⭐⭐⭐ Provides an actionable solution to the faithfulness problem in long-form generation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web](averimatec_a_dataset_for_automatic_verification_of_image-text_claims_with_eviden.md)
- [\[NeurIPS 2025\] Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)
- [\[NeurIPS 2025\] Evaluating Multiple Models Using Labeled and Unlabeled Data](evaluating_multiple_models_using_labeled_and_unlabeled_data.md)
- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)
- [\[NeurIPS 2025\] IF-GUIDE: Influence Function-Guided Detoxification of LLMs](if-guide_influence_function-guided_detoxification_of_llms.md)

</div>

<!-- RELATED:END -->
