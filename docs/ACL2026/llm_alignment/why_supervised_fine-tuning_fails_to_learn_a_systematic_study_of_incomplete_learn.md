---
title: >-
  [Paper Note] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models
description: >-
  [ACL 2026][LLM Alignment][Incomplete Learning] This paper presents the first systematic study of the "Incomplete Learning Phenomenon" (ILP) in SFT—where models fail to correctly reproduce parts of the training data even…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "Incomplete Learning"
  - "SFT Diagnosis"
  - "Knowledge Conflict"
  - "Forgetting"
  - "Fine-tuning Failure Modes"
date: 2026-05-08
content_hash: 6c84130b18a871e7
---

# Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.10079](https://arxiv.org/abs/2604.10079)  
**Code**: None  
**Area**: LLM Safety  
**Keywords**: Incomplete Learning, SFT Diagnosis, Knowledge Conflict, Forgetting, Fine-tuning Failure Modes

## TL;DR

This paper presents the first systematic study of the "Incomplete Learning Phenomenon" (ILP) in SFT—where models fail to correctly reproduce parts of the training data even after convergence. It identifies five recurring causes (missing knowledge, knowledge conflict, internal data contradiction, left-side forgetting, and insufficient optimization), and proposes a diagnostic framework along with targeted mitigation strategies.

## Background & Motivation

**Background**: SFT is the standard method for adapting LLMs to downstream tasks and is widely regarded as a reliable and efficient specialization mechanism.

**Limitations of Prior Work**: (1) Even when training loss fully converges, models frequently fail to correctly answer certain training samples—this is not an issue of overfitting or generalization, but a failure on the training set itself; (2) Unlearned samples are often non-random, corresponding to rare cases, compositional patterns, or knowledge-intensive instances; (3) Improvements in aggregate metrics can mask the persistent existence of unlearned subsets.

**Key Challenge**: While SFT datasets (especially in professional domains like law and medicine) are costly to construct, 15.3%±2.1% of samples remain unlearned after training—directly reducing data utilization efficiency.

**Goal**: Rather than proposing a new fine-tuning algorithm, this study systematically characterizes, diagnoses, and verifies the sources of incomplete learning in SFT.

**Key Insight**: Unlearned samples are treated as diagnostic signals rather than noise—understanding why these specific samples are not learned clarifies the limitations of SFT.

**Core Idea**: The five sources of ILP each require different mitigation strategies; there is no "one-size-fits-all" solution, necessitating fine-grained sample-level diagnosis.

## Method

### Overall Architecture

A three-stage framework: (1) SFT fine-tuning until convergence; (2) Converting training data into multiple-choice formats to detect unlearned samples using pass@N and BoN sampling; (3) Attributing unlearned causes through five diagnostic tests and applying targeted interventions.

### Key Designs

1.  **Unlearned Sample Detection (BoN-5 Sampling)**:
    - **Function**: Reliably identify samples that consistently fail after training.
    - **Mechanism**: SFT samples are converted into multiple-choice questions for $N$ independent inference runs to calculate the pass@N ratio. Samples with pass@5 < 0.2 and stability across seeds are marked as unlearned. The top-K=1000 most severe cases are selected for in-depth analysis.
    - **Design Motivation**: Distinguish random decoding noise from genuine learning failure—if multiple samples fail, it indicates the model has not internalized the knowledge.

2.  **Diagnosis and Intervention of Five ILP Sources**:
    - **Function**: Attribute unlearned samples to specific causes and verify causal relationships.
    - **Mechanism**: 
        - **Pre-training Knowledge Missing**: Use OpenIE to extract factual triples and BoN to probe the base model → Knowledge augmentation + Continued Pre-training (CPT).
        - **Knowledge Conflict**: Detect high-confidence incorrect answers from the base model (contradicting SFT labels) → External knowledge correction + CPT.
        - **SFT Internal Data Contradiction**: Label inconsistency among semantically similar samples → GPT evaluation + bucketed training to avoid mini-batch conflicts.
        - **Left-side Forgetting**: Early samples in sequential processing are overwritten by later ones → Random shuffling + dynamic resampling.
        - **Insufficient Optimization**: Inadequate training signals for rare or complex patterns → Increased training or re-weighting.
    - **Design Motivation**: Mitigation strategies serve as causal interventions rather than universal solutions—if strategy X works, it validates cause Y.

3.  **Knowledge State Probing (JSD Diagnosis)**:
    - **Function**: Distinguish between missing knowledge and knowledge conflict.
    - **Mechanism**: Calculate the Jensen-Shannon Divergence of the predictive distributions between the base and fine-tuned models. High JSD + base model error = Knowledge Conflict; Low JSD + fine-tuned model error = Missing Knowledge.
    - **Design Motivation**: Final accuracy alone cannot distinguish between "not knowing" and "knowing but being wrong"—distribution-level signals are required.

### Loss & Training

Standard SFT cross-entropy loss. Evaluated on Qwen, LLaMA, and OLMo2. CPT utilizes a mixed corpus $\mathcal{C}_{\text{mix}} = 0.8\mathcal{C}_{\text{general}} + 0.2\mathcal{C}_{\text{aug}}$.

## Key Experimental Results

### Main Results

**Universality of ILP (Average across 10 benchmark SFT datasets)**

| Metric | Value |
|------|------|
| Avg. Unlearned Proportion | 15.3% ± 2.1% |
| Cross-model Consistency | Observed in Qwen/LLaMA/OLMo2 |
| Cross-domain Consistency | Exists in Medical/Legal/Financial |

### Ablation Study

**Enforcement of CPT Intervention (Knowledge Missing + Conflict)**

| Domain | SFT only Acc | +CPT Acc | Gain |
|------|-------------|---------|------|
| Medical (MedQA) | baseline | Significant Increase | Validates Knowledge Missing Hypothesis |
| Law (LegalBench) | baseline | Significant Increase | Validates Knowledge Conflict Hypothesis |
| Finance (FinanceBench) | baseline | Significant Increase | — |

### Key Findings

- ILP is prevalent and heterogeneous—no single intervention solves all failures.
- Missing knowledge and knowledge conflict are the two most common causes; CPT is effective for both.
- Left-side forgetting is particularly severe in multi-task SFT—simple data shuffling mitigates most of it.
- ILP caused by internal contradictions in SFT data (label inconsistency) can be partially resolved through bucketed training.
- Improvements in aggregate metrics can mask the persistence of unlearned subsets—sample-level monitoring is necessary.

## Highlights & Insights

- The conceptualization of "ILP" itself is a major contribution—formalizing a widespread but unsystematically studied phenomenon.
- The taxonomy of five sources provides direct guidance for SFT practitioners to audit their data and models.
- A philosophy of "diagnosis before treatment"—understanding why failures occur before designing targeted fixes.

## Limitations & Future Work

- Sample-level evaluation in multiple-choice format may introduce evaluation bias.
- The computational costs of mitigation strategies (CPT, bucketed training, etc.) are not reported in detail.
- The taxonomy of five sources may be incomplete—other unidentified causes of ILP may exist.
- It remains unanalyzed whether subsequent training stages like RLHF/DPO exacerbate or alleviate ILP.

## Related Work & Insights

- **vs. Catastrophic Forgetting**: The latter focuses on losing previously learned abilities, while ILP focuses on the failure to acquire new knowledge—the directions are opposite.
- **vs. Data Quality Research**: The latter typically focuses on improving overall performance, while ILP focuses on why specific samples cannot be learned.
- **vs. Curriculum Learning (Bengio et al., 2009)**: Curriculum learning sorts training by complexity; ILP diagnosis suggests that sorting alone is insufficient—it is necessary to identify and handle five different failure modes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic study of incomplete learning in SFT with original concepts and taxonomy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model × Multi-domain + Causal intervention validation + 10 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and a logically rigorous diagnostic framework.
- Value: ⭐⭐⭐⭐⭐ Profound impact on both SFT practice and theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study](../../ICLR2026/llm_alignment/safety_subspaces_are_not_linearly_distinct_a_fine-tuning_case_study.md)
- [\[ICLR 2026\] Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence](../../ICLR2026/llm_alignment/antibody_strengthening_defense_against_harmful_fine-tuning_for_large_language_mo.md)
- [\[ACL 2026\] Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs](team-based_self-play_with_dual_adaptive_weighting_for_fine-tuning_llms.md)
- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)

</div>

<!-- RELATED:END -->
