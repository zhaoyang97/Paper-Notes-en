---
title: >-
  [Paper Note] Refining and Reusing Annotation Guidelines for LLM Annotation
description: >-
  [ACL2026][NLP Understanding][Annotation Guidelines] This paper migrates the guideline reuse and moderation processes from traditional manual annotation projects to LLM annotation. It demonstrates that explicit annotation…
tags:
  - "ACL2026"
  - "NLP Understanding"
  - "Annotation Guidelines"
  - "LLM Annotation"
  - "Biomedical NER"
  - "guideline refinement"
  - "moderation"
date: 2026-05-08
content_hash: 48bfa61c0cd84622
---

# Refining and Reusing Annotation Guidelines for LLM Annotation

**Conference**: ACL2026  
**arXiv**: [2605.20809](https://arxiv.org/abs/2605.20809)  
**Code**: https://github.com/KonWooKim/llm-guideline-moderation  
**Area**: Biomedical NLP / LLM Annotation / Annotation Guidelines  
**Keywords**: Annotation Guidelines, LLM Annotation, Biomedical NER, guideline refinement, moderation  

## TL;DR
This paper migrates the guideline reuse and moderation processes from traditional manual annotation projects to LLM annotation. It demonstrates that explicit annotation guidelines, reasoning-based models, and iterative refinement driven by a small amount of gold discrepancy can all improve the strict span+type F1 in biomedical NER.

## Background & Motivation
**Background**: Text annotation serves as the foundation for semantic retrieval, information extraction, and text mining. While LLMs perform well on zero-shot or few-shot annotation tasks, benchmark gold annotations often follow very specific annotation conventions—especially in biomedical NER, where strict rules govern entity boundaries, types, and gray-zone cases.

**Limitations of Prior Work**: Human annotation projects typically constrain annotators using annotation guidelines, but most methods for LLM-based annotation provide only simple task descriptions. While LLMs may possess domain-specific conceptual knowledge, they do not necessarily adhere to target benchmark details such as minimal spans, entity type boundaries, or composite entity conventions.

**Key Challenge**: LLMs possess strong linguistic and world knowledge, but this knowledge may not align with the specific annotation conventions of a given dataset. To obtain high-quality annotations, it is not enough for a model to "understand medicine"; it must also make decisions according to the specific rules of the gold standard.

**Goal**: To verify three hypotheses: adding original human annotation guidelines improves LLM annotation; reasoning-based models are better suited for guideline-driven annotation than non-reasoning models; and LLMs can iteratively refine guidelines through moderation with minimal gold supervision.

**Key Insight**: The authors simulate "pilot moderation" from the early stages of human annotation projects. An LLM first annotates 10 development documents using current guidelines; the system performs strict matching between predictions and gold labels to identify dominant error patterns; and an LLM moderator explains errors, generates principles, and modifies the guidelines.

**Core Idea**: Treat annotation guidelines as an intermediate representation for aligning LLM annotation behavior and use discrepancy patterns to drive guideline refinement, rather than directly fine-tuning the model.

## Method
The proposed method is an iterative closed loop: an LLM annotator labels documents using current guidelines; an evaluator uses strict span+type matching to obtain F1 scores and error sets; a discrepancy analyzer identifies the most common error groups; and an LLM moderator updates the guidelines based on error evidence before entering the next round. The process stops if a quality threshold is reached or if a new round of refinement yields no improvement, at which point ineffective modifications are discarded.

### Overall Architecture
Input for round $k$ includes the current guidelines $G_k$ and development set $D$. The LLM annotator generates predicted annotations $A_k$, and the evaluator compares $A_k$ with gold labels $A_g$ to calculate strict F1. If the $IAA_k$ (Inter-Annotator Agreement/F1) has not reached the threshold and there is room for improvement, all discrepancies are collected. The system categorizes errors into label mismatch, boundary mismatch, false negative, and false positive using soft overlap, then clusters them by predicted/gold label pairs. The most frequent groups are sent to moderation. The LLM moderator sequentially performs pattern explanation, principle generation, and guideline refinement to produce $G_{k+1}$.

### Key Designs
1. **Guideline-driven annotation**:
    - **Function**: Explicitly injects existing human annotation project guidelines into the LLM prompt to align model output with the gold convention.
    - **Mechanism**: Beyond a simple prompt-only baseline, the authors provide the LLM with lightly formatted original human guidelines and require output in PubAnnotation JSON format, evaluated using exact boundary + type.
    - **Design Motivation**: Annotation errors often stem from alignment failures regarding rules rather than a lack of conceptual knowledge; guidelines are the most direct carrier for alignment.

2. **Discrepancy-driven moderation**:
    - **Function**: Uses a small number of gold examples to identify areas where current guidelines are unclear to the LLM.
    - **Mechanism**: The system first performs soft matching and categorization of errors to select dominant error patterns. The LLM moderator explains the linguistic context of the error pattern, generates a general principle, and then inserts or rewrites rules in the guidelines. For example, in NCBI Disease, if the model misses a `DiseaseClass` in a feature list, the moderator generates a rule: "Clinical conditions listed as items in a dependency feature list should also be annotated as DiseaseClass."
    - **Design Motivation**: Allowing an LLM to freely modify guidelines can lead to divergence; constraining modifications with specific error evidence ensures refinement targets the model's actual failure modes.

3. **Minimum supervision setting**:
    - **Function**: Simulates the early stage of a real annotation project where experts provide very few gold documents.
    - **Mechanism**: For each dataset, only 10 documents are randomly sampled from the original training set for development refinement, with evaluation performed on a separate 100-document set. For NCBI Disease and BioRED, the full dev split of 100 documents is used, while for BC5CDR, 100 documents are sampled from the 500-document dev split.
    - **Design Motivation**: The goal is not to reach SOTA through large-scale statistical learning, but to test whether LLMs can induce high-level annotation rules from a small number of disagreements.

### Loss & Training
The method does not train models or fine-tune parameters. The experiments compare three prompting/moderation strategies: Prompt-only, Original-guidelines, and Guideline-refinement. Models cover three families—GPT, Gemini, and DeepSeek—distinguishing between reasoning and non-reasoning versions: GPT-5 (low/high reasoning effort), Gemini 2.5 Pro (min/max thinking budget), and deepseek-chat vs. deepseek-reasoner. All main experiments use default hyperparameters.

## Key Experimental Results

### Main Results
| Dataset / Model | Prompt-only F1 | Original-guidelines F1 | Moderation F1 | Iterations |
| :--- | :--- | :--- | :--- | :--- |
| NCBI / GPT-5 | 0.46 | 0.73 (+0.27) | 0.76 (+0.03) | 3 |
| NCBI / Gemini | 0.40 | 0.63 (+0.23) | 0.66 (+0.03) | 5 |
| NCBI / DeepSeek | 0.31 | 0.55 (+0.24) | 0.56 (+0.01) | 2 |
| BC5CDR / GPT | 0.80 | 0.85 (+0.05) | 0.86 (+0.01) | 1 |
| BC5CDR / Gemini | 0.68 | 0.76 (+0.08) | 0.77 (+0.01) | 1 |
| BC5CDR / DeepSeek | 0.58 | 0.64 (+0.06) | 0.65 (+0.01) | 1 |
| BioRED / GPT-5 | 0.74 | 0.76 (+0.02) | 0.82 (+0.06) | 2 |
| BioRED / Gemini | 0.61 | 0.67 (+0.06) | 0.69 (+0.02) | 1 |
| BioRED / DeepSeek | 0.45 | 0.53 (+0.08) | 0.54 (+0.01) | 1 |

### Comparison of Reasoning Models
| Dataset | GPT non-reason / reason | Gemini non-reason / reason | DeepSeek non-reason / reason |
| :--- | :--- | :--- | :--- |
| NCBI | 0.69 → 0.73 (+0.04) | 0.48 → 0.63 (+0.15) | 0.29 → 0.55 (+0.26) |
| BC5CDR | 0.78 → 0.85 (+0.07) | 0.70 → 0.76 (+0.06) | 0.57 → 0.64 (+0.07) |
| BioRED | 0.72 → 0.76 (+0.04) | 0.66 → 0.67 (+0.01) | 0.43 → 0.53 (+0.10) |

### Cost and Time Summary
| Dataset / Model | Iterations | Cost per round | Time per round | Est. Total Cost | Est. Total Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| NCBI / GPT-5 | 3 | $1.186 | 5.2 min | $3.557 | 15.6 min |
| NCBI / Gemini | 5 | $0.092 | 3.0 min | $0.460 | 14.8 min |
| BioRED / GPT-5 | 2 | $1.991 | 14.0 min | $3.982 | 28.0 min |
| BioRED / DeepSeek | 1 | $0.048 | 29.8 min | $0.048 | 29.8 min |

### Key Findings
- The original guidelines provide the largest improvement; on NCBI, GPT-5 increased from 0.46 to 0.73, Gemini from 0.40 to 0.63, and DeepSeek from 0.31 to 0.55.
- The absolute gain from moderation is smaller, typically +0.01 to +0.03 F1, though it reached +0.06 on BioRED with GPT-5.
- Reasoning models outperformed their non-reasoning counterparts across all datasets and model families, indicating that applying complex guidelines indeed requires reasoning capabilities.
- GPT-5 is powerful but expensive; DeepSeek is low-cost but has high latency and weaker performance; Gemini is balanced in terms of cost and stability.

## Highlights & Insights
- This paper identifies a critical issue in LLM annotation: errors do not necessarily stem from "not understanding the entity," but from "not understanding the project rules." Guidelines are a more interpretable alignment carrier than few-shot examples.
- The moderation process closely mirrors real-world annotation projects. Instead of letting the LLM directly optimize for F1, it forces the model to summarize errors into readable rules, which allows the output to be reviewed and reused by humans.
- The results are honest: moderation provides consistent positive gains, but the magnitude is modest, suggesting that a small number of gold documents can identify some rule gaps but are insufficient to cover all long-tail ambiguities.

## Limitations & Future Work
- Using only 10 development documents is risky due to sample selection bias; the dominant error patterns found may not represent those of the entire dataset.
- Stopping conditions depend on IAA/F1 changes on a small sample, which is statistically unstable and may lead to premature stopping or the retention of accidentally effective rules.
- Moderation only handles the most frequent discrepancy group per round, potentially ignoring multiple medium-frequency but important error types.
- This study focuses on span+type for biomedical NER; it is unclear if relation extraction, event extraction, or multi-label subjective annotation would benefit similarly.
- Future work could involve human experts auditing LLM-generated guideline edits to create semi-automated moderation rather than fully automatic acceptance.

## Related Work & Insights
- **vs. Direct LLM Annotation**: Prompt-only methods rely on the model's existing knowledge, which often conflicts with gold conventions; guideline prompts make project rules explicit.
- **vs. Few-shot Annotation**: Few-shot provides examples but does not necessarily explain the underlying rules; guideline refinement produces readable, reusable rule text.
- **vs. Manual Moderation**: Human moderation is more reliable but costly; LLM moderation can serve as an early draft generator to help experts quickly locate rule gaps.
- **Inspiration**: For high-requirement data annotation tasks, prioritizing a "LLM-readable and human-auditable" guideline is more sustainable than simple prompt tuning or stacking examples.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Formalizing the migration of annotation moderation to LLM annotation is highly practical; the method is simple, but the problem identification is precise.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers three biomedical datasets, three model families, and reasoning comparisons; cross-task generalization still requires more validation.
- Writing Quality: ⭐⭐⭐⭐☆ Clear hypotheses, complete experimental design and error analysis, and thorough discussion of limitations.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for building auditable and reusable LLM annotation pipelines, especially for expert-domain corpora construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition](diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)
- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)
- [\[NeurIPS 2025\] Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL](../../NeurIPS2025/nlp_understanding/planning_without_search_refining_frontier_llms_with_offline_goal-conditioned_rl.md)
- [\[ACL 2026\] MTSQL-R1: Towards Long-Horizon Multi-Turn Text-to-SQL via Agentic Training](mtsql-r1_towards_long-horizon_multi-turn_text-to-sql_via_agentic_training.md)

</div>

<!-- RELATED:END -->
