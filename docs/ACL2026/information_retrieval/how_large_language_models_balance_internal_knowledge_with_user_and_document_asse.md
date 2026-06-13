---
title: >-
  [Paper Note] How Large Language Models Balance Internal Knowledge with User and Document Assertions
description: >-
  [ACL 2026][Information Retrieval & RAG][Triple-source interaction] This paper moves beyond the binary "parametric knowledge vs. single external source" conflict paradigm to propose a triple-source interaction evaluation…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Triple-source interaction"
  - "Document preference"
  - "Sycophancy"
  - "Discrimination"
  - "Post-training impact"
date: 2026-05-08
content_hash: c92b6d4abff2f3a1
---

# How Large Language Models Balance Internal Knowledge with User and Document Assertions

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.22193](https://arxiv.org/abs/2604.22193)  
**Code**: https://github.com/shuowl/llm-source-balancing (Available)  
**Area**: RAG / Knowledge Conflict / Sycophancy  
**Keywords**: Triple-source interaction, Document preference, Sycophancy, Discrimination, Post-training impact

## TL;DR
This paper moves beyond the binary "parametric knowledge vs. single external source" conflict paradigm to propose a triple-source interaction evaluation framework (Parametric / User Assertion / Document Assertion). Evaluating 27 LLMs across two datasets reveals that most models are more credulous towards documents than users, a preference further reinforced by post-training. Furthermore, most models are "impressionable," failing to distinguish whether external information is helpful or harmful.

## Background & Motivation

**Background**: RAG and chat-based systems require LLMs to simultaneously interact with three types of information sources: knowledge within their own parameters (P), retrieved documents (D), and assertions sent by the user (U). However, existing research only investigates binary combinations: P vs D (knowledge conflict) or P vs U (sycophancy), never comparing the three within a unified framework.

**Limitations of Prior Work**: Binary settings cannot address the most common scenario in real-world systems: when the document provides one answer, the user asserts another, and the model itself holds a third—who wins? Existing literature on sycophancy and knowledge conflict operates in silos, making it impossible to directly compare the relative influence of U and D.

**Key Challenge**: The "safety" of a model's capabilities depends on its ability to distinguish "helpful external info" from "harmful external info," rather than simple compliance or rejection. However, the vast majority of evaluations only measure the former, equating "suggestibility" with "good alignment."

**Goal**: (RQ1) Quantify the relative influence of P, U, and D sources; (RQ2) assess whether models can distinguish between beneficial and harmful external information; (RQ3) determine how post-training (SFT / RLHF) shifts preferences among the three sources.

**Key Insight**: Each problem is constructed into 13 probe variants (bare / 4 single-source / 8 double-source). Logistic regression is used to simultaneously estimate the odds ratios of the three sources, which are then normalized into a "reliance ratio" to facilitate comparison between U and D influence on the same scale.

**Core Idea**: A three-layer evaluation—controlled probe design, statistical modeling, and choice behavior classification—is used to decompose "model reliance on external sources" into two independent dimensions: "reliance" (influence by presence) and "discrimination" (distinguishing helpful from harmful).

## Method

### Overall Architecture
The process involves three steps: (1) **Probe Construction**: Based on a multiple-choice question $q$ and candidate set $\mathcal{C}$, permutations of presence/absence and correctness/incorrectness of P, U, and D result in 13 variants. Linguistic factors are controlled using Tier 1 (template-based) and Tier 2 (context-aware) assertion complexities. (2) **Testing 27 LLMs** (GPT-4o, LLaMA3 series, Qwen3 series) on CommonsenseQA and GSM8K-multiple-choice. (3) **Three-level Analysis**: Macro reliance proportions → choice behavior classification → probability distribution shifts.

### Key Designs

1.  **Three-Source Probe and Logistic Regression**:
    *   **Function**: Uses a single regression to simultaneously estimate the marginal effects of P, U, and D sources on the probability of a correct answer.
    *   **Mechanism**: Fits $\log \frac{p}{1-p} = \beta_0 + \beta_P P_i + \delta_U U_{pres} + \beta_U (U_{pres} \times U_{corr}) + \delta_D D_{pres} + \beta_D (D_{pres} \times D_{corr})$. Coefficients are converted to odds ratios: Parametric OR $= e^{\beta_P}$, User OR $= e^{\delta_U + \beta_U}$, and Doc OR $= e^{\delta_D + \beta_D}$. These are normalized as Source% $= \text{Source OR} / (P+U+D)$. The U%/D% ratio is given by $e^{(\delta_U + \beta_U) - (\delta_D + \beta_D)}$.
    *   **Design Motivation**: Compared to direct accuracy differences, the multiplicative decomposition of OR separates "presence" from "correctness." Normalization makes U and D directly comparable; a ratio $<1$ indicates the document is more influential.

2.  **Probe Variant Matrix (13 types)**:
    *   **Function**: Covers all meaningful source combinations at minimal cost while controlling for linguistic style variables.
    *   **Mechanism**: One bare probe serves as a baseline; four single-source probes ($v_{u^+}, v_{u^-}, v_{d^+}, v_{d^-}$) measure discrimination via PAR/SDR; eight double-source probes test conflict scenarios (two correctness states × two orderings). Tier 1 uses uniform templates, while Tier 2 uses GPT-4o to generate natural assertions relevant to the question context.
    *   **Design Motivation**: The bare probe provides a parametric baseline. Single-source probes cleanly measure model response to only U or D. Double-source ordering detects positional bias. Tier 1/2 separates "linguistic complexity" from "source attribution."

3.  **Choice-Level Discrimination Metrics (PAR/SDR)**:
    *   **Function**: Measures if a model is "impressionable"—whether it can resist incorrect external info and adopt correct external info.
    *   **Mechanism**: $\text{PAR}^+_s = P(\hat{y}_{v_{s^-}, q} = \hat{y}_{v_{bare}, q} \mid \hat{y}_{v_{bare}, q} = y_q^*, y^{assert}_{v_{s^-}, q} \neq y_q^*)$ — whether the model maintains the correct parametric answer when source $s$ is wrong. $\text{SDR}^+_s = P(\hat{y}_{v_{s^+}, q} = y^{assert}_{v_{s^+}, q} \mid \hat{y}_{v_{bare}, q} \neq y_q^*, y^{assert}_{v_{s^+}, q} = y_q^*)$ — whether the model corrects itself when the parametric answer is wrong but the source is right.
    *   **Design Motivation**: Models are classified into four quadrants (Selective / Impressionable / Skeptical / Stubborn) based on whether PAR/SDR $\geq 0.5$. Only "Selective" models are truly robust. This refines "alignment quality" into actionable discrimination metrics.

### Loss & Training
The evaluation involves no training loss. In the mitigation phase, the authors use SFT on diverse source-interaction data to simultaneously improve PAR/SDR, demonstrating that "teaching models to choose under different source conflicts" is learnable.

## Key Experimental Results

### Main Results

| Finding | Data Highlights | Explanation |
| :--- | :--- | :--- |
| Document > User preference | Most models show U%/D% < 1 | Doc OR is higher; models prefer retrieved results over users. |
| Post-training reinforces preference | U%/D% decreases across Base / SFT / Instruct stages | Post-training makes models increasingly "trust the document." |
| Most models are impressionable | PAR$^+$ < 0.5 and SDR$^+ \geq 0.5$ is the majority | Obedient but lacks judgment; copies external info even when wrong. |
| SFT is effective | PAR / SDR rise simultaneously after SFT on diverse data | Discrimination is learnable, not just an emergent property of scale. |

### Ablation Study

| Configuration | Key Observation | Explanation |
| :--- | :--- | :--- |
| Tier 1 vs Tier 2 assertion | Trends are consistent, but Tier 2 has larger impact | Linguistic complexity itself is a source of reliance. |
| User-first vs Doc-first ordering | Trends persist with minor magnitude changes | Positional bias exists but does not flip the U < D direction. |
| GPT-4o / LLaMA3 / Qwen3 | Document preference is consistent across architectures | Not an artifact of a specific dataset, but a systemic result of training paradigms. |
| Base → SFT → Instruct | U%/D% drops continuously | Decomposing post-training effects identifies which stage amplifies preference. |

### Key Findings
*   **"Document credibility" appears to stem from training bias**: Base models are nearly balanced (U/D ratio), while Instruct models lean heavily toward D. This suggests that "cited/evidenced" samples in SFT/RLHF may teach models to equate document sources with authority.
*   **Most models are impressionable**: They are willing to adopt correct external info (high SDR) but also incorrect external info (low PAR). This is the root cause of hallucinations in RAG systems triggered by incorrect retrieved documents.
*   **SFT can improve both PAR and SDR**: Fine-tuning on data covering 4 source interaction patterns (U+D+ / U+D- / U-D+ / U-D-) teaches the model to "focus on the correctness of the source rather than its identity."
*   **Probability Distribution Drift Analysis**: KL divergence shows external info "pushes out" confidence in the correct answer—even if the prediction remains unchanged, the internal confidence is diluted by external assertions.

## Highlights & Insights
*   **Unifying sycophancy and knowledge conflict**: These two research lines were previously disconnected; this paper's use of logistic regression and odds ratios to put U and D on the same scale is a clear methodological contribution.
*   **The U%/D% ratio is highly practical**: It can serve as a "trust calibration" metric before deploying RAG/chat systems, providing more information than accuracy alone.
*   **The "impressionable" concept is transferable**: The four-quadrant classification can be applied to any "model vs. external input" scenario, such as tool use, code review, or medical Q&A.

## Limitations & Future Work
*   **Only tested on multiple-choice**: Information integration in open-ended generation is far more complex; it is unknown if the triple-source framework holds.
*   **U and D are single assertions**: Real-world RAG involves multiple documents and multi-turn user dialogues; intra-source conflicts are not modeled.
*   **No distinction between "trusted" and "untrusted" documents**: All document assertions are treated equally, leaving unanswered whether models prefer documents with specific metadata.
*   **Mitigation only verified with SFT**: Mainstream methods like DPO, RLHF, or Constitutional AI were not compared; an industrial replication path needs more completion.

## Related Work & Insights
*   **vs. Wu et al. 2024 (Context Dependence)**: They only look at P vs. context; this paper splits context into two attributed sources (U and D), representing a true extension.
*   **vs. Sharma et al. 2024 (Sycophancy)**: This paper shares the odds ratio approach with sycophancy literature but applies it differently—estimating three sources simultaneously rather than U in isolation.
*   **vs. Mallen et al. 2023 (Selective Trust)**: They propose that models decide whether to trust parameters or context; this paper refines "which side to trust" into measurable, tunable PAR/SDR metrics.

## Rating
*   Novelty: ⭐⭐⭐⭐ The unified triple-source framework and PAR/SDR classification provide clear incremental value in evaluation methodology.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage with 27 models × 2 datasets × 13 probes × dual Tiers × dual orderings.
*   Writing Quality: ⭐⭐⭐⭐ Clear three-layer structure from macro OR to mid-level behavior and micro distributions.
*   Value: ⭐⭐⭐⭐ U%/D% and PAR/SDR can be directly used as discrimination audit metrics for RAG/chat systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Retrieved Context Shapes Internal Representations in RAG](how_retrieved_context_shapes_internal_representations_in_rag.md)
- [\[ACL 2026\] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](../../ICLR2026/information_retrieval/query-level_uncertainty_in_large_language_models.md)
- [\[ACL 2026\] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)
- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](../../ICLR2026/information_retrieval/tokmem_one-token_procedural_memory_for_large_language_models.md)

</div>

<!-- RELATED:END -->
