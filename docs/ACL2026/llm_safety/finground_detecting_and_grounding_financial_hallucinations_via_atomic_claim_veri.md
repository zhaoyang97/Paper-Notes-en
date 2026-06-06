---
title: >-
  [Paper Note] FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification
description: >-
  [ACL 2026][LLM Safety][Financial QA] FinGround is a three-stage "verify-then-ground" pipeline for financial document QA: (1) finance-aware hybrid retrieval…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Financial QA"
  - "Atomic Claim Verification"
  - "Formula Reconstruction"
  - "Table Attribution"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: ecd2dfa5d58a5391
---

# FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification

**Conference**: ACL 2026  
**arXiv**: [2604.23588](https://arxiv.org/abs/2604.23588)  
**Code**: Not public  
**Area**: Information Retrieval / Financial RAG / Hallucination Detection  
**Keywords**: Financial QA, Atomic Claim Verification, Formula Reconstruction, Table Attribution, Knowledge Distillation

## TL;DR
FinGround is a three-stage "verify-then-ground" pipeline for financial document QA: (1) finance-aware hybrid retrieval; (2) decomposing answers into atomic claims and verifying them with a type-routed strategy across a six-category taxonomy (numerical, temporal, entity-attribute, comparative, regulatory, and computational—where computational claims use formula reconstruction + arithmetic re-verification); (3) grounded rewriting of unsupported claims with paragraph/cell-level citations. By distilling GPT-4o into an 8B detector, it achieves a 91.4% F1 score with 18× acceleration, reducing the end-to-end hallucination rate by 78% compared to GPT-4o+CoT.

## Background & Motivation

**Background**: LLMs in the financial industry must ground answers in specific SEC filings or financial reports. However, even GPT-4-Turbo with RAG exhibits an 81% error rate on SEC QA (Islam 2023). Furthermore, the EU AI Act mandates compliance for high-risk financial AI by August 2026, requiring "human oversight + explainability + accuracy assurance."

**Limitations of Prior Work**: General hallucination detectors like FActScore and SAFE treat all claims equally. While they can extract atomic facts such as "gross margin is 62.4%," they fail to align them with table cells for verification, missing 43% of computational errors. Rewriting methods like RARR assume a single source of evidence and often trigger 34% newly hallucinated content when rewriting computational claims without type differentiation. Table-cell attribution also suffers from 23% dangling citations if upstream chunking is structure-unaware.

**Key Challenge**: General hallucination detection seeks to be "domain-agnostic," but core errors in financial scenarios (numerical miscalculations, fabricated regulatory citations, table misalignment) necessitate "domain-awareness." The claim type dictates whether to use NLI, formula recalculation, or table matching. A one-size-fits-all NLI approach is destined to fail on ratio and margin verification.

**Goal**: (i) Unify detection and mitigation into a production-ready financial QA pipeline; (ii) design a type-routed verification strategy to specifically address computational errors; (iii) reduce costs to a level viable for deployment ($\le \$0.005/\text{query}$); (iv) propose a "retrieval-equalized evaluation" protocol to decouple retrieval gains from verification gains.

**Key Insight**: Based on an error analysis of 500 real financial hallucinations, the authors found that errors concentrate on six enumerable claim types, each having a corresponding optimal verification strategy. Therefore, the problem is not "stronger NLI models" but "routing by type."

**Core Idea**: Upgrade atomic-claim verification from a "single NLI black box" to a "multi-strategy ensemble routed by a 6-class financial claim taxonomy." Specifically, the computational class replaces traditional NLI with a three-step process: formula template matching + table cell extraction + arithmetic re-verification.

## Method

### Overall Architecture
The FinGround three-stage pipeline consists of:
**Stage 1 Finance-Aware Hybrid Retrieval** — RoBERTa-base classifies queries into Simple/Moderate/Complex tiers. Strategies include BM25, dense retrieval + table extraction (using header-aware similarity $\text{sim}(q,t)=\alpha\cdot\cos(\mathbf{q},\mathbf{t}_{\text{cell}})+(1-\alpha)\cdot\cos(\mathbf{q},\mathbf{t}_{\text{header}})$ where $\alpha=0.6$), or iterative retrieve-then-reason. Structure-aware chunking preserves row-column relationships, with each chunk carrying $\langle\text{document, section, page, element\_type}\rangle$ provenance.
**Stage 2 Atomic Financial Claim Verification** — Claim decomposition $\rightarrow$ classification $\rightarrow$ evidence alignment $\rightarrow$ type-routed judgment (supported / contradicted / unverifiable).
**Stage 3 Grounded Regeneration** — Contradicted/unverifiable claims are located in the original answer spans via fuzzy alignment (edit distance $\le 3$), followed by targeted re-retrieval and RARR-style rewriting with inline citations like `[Doc:d, §s, p.p]` or `[Doc:d, Table t, Row r, Col c]`. If $\ge 3$ claims require modification, full regeneration is triggered to avoid error compounding.

### Key Designs

1.  **6-Class Financial Claim Taxonomy + Type-Routed Verification**:
    - **Function**: Categorizes atomic claims into numerical, temporal, entity-attribute, comparative, regulatory, and computational types, applying the most suitable verification strategy for each rather than a uniform NLI approach.
    - **Mechanism**: After classification, routing occurs: numerical claims use structured extraction (value, unit, period, entity) for precise table matching; entity-attribute claims use cross-encoder NLI; regulatory claims query a rule database; and computational claims follow a dedicated "formula reconstruction" branch. The taxonomy was derived from 500 real hallucinations; empirical results showed the 6-class system outperformed a 3-class system by 4.3 F1 and showed no significant difference from a 10-class system ($p=0.23$).
    - **Design Motivation**: General NLI cannot perform arithmetic for ratios or margins; forcing NLI to verify "gross margin = 62.4%" is equivalent to "asking an innumerate person to audit accounts." Routing after classification acknowledges that verification strategies are coupled with claim types.

2.  **Formula Reconstruction + Arithmetic Re-verification for Computational Claims**:
    - **Function**: Re-calculates "derived quantities" (e.g., gross margin, debt-to-equity ratio) instead of performing NLI.
    - **Mechanism**: Three steps: (a) identify implicit formulas using a library of 47 financial templates; (b) retrieve operand values from table cells; (c) re-calculate the derived quantity, allowing a $\pm 0.5\%$ tolerance for rounding. End-to-end computational verification reached 90.2% F1, a +18.9 F1 improvement over SelfCheckGPT.
    - **Design Motivation**: Experimental analysis revealed computational claims have the highest hallucination rate (28.4%) but are the easiest to auto-verify—provided the correct operands are found. The bottleneck is not "verification difficulty" but "routing"; treating them as NLI tasks is the root cause of the problem.

3.  **8B Distilled Detector + Retrieval-Equalized Evaluation Protocol**:
    - **Function**: (a) Distill GPT-4o on 3,200 financial QA pairs into Llama-3-8B-Instruct, reducing p95 latency from 6.1s to 340ms (18×) and achieving 91.4% F1 (retaining 96.2% teacher performance) at $\$0.003/\text{query}$; (b) mandate that all baselines use identical retrieval results during evaluation to decouple retrieval gains from verification gains.
    - **Mechanism**: Distillation uses reverse KL divergence + multi-task objectives (decomposition + alignment + verdict). Annotations used two-round consistency checks, discarding 8.4% of inconsistent samples. The retrieval-equalized protocol equips each baseline with FinGround's Stage 1 retrieval, isolating "verification" as the single variable when comparing Hallucination Rates (HalRate).
    - **Design Motivation**: (a) GPT-4o's 6.1s latency is unacceptable for real-time financial QA; (b) without retrieval-equalized evaluation, RAG papers struggle to distinguish whether improvements stem from "better evidence" or "better use of evidence"—this is a neglected methodological contribution.

### Loss & Training
Distillation employs reverse KL ($\text{KL}(p_{\text{student}} || p_{\text{teacher}})$ is more stable for mode-seeking), joint multi-task learning (decomposition + alignment + verdict), and vLLM deployment with continuous batching. The cross-encoder alignment model was fine-tuned on 8,400 TAT-QA/FinQA NLI samples, reaching 87.2% F1.

## Key Experimental Results

### Main Results: FinHalu Detection Performance (1,200 Expert-Annotated Triplets)

| System | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| SelfCheckGPT | 69.4 | 76.5 | 72.8 |
| HHEM (Vectara) | 78.9 | 73.8 | 76.3 |
| FActScore | 74.2 | 79.3 | 76.7 |
| CRAG | 80.6 | 74.9 | 77.6 |
| Self-RAG | 81.2 | 77.1 | 79.1 |
| GPT-4o (teacher) | 94.1 | 95.9 | 95.0 |
| **FinGround (8B distilled)** | **92.7** | **90.2** | **91.4** |

All improvements over baselines are significant at $p < 0.01$. The FinGround 8B model retains 96.2% of the teacher's F1 with a p95 latency of 340ms (vs. 6.1s).

### Ablation Study: End-to-End HalRate (Key FinanceBench Highlights)

| System | FinQA HalRate↓ | TAT-QA HalRate↓ | FinanceBench HalRate↓ | Uncond. Acc |
| :--- | :---: | :---: | :---: | :---: |
| Vanilla RAG | 34.7 | 31.5 | 43.8 | 63.9 |
| FActScore | 25.3 | 22.7 | 32.4 | 66.2 |
| Self-RAG | 22.1 | 18.4 | 28.5 | 68.2 |
| GPT-4o + CoT | 18.6 | 15.2 | 22.4 | 71.9 |
| **FinGround (full)** | **3.6** | **3.8** | **4.9** | **71.2** |
| − regeneration | 3.6 | 3.8 | 4.9 | 63.8 |
| − taxonomy (Uniform NLI) | 7.2 | 8.1 | 11.7 | 70.5 |
| − table retrieval | 5.9 | 10.6 | 9.4 | 66.2 |

The end-to-end HalRate dropped by an average of 78% compared to GPT-4o+CoT. Under retrieval-equalized settings, it still achieved a 68–76% reduction ($p < 0.01$), proving that verification contribution is independent of retrieval.

### Key Findings
- **Removing taxonomy doubles the HalRate**: On FinanceBench, it rose from 4.9% to 11.7%, proving that "type-routed verification" is a core contribution, not just a gimmick.
- **Removing table retrieval hurts performance on table-heavy datasets**: In TAT-QA, the HalRate jumped from 3.8% to 10.6%, highlighting that table evidence is indispensable in financial QA.
- **Computational claims are the most difficult yet the easiest**: They represent the highest hallucination category (28.4%) but also saw the largest improvement from formula reconstruction (+18.9 F1)—evidence that the bottleneck is routing, not verification difficulty.
- **Strong cross-generator generalization**: Even with Llama-3-70B and Claude-3.5-Sonnet, the F1 remains at 87–89%, indicating that FinGround is a generator-agnostic verification layer.
- **Hedged language accounts for 52% of false positives**: Vague expressions like "approximately" or "roughly" cause the verifier to misjudge, marking a clear point for future improvement.
- **4-week pilot with 24 analysts**: Incomplete retrieval recall led to 3.8% false negatives, 56% of which occurred because computational operands fell outside the retrieval window. Verification is not a panacea; it must be paired with better retrieval.

## Highlights & Insights
- **Type-routed verification is the most natural evolution after FActScore**: FActScore decomposes answers into atomic facts for independent verification but ignores that different facts require different verification methods. FinGround completes this idea. This strategy can be transferred to legal or medical QA domains (e.g., verifying dosages, side effects, or case law citations).
- **Computational claims use formula recalculation instead of NLI**: Making the model "calculate" rather than "guess" is a practical implementation of embedding symbolic execution into RAG verification. Moreover, a library of 47 templates is more comprehensive than it seems, as financial ratios are finite.
- **Retrieval-equalized evaluation is a neglected methodological contribution**: Decoupling "retrieval gain" from "verification gain" should become the new standard for RAG research to clarify mixed results.
- **8B distillation to $0.003/query**: This isn't just a technical flex; it is a hard requirement for deployment across 40 banks. While many NLP papers stop at the GPT-4o teacher, FinGround completes the production-grade "last mile."

## Limitations & Future Work
- The template library is limited to 47 formulas. Derived quantities outside this library fallback to NLI, leading to a drop in accuracy.
- Handling of hedged language is weak; words like "approximately" trigger 52% of false positives, necessitating more granular uncertainty modeling.
- Operands outside the retrieval window cause 3.8% false negatives, proving that "even perfect verification" cannot save a "retrieval miss"—pipeline performance is bounded by Stage 1 recall.
- FinHalu consists of only 1,200 expert-labeled samples ($\kappa=0.83$), which is relatively small and heavily dependent on the financial expertise of annotators.
- Evaluations were conducted only on English SEC domains (FinQA/TAT-QA/FinanceBench). Generalization across languages and regulatory systems (e.g., MiFID II, SAC) remains unknown.

## Related Work & Insights
- **vs. FActScore (Min 2023)**: Both decompose answers into atomic facts. FActScore uses uniform NLI verification (76.7 F1), while FinGround adds 6-class taxonomy routing + formula reconstruction, gaining +14.7 F1 in financial scenarios. The fundamental difference is "type-awareness vs. type-agnosticism."
- **vs. SelfCheckGPT (Manakul 2023)**: SelfCheckGPT relies on sampling consistency without external evidence. FinGround requires retrieved evidence but provides grounded citations. The former is better for open-domain tasks, while the latter is better for regulated domains.
- **vs. Self-RAG / CRAG (Asai 2024, Yan 2024)**: Both improve the RAG process itself via adaptive retrieval. FinGround is a verification layer that can be layered on top of them (Table 3 shows FinGround further reduced the HalRate by 68% after Stage 1 retrieval already provided a 37% improvement).
- **Inspiration**: (a) Any grounded QA scenario should cluster error types before designing verification; (b) embedding symbolic execution/arithmetic into RAG verification is low-hanging fruit; (c) production papers should strictly decouple the evaluation of "retrieval," "verification," and "rewriting."

## Rating
- **Novelty**: ⭐⭐⭐⭐ The 6-class taxonomy + formula reconstruction is a substantial contribution to the financial domain; the retrieval-equalized protocol is an insightful methodological addition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes 3 public + 1 custom benchmark, cross-generator transfer, a 4-week 24-person pilot, and detailed ablations, though the FinHalu size (1,200) is somewhat small.
- **Writing Quality**: ⭐⭐⭐⭐ Clear three-stage structure, data-backed claims, and honest limitations; however, formulas and tables are dense.
- **Value**: ⭐⭐⭐⭐ Directly addresses the EU AI Act 2026 compliance deadline with a $\$0.003/\text{query}$ deployment plan; the taxonomy routing logic is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ICML 2026\] From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity](../../ICML2026/llm_safety/from_flat_facts_to_sharp_hallucinations_detecting_stubborn_errors_via_gradient_s.md)
- [\[CVPR 2026\] Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](../../CVPR2026/llm_safety/beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)
- [\[ACL 2026\] FaithLens: Detecting and Explaining Faithfulness Hallucination](faithlens_detecting_and_explaining_faithfulness_hallucination.md)
- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)

</div>

<!-- RELATED:END -->
