---
title: >-
  [Paper Note] ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation
description: >-
  [AAAI 2026][Hallucination Detection][ESG Reports] This paper constructs ESG-Bench — 270 manually annotated QA pairs from 94 real ESG reports (2020–2024) — and proposes a three-stage hallucination mitigation pipeline: SFT…
tags:
  - "AAAI 2026"
  - "Hallucination Detection"
  - "ESG Reports"
  - "Long-Context"
  - "CoT Fine-tuning"
  - "Compliance"
date: 2026-05-08
content_hash: 3d63221087cd1e2e
---

# ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation

**Conference**: AAAI 2026
**arXiv**: [2603.13154](https://arxiv.org/abs/2603.13154)  
**Code**: Available  
**Area**: Hallucination Detection
**Keywords**: ESG Reports, Long-Context, Hallucination Detection, CoT Fine-tuning, Compliance

## TL;DR
This paper constructs ESG-Bench — 270 manually annotated QA pairs from 94 real ESG reports (2020–2024) — and proposes a three-stage hallucination mitigation pipeline: SFT (with grounded answers + "Not Provided" abstention labels) → CoT Prompting (2/4-step prompt templates) → CoT Fine-tuning (with human-annotated reasoning chains). The 4-step CoT fine-tuned Llama-3 achieves 92.52% with-answer (WA) accuracy and 99.37% without-answer (WoA) accuracy (balanced 96%), with generalization gains on HaluEval and BioASQ.

## Background & Motivation

**Background**: ESG (Environmental, Social, and Governance) reports have become a legal requirement in regions such as the EU, yet individual reports can span hundreds of pages containing complex narratives, tables, and figures. LLMs are increasingly employed for automated ESG report analysis.

**Limitations of Prior Work**:
- LLMs are prone to hallucination on long-context ESG reports — fabricating non-existent environmental metrics or governance claims.
- Hallucinations are particularly dangerous in high-stakes compliance scenarios, where erroneous ESG data may lead to flawed investment decisions and legal liability.
- No dedicated long-context QA benchmark exists for the ESG domain.

**Key Challenge**: ESG report analysis requires precise fact extraction from hundreds of pages and correct abstention (e.g., "not mentioned in the report"), yet LLMs tend to fabricate answers rather than acknowledge uncertainty.

**Goal**: Construct a long-context QA benchmark for the ESG domain and develop effective hallucination mitigation methods.

**Key Insight**: A three-stage progressive approach — from basic SFT → CoT Prompting → CoT Fine-tuning — incrementally training models to answer correctly and abstain appropriately in ESG scenarios.

**Core Idea**: 4-step CoT (extract topic → search report → assess answerability → generate answer) combined with grounded supervision = ESG long-context hallucination mitigation.

## Method

### Overall Architecture
ESG-Bench construction: 94 real ESG reports → GPT-4o-generated questions → human-annotated answers and hallucination labels (46.7% correct / 34.8% incomplete / 15.6% hallucinated / 3.0% not found) → evaluation under a 3-stage mitigation framework.

### Key Designs

1. **SFT Baseline**:

    - Function: Fine-tune the model on grounded answers with "Not Provided" abstention labels.
    - Performance: WA 80.99% / WoA 99.0% / F1 73.68%

2. **CoT Prompting (2-step / 4-step)**:

    - Function: Guide the LLM with structured reasoning templates.
    - 2-step: "Can this question be answered from the report? → If yes, answer it."
    - 4-step: "Extract the topic → Search the report for relevant passages → Assess answerability → Answer based on evidence."
    - Design Motivation: The 4-step variant outperforms the 2-step variant — intermediate steps (search + answerability assessment) force the model to verify before answering.

3. **CoT Fine-tuning**:

    - Function: Fine-tune on human-annotated reasoning chains.
    - Mechanism: Rather than supervising only the final answer, the model is trained on the full reasoning process, learning the 4-step inference pattern.
    - Performance: **4-step CoT fine-tuned Llama-3 achieves WA 92.52% + WoA 99.37%.**

### Loss & Training
- Models: Llama-3.2-3B / Gemma-2-2B / Mistral-7B
- Train/test split by report to prevent data leakage.

## Key Experimental Results

### Main Results (ESG-Bench Test Set)

| Method | WA Acc↑ | WoA Acc↑ | Balanced↑ | F1↑ |
|--------|---------|---------|-----------|-----|
| Llama-3 Baseline | 67.61 | 83.54 | 76.00 | 65.23 |
| + SFT | 80.99 | 99.00 | 90.67 | 73.68 |
| + CoT-2step | 88.73 | 97.47 | 93.33 | 76.35 |
| **+ CoT-4step** | **92.52** | **99.37** | **96.00** | **78.62** |

### Ablation Study: Cross-Domain Transfer

| Method | HaluEval Acc | BioASQ |
|--------|-------------|--------|
| Mistral Baseline | 90.30% | Baseline |
| **Mistral CoT-4** | **95.91%** | Improved |

### Key Findings
- **4-step CoT is optimal**: 92%+ WA AND 99%+ WoA — the model both answers correctly and abstains appropriately.
- **Answerability assessment is the critical step**: 2-step (direct judgment) underperforms 4-step (search first, then judge).
- **Cross-domain transfer is effective**: CoT ability trained on ESG generalizes to HaluEval (+5.6 pp).
- **15.6% of ESG report responses are hallucinated**: Models tend to fabricate answers for questions with uncertain grounding.

## Highlights & Insights
- The **4-step "search → assess answerability → answer" structure** is broadly applicable to any long-context QA setting.
- **Simultaneously high WA and WoA accuracy is the true requirement in compliance scenarios** — answering correctly is insufficient; knowing when to abstain is equally critical.
- ESG analysis represents a high-value application domain for LLM hallucination mitigation, given the legal demand for accuracy.

## Limitations & Future Work
- Only 270 QA pairs — the dataset is small in scale.
- Only 3–7B models are evaluated — the behavior of larger models remains unknown.
- Limited analysis of multimodal content (tables and figures).
- GPT-4o-generated questions may introduce selection bias.

## Related Work & Insights
- **vs. HaluEval**: A general-purpose hallucination benchmark. ESG-Bench targets a specific, high-risk domain.
- **vs. LongBench**: A long-context benchmark that does not evaluate abstention capability.
- The 4-step CoT method is transferable to legal and medical document analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ — First long-context hallucination mitigation benchmark in the ESG domain, filling a significant gap in financial NLP.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three models, three-stage method, and cross-domain transfer validation, though dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐ — The progressive method design is clearly presented, with a well-motivated transition from baseline to chunking to CoT.
- Value: ⭐⭐⭐⭐ — Practically valuable for compliance AI and long-context hallucination mitigation; the 4-step CoT is generalizable to legal and medical document analysis.

## Additional Notes
- ESG report analysis is an important application area in financial NLP. This benchmark fills a gap in long-context compliance evaluation and serves as a useful reference for financial AI applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[ICML 2026\] Building Reliable Long-Form Generation via Hallucination Rejection Sampling](../../ICML2026/hallucination/building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](../../ACL2026/hallucination/enhancing_hallucination_detection_via_future_context.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](../../ACL2026/hallucination/benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](inex_hallucination_mitigation_via_introspection_and_cross-mo.md)

</div>

<!-- RELATED:END -->
