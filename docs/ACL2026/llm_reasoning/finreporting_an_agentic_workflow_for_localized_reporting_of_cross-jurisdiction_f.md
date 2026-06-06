---
title: >-
  [Paper Note] FinReporting: An Agentic Workflow for Localized Reporting of Cross-Jurisdiction Financial Disclosures
description: >-
  [ACL2026][LLM Reasoning][Financial Disclosures] FinReporting decomposes localized reporting across the US, Japan, and China into an auditable agentic workflow consisting of "rule-based extraction + ontology mapping + con…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "Financial Disclosures"
  - "Cross-jurisdiction"
  - "agentic workflow"
  - "canonical ontology"
  - "LLM guardrail"
date: 2026-05-08
content_hash: 096dbea8d81376c1
---

# FinReporting: An Agentic Workflow for Localized Reporting of Cross-Jurisdiction Financial Disclosures

**Conference**: ACL2026  
**arXiv**: [2604.05966](https://arxiv.org/abs/2604.05966)  
**Code**: Demo: https://huggingface.co/spaces/BoomQ/FinReporting-Demo  
**Area**: Financial NLP / LLM Agent  
**Keywords**: Financial Disclosures, Cross-jurisdiction, agentic workflow, canonical ontology, LLM guardrail  

## TL;DR
FinReporting decomposes localized reporting across the US, Japan, and China into an auditable agentic workflow consisting of "rule-based extraction + ontology mapping + constrained LLM verification/repair + manual review." It utilizes a unified IS/BS/CF schema to mitigate inconsistencies in financial disclosure formats and accounting semantics across different jurisdictions.

## Background & Motivation
**Background**: Financial NLP has expanded from sentiment analysis and risk prediction to financial report QA, structured extraction, XBRL querying, and financial agents. LLMs can assist users in extracting metrics from lengthy reports, summarizing disclosures, and answering financial questions, thereby reducing the cost of reading full annual reports.

**Limitations of Prior Work**: Many systems assume a single-market scenario where users are familiar with local accounting standards, disclosure formats, and taxonomies, only needing to retrieve information within a known framework. However, global investors often need to understand reports from foreign markets. While the US and Japan rely heavily on XBRL with high machine readability, Chinese annual reports are primarily PDF tables characterized by layout variations, table fragmentation, OCR noise, and company-specific custom items. The same label may have different meanings across markets, or the same concept may be represented by different labels.

**Key Challenge**: Cross-jurisdiction "localized reporting" is not a simple translation of field names or the mere extraction of PDF tables. The true difficulty lies in semantic alignment and aggregation conventions: which home-market canonical concept should a specific line item map to? If extraction is missing or suspected to be erroneous, the system must clearly flag, repair, or delegate to experts, rather than allowing the LLM to freely generate a plausible-looking figure.

**Goal**: The authors propose FinReporting to map core financial items from US, Japanese, and Chinese annual reports to a unified Income Statement (IS), Balance Sheet (BS), and Cash Flow (CF) schema. The goal is to provide an audit trail, quality signals, and anomaly logs at each step to support cross-market comparisons and downstream financial QA.

**Key Insight**: The paper positions the LLM as a constrained verifier rather than an extractor or generator. A rule layer first generates reproducible candidate values, and the LLM executes KEEP, REPAIR, or NEED_REVIEW decisions only within explicit evidence and decision spaces. Final high-impact or evidence-insufficient cases are deferred to human experts.

**Core Idea**: Use a canonical financial ontology to unify cross-market semantics and place the LLM within a verification/repair layer equipped with guardrails, making cross-jurisdiction reporting localization both automated and auditable.

## Method
FinReporting is a systems paper where the methodology consists of three layers: a deterministic rule processing layer, an LLM guardrail layer, and a conditional expert review layer. 

### Overall Architecture
The input consists of a specific market, company, and filing. The system first performs Filing Acquisition and Statement Identification: for US and Japanese markets, it directly reads XBRL tagged facts; for the Chinese market, it locates PDF pages and tables related to IS/BS/CF. Next is Extraction: XBRL-native markets involve selecting the correct reporting context (e.g., consolidated vs. separate, period length, instant/duration). PDF-centric markets require document decomposition, table parsing, fallback column selection, and per-field status labeling.

Post-extraction, the system uses a global ontology to map local line items to a unified canonical schema. This schema covers core concepts of the IS, BS, and CF while retaining metadata such as local labels, units, currencies, and accounting standards. Outputs include localized financial statements, anomaly logs, audit trails, and structured workbooks.

### Key Designs
1.  **Cross-market canonical ontology**:
    - **Function**: Provides a unified concept inventory for financial report items across the US, Japan, and China, enabling aligned comparisons.
    - **Mechanism**: Defines core fields for IS, BS, and CF and maps different market labels to canonical concepts. In experiments, the shared subset includes 18 target fields: 5 for IS, 7 for BS, and 6 for CF.
    - **Design Motivation**: Without an ontology layer, the system would only perform local extraction, failing to correctly handle "same name, different meaning" or "different name, same meaning" items, thus hindering cross-market QA and benchmarking.

2.  **Rule-priority reproducible extraction layer**:
    - **Function**: Generates stable, interpretable, and reproducible candidate values to minimize LLM hallucinations in numerical data.
    - **Mechanism**: US/JP markets use XBRL tagged facts and reporting context selection; CN uses PDF table parsing with fallbacks, labeling each field with statuses like OK, MISSING, PARSE_ERROR, or NOT_APPLICABLE.
    - **Design Motivation**: Financial value extraction cannot rely on free-text generation. While the rule layer has finite coverage, its errors are locatable, and status labels make missing data and uncertainty explicit.

3.  **Constrained LLM verifier / repairer**:
    - **Function**: Identifies suspected errors in rule outputs, completes repairable fields, and decides if manual review is required.
    - **Mechanism**: LLM decision space is restricted to KEEP, REPAIR, and NEED_REVIEW. REPAIR is only permitted when a field is fixable, evidence is clearly present in the filing context, and the candidate value is consistent with the evidence; otherwise, it falls back to NEED_REVIEW. All decisions record evidence and failure reasons.
    - **Design Motivation**: Free LLM extraction can produce unmarked errors—the most dangerous being "plausible but incorrect" figures. A constrained verifier makes the LLM responsible for reasoning and evidence verification rather than original generation.

### Loss & Training
This paper does not train a new model but focuses on the system workflow and evaluation. The LLM guardrail layer used GPT-4o in experiments, comparing it with backbones like GPT-5.2, GPT-5 mini, Gemini-2.5-Flash, Gemini-2.5-Flash-Lite, and DeepSeek-Chat. Evaluation metrics include Filled Rate (FR), Conflict Rate (CR), and Accuracy (ACC). FR is the ratio of non-empty outputs, CR is the ratio of triggers for human review due to rule-LLM conflicts or insufficient evidence, and ACC is accuracy relative to human annotation.

## Key Experimental Results

### Main Results
| Jurisdiction | Metric | LLMReporting | FinReporting | Observation |
|--------|------|------|----------|------|
| US | FR | 94.44 | 95.56 | High XBRL standardization leads to highest coverage |
| US | CR | 5.56 | 15.56 | FinReporting proactively exposes conflict/review signals |
| US | ACC | 89.38 | 90.23 | Slight improvement |
| JP | FR | 84.44 | 84.44 | Japanese XBRL still shows greater label/report variation |
| JP | CR | 15.56 | 15.56 | Identical conflict rates |
| JP | ACC | 88.36 | 88.36 | No improvement observed |
| CN | FR | 63.33 | 63.33 | Coverage remains lower in PDF-centric environments |
| CN | CR | 26.67 | 40.56 | More NEED_REVIEW / conflict exposure |
| CN | ACC | 78.15 | 82.11 | Largest gain in the most difficult market |

### LLM backbone comparison (US filings)
| Backbone | FR | CR | ACC | Cost ($) |
|------|---------|------|------|------|
| GPT-5.2 | 95.56 | 8.89 | 90.23 | 36.96 |
| GPT-5 mini | 95.56 | 15.00 | 90.23 | 17.77 |
| GPT-4o | 95.56 | 15.56 | 90.00 | 34.04 |
| Gemini-2.5-Flash | 95.56 | 12.78 | 90.23 | 7.27 |
| Gemini-2.5-Flash-Lite | 95.56 | 8.89 | 90.00 | 1.47 |
| DeepSeek-Chat | 95.56 | 100.00 | 90.23 | 2.41 |

### Key Findings
- Coverage is primarily determined by the pipeline and data source structure rather than the LLM backbone: FR remains 95.56 across all backbones in the US table.
- Accuracy in the CN scenario improved from 78.15 to 82.11, demonstrating the value of verification/repair in difficult contexts, though FR remains at 63.33.
- Stronger/more expensive models do not necessarily yield higher ACC. Gemini-2.5-Flash-Lite achieves an ACC of 90.00 at a cost of $1.47, nearly matching GPT-5.2's 90.23.
- DeepSeek-Chat has a CR of 100.00, suggesting that certain backbones may over-trigger conflicts or fail to complete constrained verification under this guardrail setting.

## Highlights & Insights
- **LLM placed in the correct role**: Instead of having the LLM directly read reports to generate values, the rule layer handles candidate extraction while the LLM focuses on evidence checking and limited repair. This is a more trustworthy architecture for high-risk financial scenarios.
- **CR is not a purely negative indicator**: FinReporting shows higher CR in US/CN markets; while this seems like more conflict, it indicates the system's willingness to expose uncertainty for human review rather than outputting silent errors.
- **PDF-centric markets are the true stress test**: US/JP XBRL already provides machine-readable structures. CN layout variations, table fragmentation, and OCR noise are closer to real-world Document AI challenges, making the improvements on CN more significant.
- **Cost analysis has deployment value**: If coverage is driven by the rule pipeline and the LLM only acts as a verifier, cheaper models may suffice. This is crucial for enterprise-grade financial report processing.

## Limitations & Future Work
- Currently covers only three jurisdictions (US, Japan, China) and evaluates only annual filings, non-financial enterprises, consolidated statements, and 18 core fields.
- Coverage in CN PDF scenarios remains insufficient (FR 63.33). Layout variations, fragmented tables, OCR noise, and company-specific disclosure habits still lead to MISSING, PARSE_ERROR, or NEED_REVIEW statuses.
- The canonical ontology is manually predefined and may fail to cover long-tail taxonomies, company-specific metrics, or market-specific accounting semantics, leading to interpretative bias.
- The system is an auditable assistant, not a fully autonomous financial reporting tool. High-risk investment, auditing, and regulatory decisions still require human verification of original filings.
- Future work could extend to footnotes, segment disclosures, more periods, and market-specific long-tail fields, incorporating stricter provenance tracking and numerical consistency checks.

## Related Work & Insights
- **vs XBRL Agent / XBRL-centered systems**: These are suitable for machine-readable disclosures but often assume a fixed taxonomy; FinReporting explicitly handles cross-jurisdiction heterogeneity and PDF-centric markets.
- **vs FinQA / TAT-QA / ConvFinQA**: These benchmarks focus on financial QA and numerical reasoning; FinReporting targets upstream structuring and localization, converting heterogeneous disclosures into a unified schema.
- **vs Free LLM financial annotator**: Free generation is more flexible but higher risk. The KEEP/REPAIR/NEED_REVIEW design is worth migrating to other high-risk extraction tasks like medical insurance, legal compliance, and auditing.
- **Insight**: The key to high-risk agent systems is not "automating every step," but making the evidence, status, repairs, and failure reasons of every step traceable.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The system design of canonical ontology + guardrailed LLM verifier is solid, even if individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐☆☆ Covers 90 companies across three markets, but the number of fields is relatively small, and no gain was shown for the JP market. Long-tail scenarios are still missing.
- Writing Quality: ⭐⭐⭐☆☆ Clear logic, though some experimental details were compressed.
- Value: ⭐⭐⭐⭐☆ Highly valuable for engineering financial NLP agents and high-risk document structuring, particularly the "LLM as an evidence-based verifier" approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval](towards_effective_in-context_cross-domain_knowledge_transfer_via_domain-invarian.md)
- [\[AAAI 2026\] L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](../../AAAI2026/llm_reasoning/l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)
- [\[ACL 2026\] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)
- [\[ICML 2026\] Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs](../../ICML2026/llm_reasoning/deliberate_evolution_agentic_reasoning_for_sample-efficient_symbolic_regression_.md)
- [\[NeurIPS 2025\] SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction](../../NeurIPS2025/llm_reasoning/sql-of-thought_multi-agentic_text-to-sql_with_guided_error_correction.md)

</div>

<!-- RELATED:END -->
