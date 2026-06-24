---
title: >-
  [Paper Note] AIMSCheck: Leveraging LLMs for AI-Assisted Review of Modern Slavery Statements Across Jurisdictions
description: >-
  [ACL 2025][LLM (Other)][Compliance Detection] This work proposes AIMSCheck—an end-to-end framework for corporate modern slavery statement compliance assessment. It decomposes the evaluation task into three tiers: sentence-level multi-label classification, token-level SHAP explanation, and evidence status tracking. It also constructs two new annotated datasets, AIMS.uk and AIMS.ca, validating that models fine-tuned on Australian data can effectively generalize across jurisdict…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Compliance Detection"
  - "Modern Slavery"
  - "Cross-Jurisdictional Generalization"
  - "Multi-Label Sentence Classification"
  - "SHAP Explainability"
  - "Evidence Tracking"
date: 2026-05-08
content_hash: 72f6d246663d0f9d
---

# AIMSCheck: Leveraging LLMs for AI-Assisted Review of Modern Slavery Statements Across Jurisdictions

**Conference**: ACL 2025  
**arXiv**: [2506.01671](https://arxiv.org/abs/2506.01671)  
**Area**: LLM/NLP, Legal Compliance  
**Keywords**: Compliance Detection, Modern Slavery, Cross-Jurisdictional Generalization, Multi-Label Sentence Classification, SHAP Explainability, Evidence Tracking

## TL;DR

This work proposes AIMSCheck—an end-to-end framework for corporate modern slavery statement compliance assessment. It decomposes the evaluation task into three tiers: sentence-level multi-label classification, token-level SHAP explanation, and evidence status tracking. It also constructs two new annotated datasets, AIMS.uk and AIMS.ca, validating that models fine-tuned on Australian data can effectively generalize across jurisdictions.

## Background & Motivation

**Background**: More than 50 million people worldwide are affected by modern slavery. The UK (2015), Australia (2018), and Canada (2024) have sequentially enacted Modern Slavery Acts (MSA), requiring large corporations to publish annual statements disclosing their efforts to combat modern slavery in their operations and supply chains. The three countries expect approximately 12,000, 3,000, and 6,000 statements, respectively, to be submitted to government registries annually.

**Limitations of Prior Work**:
1. **Bottleneck of Manual Review**: With approximately 80,000 cumulative statements globally, manual review scales poorly—it took WikiRate 8 years to annotate just 3,500 statements.
2. **Scarcity of Annotated Data**: Most studies only manually review 100-200 statements, and NLP annotated datasets remain extremely scarce.
3. **Uncertainty of Cross-Jurisdictional Generalization**: The only previously available dataset, AIMS.au, covers only Australia, leaving its generalizability to other legal systems unverified.
4. **Lack of End-to-End Systems**: Existing approaches lack interpretability and evidence tracking capabilities; simple binary classification is insufficient to support real-world compliance decision-making.

**Mechanism**: Construct cross-jurisdictional annotated datasets and design a multi-tier end-to-end evaluation framework. This framework decomposes compliance assessment into a workflow of sentence classification $\rightarrow$ token explanation $\rightarrow$ evidence status tracking, supporting human-in-the-loop collaborative review.

## Method

### Overall Architecture

AIMSCheck consists of three concatenated levels:

1. **Sentence-Level Prediction**: Performs multi-label binary classification on each sentence in a statement to determine its relevance to nine compliance criteria.
2. **Token-Level Explanation**: Uses SHAP values to quantify the contribution of each token to the classification decision, providing explainability support.
3. **Evidence Status Tracking**: Further classifies relevant sentences into "implemented", "future commitment", or "disclosed denial/omission", supporting longitudinal compliance monitoring.

### Key Designs

#### Key Design 1: Cross-Jurisdictional Dataset Construction and Criteria Mapping

- **AIMS.uk** (50 UK statements, 2,807 sentences) and **AIMS.ca** (50 Canadian statements, 3,658 sentences) were constructed, following the same preprocessing and annotation guidelines as AIMS.au.
- Legal experts established a mapping of compliance criteria across the three jurisdictions, extracting **9 common criteria** from their respective regulations (e.g., statement approval, signature, description of organizational structure/operations/supply chains, risk identification, risk mitigation/remediation, and effectiveness assessment).
- Annotation achieved Cohen's Kappa = 0.776, Jaccard Similarity = 0.813, indicating "substantial agreement".

#### Key Design 2: Sentence-Level Classification with Multi-Model Comparison

- Fine-tuned models: BERT (full parameter) and Llama3.2 3B (LoRA), trained on the AIMS.au training set.
- Zero-shot/Few-shot models: GPT-4o (zero-shot + CoT + few-shot), DeepSeek-R1 (2.51-bit quantized).
- Two input configurations: no context vs. 100-word context window (50 words before and after the target sentence).
- Core finding: Fine-tuned models consistently outperform zero-shot/few-shot models, with a 100-word context offering the optimal balance between performance and efficiency.

#### Key Design 3: Evidence Status Tracking Mechanism

- **Future Action Detection**: Based on an NLTK tense classifier combined with keyword matching (e.g., "plan to", "aim to").
- **Negative Evidence Detection**: Uses a zero-shot BART-MNLI model to construct two hypotheses for each sentence (denial/avoidance vs. acknowledgment), selecting the one with the higher probability. The classification threshold is lowered from 0.5 to 0.35 to increase sensitivity.
- This design allows reviewers to distinguish among three states—"existing policy", "planned implementation", and "explicit denial"—supporting longitudinal compliance trend analysis.

## Key Experimental Results

### Table 1: Overall F1 Score Comparison Across Models

| Model | Context Words | AIMS.au | AIMS.ca | AIMS.uk |
|------|-----------|---------|---------|---------|
| **Llama3.2 3B** | 100 | **0.738** | **0.719** | **0.686** |
| Llama3.2 3B | 0 | 0.726 | 0.716 | 0.672 |
| BERT | 100 | 0.719 | 0.700 | 0.669 |
| BERT | 0 | 0.694 | 0.677 | 0.653 |
| GPT-4o CoT (few-shot) | 100 | 0.617 | 0.614 | 0.573 |
| GPT-4o | 100 | 0.601 | 0.582 | 0.542 |
| DeepSeek-R1 | 100 | 0.548 | 0.550 | 0.505 |

### Table 2: Category-wise F1 of the Best Model (Llama3.2 3B + 100-word context)

| Compliance Criteria | AIMS.au | AIMS.ca | AIMS.uk |
|---------|---------|---------|---------|
| Statement Approval (Approval) | 0.864 | 0.947 | 0.783 |
| Description of Supply Chain | 0.805 | 0.656 | 0.704 |
| Signature (Signature) | 0.790 | 0.816 | 0.686 |
| Description of Operations (Operations) | 0.769 | 0.803 | 0.789 |
| Organizational Structure (Structure) | 0.749 | 0.741 | 0.773 |
| Risk Description | 0.738 | 0.596 | 0.622 |
| Risk Mitigation (Mitigation) | 0.669 | 0.674 | 0.646 |
| Remediation Measures (Remediation) | 0.667 | 0.567 | 0.651 |
| Effectiveness Assessment (Effectiveness) | 0.592 | 0.526 | 0.525 |

**Key Findings**:
- Fine-tuned models exhibit only a slight performance decline ($<5\%$) when transferred from Australia to the UK/Canada, demonstrating strong cross-jurisdictional generalization.
- Well-defined criteria (such as Approval) show high performance, while subjective and ambiguous criteria (such as Effectiveness) remain challenging.
- Jensen-Shannon divergence analysis confirms that the vocabulary distribution gap between training and testing sets is minimal, explaining the stable cross-domain performance.

## Highlights & Insights

- **First Cross-Jurisdictional Compliance Detection Framework**: Simultaneously covers Australian, UK, and Canadian laws, establishing a methodology for criteria mapping.
- **Highly Practical Three-Tier Decomposed Design**: Sentence classification $\rightarrow$ token explanation $\rightarrow$ evidence tracking forms a collaborative workflow of "AI assistance + human final review".
- **Validated Cross-Domain Generalization Feasibility**: Training solely on Australian data effectively serves the UK and Canada, significantly reducing annotation costs.
- **Calibration Analysis Shows Reliable Fine-Tune Model Probabilities**: Predicted probabilities closely align with empirical accuracy, allowing direct use as confidence metrics.
- **Open-Sourced Datasets and Framework**: Datasets are released on HuggingFace, with code and model weights publicly available.

## Limitations & Future Work

- **Limited Dataset Scale**: Only 50 statements each for the UK and Canada, representing a very small subset of the total volume.
- **Single Annotator**: Although iteratively refined, the annotation was primarily conducted by a single domain expert, presenting a risk of subjective bias.
- **English Only**: The three legal jurisdictions happen to be English-speaking, leaving multilingual scenarios unaddressed (e.g., France's Corporate Duty of Vigilance Law, Germany's Supply Chain Act).
- **Fragile Preprocessing Pipeline**: Errors in OCR and sentence segmentation lead to formatting issues in bullet points, lists, and signatures, impacting classification accuracy.
- **Simplistic Evidence Tracking**: Future action detection relies on basic tense/keyword matching, and negative evidence detection remains insufficient under complex corporate jargon.
- **Difficulty in Distinguishing Similar Criteria**: Higher confusion rates are observed between closely related criteria, such as "Organizational Structure" vs. "Operations", and "Risk Description" vs. "Risk Mitigation".

## Related Work

- **Legal NLP Tasks**: LEGAL-BERT (Chalkidis et al., 2020) pre-trained on legal text; LegalBench (Guha et al., 2023) providing legal reasoning benchmarks; ClimateBERT focusing on climate change texts. None of these address modern slavery statement analysis.
- **Modern Slavery Text Analysis**: Nersessian & Pachamanova (2022) used unsupervised topic modeling to analyze trends in UK statements; Bora (2019) analyzed statements with augmented intelligence techniques; AIMS.au (Bora et al., 2025) provided the first sentence-level annotated dataset, but exclusively covered Australia.
- **Fine-Tuning & Prompting Techniques**: LoRA parameter-efficient fine-tuning (Xu et al., 2024), context-aware modeling (Tian et al., 2017; Yang et al., 2021), and Chain-of-Thought reasoning (Wei et al., 2022).
- **Explainability Methods**: SHAP (Lundberg & Lee, 2017) calculates token-level contributions based on game theory; the BERT+SHAP combination (Kokalj et al., 2021) is applied for Transformer classifier explanation.

## Rating

⭐⭐⭐

> This work holds clear real-world significance at the social impact level—scaling compliance review from purely manual processes to AI-assisted large-scale processing. The three-tier decomposed design is pragmatic and possesses strong engineering value. However, the core technical innovation is limited: sentence classification utilizes standard fine-tuning/prompting methods; SHAP explanations and BART-MNLI zero-shot detection are direct applications of existing tools; and the tense/keyword matching method for evidence tracking is oversimplified. While the verification of cross-jurisdictional generalization is valuable, the legal frameworks of the three English-speaking nations are inherently highly similar, which limits the persuasiveness of the model's broader generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs](input_dependent_soft_prompting.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](../../AAAI2026/llm_nlp/position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)

</div>

<!-- RELATED:END -->
