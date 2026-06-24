---
title: >-
  [Paper Note] Towards Robust ESG Analysis Against Greenwashing Risks: A3CG
description: >-
  [ACL 2025][LLM (Other)][ESG analysis] This work proposes the A3CG dataset and the Aspect-Action Analysis task (extracting aspects and their action types from sustainability claims: Implemented, Planning, or Indeterminate) to evaluate the robustness of NLP methods against greenwashing risks under cross-category generalization settings. It finds that supervised learning methods (GRACE F1 = 47.51) outperform LLMs (Claude 3.5 F1 = 42.03) but exhibit worse generalization efficienc…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "ESG analysis"
  - "Greenwashing detection"
  - "Aspect-Action analysis"
  - "Cross-category generalization"
  - "Sustainability reports"
date: 2026-05-08
content_hash: 081c949de22964f7
---

# Towards Robust ESG Analysis Against Greenwashing Risks: A3CG

**Conference**: ACL 2025  
**arXiv**: [2502.15821](https://arxiv.org/abs/2502.15821)  
**Code**: [https://github.com/keanepotato/a3cg_greenwash](https://github.com/keanepotato/a3cg_greenwash)  
**Area**: Others  
**Keywords**: ESG analysis, Greenwashing detection, Aspect-Action analysis, Cross-category generalization, Sustainability reports  

## TL;DR
This work proposes the A3CG dataset and the Aspect-Action Analysis task (extracting aspects and their action types from sustainability claims: Implemented, Planning, or Indeterminate) to evaluate the robustness of NLP methods against greenwashing risks under cross-category generalization settings. It finds that supervised learning methods (GRACE F1 = 47.51) outperform LLMs (Claude 3.5 F1 = 42.03) but exhibit worse generalization efficiency.

## Background & Motivation
**Background**: Sustainability reports serve as key tools for evaluating corporate ESG performance. While NLP methods (topic analysis, RAG, etc.) can automatically extract ESG insights, existing methods fail to consider greenwashing—instances where companies publish misleading or exaggerated sustainability claims.

**Limitations of Prior Work**: Insights extracted by current NLP methods may reflect vague rhetoric rather than verifiable sustainability actions. Simultaneously, corporations may strategically alter report contents to favor certain areas, causing models trained on specific categories to fail.

**Key Challenge**: Merely extracting "aspects" (e.g., "carbon emissions") is insufficient; it is also necessary to determine the action type associated with that aspect—whether it is "implemented", "planned", or "indeterminate". The latter is a primary characteristic of greenwashing.

**Goal**: (1) How to extract fine-grained aspect-action pairs from sustainability claims? (2) Can models generalize when reports involve sustainability categories unseen during training?

**Key Insight**: Transferring the ABSA (Aspect-Based Sentiment Analysis) paradigm to the ESG domain by replacing "sentiment" with "action type," while introducing cross-category generalization evaluation.

**Core Idea**: Decomposing sustainability claims into (aspect, action) pairs and distinguishing three action types transparently flags "indeterminate" actions as potential greenwashing, anchoring ESG analysis to verifiable actions instead of vague rhetoric.

## Method

### Overall Architecture
A3CG Task: Given a sustainability claim, extract all (aspect, action) pairs. The aspect is a sustainability entity/target/activity mentioned in the claim (e.g., "carbon emissions reduction"), and the action is a three-way classification: Implemented, Planning, or Indeterminate. The latter highlights vague/non-committal language associated with greenwashing risks.

### Key Designs

1. **Dataset Construction (A3CG)**:

    - **Function**: Collects 2,004 claims from 1,679 sustainability reports of SGX-listed companies (2017-2022).
    - **Mechanism**: Employs 5 sustainability PhD/postdoc annotators and 3 verifiers. Annotators first underwent multiple rounds of trial annotation until reaching an accuracy $\ge 95\%$, followed by daily annotations, discussing indeterminate cases every 3 days (majority vote), and auditing a random 20% sample every 3 days for verification.
    - **Design Motivation**: 2,723 aspect-action pairs covering 10 sustainability categories. 33.2% of claims contain no aspect-action pairs (simulating real-world scenarios). Action distribution: Implemented (53.6%), Indeterminate (32.5%), Planning (13.9%).

2. **Cross-Category Generalization Evaluation Design**:

    - **Function**: Tests model performance on sustainability categories unseen during training.
    - **Mechanism**: Splits the dataset into 3 folds, where each fold excludes 3–4 categories to construct the Unseen Test set (US), alongside a control Seen Test set (S) of seen categories. Excluded categories across the 3 folds do not overlap to ensure comprehensive coverage.
    - **Design Motivation**: Companies may emphasize different ESG areas in different years, meaning models need to remain effective on newly emerging sustainability topics. Distinct from cross-domain generalization, cross-category represents distribution shifts within the same domain.

3. **Contrastive Learning vs. Adversarial Learning**:

    - **Function**: Evaluates the efficacy of two generalization strategies under the cross-category setting.
    - **Mechanism**: Contrastive Learning (CL) uses supervised contrastive loss pre-training to cluster same-category samples and disperse different-category samples; Adversarial Learning (AL) utilizes a gradient reversal layer to learn category-invariant features.
    - **Design Motivation**: Generalization under CL consistently outperforms AL (T5+CL US Avg = 43.99 vs. T5+AL = 42.82) because CL learns transferable category-distinguishing features, whereas AL's feature collapse may suppress useful category-specific information.

## Key Experimental Results

### Main Results
Average F1 of Aspect-Action Analysis (AAA) on unseen categories:

| Method Type | Method | US Avg F1 | $\Delta$ (US-S) |
|----------|------|-----------|----------|
| Supervised | GRACE | **47.51** | -14.45 |
| Supervised | CONTRASTE | 46.34 | -21.13 |
| Supervised | T5+CL | 43.99 | -22.91 |
| LLM | Claude 3.5+FS | 42.03 | -0.92 |
| LLM | DeepSeek V3 | 41.08 | -0.88 |
| LLM | GPT-4o+FS | 40.41 | +0.99 |
| LLM | Llama 3 70B | 20.67 | -0.64 |

### Sub-task Analysis

| Finding | Details |
|------|------|
| ATE vs. AC Difficulty | Action Classification (AC) F1 is significantly lower than Aspect Term Extraction (ATE) across all methods. |
| Main Issues of Supervised Models | ATE recall plunges (even semantically obvious sustainability aspects in unseen categories are undetected). |
| Main Issues of LLMs | Recall for environmental aspects is higher than non-environmental ones (pre-training bias), and pragmatic reasoning is weak. |
| CL vs. AL | CL outperforms AL consistently, achieving US +2.87 on T5 and +11.87 on BERT-ST. |

### Key Findings
- **Supervised models achieve the best absolute performance but the worst generalization efficiency**: GRACE achieves the highest F1 of 47.51, but drops by $\Delta = -14.45$ (from seen to unseen), whereas the $\Delta$ for LLMs is nearly 0.
- **LLMs exhibit environmental biases**: ATE recall is systematically higher in environment-related categories (resource optimization, emission control, environmental protection) than in non-environmental categories, likely due to stereotypical associations of "sustainability $\approx$ environmental protection" in pre-training data.
- **Pragmatic reasoning is a bottleneck for LLMs**: Modality misinterpretation (treating hesitant expressions as firm commitments), negation handling, unattributed claims, and future-dependent judgments—all of which are common linguistic strategies in greenwashing.
- **Syntactic reasoning is a bottleneck for supervised models**: Ellipses and ambiguous syntax lead to action classification errors.

## Highlights & Insights
- **Transferring the ABSA paradigm to the ESG domain is a natural and effective design**: Replacing "sentiment polarity" with "action type" allows mature ABSA methods to be reused, lowering the barrier to methodological development.
- **Cross-category generalization is an overlooked yet crucial evaluation dimension**: Unlike cross-domain settings, category shift within the same domain is harder to detect but equally impacts performance. This evaluation setup can be generalized to other domains.
- **Complementary error patterns of supervised models and LLMs are identified**: Supervised models excel in pragmatics but struggle with syntax, while LLMs exhibit the opposite behavior. This suggests that hybrid methods could combine their strengths.

## Limitations & Future Work
- Only covers English sustainability reports, lacking multilingual support.
- Does not test the effect of LoRA fine-tuning on smaller LLMs (e.g., Llama 3 8B).
- The "indeterminate" action category does not equate directly to greenwashing; real greenwashing detection requires more external verification.
- Data sources are limited to SGX-listed companies, offering limited sectoral and regional representation.

## Related Work & Insights
- **vs. Stammbach et al. (2022)**: Environmental claim detection only performs binary classification (whether it is an environmental claim), whereas A3CG is more fine-grained, simultaneously extracting aspects and identifying action types.
- **vs. Standard ABSA Datasets (e.g., Rest15)**: A3CG has a comparable scale (2,004 vs. ~1,500) but differs in domain and label hierarchy, validating the transferability of ABSA methods.
- Application of NLP in the financial and ESG domains is increasingly crucial; A3CG fills the dataset gap for robust anti-greenwashing analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ The first ESG aspect-action analysis dataset targeting greenwashing risks, featuring a novel cross-category generalization design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison between supervised methods and LLMs, featuring learning paradigm ablations and detailed error analysis.
- Writing Quality: ⭐⭐⭐⭐ Structured clearly with in-depth error analysis and rich tables/figures.
- Value: ⭐⭐⭐ Although the application scope is relatively narrow (ESG report analysis), the methodology serves as a reference for other fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SDD: Self-Degraded Defense against Malicious Fine-tuning](sdd_self-degraded_defense_against_malicious_fine-tuning.md)
- [\[ACL 2025\] Enabling LLM Knowledge Analysis via Extensive Materialization](enabling_llm_knowledge_analysis_via_extensive_materialization.md)
- [\[ACL 2025\] Concreteness Versus Abstractness: A Selectivity Analysis in LLMs](concreteness_versus_abstractness_a_selectivity_analysis_in_llms.md)
- [\[ACL 2025\] Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs](sarft_roleplay_safety.md)
- [\[ACL 2025\] Robust Utility-Preserving Text Anonymization Based on Large Language Models](robust_utility-preserving_text_anonymization_based_on_large_language_models.md)

</div>

<!-- RELATED:END -->
