---
title: >-
  [Paper Note] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals
description: >-
  [NeurIPS 2025][Social Computing][RAG robustness] This paper introduces RAGuard, the first benchmark dataset to systematically evaluate the robustness of RAG systems against misleading retrieved content. By constructing a…
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "RAG robustness"
  - "fact-checking"
  - "misleading retrieval"
  - "benchmark"
  - "misinformation"
date: 2026-05-08
content_hash: 1403fb8793a2f874
---

# Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals

**Conference**: NeurIPS 2025
**arXiv**: [2502.16101](https://arxiv.org/abs/2502.16101)  
**Code**: [HuggingFace Dataset](https://huggingface.co/datasets/UCSC-IRKM/RAGuard)  
**Area**: Information Retrieval
**Keywords**: RAG robustness, fact-checking, misleading retrieval, benchmark, misinformation

## TL;DR

This paper introduces RAGuard, the first benchmark dataset to systematically evaluate the robustness of RAG systems against misleading retrieved content. By constructing a realistic retrieval corpus from Reddit — containing supporting, misleading, and unrelated documents — it demonstrates that all tested LLM-RAG systems perform **worse than a zero-shot baseline** when exposed to misleading retrievals, whereas human annotators maintain consistent judgment.

## Background & Motivation

Retrieval-Augmented Generation (RAG) is widely regarded as an effective remedy for LLM hallucination, yet existing research contains a critical blind spot:

**Idealized Evaluation Assumptions**: Mainstream RAG benchmarks evaluate under "clean retrieval" settings (using gold-standard documents) or inject noise artificially, neither of which reflects real-world conditions.

**The Challenge of Misleading Information**: In highly polarized domains such as politics, retrieved content may involve selective framing, factual distortion, and bias. The capacity of LLMs to handle such content has not been systematically evaluated.

**Limitations of Existing Datasets**:
   - FEVER/FEVEROUS: Contains only supporting evidence rewritten from Wikipedia.
   - Liar/Mocheg: Either lacks conflicting evidence or relies solely on gold-standard documents.
   - RAAT/NoiserBench: Uses artificial counterfactuals or fabricated noise, lacking realism.
   - QACC: Manually annotated, but conflict is defined as binary, missing subtle misleading forms.

Core Problem: **How does misleading retrieved content in the real world affect the reasoning capability of RAG systems?**

## Method

### Overall Architecture

The RAGuard dataset is constructed in three stages:

1. **Claim and Verdict Collection**: Political claims are scraped from PolitiFact.
2. **Knowledge Base Construction**: Relevant discussions are retrieved from Reddit.
3. **LLM-Guided Annotation**: Document types are labeled based on model behavior.

### Key Design 1: Document Taxonomy

Prior work's inconsistent terminology is unified into a structured classification framework:

- **Supporting**: Documents that help the model reach the correct verdict, including those that supply contextual cues (not limited to direct answer statements).
- **Misleading**: Documents that cause the model to produce an incorrect verdict through framing effects, omissions, or biased presentation. A critical definitional choice: **misleadingness is defined relative to the language model, not to humans** — such content typically contains no overt fallacies but presents facts in subtly distorting ways.
- **Unrelated**: Topically relevant but uninformative documents.

The key distinction from prior datasets: misleading information originates from **naturally occurring** user-generated content rather than artificial synthesis.

### Key Design 2: Dataset Construction Pipeline

**Claim Collection**: Restricted to statements made by U.S. presidential candidates between 2000 and 2024; PolitiFact's 6-level veracity scale (true → pants on fire) is binarized into true/false.

**Knowledge Base Construction**:
- GPT-4 is used to generate keyword variants for each claim.
- Google Search is restricted to Reddit, collecting the top-10 posts per claim.
- Reddit is chosen as the data source because user-generated content naturally contains persuasive writing regardless of factual accuracy.

**LLM-Guided Annotation**:
- The RAG pipeline is simulated by providing GPT-4 with a single document as context for claim verification.
- If the document helps the model reach the correct verdict → **Supporting**.
- If the document causes the model to reach an incorrect verdict → **Misleading**.
- If the model deems the document irrelevant to the verdict → **Unrelated**.

Cross-model validation: a subset is re-annotated with Claude 3.5 Sonnet ($\kappa = 0.789$) and Gemini 1.5 Flash ($\kappa = 0.650$) to confirm annotation consistency.

### Key Design 3: Evaluation Task Design

1. **Zero-Context Prediction**: Zero-shot baseline without retrieval.
2. **Standard RAG**: Real-time retrieval from the corpus (RAG-1 takes top-1; RAG-5 takes top-5).
3. **Oracle Retrieval (All)**: All associated documents (supporting/misleading/unrelated) are provided directly.
4. **Oracle Retrieval (Misleading)**: Only misleading documents are provided.

### Dataset Statistics

- Total claims: 2,648 (True 50.3% / False 49.7%)
- Total documents: 16,331 (Supporting 16.4% / Misleading 11.1% / Unrelated 72.5%)
- Average documents per claim: 6.2
- Claims with supporting documents: 36.1%
- Claims with misleading documents: 29.8%

## Key Experimental Results

### Main Results: Multi-Model Multi-Setting Comparison

| Setting | OLMo-1B | Llama 3 | Mistral | Gemini 1.5 | GPT-4o | Claude 3.5 | DeepSeek R1 | o4-mini |
|---|---|---|---|---|---|---|---|---|
| Zero-Context | 56.87 | 62.50 | 63.97 | 61.06 | 67.33 | **74.51** | 69.98 | 63.67 |
| RAG-1 | 52.68 | 59.40 | 59.14 | 56.68 | 64.80 | 70.09 | 66.88 | 62.76 |
| RAG-5 | 49.74 | 61.37 | 58.91 | 57.59 | 65.90 | 68.58 | 57.81 | 63.14 |
| Oracle (All) | 53.89 | 61.09 | 51.55 | 52.38 | 53.22 | 52.56 | 50.06 | 51.88 |
| Oracle (Misleading) | 44.04 | 36.81 | 26.88 | 30.57 | **45.97** | 35.98 | 38.25 | 33.39 |

All models achieve their highest accuracy under Zero-Context — **retrieval consistently degrades performance**.

### Ablation Analysis: Impact of Misleading Documents

| Analysis Dimension | Key Result |
|---|---|
| Average accuracy drop under Oracle (Misleading) | **46.5%** (relative to zero-shot) |
| Oracle (Misleading) accuracy across all models | **All below 50%** (binary classification task) |
| Misleading document recall under RAG-1 | 21.3% |
| Misleading document recall under RAG-5 | 44.8% |
| CRAG method vs. Llama 2 zero-shot | 37.24% vs. 50.57% (worse) |

### Human Study

| Evaluator Type | Zero-shot Accuracy | Accuracy with Documents | Change |
|---|---|---|---|
| Experts (PhD-level) | High | High (stable) | Negligible |
| Lay participants | Moderate | Higher | **Positive gain** |
| LLMs (multi-model average) | Highest | Lower | **Negative drop** |

Core contrast: **Humans benefit from, or are at least unaffected by, additional information, whereas LLMs are misled by the same content.**

### Key Findings

1. **Retrieval is not always helpful**: All RAG variants underperform the zero-shot baseline, challenging the assumption that retrieval necessarily improves accuracy.
2. **More retrieval ≠ better performance**: RAG-5 frequently underperforms RAG-1, especially for stronger models such as Claude and DeepSeek.
3. **GPT-4o is most robust**: It drops only 31.7% under Oracle (Misleading), compared to 51.8% for Claude and 58.0% for Mistral.
4. **Strong zero-shot ≠ strong robustness**: Claude 3.5 achieves the highest zero-shot accuracy yet exhibits the largest robustness degradation.
5. **Reasoning models are not immune**: o4-mini and DeepSeek R1 drop 45.3% and 47.6%, respectively.
6. **Existing robustness methods are ineffective**: CRAG, designed to mitigate retrieval errors, actually performs worse than zero-shot (37.24% vs. 50.57%).

## Highlights & Insights

1. **First benchmark with realistic misleading retrievals**: Rather than relying on synthetic noise, RAGuard captures naturally occurring misleading information in Reddit user-generated content, marking a milestone in RAG robustness evaluation.
2. **Model-behavior-centric definition of misleadingness**: Defining "misleading" in terms of its effect on model predictions rather than human judgment is a novel contribution that makes the dataset targeted and scalable.
3. **Counterintuitive core finding**: Retrieval augmentation **consistently degrades performance** across all models, directly challenging a foundational assumption of RAG research.
4. **Quantification of the human–machine reasoning gap**: The starkly different responses of humans and LLMs to the same misleading information reveal a fundamental weakness in LLM reasoning.
5. **Two characteristic LLM failure modes**:
    - *Conflating opinion with fact*: The model misreads subjective tone as factual evidence.
    - *Lack of temporal reasoning*: The model fails to distinguish information from different time points.

## Limitations & Future Work

1. **GPT-4 annotation bias**: Although mitigated through cross-model consistency validation, labeling based on a single model's behavior carries residual bias risk.
2. **Domain specificity**: Only political claims are covered; applicability to other high-stakes domains such as medicine and law remains unverified.
3. **Small-scale human study**: Only 4 annotators and a 64-instance subset make strong statistical conclusions difficult.
4. **Binarized verdicts**: Simplifying PolitiFact's 6-level scale to binary labels discards nuance in intermediate ratings.
5. **Single source for misleading content**: Misleading information is drawn exclusively from Reddit, excluding other platforms such as Twitter/X and news websites.
6. **No proposed remedies**: The paper diagnoses the problem but does not propose effective robustification methods.

## Related Work & Insights

- **Distinction from AVeriTeC**: AVeriTeC requires models to **abstain** when confronted with conflicting evidence, whereas RAGuard requires models to **reason through misleading information to reach the correct verdict** — a stricter and more practically relevant task.
- **Distinction from RAAT/NoiserBench**: The latter employs artificial counterfactuals (e.g., direct numerical substitution), whereas RAGuard's misleading content consists of naturally occurring, subtly distorted information.
- **Implications for RAG system design**: The field needs to shift from "more retrieval = better" toward retrieval quality awareness, developing mechanisms capable of assessing the reliability of retrieved documents.
- **Adversarial retrieval training**: Exposing models to misleading evidence during training may improve robustness.
- **Multi-step reasoning and document subjectivity classification**: Post-retrieval classification of document subjectivity may help mitigate misleading effects.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic real-world misleading retrieval RAG benchmark; problem definition is clear and significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 8 models, 5 settings, human comparison; comprehensive coverage, with minor deduction for the small-scale human study.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous terminology unification, clear taxonomy, and persuasive argumentation.
- Value: ⭐⭐⭐⭐⭐ — Directly addresses a core weakness of RAG systems; dataset is publicly available and holds significant value for advancing robust RAG research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[NeurIPS 2025\] Noise-Robustness Through Noise: A Framework Combining Asymmetric LoRA with Poisoning MoE](noise-robustness_through_noise_a_framework_combining_asymmetric_lora_with_poison.md)
- [\[ACL 2026\] VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking](../../ACL2026/social_computing/veritas_the_first_dynamic_benchmark_for_multimodal_automated_fact-checking.md)
- [\[AAAI 2026\] Fact2Fiction: Targeted Poisoning Attack to Agentic Fact-checking System](../../AAAI2026/social_computing/fact2fiction_targeted_poisoning_attack_to_agentic_fact-check.md)

</div>

<!-- RELATED:END -->
