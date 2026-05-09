---
title: >-
  [Paper Note] ExpGuard: LLM Content Moderation in Specialized Domains
description: >-
  [ICLR2026][Medical Imaging][LLM safety] This paper proposes ExpGuard, a safety guardrail model targeting specialized domains such as finance, healthcare, and law, along with a companion dataset ExpGuardMix (58,928 samples). ExpGuard achieves prompt classification F1 exceeding WildGuard by 8.9% and response classification by 15.3% on domain-specific test sets, while maintaining state-of-the-art performance on general safety benchmarks.
tags:
  - ICLR2026
  - Medical Imaging
  - LLM safety
  - guardrail model
  - content moderation
  - domain-specific
  - financial/medical/legal
date: 2026-05-08
content_hash: 4fd5853246aca8a3
---

# ExpGuard: LLM Content Moderation in Specialized Domains

**Conference**: ICLR2026  
**arXiv**: [2603.02588](https://arxiv.org/abs/2603.02588)  
**Code**: [brightjade/ExpGuard](https://github.com/brightjade/ExpGuard)  
**Area**: Medical Imaging  
**Keywords**: LLM safety, guardrail model, content moderation, domain-specific, financial/medical/legal

## TL;DR
This paper proposes ExpGuard, a safety guardrail model targeting specialized domains such as finance, healthcare, and law, along with a companion dataset ExpGuardMix (58,928 samples). ExpGuard achieves prompt classification F1 exceeding WildGuard by 8.9% and response classification by 15.3% on domain-specific test sets, while maintaining state-of-the-art performance on general safety benchmarks.

## Background & Motivation
As LLMs are increasingly deployed in high-stakes specialized domains such as finance, healthcare, and law, existing safety guardrail models face significant challenges:

- **Blind spots of general-purpose guardrails**: Existing guardrails (e.g., Llama-Guard, WildGuard) are primarily designed for general human–computer interaction scenarios and lack comprehension of domain-specific terminology and concepts. For instance, the financial term "haircut" (a discount applied to asset valuation) can be exploited in adversarial prompts to bypass general-purpose guardrails.
- **Near-complete failure of API-based tools**: Detoxify, Perspective API, and OpenAI Moderation achieve only 0.3%–14.1% F1 on domain-specific test sets, rendering them nearly incapable of identifying domain-specific harmful content.
- **Limitations of internal alignment**: Techniques such as RLHF are resource-intensive and difficult to extend to domain-specific risks; external guardrail models thus serve as a necessary complementary layer.

## Core Problem
How to build a safety guardrail model that handles both general safety detection and the effective identification of harmful content disguised through technical terminology in specialized domains such as finance, healthcare, and law?

## Method

### 1. ExpGuardMix Dataset Construction (58,928 Samples)

The dataset is divided into ExpGuardTrain (56,653 samples for training) and ExpGuardTest (2,275 samples for evaluation).

**Phase 1: Domain Terminology Mining**

- Domain-specific terms are extracted by recursively crawling Wikipedia category pages in finance, healthcare, and law.
- The Wikidata API is used to filter out non-technical entities (e.g., person names, organizations, countries).
- GPT-4o is applied to exclude non-sensitive or irrelevant terms.
- Manual verification is conducted by three annotators via majority vote, yielding a final set of 2,646 terms (finance: 989, healthcare: 1,012, law: 645).

**Phase 2: Prompt and Response Generation**

- **Harmful prompts**: For each term, GPT-4o generates harmful prompts targeting the risk scenarios associated with that term. Safety mechanisms are bypassed by prepending the prefix "I have an idea for a prompt:". Both long and short variants are generated, with random sampling from 100+ preset instruction templates and few-shot examples.
- **Benign prompts**: Wikipedia documents are converted into instruction–response pairs, retaining only the instruction component. Although these involve sensitive topics, they are inherently safe and are included to mitigate over-refusal behavior.
- **In-the-wild data**: Subsampled from LMSYS-Chat-1M and WildChat, supplemented with DAN jailbreak prompts and human-written data from HH-RLHF and Aegis 2.0.
- **Response generation**: Compliant responses are generated using Mistral-7B-Instruct-v0.1 (an older model more prone to following harmful instructions), while refusal responses are generated using Gemma-3-27B-IT.

**Phase 3: Classification Annotation and Filtering**

- Thirteen harmful categories plus one "harmless" pseudo-category are defined, covering violence, pornography, discrimination, privacy violations, financial fraud, illegal substances, and others.
- Three-model ensemble annotation is performed using Claude 3.7 Sonnet, Gemini 2.0 Flash, and Qwen2.5-Max, each required to produce chain-of-thought reasoning before assigning a category label.
- **Strict consensus filtering**: At least 2 out of 3 models must assign exactly the same category index (not merely "safe/unsafe"); 4.8% of ambiguous samples are discarded.
- Near-duplicate samples with Sentence-BERT cosine similarity > 0.9 are removed.

### 2. ExpGuardTest (2,275 Samples)

- Distribution: finance (964), healthcare (771), law (540).
- Initially annotated via LLM ensemble, then verified by domain experts.
- The finance subset is reviewed by banking professionals, achieving Cohen's Kappa of 0.89 (prompt) / 0.98 (response), indicating near-perfect agreement.

### 3. ExpGuard Model Training

- Fine-tuned from a 7B-parameter LLM on ExpGuardTrain in a multi-task setting.
- When only a prompt is provided, the model predicts prompt harmfulness; when a prompt–response pair is provided, it predicts the harmfulness of both.
- Output is a binary label (safe/unsafe).

## Key Experimental Results

### Main Results on ExpGuardTest (F1%)

| Model | Prompt F1 (Overall) | Response F1 (Overall) |
|-------|--------------------|-----------------------|
| Detoxify / Perspective / OpenAI Mod | 0.3–0.5 | 0.6 |
| Azure | 14.1 | 2.6 |
| Llama-Guard3 (8B) | 71.1 | 84.2 |
| Aegis-Guard-D (7B) | 82.9 | 87.2 |
| WildGuard (7B) | 84.4 | 77.4 |
| **ExpGuard (7B)** | **93.3** | **92.7** |

- Prompt classification surpasses WildGuard by **+8.9%**; response classification by **+15.3%**.
- ExpGuard leads across all three sub-domains: finance, healthcare, and law.

### Results on Public Safety Benchmarks (Average F1% across 8 Benchmarks)

| Model | Prompt Avg. | Response Avg. |
|-------|-------------|---------------|
| WildGuard | 84.2 | 78.8 |
| **ExpGuard** | **85.7** | 78.5 |

- ExpGuard matches or slightly surpasses state-of-the-art on general benchmarks, demonstrating that domain specialization does not compromise general-purpose safety performance.

### Ablation Study

- Removing domain-specific data: ExpGuardTest prompt F1 drops from 93.3% to 85.3% (−8.0%).
- Removing in-the-wild data: public benchmark prompt F1 drops from 85.7% to 84.1%.
- Removing human-written data: public benchmark response F1 drops from 78.5% to 73.9% (largest impact).

### Jailbreak Robustness

- ExpGuard remains competitive under standard jailbreak attacks (CipherChat, AutoDAN-Turbo, FlipAttack, GASP).
- The ExpGuard+ variant, augmented with 270 domain-specific adversarial samples, significantly outperforms all baselines on domain-specific jailbreak attacks.

## Highlights & Insights

1. **First safety guardrail dataset and model targeting specialized domains**: Fills a critical gap in LLM content moderation for finance, healthcare, and law.
2. **Reusable data construction pipeline**: The Wikipedia-based terminology mining → LLM generation → three-model ensemble annotation → expert verification pipeline is extensible to other domains.
3. **Rigorous quality control**: Three-model exact category consensus (rather than binary safe/unsafe consensus) combined with domain expert validation of the finance subset (Kappa 0.89/0.98).
4. **Domain specialization without general degradation**: ExpGuard achieves substantial gains on ExpGuardTest while maintaining or exceeding state-of-the-art across 8 public benchmarks.
5. **Quantifies the severe inadequacy of API-based tools**: Empirically demonstrates that mainstream APIs are nearly completely ineffective in professional scenarios.

## Limitations & Future Work

- **Limited domain coverage**: Only finance, healthcare, and law are covered; other specialized domains (e.g., cybersecurity, chemical engineering) remain to be addressed.
- **English-only support**: Multilingual domain-specific content moderation is an important direction for future work.
- **Limitations of synthetic data**: Despite various augmentation strategies, synthetic data may not fully capture the diversity of real-world user interactions.
- **Need for dynamic updates**: Harmful content and adversarial techniques evolve rapidly, necessitating continuous dataset updates.
- **Incomplete expert validation**: Only the finance subset underwent expert review; reliability of the healthcare and law subsets is inferred from the LLM ensemble annotation.

## Related Work & Insights

| Dimension | WildGuard | Llama-Guard Series | ExpGuard |
|-----------|-----------|-------------------|----------|
| Domain Coverage | General | General | General + Finance/Healthcare/Law |
| Training Data | WildGuardMix (92K) | Internal safety data | ExpGuardMix (58.9K) |
| Domain-specific F1 | 84.4 / 77.4 | 71.1 / 84.2 | **93.3 / 92.7** |
| General Benchmarks | **84.2** / 78.8 | 78.9 / 66.8 | 85.7 / 78.5 |
| Data Construction | LLM generation + in-the-wild | Undisclosed | Term mining + RAG generation + expert verification |

A key distinction from the "generate-then-filter" pipelines of An et al. (2024) and Cui et al. (2025) is that those works focus on reducing false positives (over-refusal), whereas this paper targets reducing false negatives (missed harmful content), and additionally introduces domain expert verification.

The terminology mining → RAG generation → multi-model ensemble annotation → expert verification pipeline is highly transferable and applicable to constructing safety datasets in domains such as cybersecurity and biochemistry. The experiments compellingly demonstrate the necessity of open-source LLM guardrail models over commercial APIs in professional scenarios. ExpGuard, as an external moderation layer, complements internal alignment methods such as RLHF to form a dual-safety architecture suitable for industrial deployment.

## Rating
- Novelty: 8/10 — First systematic treatment of domain-specific LLM safety guardrails; the data construction methodology is innovative.
- Experimental Thoroughness: 9/10 — Comprehensive evaluation across 13 baselines, 9 benchmarks, ablation studies, and jailbreak analysis.
- Writing Quality: 8/10 — Well-structured with detailed pipeline descriptions and rich figures and tables.
- Value: 8/10 — Addresses an important gap, though domain and language coverage remain limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](../../CVPR2026/medical_imaging/moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [\[ICLR 2026\] Scaling with Collapse: Efficient and Predictable Training of LLM Families](scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [\[NeurIPS 2025\] Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](../../NeurIPS2025/medical_imaging/pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)
- [\[ICLR 2026\] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [\[ACL 2026\] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](../../ACL2026/medical_imaging/can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)

</div>

<!-- RELATED:END -->
