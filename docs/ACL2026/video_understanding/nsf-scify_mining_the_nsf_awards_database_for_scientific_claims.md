---
title: >-
  [Paper Note] NSF-SciFy: Mining the NSF Awards Database for Scientific Claims
description: >-
  [ACL2026][Video Understanding][Scientific Claim Extraction] NSF-SciFy extracts 2.8M scientific claims and investigation proposals from NSF award abstracts…
tags:
  - "ACL2026"
  - "Video Understanding"
  - "Scientific Claim Extraction"
  - "NSF Award Abstracts"
  - "Scientific Feasibility"
  - "LoRA Fine-Tuning"
  - "Metascience"
date: 2026-05-08
content_hash: beade2f4116f91f7
---

# NSF-SciFy: Mining the NSF Awards Database for Scientific Claims

**Conference**: ACL2026  
**arXiv**: [2503.08600](https://arxiv.org/abs/2503.08600)  
**Code**: https://github.com/darpa-scify/NSFSciFy  
**Area**: Scientific Text Mining / Dataset Construction  
**Keywords**: Scientific Claim Extraction, NSF Award Abstracts, Scientific Feasibility, LoRA Fine-Tuning, Metascience

## TL;DR
NSF-SciFy extracts 2.8M scientific claims and investigation proposals from NSF award abstracts, constructing a resource several orders of magnitude larger than existing scientific claim datasets, and demonstrates that it significantly enhances claim/proposal extraction models.

## Background & Motivation

**Background**: Scientific claim verification already utilizes datasets such as SciFACT, PubHEALTH, CLIMATE-FEVER, and HealthVer. However, most originate from papers, news, or fact-checking sites, with scales typically ranging from hundreds to over ten thousand claims, and are often restricted to specific topics like biomedicine, public health, or climate.

**Limitations of Prior Work**: Scientific literature is growing rapidly, with a general annual growth rate of approximately 4% and a doubling time of about 17 years. Manually tracking "which scientific claims are being made versus which are merely research plans to be verified" is becoming increasingly infeasible. Existing datasets are not only small in scale but also rarely cover early-stage scientific claims and future research plans within grant proposals.

**Key Challenge**: Scientific grant abstracts contain both knowledge that authors claim to be true and "future-looking proposals" intended for research. If extraction systems do not distinguish between these two types of statements, they risk misidentifying uncompleted research plans as established scientific facts. Conversely, extracting only claims ignores critical clues for understanding the evolution of scientific activity.

**Goal**: The authors aim to leverage the NSF Awards database to build a large-scale resource across scientific and mathematical domains. This resource includes both scientific claims and investigation proposals, and its utility is validated across three tasks: technical-to-non-technical summarization, claim extraction, and proposal extraction.

**Key Insight**: NSF award abstracts possess several natural advantages: they cover a broad range of basic research fields, undergo expert review, are publicly available, and for recent projects, may link to subsequent publications. They are closer to the "upstream" source where scientific ideas are funded and formed than published papers are.

**Core Idea**: Use zero-shot LLM prompting to jointly extract claims and investigation proposals from NSF grant abstracts, and then use this high-precision, large-scale weakly labeled data to train smaller open-source models.

## Method

The NSF-SciFy method does not propose a complex model but rather constructs a reusable data generation and evaluation pipeline: first, the NSF Awards XML database is crawled and parsed into structured award records; next, Claude-3.5 is used for joint extraction; followed by human/LLM-assisted evaluation of extraction quality; finally, Mistral-7B and Qwen2.5-7B are trained on a materials science subset to verify the training value of the data for downstream tasks.

### Overall Architecture

The data source is the NSF Awards database, covering 1970 to September 2024, with the raw XML containing over 0.5M awards. After parsing, 412,155 usable awards were obtained, forming the main body of NSF-SciFy. The paper focuses analysis on two subsets: NSF-SciFy-MatSci, from the Division of Materials Research, and NSF-SciFy-20K, sampled stratifically from five NSF directorates.

Each record typically contains an award ID, title, year, directorate/division, technical abstract, non-technical abstract, claims, investigation proposals, and links to subsequent publications for recent awards. Non-technical abstracts are not simple copies of technical ones: among 13,025 technical/non-technical abstract pairs, only 202 (1.5%) had a symmetric BLEU similarity exceeding 0.6.

The extraction phase uses Claude-3.5-Sonnet-20240620 with the temperature set to 0. The prompt requires the model to return JSON including the award ID, technical abstract, non-technical abstract, a list of claims, and a list of investigation proposals. The authors emphasize that joint extraction is crucial: if only claims are extracted, the model is more likely to mislabel forward-looking investigation statements as established claims.

### Key Designs

1.  **Using NSF grant abstracts as claim sources**:

    *   Function: Expands scientific claim datasets from "published papers/news" to the grant application and award abstract stage.
    *   Mechanism: Parses NSF Awards XML while retaining technical/non-technical abstracts, disciplinary catalogs, award years, and subsequent publication links to form a scientific claim database suitable for longitudinal analysis.
    *   Design Motivation: Grant abstracts capture knowledge assumptions and plans at the time of funding, which is more appropriate for studying the early formation and evolution of scientific ideas than looking at publications alone.

2.  **Joint extraction of claims and investigation proposals**:

    *   Function: Simultaneously extracts statements "claimed as true by authors" and statements "authors plan to investigate."
    *   Mechanism: The prompt explicitly distinguishes claims from forward-looking proposals and requires structured JSON output; the temperature is set to 0 to improve consistency.
    *   Design Motivation: Numerous sentences in scientific abstracts are of the form "we will study/develop/test." Without joint modeling of proposals, extractors mistake plans for facts, reducing the quality of claim data.

3.  **Verifying trainability with NSF-SciFy**:

    *   Function: Demonstrates that the dataset is not just large but can also be used to train functional models.
    *   Mechanism: After deduplication and filtering on NSF-SciFy-MatSci, 11,141 samples were obtained and split into 8,641 / 500 / 2,000 for train / validation / test sets. Mistral-7B-instruct-v0.3 and Qwen2.5-7B-Instruct were fine-tuned using LoRA.
    *   Design Motivation: If fine-tuning can significantly improve claim/proposal extraction, it indicates that the large-scale data obtained from zero-shot LLM extraction can serve as a bootstrapping resource.

### Loss & Training

The paper does not propose a new loss function but employs LoRA to fine-tune 7B models. The LoRA rank is 128, `lora_alpha=64`, and the learning rate is $1 \times 10^{-5}$ with a linear schedule. The query, key, value, output projection, as well as MLP gate, up, and down projections are updated. Training is conducted for 3 epochs with 100 warmup steps, a batch size of 2, and gradient accumulation of 4, taking approximately 1 hour per epoch on an A100 GPU.

For evaluation, BERTScore and ROUGE are used for technical-to-non-technical abstract summarization. For claim/proposal extraction, a pairwise boolean judge function defined by GPT-4o-mini is used to calculate precision, recall, and F1, with human-annotated samples confirming its judgments closely match human ones.

## Key Experimental Results

### Dataset Scale

| Dataset | awards / abstracts | claims | investigation proposals | Coverage |
| :--- | :--- | :--- | :--- | :--- |
| NSF-SciFy | 412,155 awards | 2.8M | Not specified in cache | 1970-2024, all science and math domains |
| NSF-SciFy-MatSci | 16,042 awards | 114K | 145K | Materials Science; ~7 claims and 9 proposals per pair |
| NSF-SciFy-20K | 20,001 awards | 135K | 139K | Five directorates: MPS, GEO, ENG, CSE, BIO |
| MatSci Training Subset | 11,141 samples | Used for training | Used for training | train / val / test = 8,641 / 500 / 2,000 |

### Main Results

| Task | Model | Precision / BERTScore-F1 | Recall | F1 / Other Metrics | Key Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Tech to Non-tech Summary | Mistral-7B | BERTScore-F1 0.8561 | - | ROUGE-L 0.1273 | Small gain from fine-tuning; base model already strong |
| Tech to Non-tech Summary | Qwen2.5-7B | BERTScore-F1 0.8437 | - | ROUGE-L 0.1466 | Higher ROUGE-L than Mistral, but Mistral stronger overall |
| Scientific claim extraction | Mistral-7B | 0.7450 (+116.7% Gain) | 0.7098 (+59.5% Gain) | 0.7097 (+101.8% Gain) | Fine-tuning roughly doubles F1 |
| Scientific claim extraction | Qwen2.5-7B | 0.6839 (+107.1% Gain) | 0.6611 (+7.8% Gain) | 0.6541 (+63.3% Gain) | Also benefits significantly, but weaker than Mistral |
| Investigation proposal extraction | Mistral-7B | 0.7351 (+18.24% Gain) | 0.7539 (+127.24% Gain) | 0.7261 (+90.97% Gain) | Proposal task also strongly dependent on fine-tuning |
| Investigation proposal extraction | Qwen2.5-7B | 0.7245 (+70.07% Gain) | 0.6865 (+81.57% Gain) | 0.6827 (+112.60% Gain) | High relative gain, but absolute F1 lower than Mistral |

### Quality and Error Analysis

| Analysis Item | Number | Description |
| :--- | :--- | :--- |
| High similarity tech/non-tech pairs | 202 / 13,025 = 1.5% | Suggests non-technical abstracts are not simple rewrites |
| SVM distinguishing tech/non-tech | F1 90.99 (SPECTER), 88.42 (STEL) | Both abstract types are distinguishable in content and style |
| Claim categories top-3 | Method/Tech 32.8%, Gaps 21.0%, Observation 18.9% | Based on 810 claims / 120 awards |
| Proposal categories top-3 | Theory 36.9%, Exp. Tech 16.8%, Education 12.8% | Based on 833 proposals |
| Mistral claim error rate | 2.6% | Based on 802 claims; errors include overconfidence, mixed info |
| Claude claim error rate | 2.1% | Mostly administrative hallucinations |
| Mistral proposal error rate | 2.4% | Generation when none exist, mismatch, over-specification |

### Key Findings

*   The scale of NSF-SciFy far exceeds existing datasets: SciFACT has only 1.4K claims and PubHEALTH 11.8K, while the NSF-SciFy-MatSci subset alone contains 114K claims.
*   For claim/proposal extraction, the gain from fine-tuning is much larger than for the summarization task, suggesting that the core value of NSF-SciFy lies in teaching models to recognize scientific statement structures.
*   The extraction pipeline prioritizes high precision, though recall remains relatively low; the authors view this as a direction for improvement via multi-turn extraction, ensembles, and active annotation.

## Highlights & Insights

*   The most valuable contribution is the choice of data source: grant abstracts represent the "upstream state" of scientific claims, allowing for the observation of assertions and plans before research is published. This is vital for tracking scientific discovery and metascience analysis.
*   Joint extraction of claims and proposals is a small but critical design choice. The verb tenses and moods in many scientific abstracts can easily cause models to confuse "what is known" with "what is to be researched"; joint extraction forces the model to distinguish them explicitly.
*   The paper does not exaggerate the perfection of zero-shot extraction; instead, it acknowledges high precision and low recall, demonstrating through fine-tuning that the data can continue to improve models. This bootstrapping narrative is highly credible.
*   The paired technical/non-technical abstract data is also highly valuable. It can be used not just for science communication, but also to study how the same scientific content is translated between expert and public registers.

## Limitations & Future Work

*   **Data source biased toward the US NSF**: While NSF covers about 25% of federally supported basic research in the US, it excludes unfunded proposals, international funds, and non-public applications.
*   **Trade-off between high precision and low recall**: Zero-shot extraction prioritizes reliability, leading to lower recall for claims. Future work needs multi-turn extraction, model ensembles, or active annotation to capture missing claims.
*   **LLM-as-judge requires more validation**: While GPT-4o-mini evaluation showed high consistency with humans in samples, it still requires community validation across more disciplines and varying claim complexities.
*   **Non-uniform coverage of time and publication links**: Subsequent publications are more commonly updated for recent years, leading to a temporal bias in longitudinal "claim-to-paper" tracking.
*   **Fact-checking loop is not yet closed**: The dataset contains claims and proposals but does not directly provide supporting/refuting evidence or final truth labels. True claim verification still requires evidence retrieval and annotation.

## Related Work & Insights

*   **vs SciFACT / SciFACT-Open**: SciFACT focuses on biomedical paper claim verification with 1.4K claims; NSF-SciFy covers grant abstracts, reaching 2.8M claims, though it lacks direct evidence labels.
*   **vs PubHEALTH / CLIMATE-FEVER / HealthVer**: These datasets originate from public health, climate, or news fact-checking oriented toward public discourse; NSF-SciFy is closer to scientific funding and plan texts.
*   **vs individual claim extraction**: This work jointly extracts proposals, reducing the risk of treating future research plans as facts and providing structured data for studying how science plans transform into paper results.
*   **Insights**: Much academic NLP data can be mined from administrative scientific texts, not just paper bodies. Grant applications, peer reviews, and project reports may contain scientific knowledge states from different stages.

## Rating
*   Novelty: ⭐⭐⭐⭐☆ The data source and joint claim/proposal extraction are quite novel; the model methodology is primarily engineering-focused data construction and fine-tuning validation.
*   Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes scale statistics, human quality analysis, three downstream tasks, and error analysis; the lack of a full claim verification evidence chain is a shortcoming.
*   Writing Quality: ⭐⭐⭐⭐☆ The structure is clear and numerical data is abundant; individual table descriptions are slightly long, but the overall data pipeline is easy to replicate.
*   Value: ⭐⭐⭐⭐⭐ The data resource value is very high, particularly suitable for scientific claim mining, metascience, science communication, and early scientific trend analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] APB-V: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention](apb-v_accelerating_long-video_understanding_via_sequence-parallelism-aware_appro.md)
- [\[ACL 2026\] Automated Knowledge Component Generation and Interpretable Knowledge Tracing in Coding Problems](automated_knowledge_component_generation_for_interpretable_knowledge_tracing_in_.md)
- [\[ACL 2026\] Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning](rethinking_the_idiomaticity_decomposability_hypothesis_evidence_from_distributio.md)
- [\[ACL 2026\] CRAFT: Critic-Refined Adaptive Key-Frame Targeting for Multimodal Video Question Answering](craft_critic-refined_adaptive_key-frame_targeting_for_multimodal_video_question_.md)
- [\[ACL 2026\] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)

</div>

<!-- RELATED:END -->
