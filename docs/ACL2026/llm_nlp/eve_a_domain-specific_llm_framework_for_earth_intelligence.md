---
title: >-
  [Paper Note] EVE: A Domain-Specific LLM Framework for Earth Intelligence
description: >-
  [ACL 2026][LLM/NLP][Earth Observation] Ours proposes EVE—the first open-source end-to-end LLM framework for Earth Observation / Earth Sciences led by the ESA Φ-lab. It includes the 24B domain-adapted EVE-Instruct (based…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Earth Observation"
  - "Domain LLM"
  - "RAG"
  - "Hallucination Detection"
  - "Mistral Small 3.2"
  - "ESA"
date: 2026-05-08
content_hash: 560c824e88d37666
---

# EVE: A Domain-Specific LLM Framework for Earth Intelligence

**Conference**: ACL 2026  
**arXiv**: [2604.13071](https://arxiv.org/abs/2604.13071)  
**Code**: <https://github.com/eve-esa> (Models at <https://huggingface.co/eve-esa>)  
**Area**: Domain LLM / Earth Sciences / RAG  
**Keywords**: Earth Observation, Domain LLM, RAG, Hallucination Detection, Mistral Small 3.2, ESA

## TL;DR
Ours proposes EVE—the first open-source end-to-end LLM framework for Earth Observation / Earth Sciences led by the ESA Φ-lab. It includes the 24B domain-adapted EVE-Instruct (based on Mistral Small 3.2 + interleaved IFT/CPT with 10.7B synthetic tokens + 10-checkpoint merging), the first human-annotated EO benchmark with 5,693 samples, and a RAG + hallucination detection pipeline, serving 350 users in a 6-month pilot.

## Background & Motivation

**Background**: Earth Observation (EO) and Earth Sciences generate massive volumes of high-value knowledge daily. However, this knowledge is fragmented across heterogeneous sources (satellite imagery, scientific papers, proprietary publisher databases, and internal ESA documents), requiring deep expertise to synthesize. General LLMs lack domain specialization and rigorous evaluation, failing to meet the scientific rigor required for "Earth Action" decision-making.

**Limitations of Prior Work**: (i) Existing domain LLM research either focuses solely on corpora and CPT (INDUS, K2, AstroLLaMA, COSMOSAGE) while lacking end-to-end deployment, or focuses on spatial reasoning tool integration (GeoLLM, GeoGPT, ChatGeoAI) without genuine domain SFT. (ii) Earth Sciences lack standardized dialogue/NLP benchmarks, preventing horizontal model comparison. (iii) The core conflict for production deployment is how "medium-sized" models like 24B can achieve domain adaptation without sacrificing general capabilities (tool calling, IF, chat quality).

**Key Challenge**: To build a truly usable domain assistant, one must simultaneously address data (high-quality EO corpora), training (avoiding catastrophic forgetting), evaluation (domain benchmarks), and deployment (RAG grounding and hallucination control). Prior works typically only cover one or two of these areas.

**Goal**: (i) Construct high-quality EO corpora totaling 5.3B tokens and 10.7B synthetic training tokens; (ii) Implement a training recipe using interleaved IFT/long-form text + replay + 10-checkpoint merging to achieve domain adaptation while retaining general abilities; (iii) Release the first EO evaluation benchmark with 5,693 samples (MCQA, open-ended QA, and hallucination detection); (iv) Integrate an end-to-end RAG + hallucination detection pipeline into production for 350 users.

**Key Insight**: The authors found that LoRA is insufficient for a "medium-scale" corpus of 5.3B tokens, while pure CPT degrades instruction-following. Therefore, a hybrid "interleaved IFT + long-form text + replay data" training strategy is chosen, interleaving long-form and instruct data within the same run and using an active reading pipeline for self-synthetic corpus enhancement.

**Core Idea**: A five-fold approach—small-scale high-quality corpora, large-scale synthetic data, interleaved IFT/long-form training, replay, and checkpoint merging—transforms a 24B general model into a domain expert without increasing parameters.

## Method

### Overall Architecture
EVE is a production system consisting of four modules: (i) **EVE-Instruct**—the core LLM fine-tuned from Mistral Small 3.2 (24B, 128k context), responsible for answer generation, query rewriting, and summarization; (ii) **Knowledge Bases**—a multi-source KB of ~365k documents (Open Access + Wiley proprietary + ESA documents) supporting hybrid semantic and metadata retrieval; (iii) **Retrieval Pipeline**—selects relevant documents based on queries and filters, followed by reranking via Qwen3-Reranker-4B; (iv) **Chat System + Hallucination Detection**—manages conversation states, performs fact-checking, and triggers a "rewrite-answer" loop when necessary.

### Key Designs

1.  **Dual-Track EO Corpus and Synthetic Data (5.3B raw + 10.7B synthetic)**:
    *   **Function**: Balances the gap between "insufficient raw data for CPT" and the potential "drift" of synthetic data.
    *   **Mechanism**: The raw portion uses a custom scraper to collect data from 172 sources across 22 trusted publishers (4.2B Open Access + 1.1B Wiley proprietary), processed via Trafilatura/Nougat OCR, SHA-256 + MinHash LSH deduplication, Presidio anonymization, and CrossRef metadata completion. The synthetic portion follows two paths: (a) long-form text is reorganized using an Active Reading pipeline (Mistral Medium 3.1 selects strategies, Mistral Small 3.2 generates); (b) instruction text is generated by 7 high-quality models (Mistral Large 3, GPT-4o Mini, Qwen3-235B, DeepSeek-R1, etc.) across five categories: ContextQA, SelfQA, LongQA, MultiHop QA, and self-referential alignment. From ~21B initial synthetic tokens, 10.7B are filtered via LLM-as-judge for the final set.
    *   **Design Motivation**: The raw corpus of 5.3B tokens sits between "sufficient for SFT" and "sufficient for CPT"; pure CPT would damage instruction-following. Introducing large-scale synthetic data with strict quality filters provides scale while ensuring diversity and quality.

2.  **Interleaved IFT/Long-form Training + Replay + 10-Checkpoint Merging**:
    *   **Function**: Enables the 24B model to adapt to the domain without losing general capabilities.
    *   **Mechanism**: Instruction-formatted text and long-form text are injected alternately within the same training run. Within each category, general-domain replay data is mixed at 50/50 or 60/40 ratios. The learning rate is set at an intermediate value between IFT and CPT to balance "factual integration" and "alignment stability." Finally, 10 training runs with different mixing ratios are conducted, and uniform parameter interpolation (merging) is used to average the "domain-strong/general-weak" and "general-strong/domain-weak" checkpoints into a single trade-off optimized model. The process concludes with Online DPO for alignment.
    *   **Design Motivation**: Experiments showed that a single mixing ratio always results in a static trade-off between domain and general performance. Checkpoint merging serves as a low-cost ensemble alternative—improving robustness without needing multi-model inference. It proves more stable than LoRA or regularization-based anti-forgetting methods.

3.  **First EO Benchmark + Hallucination-aware RAG Closed-Loop**:
    *   **Function**: Quantifies the effectiveness of the domain LLM and allows production hallucinations to be automatically captured and corrected.
    *   **Mechanism**: The benchmark includes 5 task types (1,261 Single-choice MCQA, 431 Multi-choice MCQA, 1,257 Open-ended without context, 418 Open-ended with context, and 2,326 Hallucination detection), annotated by 25 EO experts via LLM/Human dual-source generation and independent auditing. The RAG side uses ~512-word chunks + Qwen3-Embedding-4B + Qdrant binary quantization. Retrieval involves query rewriting, taking top 2K candidates per KB, and reranking to top K via Qwen3-Reranker-4B. The hallucination detection workflow involves: EVE-Instruct self-evaluation → if hallucination is flagged → rewrite query with justification → re-retrieval → generate revised version → self-evaluation → selection of the best version.
    *   **Design Motivation**: General hallucination benchmarks (FEVER, TruthfulQA, HaluEval) do not cover EO domain knowledge. Production systems require a lightweight "detect-then-repair" loop for controllable latency. Integrating detection into the LLM itself (using EVE-Instruct as a judge) reduces deployment costs compared to an independent verifier.

### Loss & Training
The base model is Mistral Small 3.2 (24B, 128k context). Training mixes 30% long-form + 70% instruction data, with 50/50 or 60/40 replay data added internally (see Table 3). The learning rate is between typical IFT and CPT values. 10 checkpoints from varying mix ratios are merged. Alignment uses Online DPO (following the Liu et al. 2026 recipe). Training costs are estimated at approximately 38 tons of $CO_2eq$.

## Key Experimental Results

### Main Results (EO benchmark zero-shot)

| Model | Size | MCQA Multi-choice IoU | MCQA Single-choice Acc | Hallucination F1 | Open QA Judge | Avg. Rank ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Llama4 Scout | 109-B | 80.32 | 71.23 | 66.08 | 87.37 | 3.67 |
| Qwen3 | 30-B | 78.40 | 66.36 | 81.30 | 94.92 | 2.67 |
| Gemma3 | 27 | 73.60 | 57.54 | 75.07 | 94.41 | 3.83 |
| Mistral Small 3.2 (parent) | 24 | 80.19 | 70.30 | 82.19 | 91.78 | 3.50 |
| **EVE-Instruct** | **24** | **86.12** | **77.73** | **84.70** | **96.40** | **1.33** |

EVE-Instruct ranked first in 4 out of 5 tasks, with an average rank of 1.33. The Multi-choice MCQA IoU is 86.12 (6 points higher than the parent Mistral Small 3.2). The Hallucination F1 of 84.70 significantly outperforms Llama4 Scout's 66.08, demonstrating the discriminative capability gained through domain adaptation.

### Ablation Study (Impact on General Capabilities)

| Category | Small 3.2 | EVE-Instruct | Δ |
| :--- | :--- | :--- | :--- |
| Math & Reasoning | 50.8 | 54.9 | +4.1 |
| Coding | 55.6 | 56.5 | +0.9 |
| Knowledge | 67.7 | 69.0 | +1.3 |
| Tool Calling | 87.9 | 90.9 | +3.0 |
| Instruction Following | 80.1 | 81.2 | +1.1 |
| Chat Quality | 90.8 | 91.7 | +0.9 |
| **Overall** | **72.2** | **74.0** | **+1.8** |

All general capability sub-items showed improvement, proving that the interleaved IFT/long-form + replay + 10-checkpoint merging recipe successfully resolves the "domain vs. general" trade-off.

### Key Findings
- **24B can outperform 109B**: EVE-Instruct consistently beats Llama4 Scout (109B MoE) on EO tasks, showing that refined data and training strategies in vertical domains outweigh mere parameter scaling.
- **Domain adaptation does not require sacrificing generality**: All Δ values are positive, debunking the traditional belief that domain LLMs inevitably lose performance on general tasks.
- **Judge bias verified**: Replacing LLM-judge with Claude Sonnet 4.6 and Gemini 2.5 Flash resulted in rank shifts of only ±0.25, confirming evaluation robustness.
- **Production viability**: A 6-month pilot with 350 users using RunPod serverless + Qdrant binary quantization + AWS backend proves that a 24B domain assistant is economically feasible.
- **Open-Ended QA with Context**: While Qwen3 scored slightly higher on judge scores, EVE achieved the best Win Rate, indicating comparable quality in RAG-enhanced scenarios.

## Highlights & Insights
- **"10-checkpoint Merging" as a Low-Cost Ensemble**: Using uniform parameter interpolation instead of inference-time ensembling addresses the "domain vs. general" trade-off during post-training. This is a valuable engineering trick for other domain LLM projects—increasing training compute while keeping deployment costs constant.
- **Active Reading Pipeline for Synthetic Long-form Text**: Unlike traditional paraphrasing, Active Reading allows the LLM to reorganize "important content + key terminology," which is particularly effective for domain adaptation by reinforcing schema and jargon over syntax.
- **First EO Benchmark + 25-Expert Annotation**: This open release provides the Earth Sciences NLP community with standardized evaluation, representing a community-level contribution.
- **Hallucination Self-Evaluation + Self-Repair Loop**: Consolidating detection, revision, and selection into a single model (single RT) rather than a multi-model pipeline is latency-friendly and represents a successful engineering of RARR/SelfCheckGPT methods for production.

## Limitations & Future Work
- Due to Wiley agreements, 1.1B tokens (~21% of the corpus) cannot be released, affecting external reproducibility.
- Evaluation still relies heavily on LLM-as-judge; human evaluation coverage is limited due to the scarcity of EO experts.
- EVE is currently text-only and cannot process satellite imagery or structured geospatial data directly. The authors plan to develop a multimodal agentic platform integrating Geospatial Foundation Models.
- RAG timeliness depends on manual KB refreshes, leaving the engineering of automated knowledge updates unresolved.
- The training cost of 38 tons of $CO_2eq$ remains high for academic institutions; the open-source strategy prioritizes reuse over replication.

## Related Work & Insights
- **vs. INDUS / K2 / AstroLLaMA / COSMOSAGE**: Those works primarily focused on corpora and CPT; EVE upgrades to a full end-to-end system (data, training, eval, RAG, deployment) and provides the first domain benchmark.
- **vs. GeoGPT / GeoLLM / ChatGeoAI**: Those models utilize LLM + GIS tool calling for spatial reasoning. EVE focuses on text understanding, scientific QA, and hallucination control, making them complementary.
- **vs. SelfCheckGPT / RARR**: The hallucination self-evaluation and rewrite loop borrows from RARR but consolidates it into a single model for production.
- **vs. LoRA / Selective Parameter Freezing**: The authors abandoned LoRA after experiments, proving that for "medium-scale" corpora (5.3B tokens), full-parameter interleaved fine-tuning + checkpoint merging offers a superior trade-off.

## Rating
- Novelty: ⭐⭐⭐ Individual components are not new, but the first end-to-end EO system + public benchmark is a milestone.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-judge, cross-family evaluation, general capability comparisons, and 6-month deployment are rare in the field.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with comprehensive methodology, evaluation, and deployment details.
- Value: ⭐⭐⭐⭐⭐ Open-source models, corpora, benchmarks, and code provide infrastructure-level contributions to the EO NLP community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/synergy_over_discrepancy_a_partition-based_approach_to_multi-domain_llm_fine-tun.md)
- [\[ICLR 2026\] BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning](../../ICLR2026/llm_nlp/bots_a_unified_framework_for_bayesian_online_task_selection_in_llm_reinforcement.md)
- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](../../ICLR2026/llm_nlp/ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[ACL 2026\] When TableQA Meets Noise: A Dual Denoising Framework for Complex Questions and Large Tables](when_tableqa_meets_noise_a_dual_denoising_framework_for_complex_questions_and_la.md)
- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)

</div>

<!-- RELATED:END -->
