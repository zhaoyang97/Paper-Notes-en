---
title: >-
  [Paper Note] Do Retrieval Augmented Language Models Know When They Don't Know?
description: >-
  [AAAI 2026][Information Retrieval & RAG][RAG] This paper systematically analyzes the refusal calibration problem in RAG models, finding that RALMs exhibit an over-refusal rate exceeding 55% when all retrieved documents a…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Calibration"
  - "Over-Refusal"
  - "Uncertainty Estimation"
  - "Retrieval Augmentation"
date: 2026-05-08
content_hash: 6475a647c8d9e54a
---

# Do Retrieval Augmented Language Models Know When They Don't Know?

**Conference**: AAAI 2026
**arXiv**: [2509.01476](https://arxiv.org/abs/2509.01476)  
**Code**: [GitHub](https://github.com/zuochao912/refusal-ability-of-retrieval-augmented-LLMs)  
**Area**: Information Retrieval
**Keywords**: RAG, Calibration, Over-Refusal, Uncertainty Estimation, Retrieval Augmentation

## TL;DR

This paper systematically analyzes the refusal calibration problem in RAG models, finding that RALMs exhibit an over-refusal rate exceeding 55% when all retrieved documents are irrelevant (even when the model's internal knowledge suffices to answer), and proposes a mechanism combining uncertainty estimation with refusal-aware fine-tuning to balance refusal behavior and answer quality.

## Background & Motivation

- **Background**: RAG systems augment LLMs by retrieving external documents, while refusal post-training teaches models to proactively decline answering under uncertainty. Both techniques are widely adopted but have rarely been studied jointly.
- **Limitations of Prior Work**: When all retrieved documents are irrelevant, RALMs with refusal training tend to reject all queries—even when the model's parametric knowledge (internal knowledge) is sufficient for a correct answer. This "over-refusal" phenomenon has been largely overlooked.
- **Key Challenge**: Improving refusal rates does not equate to improving calibration quality. Existing refusal training methods (e.g., R-tuning) may in fact exacerbate over-refusal. The fundamental issue is a calibration imbalance between the model's internal knowledge and externally retrieved knowledge.
- **Key Insight**: The paper categorizes RALM knowledge sources into four quadrants (internal knowledge present/absent × external knowledge present/absent) and systematically investigates refusal behavior and uncertainty calibration across different knowledge states.

## Method

### Overall Architecture

The study is organized around three progressive research questions: (RQ1) How well are RALMs calibrated under different knowledge states? (RQ2) What is the relationship between refusal ability and calibration quality? (RQ3) Can uncertainty estimation be leveraged to mitigate over-refusal?

### Key Designs

1. **Four-Quadrant Knowledge State Analysis (RQ1)**

    - Each question is classified into one of four knowledge states: *highlyknown*, *maybeknown*, *weaklyknown*, or *unknown*
    - Classification is based on consistency between temperature-sampled and greedy-decoded outputs
    - Refusal behavior under different retrieval configurations (0 positive/10 negative, 1 positive/9 negative, 5 positive/5 negative, etc.) is analyzed for each knowledge state
    - Finding: Even *highlyknown* questions are subject to over-refusal under fully negative retrieval conditions

2. **Comparative Analysis of Refusal Training Methods (RQ2)**

    - **R-tuning**: Detects questions the model cannot answer and trains it to output "I don't know"—found to exacerbate over-refusal
    - **In-Context Fine-Tuning (ICFT)**: Inserts both positive and negative contexts into fine-tuning data simultaneously, with training targets determined by the knowledge quadrant—effectively mitigates over-refusal
    - Key finding: ICFT improves refusal behavior but does not necessarily improve calibration or accuracy, due to changes in robustness and context faithfulness

3. **Uncertainty-Based Refusal Mechanism (RQ3)**

    - Uses uncertainty estimates and their changes to infer the RALM's knowledge state
    - Based on the inferred state, decides to: answer using retrieved context / answer without retrieved context / refuse to answer
    - Three categories of uncertainty estimation methods: Verbalization-based (model self-reported confidence, 4 prompt variants), Consistency-based (Agreement/Entropy/FSD metrics), and Similarity Matrix-based (eigenvalue/degree measures)

4. **Experimental Rigor**

    - High-quality negative retrieval samples are constructed via Milvus hybrid search with re-ranking
    - Strict answer evaluation: LLM-as-judge + exact match + refusal keyword filtering
    - Qwen evaluated on Chinese datasets (CRUD, RGB_zh); LLaMA evaluated on English datasets (NQ, RGB_en)
    - Generation temperature of 0.5 with 16 sampling runs

### Loss & Training

- R-tuning: Two-stage pipeline—first detecting questions outside the model's knowledge boundary, then fine-tuning with "should refuse" labels
- ICFT: Both positive and negative contexts are inserted into each training sample simultaneously; training targets are set as correct answers or refusal expressions based on the RALM's knowledge quadrant

## Key Experimental Results

### Main Results (RQ1 Calibration Analysis, Lower Brier Score Is Better)

| Uncertainty Estimation Method | RGB_en (0p10n) | RGB_en (5p5n) | RGB_zh (0p10n) |
|-------------------------------|---------------|--------------|---------------|
| Verb-1s-1 | 0.139 | 0.023 | 0.441 |
| Entropy | 0.305 | 0.009 | 0.256 |
| Agree | 0.192 | 0.010 | 0.261 |
| Eigv (Similarity Matrix) | 0.232 | 0.271 | 0.282 |

### Ablation Study (RQ2 Refusal Training Effectiveness)

| Method | Over-Refusal Rate | Overall Accuracy | Calibration Quality |
|--------|------------------|-----------------|-------------------|
| Original RALM | >55% | Baseline | Baseline |
| R-tuning | Worsened | Degraded | Not improved |
| ICFT | Significantly reduced | Maintained/slightly improved | Not necessarily improved |
| ICFT + Uncertainty Refusal | Best | Significantly improved | Improved |

### Key Findings

- **Over-refusal rate exceeds 55% under fully irrelevant retrieval**: Models possess sufficient internal knowledge to answer correctly, yet refuse due to poor retrieval results
- **R-tuning is counterproductive**: Refusal training causes models to refuse in more cases, exacerbating over-refusal
- **ICFT is effective but limited**: It mitigates over-refusal but does not automatically improve calibration quality, suggesting that refusal behavior and calibration are two independent dimensions
- **Consistency-based methods achieve the best calibration when positive documents are present** (Brier = 0.009), but degrade significantly under fully negative retrieval settings
- **Changes in uncertainty (the difference between with/without retrieval) serve as effective indicators of knowledge state**

## Highlights & Insights

- **Systematic characterization of "over-refusal"**: This paper is among the first to situate RAG refusal ability within an uncertainty calibration framework, revealing a widely overlooked but practically significant problem
- **Key insight that refusal ≠ calibration**: Improving a model's refusal behavior does not equate to improving its calibration quality; the two dimensions require independent optimization
- **Practical utility of the four-quadrant knowledge state framework**: Provides a clear analytical structure for dynamic strategy selection in RAG systems

## Limitations & Future Work

- Experiments are primarily based on simple factual QA (single-hop); over-refusal patterns on complex reasoning tasks may differ
- The choice of uncertainty estimation method substantially impacts results, and no mechanism for automatically selecting the optimal method is provided
- Dynamic retrieval still relies on static thresholds; adaptive threshold mechanisms warrant further exploration
- Experiments are limited to models of approximately 7B parameters; over-refusal behavior in larger models remains to be verified

## Related Work & Insights

- **vs. Dynamic RAG methods (Self-RAG, FLARE, etc.)**: These methods determine when to retrieve based on uncertainty, but assume the model is well-calibrated; this paper demonstrates that assumption does not hold
- **vs. LLM knowledge boundary research (Li et al. 2025)**: Prior work focuses primarily on internal knowledge; this paper extends the analysis to the joint interplay of internal and external knowledge

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic characterization of over-refusal and the four-quadrant analysis framework constitute original contributions
- Experimental Thoroughness: ⭐⭐⭐⭐ Three progressive RQs, multiple UE methods, bilingual evaluation (Chinese and English), and diverse retrieval configurations
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; research questions are developed with strong logical progression
- Value: ⭐⭐⭐⭐ Offers direct guidance for improving RAG system reliability and designing intelligent refusal strategies

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Positional Bias in Multimodal Embedding Models: Do They Favor the Beginning, the Middle, or the End?](positional_bias_in_multimodal_embedding_models_do_they_favor_the_beginning_the_m.md)
- [\[AAAI 2026\] When Small Models Are Right for Wrong Reasons: Process Verification for Trustworthy Agents](when_small_models_are_right_for_wrong_reasons_process_verification_for_trustwort.md)
- [\[AAAI 2026\] Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-Based Machine Translation](exposing_the_cracks_vulnerabilities_of_retrieval-augmented_llm-based_machine_tra.md)
- [\[AAAI 2026\] Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation](cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)

</div>

<!-- RELATED:END -->
