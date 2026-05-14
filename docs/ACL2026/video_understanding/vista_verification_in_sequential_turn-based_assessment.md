---
title: >-
  [Paper Note] VISTA: Verification In Sequential Turn-based Assessment
description: >-
  [ACL 2026][Video Understanding][Hallucination Detection] VISTA proposes a multi-turn dialogue factuality assessment framework based on claim-level decomposition and sequential consistency tracking…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Hallucination Detection"
  - "Dialogue Factuality"
  - "Claim-level Verification"
  - "Sequential Consistency Tracking"
  - "Multi-turn Dialogue"
date: 2026-05-08
content_hash: c2286ee5700398d8
---

# VISTA: Verification In Sequential Turn-based Assessment

**Conference**: ACL 2026  
**arXiv**: [2510.27052](https://arxiv.org/abs/2510.27052)  
**Code**: [https://github.com/ashleylew/VISTA](https://github.com/ashleylew/VISTA)  
**Area**: Video Understanding  
**Keywords**: Hallucination Detection, Dialogue Factuality, Claim-level Verification, Sequential Consistency Tracking, Multi-turn Dialogue

## TL;DR

VISTA proposes a multi-turn dialogue factuality assessment framework based on claim-level decomposition and sequential consistency tracking, subdividing unverifiable content into four categories: subjective, contradicted, lacking evidence, and abstention. It significantly outperforms FActScore and LLM-as-Judge baselines across four dialogue benchmarks and eight LLMs.

## Background & Motivation

**State of the Field**: Hallucination detection is a major obstacle to deploying conversational AI systems. Existing methods like FActScore decompose text into atomic facts and verify them individually, while LLM-as-Judge directly uses LLMs for holistic judgment, achieving some progress in single-turn evaluation scenarios.

**Limitations of Prior Work**: Existing metrics suffer from two core deficiencies—(1) they treat each generation as isolated text, ignoring the sequential and pragmatic features of dialogue, where earlier claims constrain subsequent content; (2) they categorize all unverifiable content (subjective expressions, abstentions, etc.) uniformly as hallucinations, unable to distinguish between "genuine errors" and "reasonable uncertainty."

**Root Cause**: Factuality in dialogue is a dynamically evolving property rather than a static textual feature, yet existing evaluation methods treat it as a static correctness problem.

**Paper Goals**: (1) Redefine hallucination detection as a sequential claim verification process; (2) provide fine-grained classification of unverifiable content; (3) achieve cross-turn factual consistency tracking in RAG scenarios for dialogue.

**Starting Point**: Drawing on ideas from "common ground" in linguistics and discourse representation theory, the work models factual reliability as a dynamic process progressively constructed during dialogue, implementing cross-turn verification by maintaining an accumulating knowledge store.

**Core Idea**: Replace monolithic judgment with a structured multi-stage pipeline (claim decomposition → verification → unverifiable classification → sequential memory), upgrading dialogue factuality from "point detection" to "trajectory tracking."

## Method

### Overall Architecture

VISTA is a sequential evaluation pipeline that processes each assistant response in dialogue turn order. Input consists of multi-turn dialogue and reference documents; output includes verification categories for each claim and a dialogue-level factuality score. The pipeline comprises five steps: initialize knowledge store → claim decomposition → verification → unverifiable classification → sequential memory & aggregation.

### Key Designs

1. **Claim Decomposition**:

    - Function: Decompose each assistant response into atomic factual claims
    - Mechanism: Unlike FActScore's sentence-level splitting, VISTA operates at the turn level, taking the complete response as input and using dialogue history as context with few-shot templates (n=6) to generate numbered lists. The key lies in explicitly handling presupposition reasoning and coreference resolution—for example, "I don't know that embroidery is a needlework technique" is decomposed into "embroidery is a needlework technique" and "the assistant does not know that embroidery is a needlework technique"
    - Design Motivation: Pre-splitting sentences reduces recall of implicit or coreferent content; experiments show removing FActScore's sentence splitting step actually improves its performance (11.4% gain for DeepSeek, 4.4% for GPT-4o)

2. **Verification & Categorization**:

    - Function: Label each claim as verified or unverifiable, then further categorize unverifiable claims into four types
    - Mechanism: Verification stage uses two evidence sources—(a) accumulated set of verified and out-of-scope claims from previous turns, (b) reference text from current turn. Claims are marked VERIFIED only when they have direct textual support. Unverifiable claims enter classification stage and are categorized into: Out-of-Scope (subjective/experiential content), Contradicted (explicitly refuted by reference materials or prior facts), Lacking Evidence (potentially factual but unsupported), Abstention (expressing uncertainty or refusal to answer)
    - Design Motivation: FActScore only verifies against static reference documents, unable to detect cross-turn contradictions; LLM-as-Judge conflates all unverifiable content. Four-category subdivision provides more diagnostic labels

3. **Sequential Memory & Aggregation**:

    - Function: Maintain dynamic factual memory to support cross-turn consistency tracking
    - Mechanism: Verified and out-of-scope claims from each turn are appended to a running knowledge store, forming dynamic factual memory. Verification in subsequent turns is conditioned on this—verified claims reinforce prior information, contradicted claims indicate factual drift. At evaluation end, all claim-level results are aggregated to generate VISTA Score
    - Design Motivation: Correctness of a claim in dialogue may depend on information established in previous turns (e.g., the Elvis Presley "King of Rock and Roll" example); without sequential memory, correct cross-turn references would be misclassified as "lacking evidence"

### Loss & Training

VISTA requires no training—it is a prompt-based evaluation framework that invokes LLMs through structured prompt templates to complete subtasks at each stage. It supports both zero-shot and few-shot configurations and provides a model-agnostic unified interface.

## Key Experimental Results

### Main Results

**Automatic Evaluation (Accuracy % for Unverifiable Turn Detection)**

| Dataset | Model | VISTA | FActScore | LLM-as-Judge |
|--------|------|-------|-----------|--------------|
| AIS | GPT-4o | **63.00** | 56.80 | 56.80 |
| BEGIN | GPT-4o | **83.20** | 65.80 | 70.40 |
| FaithDial | DeepSeek | **81.70** | 63.75 | 55.45 |
| FADE | Llama-70B | **65.10** | 56.65 | 62.28 |
| FaithDial | Qwen-32B | **75.73** | 58.41 | 35.89 |
| BEGIN | Mistral-7B | **72.00** | 53.80 | 57.40 |

**Human Evaluation (Alignment with Consensus Labels)**

| Model | Turn Accuracy | Claim Accuracy | Macro F1 |
|------|-----------|-----------|----------|
| GPT-5 | 92.51 | 81.53 | 69.09 |
| GPT-4o | 91.19 | 75.68 | 62.41 |
| DeepSeek | 92.51 | 79.73 | 67.15 |

### Ablation Study

| Config | FaithDial Accuracy | Note |
|------|-----------------|------|
| VISTA (full model) | 81.70 | Full model |
| Remove accumulated claims | 81.74 | Minimal impact, most claims verifiable from current document |
| Remove dialogue history | 77.24 | 4.5% drop, dialogue context crucial for decomposition |
| Zero-shot | 70.17 | 11.5% drop, few-shot examples important for modeling dialogue phenomena |

### Key Findings

- VISTA's advantages primarily stem from dialogue contextualization and few-shot examples rather than the accumulated claims mechanism—this relates to most claims in FaithDial being verifiable from current reference documents
- On contradiction detection tasks, performance improves dramatically with full dialogue history (DeepSeek: 60.0% to 77.0%, GPT-5: 54.2% to 86.0%), with accumulated claims being the key driver
- Human annotators disagreed with original dataset labels on 26.4% of turns, with 86.7% being original annotations incorrectly marking unverifiable content as verifiable—indicating claim-level decomposition improves annotation quality
- Abstention identification accuracy reaches 90.6%, demonstrating VISTA can reliably distinguish refusals to answer from hallucinations

## Highlights & Insights

- The paradigm shift from modeling dialogue factuality as a dynamic process rather than static property, with elegant alignment to linguistic common ground theory—this represents theory-driven system design
- Claim-level decomposition not only improves automatic evaluation accuracy but also enhances human annotation consistency (Krippendorff's α = 0.832); this "byproduct" may have more practical value than the main results
- The design philosophy of four-category unverifiable classification can directly transfer to other NLG evaluation tasks, such as faithfulness assessment for summarization and translation

## Limitations & Future Work

- The four benchmark datasets currently used contain few contradiction and abstention cases, leaving VISTA's robustness in these scenarios insufficiently validated
- The accumulated claims mechanism shows limited effectiveness in short dialogues; longer multi-turn dialogue benchmarks are needed to fully demonstrate its value
- VISTA relies on LLMs to execute subtasks at each stage, introducing cascading error problems—errors in claim decomposition propagate to subsequent verification
- Focus is limited to RAG scenarios, without addressing open-domain fact verification

## Related Work & Insights

- **vs FActScore**: Both perform claim-level decomposition and verification, but FActScore verifies in isolation on single documents, while VISTA adds dialogue context and cross-turn tracking
- **vs LLM-as-Judge**: LLM-as-Judge provides holistic judgment but lacks interpretability; VISTA offers claim-level diagnostics through a structured pipeline

## Rating

- Novelty: ⭐⭐⭐⭐ Modeling dialogue factuality as a dynamic process represents a meaningful perspective shift, though individual components (decomposition, verification) are not novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four benchmarks, eight models, human evaluation, ablation analysis, specialized contradiction/abstention tests—comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation derivation, rigorous experimental logic, tight integration of theory and practice
- Value: ⭐⭐⭐⭐ Provides an evaluation framework more suitable for dialogue scenarios than FActScore, though actual deployment costs are high

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Quantifying Conversational Reliability of Large Language Models under Multi-Turn Interaction](../../AAAI2026/video_understanding/quantifying_conversational_reliability_of_large_language_models_under_multi-turn.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](../../NeurIPS2025/video_understanding/sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[ICLR 2026\] AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment](../../ICLR2026/video_understanding/anveshanaai_a_multimodal_platform_for_adaptive_aiml_education_through_automated_.md)
- [\[ACL 2026\] VC-Inspector: Advancing Reference-free Evaluation of Video Captions with Factual Analysis](vc-inspector_advancing_reference-free_evaluation_of_video_captions_with_factual_.md)
- [\[ACL 2026\] GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents](gameplayqa_a_benchmarking_framework_for_decision-dense_pov-synced_multi-video_un.md)

</div>

<!-- RELATED:END -->
