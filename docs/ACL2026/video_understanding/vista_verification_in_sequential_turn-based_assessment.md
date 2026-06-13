---
title: >-
  [Paper Note] VISTA: Verification In Sequential Turn-based Assessment
description: >-
  [ACL 2026][Video Understanding][Hallucination Detection] VISTA proposes a multi-turn dialogue factuality assessment framework based on claim-level decomposition and sequential consistency tracking. It subdivides unverifi…
tags:
  - "ACL 2026"
  - "Video Understanding"
  - "Hallucination Detection"
  - "Dialogue Factuality"
  - "Claim-level Verification"
  - "Sequential Consistency Tracking"
  - "Multi-turn Dialogue"
date: 2026-05-08
content_hash: 8e483013ef5ea245
---

# VISTA: Verification In Sequential Turn-based Assessment

**Conference**: ACL 2026  
**arXiv**: [2510.27052](https://arxiv.org/abs/2510.27052)  
**Code**: [https://github.com/ashleylew/VISTA](https://github.com/ashleylew/VISTA)  
**Area**: Video Understanding  
**Keywords**: Hallucination Detection, Dialogue Factuality, Claim-level Verification, Sequential Consistency Tracking, Multi-turn Dialogue

## TL;DR

VISTA proposes a multi-turn dialogue factuality assessment framework based on claim-level decomposition and sequential consistency tracking. It subdivides unverifiable content into four categories—subjective, contradicted, lacking evidence, and abstention—significantly outperforming FActScore and LLM-as-judge baselines across four dialogue benchmarks and eight LLMs.

## Background & Motivation

**Background**: Hallucination detection is a primary obstacle to the deployment of dialogue AI systems. Existing methods such as FActScore decompose text into atomic facts for individual verification, while LLM-as-Judge uses LLMs for holistic judgment, showing progress in single-turn evaluation scenarios.

**Limitations of Prior Work**: Existing metrics suffer from two core defects: (1) they treat each generation as isolated text, ignoring the sequential and pragmatic characteristics of dialogue where previous claims constrain subsequent content; (2) they treat all unverifiable content (subjective expressions, abstentions, etc.) uniformly as hallucinations, failing to distinguish between genuine errors and reasonable uncertainty.

**Key Challenge**: Factuality in dialogue is a dynamically evolving property rather than a static feature of text, yet existing evaluation methods treat it as a static correctness problem.

**Goal**: (1) Redefine hallucination detection as a sequential claim verification process; (2) implement fine-grained classification of unverifiable content; (3) achieve cross-turn fact consistency tracking in dialogue RAG scenarios.

**Key Insight**: Drawing on ideas from linguistic "common ground" and Discourse Representation Theory, factuality reliability is modeled as a dynamic process constructed step-by-step during dialogue. This is realized by maintaining an accumulating knowledge store for cross-turn verification.

**Core Idea**: Replace monolithic judgment with a structured multi-stage pipeline (claim decomposition $\rightarrow$ verification $\rightarrow$ unverifiable classification $\rightarrow$ sequential memory), upgrading dialogue factuality from "point detection" to "trajectory tracking."

## Method

### Overall Architecture

VISTA is a sequential evaluation pipeline that processes each assistant response in dialogue turn order. The inputs are multi-turn dialogues and reference documents; the outputs are verification categories for each claim and turn-level factuality scores. The pipeline comprises five steps: knowledge store initialization $\rightarrow$ claim decomposition $\rightarrow$ verification $\rightarrow$ unverifiable classification $\rightarrow$ sequential memory and aggregation.

### Key Designs

1.  **Claim Decomposition**:
    - **Function**: Decomposes each assistant response into atomic factual claims.
    - **Mechanism**: Unlike the sentence-level splitting in FActScore, VISTA operates at the turn level, taking the full response as input and utilizing dialogue history as context. It uses few-shot templates ($n=6$) to generate numbered lists. Key to this is explicitly handling presupposition reasoning and coreference resolution—for instance, "I didn't know embroidery was a needlework technique" is split into "Embroidery is a needlework technique" and "The assistant did not know embroidery was a needlework technique."
    - **Design Motivation**: Pre-splitting sentences reduces the recall of implicit or coreferential content; in experiments, removing FActScore's sentence-splitting step actually improved its performance (DeepSeek increased by 11.4%, GPT-4o by 4.4%).

2.  **Verification & Categorization**:
    - **Function**: Tags each claim as verified or unverifiable and further categorizes unverifiable claims into four types.
    - **Mechanism**: The verification phase uses two evidence sources: (a) the collection of verified and out-of-scope claims accumulated from previous turns, and (b) the reference text for the current turn. A claim is marked as VERIFIED only if there is direct textual evidence. Unverifiable claims enter a classification phase and are assigned to one of four categories: Out-of-Scope (subjective/experiential content), Contradicted (explicitly refuted by reference material or prior facts), Lacking Evidence (potentially factual but unsupported), or Abstention (expressing uncertainty or refusing to answer).
    - **Design Motivation**: FActScore only verifies against static reference documents and cannot detect cross-turn contradictions; LLM-as-Judge conflates all unverifiable content. The four-way breakdown provides labels with greater diagnostic value.

3.  **Sequential Memory & Aggregation**:
    - **Function**: Maintains dynamic factual memory to support cross-turn consistency tracking.
    - **Mechanism**: Verified and out-of-scope claims from each turn are appended to a running knowledge store, forming a dynamic factual memory. Verification of subsequent turns is conditioned on this—verified claims reinforce prior information, while contradicted claims indicate factual drift. At the end of the assessment, all claim-level results are aggregated to generate the VISTA Score.
    - **Design Motivation**: The correctness of a claim in dialogue may depend on information established in earlier turns (e.g., the example of Elvis Presley being called the "King of Rock and Roll"); without sequential memory, correct cross-turn references would be misjudged as "Lacking Evidence."

### Loss & Training

VISTA does not require training—it is a prompt-based evaluation framework that invokes LLMs to complete sub-tasks at each stage via structured prompt templates. It supports zero-shot and few-shot configurations and provides a model-agnostic unified interface.

## Key Experimental Results

### Main Results

**Automatic Evaluation (Unverifiable Turn Detection Accuracy %)**

| Dataset | Model | VISTA | FActScore | LLM-as-Judge |
| :--- | :--- | :--- | :--- | :--- |
| AIS | GPT-4o | **63.00** | 56.80 | 56.80 |
| BEGIN | GPT-4o | **83.20** | 65.80 | 70.40 |
| FaithDial | DeepSeek | **81.70** | 63.75 | 55.45 |
| FADE | Llama-70B | **65.10** | 56.65 | 62.28 |
| FaithDial | Qwen-32B | **75.73** | 58.41 | 35.89 |
| BEGIN | Mistral-7B | **72.00** | 53.80 | 57.40 |

**Human Evaluation (Alignment with Consensus Labels)**

| Model | Turn Accuracy | Claim Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- |
| GPT-5 | 92.51 | 81.53 | 69.09 |
| GPT-4o | 91.19 | 75.68 | 62.41 |
| DeepSeek | 92.51 | 79.73 | 67.15 |

### Ablation Study

| Configuration | FaithDial Accuracy | Description |
| :--- | :--- | :--- |
| VISTA (Full) | 81.70 | Full model |
| Remove Accumulated Claims | 81.74 | Minimal impact; most claims verifiable from current doc |
| Remove Dialogue History | 77.24 | 4.5% decrease; dialogue context is vital for decomposition |
| Zero-shot | 70.17 | 11.5% decrease; few-shot examples are important for modeling dialogue phenomena |

### Key Findings

- VISTA's advantage stems mainly from dialogue contextualization and few-shot examples rather than the claim accumulation mechanism—this correlates with FaithDial where most claims are verifiable from current reference documents.
- In contradiction detection tasks, performance improved significantly after adding full dialogue history (DeepSeek from 60.0% to 77.0%, GPT-5 from 54.2% to 86.0%); accumulated claims were the key driver.
- Human annotators and original dataset labels disagreed on 26.4% of turns, with 86.7% of those cases being original labels incorrectly marking unverifiable content as verifiable—indicating that claim-level decomposition improves annotation quality.
- Abstention recognition accuracy reached 90.6%, showing that VISTA can reliably distinguish between refusal to answer and hallucinations.

## Highlights & Insights

- The paradigm shift of modeling dialogue factuality as a dynamic process rather than a static property aligns elegantly with linguistic Common Ground theory—this is a theory-driven system design.
- Claim-level decomposition not only boosts automatic evaluation accuracy but also improves human annotation consistency (Krippendorff’s $\alpha = 0.832$); this "byproduct" may possess more practical value than the primary results.
- The design of the four-way unverifiable classification can be directly transferred to other NLG evaluation tasks, such as faithfulness assessment in summarization and translation.

## Limitations & Future Work

- Current benchmarks contain few contradiction and abstention cases; thus, VISTA's robustness in these scenarios is insufficiently validated.
- The accumulated claim mechanism has limited effect in short dialogues, requiring longer multi-turn dialogue benchmarks to fully demonstrate its value.
- VISTA relies on LLMs for sub-tasks across stages, leading to potential error propagation—faults in claim decomposition propagate to subsequent verification.
- The focus is solely on RAG scenarios, with no handling of open-domain fact verification.

## Related Work & Insights

- **vs FActScore**: Both use claim-level decomposition, but FActScore performs isolated verification on single documents, while VISTA adds dialogue context and cross-turn tracking.
- **vs LLM-as-Judge**: LLM-as-Judge provides holistic judgment but lacks interpretability; VISTA provides claim-level diagnostics through a structured pipeline.

## Rating

- Novelty: ⭐⭐⭐⭐ Modeling dialogue factuality as a dynamic process is a meaningful perspective shift, though individual components (decomposition, verification) are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across four benchmarks, eight models, human evaluation, ablation analysis, and specific contradiction/abstention testing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous experimental logic, and close integration of theory and practice.
- Value: ⭐⭐⭐⭐ Provides an evaluation framework better suited for dialogue scenarios than FActScore, though actual deployment costs remain high.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Confidence Estimation for LLMs in Multi-turn Interactions](confidence_estimation_for_llms_in_multi-turn_interactions.md)
- [\[ACL 2026\] DualFact: A Multimodal Fact Verification Framework for Procedural Video Understanding](dualfact_a_multimodal_fact_verification_framework_for_procedural_video_understan.md)
- [\[CVPR 2026\] Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline](../../CVPR2026/video_understanding/pioneering_perceptual_video_fluency_assessment_a_novel_task_with_benchmark_datas.md)
- [\[ICML 2026\] Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](../../ICML2026/video_understanding/video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](../../NeurIPS2025/video_understanding/sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)

</div>

<!-- RELATED:END -->
