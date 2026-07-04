---
title: >-
  [Paper Note] InspireDebate: Multi-Dimensional Evaluation-Guided Reasoning for Debating
description: >-
  [ACL 2025][Information Retrieval & RAG][Debate optimization] A two-component framework is proposed: InspireScore (a debate evaluation system integrating 4 subjective dimensions and 2 objective dimensions) and InspireDebate (a debate framework optimized through a three-stage process of CoT-SFT + multi-dimensional DPO + Web-RAG). This evaluation system improves correlation with expert judgment by 44%, and the debate performance surpasses the baseline by 57%.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Debate optimization"
  - "multi-dimensional evaluation"
  - "DPO"
  - "CoT reasoning"
  - "fact verification"
date: 2026-05-08
content_hash: dbbaa749354f7f65
---

# InspireDebate: Multi-Dimensional Evaluation-Guided Reasoning for Debating

**Conference**: ACL 2025  
**arXiv**: [2506.18102](https://arxiv.org/abs/2506.18102)  
**Code**: [https://github.com/fywang12/InspireDebate](https://github.com/fywang12/InspireDebate)  
**Area**: Others  
**Keywords**: Debate optimization, multi-dimensional evaluation, DPO, CoT reasoning, fact verification  

## TL;DR
A two-component framework is proposed: InspireScore (a debate evaluation system integrating 4 subjective dimensions and 2 objective dimensions) and InspireDebate (a debate framework optimized through a three-stage process of CoT-SFT + multi-dimensional DPO + Web-RAG). This evaluation system improves correlation with expert judgment by 44%, and the debate performance surpasses the baseline by 57%.

## Background & Motivation
**Background**: LLMs have made progress in debate tasks, including argument quality assessment and debate process simulation. Works like Debatrix have advanced debate-level automatic evaluation.

**Limitations of Prior Work**: (a) Existing evaluation systems focus on subjective dimensions (e.g., emotion, clarity), neglecting objective dimensions (e.g., factual truthfulness, logical validity), and cannot detect hallucinations and logical fallacies; (b) Debate systems lack the representation of structured reasoning processes; (c) Optimization methods lack evaluation-driven iterative self-improvement.

**Key Challenge**: Debating requires simultaneously satisfying rhetorical persuasiveness (subjective) and logical/factual correctness (objective). These two dimensions may conflict—an argument can be emotionally strong but logically flawed. Existing methods cannot optimize both dimensions simultaneously.

**Goal**: How to construct a unified subjective-objective debate evaluation system and use it to guide the multi-dimensional optimization of LLM debating capabilities?

**Key Insight**: Using first-order logic to evaluate logical validity, external search to verify factual truthfulness, and DPO with multi-dimensional scores as reward signals to optimize the model.

**Core Idea**: First establish InspireScore, an evaluation system integrating subjective (emotional appeal, clarity, arrangement, and relevance) and objective (logical symbolic reasoning + search-based fact verification) dimensions, and then iteratively optimize the debating LLM via DPO using multi-dimensional feedback from InspireScore.

## Method

### Overall Architecture
Two components are connected in series:
1. **InspireScore** (evaluation system): 4 subjective dimensions + 2 objective dimensions
2. **InspireDebate** (optimization framework): SFT (CoT integration) → DPO (guided by InspireScore) → Web-RAG (real-time fact augmentation)

### Key Designs

1. **InspireScore Subjective Evaluation**:

    - Emotional Appeal: Whether the arguments elicit a sense of agreement/empathy
    - Argument Clarity: Whether the expression is clear and concise
    - Argument Arrangement: Whether the order and structure of the arguments are reasonable
    - Topic Relevance: Whether the arguments closely adhere to the debate topic
    - Implementation: Designing structured evaluation prompts for LLMs to score across different dimensions

2. **InspireScore Objective Evaluation — Logical Validity**:

    - **Function**: Evaluates whether the reasoning in the debate logically supports the arguments
    - **Mechanism**: A two-step method—(1) LLM translates natural language arguments into first-order logic (FOL) symbolic representations; (2) Applies logical deduction rules to verify whether each step of the reasoning correctly derives the conclusion
    - **Metric**: $S_{LV} = \frac{\sum_{i=1}^{m}\sum_{j=1}^{N_i} v(\text{FOL}_i^j)}{\sum_{i=1}^{m} N_i}$, representing the proportion of argument expressions that can be correctly deduced
    - **Design Motivation**: Directly detecting logical fallacies in reasoning rather than merely evaluating whether it "sounds" reasonable

3. **InspireScore Objective Evaluation — Factual Truthfulness**:

    - **Function**: Deconstructs debate responses into independent factual statements and verifies their truthfulness using a search engine
    - **Mechanism**: Optimized based on the SAFE method, extracting facts → constructing search queries → retrieving external evidence → LLM judging whether the facts are supported
    - **Metric**: $S_{FA} = \frac{\text{number of facts verified as true}}{\text{total number of independent facts}}$

4. **InspireDebate Optimization Framework**:

    - **SFT + CoT**: Constructs structured training data containing reasoning processes and argument outputs using GPT-4o, solving the refusal behavior issue of open-source models
    - **Multi-dimensional DPO**: Two LLMs play the affirmative and negative sides respectively to debate, using InspireScore to evaluate each debate → constructing preference pairs $(y_w, y_l)$ → DPO optimization
    - **Web-RAG**: Real-time keyword extraction during debates → search engine retrieval → integrating retrieved information into argument generation

### Loss & Training
- SFT Data: 100 debate topics $\times$ CoT structured debates generated by GPT-4o
- DPO Data: 510 debate topics $\times$ self-play debates $\times$ filtered by InspireScore scoring
- Hardware: 2$\times$V100 (32G), training for 2-3 hours
- LoRA fine-tuning, lr=1e-5, 3 epochs

## Key Experimental Results

### InspireScore Evaluation Quality

| Evaluation System | Correlation with Expert Judgment | Description |
|----------|----------------|------|
| Debatrix | Baseline | Subjective dimensions only |
| **InspireScore** | **+44%** Gain in correlation | Subjective + objective dimensions |

### InspireDebate Debate Performance

| Method | Overall Gain | Description |
|------|---------|------|
| Baseline Model | - | LLaMA-8B/Qwen-1.5B etc. |
| **Inspire Version** | **+57%** | SFT+DPO+Web-RAG |

### Key Findings
- **Objective dimensions are key differentiating factors**: The inclusion of logical validity and factual truthfulness significantly improves the alignment of evaluation with expert judgments.
- **Subjective and objective dimensions may conflict**: Models may generate emotionally strong but logically flawed arguments. A unified framework can balance this tension.
- **Web-RAG improves factual reliability**: Real-time retrieval reduces factual hallucinations in debates.
- **Small models also benefit**: Small models, such as Qwen-1.5B and Phi-3.6B, also achieve significant improvements after optimization through InspireDebate.

## Highlights & Insights
- **The approach of using first-order logic to verify debate validity is highly creative**: Translating debate arguments into symbolic representations and verifying them using logical deduction rules is a good example of combining formal methods with LLMs.
- **Evaluation-driven optimization closed-loop**: InspireScore serves as both an evaluation tool and the source of reward signals for DPO, forming an iterative improvement loop of "evaluation → optimization → re-evaluation".
- **Unified subjective + objective evaluation** fills a gap in the field of debate evaluation.

## Limitations & Future Work
- Logical validity evaluation relies on the LLM to translate natural language into FOL; translation errors will affect the accuracy of the evaluation.
- Fact verification relies on search engines, which can limit the quality and coverage of search results.
- The construction of DPO preference pairs relies on the accuracy of InspireScore itself, posing a bootstrapping risk.
- Evaluation was conducted only on English debates.

## Related Work & Insights
- **vs Debatrix (Liang et al. 2024)**: Debatrix only evaluates subjective dimensions such as arguments, sources, and language; InspireScore adds logical validity and factual truthfulness.
- **vs MAD (Liang et al. 2024)**: MAD utilizes debate to enhance reasoning but lacks objective feedback; InspireDebate provides multi-dimensional feedback via InspireScore.
- **vs DebateTune (Li et al. 2024)**: DebateTune enhances argument diversity but lacks evaluation-driven optimization.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of unified subjective-objective evaluation is valuable, and the design of verifying argument validity with first-order logic is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 4 open-source + 2 closed-source models, using both automatic and human evaluations.
- Writing Quality: ⭐⭐⭐ The framework is complex but the writing is relatively clear, though some formula symbols are dense.
- Value: ⭐⭐⭐⭐ Provides a comprehensive toolchain of evaluation + optimization for LLM debating.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations](ldir_low-dimensional_dense_and_interpretable_text_embeddings_with_relative_repre.md)
- [\[ACL 2025\] Unanswerability Evaluation for Retrieval Augmented Generation](unanswerability_evaluation_for_retrieval_augmented_generation.md)
- [\[ACL 2025\] SGIC: A Self-Guided Iterative Calibration Framework for RAG](sgic_a_self-guided_iterative_calibration_framework_for_rag.md)
- [\[ACL 2025\] GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation](garage_a_benchmark_with_grounding_annotations_for_rag_evaluation.md)
- [\[ACL 2025\] HELIOS: Harmonizing Early Fusion, Late Fusion, and LLM Reasoning for Multi-Granular Table-Text Retrieval](helios_harmonizing_early_fusion_late_fusion_and_llm_reasoning_for_multi-granular.md)

</div>

<!-- RELATED:END -->
