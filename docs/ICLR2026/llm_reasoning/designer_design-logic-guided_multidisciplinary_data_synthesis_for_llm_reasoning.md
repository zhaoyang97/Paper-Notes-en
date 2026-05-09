---
title: >-
  [Paper Note] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][data synthesis] This paper introduces Design Logic—reusable meta-knowledge reverse-engineered from real exam questions—to guide the synthesis of multidisciplinary reasoning problems from raw text. A dataset of 4.7 million questions spanning 75 disciplines is constructed, and base models fine-tuned via SFT on this data surpass their officially post-trained counterparts.
tags:
  - ICLR 2026
  - LLM Reasoning
  - data synthesis
  - design logic
  - multidisciplinary reasoning
  - question generation
  - SFT
date: 2026-05-08
content_hash: 3b87508769665f3b
---

# DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning

**Conference**: ICLR 2026  
**arXiv**: [2508.12726](https://arxiv.org/abs/2508.12726)  
**Code**: [https://attention-is-all-i-need.github.io/Design-Logic-Reasoning](https://attention-is-all-i-need.github.io/Design-Logic-Reasoning)  
**Area**: LLM Reasoning  
**Keywords**: data synthesis, design logic, multidisciplinary reasoning, question generation, SFT

## TL;DR
This paper introduces Design Logic—reusable meta-knowledge reverse-engineered from real exam questions—to guide the synthesis of multidisciplinary reasoning problems from raw text. A dataset of 4.7 million questions spanning 75 disciplines is constructed, and base models fine-tuned via SFT on this data surpass their officially post-trained counterparts.

## Background & Motivation

**Background**: LLMs have achieved remarkable gains in mathematical and programming reasoning (aided by abundant open-source competition problems), yet they still lag behind human experts in university-level cross-disciplinary reasoning. The core bottleneck is a severe shortage of high-quality multidisciplinary reasoning training data.

**Limitations of Prior Work**: (a) **Query-centric methods** (e.g., Evol-Instruct) expand data by paraphrasing seed questions, constrained by seed coverage and model bias; (b) **Document-centric methods** generate questions from text but struggle to control difficulty and diversity, often degenerating into factual recall; (c) Existing datasets are heavily skewed in disciplinary distribution (mathematics dominates), with insufficient cross-disciplinary coverage.

**Key Challenge**: How can one synthesize exam-level questions at scale from raw text (books, web pages) with multi-step reasoning depth, controllable difficulty, and high diversity? The absence of guiding principles leaves LLMs without a mechanism to transform knowledge into complex problems.

**Goal**: To provide a systematic multidisciplinary reasoning data synthesis pipeline that synthesizes not only questions but also the underlying "question-design methodology."

**Key Insight**: Human educational experts follow a structured design process when constructing exam questions (identify objectives → build context → design reasoning path → craft distractors → validate). If this "design logic" can be extracted from real exam questions, it can serve as a reusable template applicable to new source texts.

**Core Idea**: Reverse-engineer 125,000 Design Logic entries (question-construction meta-knowledge) from real exam questions, then apply a retrieve-and-generate approach to match these logic entries with raw text, guiding the LLM to generate new questions following the same reasoning patterns from entirely new material.

## Method

### Overall Architecture
A three-stage pipeline: (1) **Data processing**—multidimensional annotation and filtering of question banks, book corpora, and web corpora; (2) **Design Logic extraction**—reverse-engineering 125,000 structured question-design logic entries from 132,000 curated real exam questions; (3) **Question synthesis**—a two-stage retrieve-and-generate process that matches Design Logic with source text and generates questions.

### Key Designs

1. **Design Logic Extraction**:

    - **Function**: Extract reusable question-construction meta-knowledge from real exam questions.
    - **Mechanism**: DeepSeek-R1 is used to analyze each real exam question by (i) inferring the question designer's thought process, (ii) tracing the construction pathway from knowledge points to the final question, and (iii) abstracting the result into structured design principles (Mermaid format). Semantic deduplication is then applied via graph clustering based on Qwen3-Embedding (threshold $\tau=0.85$), yielding 125,328 unique Design Logic entries.
    - **Design Motivation**: Design Logic constitutes meta-knowledge decoupled from specific disciplinary content—the same question-design logic can be applied to different knowledge points across different disciplines, enabling the transfer of question-construction capability.

2. **High-Quality Source Text Repository Construction**:

    - **Book corpus**: Chapter-level processing, MinHash deduplication, ModernBERT classifier for discipline annotation, BERT for readability assessment, and fineweb-edu-classifier for educational value scoring. This yields 3 million high-quality text segments.
    - **Web corpus**: Qwen3-30B is used to apply five-level quality scoring on 6.5B texts from FineFineWeb (retaining scores ≥ 3), with discipline labels re-annotated to align with the 75-discipline taxonomy.

3. **Two-Stage Retrieve-and-Generate Question Synthesis**:

    - **Function**: Identify the most compatible Design Logic for each source text segment and generate questions accordingly.
    - **Stage 1: Coarse retrieval**—compute cosine similarity between source text and Design Logic embeddings; retrieve top-5 candidates.
    - **Stage 2: Fine matching + generation**—DeepSeek-R1 selects the most suitable logic from the top-5 candidates and strictly follows its steps to generate graduate-level exam questions with reference answers from the source text.
    - **Design Motivation**: Avoids the combinatorial explosion of exhaustive matching; the coarse-to-fine strategy ensures matching quality.

4. **Response Synthesis**:

    - Qwen3-235B-A22B-Thinking is used to generate long chain-of-thought (CoT) responses for each question.
    - Question–answer pairs are used for SFT training.

### Loss & Training
- **SFT**: Standard autoregressive loss, applied to Qwen3-Base and Llama3-Base.
- **Data scale**: DLR-Book (3.04M) + DLR-Web (1.66M) = 4.7M questions covering 75 disciplines.
- **Deduplication**: MinHash + 13-gram decontamination against all evaluation benchmarks.

## Key Experimental Results

### Main Results

| Model | MMLU | MMLU-Pro | GPQA-Diamond | SuperGPQA |
|------|------|----------|-------------|-----------|
| Llama-3.1-8B-Instruct (official) | 70.86 | 47.38 | 23.18 | 20.08 |
| **Llama-3.1-8B-SFT (DLR-Web+Book)** | **84.13** | **76.04** | **65.45** | **45.06** |
| Qwen3-4B Thinking (official) | 82.87 | 69.34 | 54.70 | 43.30 |
| **Qwen3-4B-Base-SFT (DLR-Web+Book)** | **85.00** | **73.06** | **63.69** | **46.15** |

Base models fine-tuned solely on DLR data via SFT **surpass** their officially post-trained counterparts.

### Ablation Study

| Data Source | MMLU | GPQA-Diamond | Note |
|--------|------|-------------|------|
| DLR-Web only | 83.55 | 53.74 | Web source |
| DLR-Book only | 84.73 | 62.58 | Book source performs better (greater educational depth) |
| DLR-Web + Book | **85.00** | **63.69** | Complementary combination yields best results |
| OpenThoughts3 (baseline) | -- | ~50 | Design Logic data is superior |

### Key Findings
- **DLR-synthesized data is substantially more difficult**: The proportion of "Very Hard" questions far exceeds that of all baseline datasets and evaluation benchmarks; "Easy" questions account for only 0.27%–0.72%.
- **Diversity substantially exceeds baselines**: DLR data leads across five semantic diversity metrics; 1-NN Distance is approximately twice that of baselines, indicating negligible semantic redundancy.
- **Most balanced disciplinary coverage**: 75 disciplines span STEM, humanities, social sciences, and applied fields, whereas existing datasets are heavily skewed toward mathematics.
- **Book source > Web source**: DLR-Book outperforms DLR-Web on most metrics, as textbooks provide more structured and in-depth knowledge.
- **SFT base > officially post-trained models**: The most striking finding—SFT on high-quality synthetic data alone surpasses official models trained with full post-training pipelines including RL and DPO.

## Highlights & Insights
- **Design Logic as reusable meta-knowledge**: This is a foundational contribution—rather than synthesizing data, the paper synthesizes "question-design capability." The 125,000 Design Logic entries can be reused indefinitely on new texts, enabling scalable data generation.
- **"The question matters more than the answer"**: Citing Einstein, the paper underscores the primacy of high-quality question formulation. Given a well-designed question, any model can generate a response—an insight analogous to "a good prompt matters more than a good model."
- **4.7M questions across 75 disciplines**: This constitutes the largest-scale multidisciplinary reasoning dataset to date, with quality (difficulty and diversity) exceeding all baselines.
- **Implications for post-training**: SFT data quality dominates over a full SFT+RL+DPO pipeline trained on lower-quality data, challenging the assumption that RL is indispensable.

## Limitations & Future Work
- Design Logic extraction depends on existing question banks—disciplines without available exam questions cannot yield logic entries.
- CoT answer accuracy reaches only 71.48% (due to the diversity of open-ended questions), potentially introducing noise during SFT.
- RL training is not explored—it remains an open question whether adding RL/DPO on top of DLR data would yield further gains.
- The 75-discipline classification relies on LLM annotation with 90.14% accuracy, leaving approximately 10% mislabeled.

## Related Work & Insights
- **vs. Evol-Instruct (query-centric)**: Constrained by seed coverage; cannot generalize across disciplines. DESIGNER starts from raw text and achieves far broader disciplinary coverage.
- **vs. NaturalReasoning/WebInstruct (document-centric)**: Lacks guiding principles for question construction, often degenerating into factual recall. Design Logic provides structured control over question design.
- **vs. Nemotron-Post-Training**: Relatively balanced disciplinary distribution but low difficulty. DLR outperforms on both difficulty and diversity.
- **vs. OpenThoughts3**: Strong reasoning depth but heavily biased toward mathematics. DLR holds a substantial advantage across non-mathematical disciplines.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The concept of Design Logic and the reverse-engineering methodology represent a genuinely novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-model (Qwen3 + Llama3), multi-benchmark evaluation, data quality analysis, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ The pipeline is described clearly, though the paper is engineering-oriented with limited theoretical analysis.
- **Value**: ⭐⭐⭐⭐⭐ 4.7M multidisciplinary reasoning questions and 125,000 Design Logic entries are highly valuable resources for the LLM post-training community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [\[ACL 2026\] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis](../../ACL2026/llm_reasoning/mathagent_adversarial_evolution_of_constraint_graphs_for_mathematical_reasoning_.md)
- [\[AAAI 2026\] Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](../../AAAI2026/llm_reasoning/jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)
- [\[ICLR 2026\] Scaling Generalist Data-Analytic Agents](scaling_generalist_data-analytic_agents.md)
- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](../../ACL2026/llm_reasoning/efficient_prm_training_data_synthesis_via_formal_verification.md)

</div>

<!-- RELATED:END -->
