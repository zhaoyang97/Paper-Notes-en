---
title: >-
  [Paper Note] ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries
description: >-
  [ACL 2025][Hallucination Detection][Code summarization] Proposes the Entity Tracing Framework (ETF), a hallucination detection framework that extracts code entities via static program analysis and verifies whether these entities are correctly described in the generated summaries using LLMs. Combined with the first-of-its-kind CodeSumEval dataset (~10K samples), it achieves a 73% F1 score in code summary hallucination detection.
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Code summarization"
  - "entity tracing"
  - "program analysis"
  - "LLM evaluation"
date: 2026-05-08
content_hash: fafde239ecf39206
---

# ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries

**Conference**: ACL 2025  
**arXiv**: [2410.14748](https://arxiv.org/abs/2410.14748)  
**Code**: Yes (GitHub)  
**Area**: Hallucination Detection  
**Keywords**: Code summarization, hallucination detection, entity tracing, program analysis, LLM evaluation

## TL;DR

Proposes the Entity Tracing Framework (ETF), a hallucination detection framework that extracts code entities via static program analysis and verifies whether these entities are correctly described in the generated summaries using LLMs. Combined with the first-of-its-kind CodeSumEval dataset (~10K samples), it achieves a 73% F1 score in code summary hallucination detection.

## Background & Motivation

With the widespread application of LLMs in Code Summarisation tasks (e.g., Amazon Q Developer, IBM Watsonx), the generated code documentation, though highly fluent, may contain hallucinations—meaning the summary content does not align with the actual functionality of the source code. This problem is particularly dangerous because:

**Hallucinations in code summaries are difficult to detect**: The generated text looks plausible but is actually incorrect, which may even mislead novice developers.

**Complex interactions between code and natural language**: Models might infer functionality based on identifier names (e.g., hallucinating a database query when seeing `getJobID()`), a phenomenon known as "identifier name bias".

**Existing hallucination detection methods are not directly applicable**: Hallucination detection in the natural language processing (NLP) domain typically does not require alignment with a reference text (code), whereas code summarization must treat the source code as the ground truth.

The paper points out that there are currently no datasets or methods specifically designed for hallucination detection in code summarization, highlighting a critical gap.

## Method

### Overall Architecture

The core idea of ETF is inspired by human code review behavior: when verifying documentation, developers read the code line-by-line, tracing whether entities mentioned in the documentation have corresponding counterparts in the code and whether their descriptions are accurate. This is a bottom-up process aligned with Working Memory Theory—since humans have limited working memory capacity (3-4 dependent objects), they must focus on local entity verification.

The framework consists of two main steps:

**Step 1: Entity Verification**  
Verify whether entities in the summary exist in the source code $\rightarrow$ detecting **Extrinsic Hallucinations**.

**Step 2: Entity-Intent Verification**  
For matched entities, verify whether their description in the summary is accurate $\rightarrow$ detecting **Intrinsic Hallucinations**.

### Key Designs

1. **Code-Side Entity Extraction**: Uses the Javalang static analysis tool to parse Java code into an Abstract Syntax Tree (AST), extracting all code entities such as variable names, class names, and function names. This step is entirely independent of LLMs, ensuring high reliability and accuracy.

2. **Summary-Side Entity Extraction**: Employs an LLM (GPT-4-Omni) for Named Entity Recognition (NER) based on the tagset from Tabassum et al. (2020). An LLM is used instead of rule-based systems because summaries suffer from severe polysemy issues—for instance, "list", "while", and "if" can function as code entities or common natural language words. A filtering step is also introduced to remove entities hallucinated by the LLM.

3. **Entity Matching**: Maps summary entities back to code using heuristic string matching rules. Sentences containing unmatched entities are flagged as extrinsic hallucinations. Matched entities proceed to the next validation step.

4. **Intent Extraction and Verification**: For each matched entity, string matching (instead of querying an LLM/prompting) is used to extract sentences describing the entity from the summary. String matching is preferred over LLM prompting because prompting often introduces additional hallucinations (such as fabricated sentences or inaccurate extractions). Then, a <code, entity, relevant sentences> triplet is constructed, and an LLM is used to verify the accuracy of the description in a zero-shot manner.

5. **Summary-Level Aggregation**: If $\ge 1$ entity in a summary is flagged as hallucinated, the entire summary is classified as hallucinated. Setting the threshold to 1 is based on human-annotated data, where summaries rated as "FAIR/POOR" contain an average of 1.33 hallucinated entities.

### CodeSumEval Dataset

Selects 600 Java code snippets from the CodeXGLUE dataset and generates summaries using 7 different LLMs (including Granite-20B/34B, Llama3-8B/70B, CodeLlama-7B/34B, and Mistral-7B). It produces an average of 4.11 summaries per snippet, which are further divided into 9,933 entity-level samples.

The annotation process was conducted by 8 Java experts (all holding master's degrees or above with 4+ years of Java experience) across three levels of annotation: NER, entity description verification (CORRECT/INCORRECT/IRRELEVANT), and overall summary quality assessment. The obtained Cohen's Kappa is 0.72, indicating high agreement.

## Key Experimental Results

### Main Results: ETF vs Direct Methods (Instance-Level Hallucination Detection)

| Model | Direct P/R/F1 | ETF P/R/F1 |
|------|---------------|------------|
| GPT4-Omni | 0.48/0.50/0.28 | **0.72/0.74/0.73** |
| Gemini-2.0 | 0.51/0.50/0.42 | **0.64/0.65/0.64** |
| Mixtral-8x22B | 0.48/0.48/0.45 | **0.62/0.61/0.61** |
| Llama-3.1-70B | 0.57/0.54/0.38 | **0.62/0.62/0.54** |
| Llama3-8B | 0.59/0.51/0.26 | **0.51/0.55/0.50** |
| Mistral-7B | 0.16/0.50/0.24 | **0.51/0.50/0.41** |

### Ablation Study: Entity Mapping Statistics of Summaries Generated by Different Models

| Gen Model | Avg Summary Length | Entity Count | Mapping Rate ↑ | Unmapped Rate ↓ |
|----------|------------|--------|---------|----------|
| Llama3-70B | 313.58 | 10.01 | **88.44%** | 8.69% |
| Granite-34B | 214.50 | 6.55 | 85.69% | 12.41% |
| Llama3-8B | 257.21 | 9.45 | 84.50% | 12.77% |
| Granite-20B | 148.95 | 7.10 | 79.54% | **19.64%** |

### Key Findings

1. **ETF significantly outperforms Direct methods**: On GPT-4-Omni, the F1 score improves from 0.28 to 0.73 (+160%), demonstrating that fine-grained entity tracing is superior to coarse-grained <code, summary> judgment.
2. **Code complexity is the primary cause of hallucinations**: Within the taxonomy, Code Complexity accounts for the highest proportion, followed by Insufficient Knowledge.
3. **Larger LLMs exhibit better grounding**: Llama3-70B achieves the highest entity mapping rate (88.44%), while Granite-20B has the highest unmapped rate (19.64%), indicating that smaller models are more prone to extrinsic hallucinations.
4. **Hallucinations caused by identifier bias are particularly hard to detect**: When undefined external functions are referenced in the code, both models and human annotators exhibit ambiguity.
5. Almost all LLMs "fabricate" non-existent entities during summary-side NER extraction (with CodeLlama-7B running a fabrication rate as high as 35.68%), underscoring the absolute necessity of the filtering step.

## Hallucination Taxonomy

The paper proposes a taxonomy of four hallucination causes in code summarization:
- **HC1 Identifier Name Bias**: The model infers semantics based on variable, function, or library names rather than the actual code logic.
- **HC2 Insufficient Knowledge**: Misunderstanding unseen libraries or specific programming language features.
- **HC3 Code Complexity**: Long code snippets or complex logic (e.g., nested conditions, recursion, multiple function calls) cause comprehension failures.
- **HC4 Natural Language Context**: Comments, logging statements, or other natural language elements within the code interfere with model understanding.

## Highlights & Insights

- **Cognitive science-inspired framework design**: Formalizing the bottom-up cognitive process of human code review into an entity tracing workflow, offering a highly novel and intuitive design concept.
- **Pioneering dataset**: CodeSumEval fills a critical gap in the field of code summary hallucination detection with high-quality annotations (Kappa = 0.72).
- **High practical utility**: Unlike binary "hallucinated or not" judgments, the framework precisely localizes where the hallucination occurs in the summary and which entity is involved, providing richer interpretability.
- **Pragmatic engineering choices**: Choosing string matching over LLM prompting for intent extraction to avoid introducing further hallucinations—a realistic design decision that is highly instructive.

## Limitations & Future Work

1. Supports only Java: Highly dependent on the Javalang parser, making it inapplicable to low-resource languages (e.g., COBOL, Perl).
2. High dependency on large-scale LLMs (e.g., GPT-4) for entity detection and verification, leading to high computational costs and limited scalability.
3. The multi-stage pipeline involves multiple LLM calls, posing challenges for real-time deployment into developer tools.
4. Potential misjudgments for "creative summaries" that provide accurate but more elaborate descriptions.
5. Morphological variations of entities (e.g., "PreparedStatement" $\rightarrow$ "prepared statement") may lead to missed matches.

## Related Work & Insights

- **NL Hallucination Detection** (Manakul et al., 2023; Dhuliawala et al., 2023): Lacks reference text grounding and is not directly applicable to code scenarios.
- **Code Generation Hallucination** (Liu et al., 2024; Tian et al., 2024): Focuses on hallucinations in generated code rather than in code summaries, which is a different direction.
- **Insights**: The entity-tracing paradigm can be generalized to other scenarios requiring grounding verification (e.g., verifying scientific paper summaries against their source text).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first targeted framework specifically for code summarization hallucination detection; the cognitive science-inspired design is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Employs 7 generator models and 6 verification models, conducting multi-level evaluation (entity-level + instance-level) with rich quantitative and qualitative analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Structured clearly with abundant easy-to-understand examples and an intuitive taxonomy.
- **Value**: ⭐⭐⭐⭐ — Holds direct practical value for the quality assurance of LLM-assisted code documentation generation; the dataset has a long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence](cracking_hallucination_vhd.md)
- [\[ACL 2025\] Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge](correcting_hallucinations_in_news_summaries_exploration_of_self-correcting_llm_m.md)
- [\[NeurIPS 2025\] Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs](../../NeurIPS2025/hallucination/benfords_curse_tracing_digit_bias_to_numerical_hallucination_in_llms.md)
- [\[ACL 2025\] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)
- [\[ACL 2025\] HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs](hd-ndes_neural_differential_equations_for_hallucination_detection_in_llms.md)

</div>

<!-- RELATED:END -->
