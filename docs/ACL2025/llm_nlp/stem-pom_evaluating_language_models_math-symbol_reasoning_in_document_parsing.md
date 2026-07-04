---
title: >-
  [Paper Note] STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing
description: >-
  [ACL2025][LLM (Other)][Mathematical Symbol Classification] This paper proposes the STEM-PoM benchmark dataset (2K+ math-symbol instances), combining Part-of-Math Tagging with document parsing to systematically evaluate LLMs' capacity to classify contextual polysemy in mathematical symbols. It demonstrates that improvements in symbol classification can be transferred to enhance downstream mathematical reasoning performance.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Mathematical Symbol Classification"
  - "Part-of-Math Tagging"
  - "benchmark"
  - "Document Parsing"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: f20b8961ef702c54
---

# STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing

**Conference**: ACL2025  
**arXiv**: [2411.00387](https://arxiv.org/abs/2411.00387)  
**Code**: [jiaruzouu/STEM-PoM](https://github.com/jiaruzouu/STEM-PoM)  
**Area**: LLM/NLP  
**Keywords**: Mathematical Symbol Classification, Part-of-Math Tagging, benchmark, Document Parsing, Mathematical Reasoning

## TL;DR
This paper proposes the STEM-PoM benchmark dataset (2K+ math-symbol instances), combining Part-of-Math Tagging with document parsing to systematically evaluate LLMs' capacity to classify contextual polysemy in mathematical symbols. It demonstrates that improvements in symbol classification can be transferred to enhance downstream mathematical reasoning performance.

## Background & Motivation
1. **Contextual Polysemy of Math Symbols**: The same mathematical symbol (e.g., $y$) can represent a variable, a constant, or an operator in different formulas and contexts. LLMs struggle to determine its semantic role based solely on the symbol itself.
2. **Scarcity of Part-of-Math Tagging Data**: Analogous to Part-of-Speech Tagging, the mathematical domain has long lacked large-scale, multi-disciplinary, and multi-class benchmark datasets for symbol tagging.
3. **Limitations of Prior Work**: Existing efforts (such as annotations on DLMF) derive from a single literary source with single, self-consistent symbol classifications, failing to reflect multi-disciplinary symbol polysemy in real literature.
4. **Mathematical Understanding Bottlenecks in Document Parsing**: Traditional methods (e.g., LaTeXML) and advanced LLMs display substantial deficiencies in pattern matching and semantic understanding of abstract symbols when processing math-intensive documents.
5. **Foundational Capability for Mathematical Reasoning**: Accurately classifying math symbols is a preliminary capability for LLMs to perform more complex mathematical reasoning (such as problem-solving and proofs), but this aspect has not yet been systematically evaluated.
6. **Need for Multi-Disciplinary Coverage**: Real-world STEM literature spans mathematics, physics, chemistry, computer science, etc., requiring an evaluation benchmark with broad coverage and rich annotation hierarchies.

## Method

### Overall Architecture: STEM-PoM Benchmark Dataset Construction and Evaluation
- **Function**: Extracts mathematical symbols from math-intensive arXiv documents to construct a two-level hierarchical classification benchmark containing 2,109 annotated instances, on which various LLMs are systematically evaluated.
- **Why**: To fill the data gap of Part-of-Math Tagging in large-scale multi-disciplinary scenarios, providing a standardized tool to evaluate and improve the mathematical symbol understanding of LLMs.
- **How**: Randomly samples papers from 10,000 arXiv articles, pre-filters symbol sets using MTDE, and employs 33 domain experts to annotate using the self-developed STEM-PoM Labeler tool. Ultimately, 2,109 symbol instances are extracted from 453 papers, averaging 4.7 symbols per paper.

### Key Designs 1: Two-Level Hierarchical Attribute Classification System
- **Function**: Defines 4 first-level main attributes (Variable / Constant / Operator / Unit Descriptor) and 6 second-level sub-attributes (Scalar/Vector/Matrix for variables; Local/Global/Discipline-Specific for constants and operators).
- **Why**: The semantics of mathematical symbols depend not only on "what type they are", but also on more fine-grained dimension and scope information. Two-level classification enables a more comprehensive evaluation of the model's depth of understanding.
- **How**: Experts first annotate the main attributes for each symbol, and then annotate sub-attributes based on context. The annotation process is validated through consistency checks and inter-annotator agreement verification (average Cohen's Kappa of 0.903).

### Key Designs 2: Multi-Granularity Context Evaluation Strategy
- **Function**: Provides three context lengths (single sentence / ten sentences / full text) for each symbol to evaluate the model's classification accuracy respectively.
- **Why**: To investigate the impact of contextual information volume on LLMs' mathematical symbol understanding, and the differences in context utilization efficiency among models of different scales.
- **How**: Domain experts select the complete sentences most relevant to the symbols as context through pre-defined windows, ensuring the accuracy and relevance of input information.

### Key Designs 3: Downstream Mathematical Reasoning Transfer Validation
- **Function**: Evaluates changes in model reasoning performance on GSM8K, MATH, and OlympiadBench after LoRA fine-tuning on STEM-PoM.
- **Why**: To verify whether "improvements in mathematical symbol classification capability can transfer to mathematical reasoning tasks", thereby proving the practical value of STEM-PoM.
- **How**: Llama2-13B, Mixtral-8x7B, Llama3.1-70B, and GPT-4o are first LoRA fine-tuned on STEM-PoM, and then evaluated on downstream tasks using 3-shot CoT for pass@1.

## Key Experimental Results

### Experiment 1: First-level Classification Accuracy (Different Context Lengths & Models)

| Model | Single Sentence | Ten Sentences | Full Text |
|------|------|------|------|
| LSTM | 18.7% | 22.6% | - |
| Llama2-13B | 36.8% | 42.7% | 45.9% |
| Mistral-8x7B | 47.3% | 49.8% | 53.6% |
| Llama3.1-70B | 48.9% | 53.0% | 51.7% |
| Claude3.5-Sonnet | 63.7% | 65.9% | 66.7% |
| GPT-3.5-turbo | 56.8% | 58.7% | 60.6% |
| GPT-4o | 64.9% | 67.4% | 68.5% |

**Findings**: The SOTA model (GPT-4o) only achieves 68.5% accuracy under full-text context, indicating that the task is far from solved. GPT-4o stably leads Llama3.1-70B by around 16% across all three context lengths, indicating that pre-trained knowledge volume is a decisive factor. Smaller models benefit more from longer contexts.

### Experiment 2: Transfer Effect of STEM-PoM Fine-Tuning on Downstream Mathematical Reasoning

| Model | GSM8K | MATH | OlympiadBench | Average |
|------|-------|------|---------------|------|
| Llama2-13B | 42.5% | 29.1% | 11.5% | 27.7% |
| + LoRA (STEM-PoM) | 44.6% (+2.1) | 31.3% (+2.2) | 13.4% (+1.9) | 29.8% (+2.1) |
| Mixtral-8x7B | 72.4% | 32.6% | 13.7% | 39.6% |
| + LoRA (STEM-PoM) | 74.1% (+1.7) | 34.1% (+1.5) | 16.4% (+2.7) | 41.5% (+1.9) |
| Llama3.1-70B | 91.6% | 47.1% | 26.4% | 55.0% |
| + LoRA (STEM-PoM) | 93.2% (+1.6) | 48.8% (+1.7) | 28.2% (+1.8) | 56.7% (+1.7) |
| GPT-4o | 94.3% | 88.7% | 39.6% | 74.2% |
| + LoRA (STEM-PoM) | 95.2% (+0.9) | 88.9% (+0.2) | 41.2% (+1.6) | 75.1% (+0.9) |

**Findings**: All models exhibit improved downstream reasoning after fine-tuning on STEM-PoM (average +0.9 to +2.1), with the improvement particularly pronounced on the highly challenging OlympiadBench task (up to +2.7). This confirms that symbol classification capability can indeed positively transfer to mathematical reasoning.

## Highlights & Insights
- **Novel Task Definition**: Systematizes Part-of-Math Tagging and integrates it with document parsing, proposing a two-level hierarchical classification system. This is the first large-scale multi-disciplinary benchmark in this direction.
- **High-Quality Annotations**: Annotated by 33 domain experts with an average Cohen's Kappa of 0.903, ensuring extremely high data quality.
- **Revealing LLM Blind Spots**: Even GPT-4o only achieves approximately 68% accuracy, fully exposing a significant deficiency of SOTA models in understanding mathematical symbols.
- **Practical Transfer Value**: Fine-tuning on STEM-PoM transfers to enhance downstream reasoning, validating the hypothesis that symbol understanding serves as a foundational capability for mathematical reasoning.

## Limitations & Future Work
- **Limited Dataset Scale**: 2,109 instances and 453 source papers are statistically small, which may not fully cover all academic disciplines and symbol polysemy scenarios.
- **Limited Downstream Transfer Gains**: Reasoning improvement after fine-tuning is around 1-2 percentage points, showing that symbol classification is only one of many factors in mathematical reasoning and should not be over-interpreted.
- **Limited to English arXiv Literature**: The dataset does not cover non-English academic literature and other math-intensive text types such as textbooks.
- **Inadequate Sample Size for Specific Sub-categories**: Extremely scarce samples for Matrix and Discipline-Specific classes in the second-level classification (only 33 for Matrix), which may lead to unstable evaluation.

## Related Work & Insights

### vs DLMF-based PoM Tagging (Shan & Youssef, 2021/2024)
DLMF is based on a single literary source (Digital Library of Mathematical Functions), where symbol classifications are self-consistent and lack polysemy. STEM-PoM is derived from 10,000 multi-disciplinary arXiv papers, fully reflecting contextual polysemy and cross-disciplinary variations in real-world literature, making it closer to practical application scenarios.

### vs MTDE (Hamel et al., 2022)
MTDE focuses on symbol definition extraction (an NER-style task) and does not involve hierarchical classification of symbol attributes. STEM-PoM reuses the pre-filtered symbol set from MTDE but designs a two-level classification system on top of it, creating a deeper task that requires models to not only identify symbols but also understand their mathematical roles in specific contexts.

### vs Mathematical Reasoning Benchmarks (GSM8K / MATH / OlympiadBench)
These benchmarks evaluate end-to-end problem-solving capabilities, whereas STEM-PoM focuses on more foundational symbol understanding. They are complementary: experiments in this paper demonstrate that capability improvements on STEM-PoM can positively transfer to these downstream reasoning tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ — The first large-scale multi-disciplinary Part-of-Math Tagging benchmark with a novel task definition.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 7 models, 3 context lengths, fine-tuning, and downstream transfer with comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-founded motivation and experimental design.
- Value: ⭐⭐⭐⭐ — Reveals LLM shortcomings in mathematical symbol understanding, providing valuable evaluation tools and transferable training data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting](dolphin_document_image_parsing_via_heterogeneous_anchor_prompting.md)
- [\[ACL 2025\] Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes](mathneuro_math_reasoning_isolation.md)
- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)

</div>

<!-- RELATED:END -->
