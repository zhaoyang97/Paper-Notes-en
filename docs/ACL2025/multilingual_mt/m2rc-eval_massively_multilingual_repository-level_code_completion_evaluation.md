---
title: >-
  [Paper Note] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation
description: >-
  [ACL 2025][Multilingual & Machine Translation][Code completion] Proposes M2rc-Eval, a large-scale multilingual repository-level code completion benchmark covering 18 programming languages, combined with AST-based fine-grained annotations at both bucket and semantic levels, and constructs the M2rc-Instruct instruction corpus to enhance model performance.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Code completion"
  - "Repository-level code"
  - "Multilingual evaluation"
  - "Abstract Syntax Tree"
  - "Fine-grained annotation"
date: 2026-05-08
content_hash: ae6df39302da6419
---

# M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation

**Conference**: ACL 2025  
**arXiv**: [2410.21157](https://arxiv.org/abs/2410.21157)  
**Code**: [github.com/M2RC-Eval-Team/M2RC-Eval](https://github.com/M2RC-Eval-Team/M2RC-Eval)  
**Area**: Multilingual Translation  
**Keywords**: Code completion, Repository-level code, Multilingual evaluation, Abstract Syntax Tree, Fine-grained annotation

## TL;DR

Proposes M2rc-Eval, a large-scale multilingual repository-level code completion benchmark covering 18 programming languages, combined with AST-based fine-grained annotations at both bucket and semantic levels, and constructs the M2rc-Instruct instruction corpus to enhance model performance.

## Background & Motivation

Repository-level Code Completion is an active research topic in software engineering that, unlike simple in-file completion, requires understanding cross-file contextual dependencies. Existing benchmarks have the following limitations:

**Insufficient Language Coverage**: Mainstream benchmarks like CrossCodeEval only cover 4 languages (Python, Java, TypeScript, C#) and RepoBench covers only 2, failing to comprehensively evaluate the multilingual capabilities of code LLMs.

**Lack of Fine-grained Analysis**: Existing benchmarks typically only report average scores across all languages, ignoring fine-grained capability differences across various completion scenarios.

**Coarse Difficulty Classification**: Most methods only consider the number of involved files, neglecting the structural and semantic context of code within a project.

These issues create a significant blind spot when evaluating the multilingual repository-level code completion capabilities of code LLMs, making a benchmark with broader coverage and finer annotations highly desirable.

## Method

### Overall Architecture

The M2rc-Eval system consists of three core components:

1. **M2rc-Eval Benchmark**: Covers 18 programming languages (C, C#, C++, Go, HTML, Haskell, Java, JavaScript, Kotlin, Lua, Objective-C, PHP, Python, R, Ruby, Rust, Scala, TypeScript), with 100 validation samples and 500 test samples for each language.
2. **Fine-grained Annotation System**: Provides bucket-level and semantic-level annotations based on Abstract Syntax Trees (AST).
3. **M2rc-Instruct Instruction Corpus**: Contains 50,000 files per language, used for fine-tuning to boost completion performance.

### Key Designs

**Data Collection and Quality Control**:
- Data is sourced from The Stack v2 (a license-compliant GitHub repository subset), keeping repositories with $>5$ stars and comprising 10-50 files.
- The overall data pool is constructed from 431,353,244 files.
- The completion cursor position is determined by parsing the AST and randomly selecting a node rather than a random string, ensuring the integrity of identifiers and statements.

**Quality Filtering Rules**:
- The completion cursor spans no more than 5 lines.
- If the completion ground truth is shorter than 20 characters, at least 20% must be alphabetic.
- Repositories in M2rc-Eval do not appear in M2rc-Instruct.
- 30% of the completion ground truths are at least 2 lines.
- Samples that DeepSeekCoder-1.3B can accurately predict without cross-file context are filtered out.

**Bucket-level Annotation**:
- Divides the $N$ layers of the AST into $M=10$ buckets.
- The $i$-th layer belongs to the $\lceil i/(N/M) \rceil$-th bucket.
- Reflects completion depth and difficulty: deeper layers (higher bucket numbers) represent more complex code structures.

**Semantic-level Annotation**:
- Predefines 11 major semantic categories: program structure, declarations and definitions, control flow structures, expressions, data types, statements, modifiers and attributes, comments and documentation, preprocessor directives, identifiers and scopes, and special language structures.
- Specific subcategories are designed for different languages.
- Maps grammar tags from the Tree-sitter parser to semantic labels.

**Cross-file Context Retrieval**:
- Follows CrossCodeEval's methodology, extracting $L$-line contiguous code segments from the same repository.
- Ranked based on Jaccard similarity.
- Appended in descending order of relevance to the in-file context until reaching the 4,096-token limit.

### Loss & Training

Fine-tuning on M2rc-Instruct adopts the standard code completion objective—predicting the completion ground truth given the in-file context and cross-file context. Training uses the Fill-In-the-Middle (FIM) paradigm, consistent with the pre-training strategies of models like Code Llama and DeepSeek-Coder.

## Key Experimental Results

### Main Results

Evaluated on three code LLMs (Code Llama-7B, StarCoder-7B, and DeepSeekCoder-6.7B) using EM (Exact Match) and ES (Edit Similarity) metrics:

**18-language average results of Baseline (in-file context only) $\rightarrow$ +Retrieval $\rightarrow$ +Retrieval & Tuning**:

| Model | Baseline EM | +Retrieval EM | +Tuning EM |
|------|-----------|-------------|----------|
| Code Llama-7B | 19.4% | ~21% | ~43% |
| StarCoder-7B | ~20% | ~23% | ~46% |
| DeepSeekCoder-6.7B | ~22% | ~26% | **~48%** |

Key Observations:
- Cross-file retrieval augmentation yields about a 2-4% EM improvement.
- Fine-tuning on M2rc-Instruct brings a massive EM improvement of about 20-25%, showing remarkable effectiveness.
- DeepSeekCoder-6.7B achieves the best performance after fine-tuning.

**Comparison with Existing Benchmarks**:

| Benchmark | # Languages | Fine-Grained Annotation | Training Set | # Test Repositories |
|------|-------|----------|-------|----------|
| RepoBench | 2 | ✗ | ✓ | 1669 |
| CrossCodeEval | 4 | ✗ | ✗ | 1002 |
| R2C2-Bench | 4 | ✗ | ✓ | 1353 |
| **M2rc-Eval** | **18** | **✓** | **✓** | **5993** |

### Ablation Study

**Bucket-level Difficulty Analysis**:
- As the bucket number increases (AST depth increases), the completion difficulty gradually rises.
- Completion EM for shallow nodes (buckets 1-3) is around 50%+, dropping to about 30% for deep nodes (buckets 8-10).
- Fine-tuning shows more pronounced improvements on deep nodes.

**Semantic-level Analysis**:
- Performance varies significantly across different languages on the same semantic categories.
- Declarations and definitions show the best completion performance.
- Control flow structures and special language structures are relatively difficult to complete.
- Different languages have distinct bottlenecks: for example, Haskell is challenging in program structure, while Go is relatively easier in control flow.

### Key Findings

1. Fine-tuning on M2rc-Instruct brings significant improvements across all 18 languages, with average EM increases of over 20 percentage points.
2. Cross-file context retrieval has limited direct impact on completion performance (about 2-4% EM), but its effect is amplified when combined with fine-tuning.
3. The complexity of cross-file dependencies varies heavily across programming languages, with HTML being the lowest, and Scala and C++ being the highest.
4. Bucket-level annotations effectively distinguish completion scenarios of different difficulty levels.
5. Semantic-level annotations reveal how model capabilities vary across different code structures, pointing the way for model improvements.

## Highlights & Insights

1. **AST-Driven Completion Cursor Selection**: Compared to choosing random strings, selecting based on AST nodes ensures the semantic integrity of the completions.
2. **Two-Level Fine-Grained Annotation System**: The bucket level (depth-wise difficulty) and semantic level (code semantic category) enable multidimensional analytical capabilities.
3. **"Reverse Filtering" in Quality Control**: Removing samples that small models can predict without cross-file context ensures the test set truly evaluates repository-level capacity.
4. **Broad Coverage**: Covering 18 programming languages spanning both mainstream and niche languages fills a critical gap in multilingual code completion evaluation.
5. **Practical Value**: M2rc-Instruct provides ready-to-use, multilingual, repository-level code instruction data.

## Limitations & Future Work

1. The completion cursor position is restricted to a maximum of 5 lines, which may underestimate the difficulty of long-sequence completion.
2. Only string matching metrics (EM and ES) are used, without considering semantic equivalence (such as functionally equivalent code written differently).
3. Grammatical tags from Tree-sitter are highly fine-grained; mapping them to semantic labels might experience some information loss.
4. Only 7B-scale models are evaluated, without covering larger or more modern code LLMs.
5. The cross-file retrieval strategy remains simple (Jaccard similarity); more advanced retrieval methods could be explored.

## Related Work & Insights

- **CrossCodeEval**: A 4-language repository-level completion benchmark; this work significantly expands on it in terms of language coverage and fine-grained annotation.
- **RepoBench**: Establishes a retrieval-augmented paradigm for repository-level completion; this work adopts and extends it.
- **MultiPL-E / McEval**: Multilingual in-file code generation benchmarks; this work generalizes the multilingual approach to the repository level.
- **R2C2-Bench**: A 4-language benchmark; this work comprehensively surpasses it in scale and annotation dimensions.
- **Insight**: Fine-grained evaluation and annotation systems are crucial for diagnosing model capability bottlenecks.

## Rating

- **Novelty**: ⭐⭐⭐ — The core contributions lie in benchmark construction and annotation design; technical novelty is moderate.
- **Practicality**: ⭐⭐⭐⭐⭐ — Fills an important gap in multilingual repository-level code completion evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 18 languages $\times$ 3 models $\times$ 3 settings.
- **Writing Quality**: ⭐⭐⭐ — Detailed content, but with too many tables, making the main narrative line sometimes unclear.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages](milic-eval_benchmarking_multilingual_llms_for_chinas_minority_languages.md)
- [\[ACL 2025\] MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](maxife_multilingual_and_cross-lingual_instruction_following_evaluation.md)
- [\[ACL 2025\] KnowCoder-X: Boosting Multilingual Information Extraction via Code](knowcoder-x_boosting_multilingual_information_extraction_via_code.md)
- [\[ACL 2025\] Beyond N-Grams: Rethinking Evaluation Metrics and Strategies for Multilingual Abstractive Summarization](beyond_n-grams_rethinking_evaluation_metrics_and_strategies_for_multilingual_abs.md)

</div>

<!-- RELATED:END -->
