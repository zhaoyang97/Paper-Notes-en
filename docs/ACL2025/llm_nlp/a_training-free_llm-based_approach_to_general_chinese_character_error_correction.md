---
title: >-
  [Paper Note] A Training-free LLM-based Approach to General Chinese Character Error Correction
description: >-
  [ACL 2025][LLM (Other)][Chinese spelling correction] Proposal of the Chinese Character Error Correction (C2EC) task, which covers substitution, missing, and redundant error types. By extending a training-free CSC method with Levenshtein distance and a prompt-based LLM, the proposed approach achieves performance on par with models up to 50 times larger using a 14B parameter model without direct fine-tuning.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Chinese spelling correction"
  - "CSC"
  - "training-free"
  - "LLM"
  - "Levenshtein distance"
date: 2026-05-08
content_hash: d112572975e5912f
---

# A Training-free LLM-based Approach to General Chinese Character Error Correction

**Conference**: ACL 2025  
**arXiv**: [2502.15266](https://arxiv.org/abs/2502.15266)  
**Code**: None  
**Area**: LLM / NLP  
**Keywords**: Chinese spelling correction, CSC, training-free, LLM, Levenshtein distance

## TL;DR

Proposal of the Chinese Character Error Correction (C2EC) task, which covers substitution, missing, and redundant error types. By extending a training-free CSC method with Levenshtein distance and a prompt-based LLM, the proposed approach achieves performance on par with models up to 50 times larger using a 14B parameter model without direct fine-tuning.

## Background & Motivation

Chinese Spelling Correction (CSC) is a classic NLP task aiming to automatically detect and correct character errors in Chinese text. Traditional CSC tasks **only focus on character substitution errors**, such as homophone or visually similar character substitutions caused by typing errors (e.g., miswriting "已经" as "以经").

In real-world applications, however, errors in Chinese text extend beyond substitutions. **Missing characters** (e.g., omitting "校" in "我去学") and **redundant characters** (e.g., adding an extra "去" in "我去去学校") are also common. Nevertheless, current CSC datasets typically exclude these two types of errors from annotations, and evaluations frequently neglect them. This significantly **limits the practicality** of CSC models, as they cannot handle insertion and deletion operations in real-world scenarios.

**Key Challenge**: Traditional CSC methods assume that the input and output lengths are identical (one-to-one substitution), an assumption that completely fails when encountering missing or redundant errors. Most BERT-based CSC models (such as SCOPE and ECSpell) rely on this equal-length assumption.

**Key Insight**: To achieve unified error correction for all three error types **without any training**, leveraging off-the-shelf LLMs through an ingenious inference strategy.
**Core Idea**: Extend the training-free prompt-free CSC method to variable-length scenarios, utilizing Levenshtein distance to handle length changes, followed by verification and enhancement via a prompt-based LLM.

## Method

### Overall Architecture

The system adopts a two-stage pipeline architecture:
1. **Stage 1 (Prompt-free Correction)**: Uses the language model probabilities of the LLM to locate and correct erroneous characters without prompts. By calculating character-level perplexities, potential error locations are identified, followed by candidate correction generation.
2. **Stage 2 (Prompt-based Enhancement)**: Takes the corrections from the first stage as input, utilizing a well-designed prompt to guide the LLM to perform secondary verification and correction, further improving accuracy.

The input is the erroneous Chinese text, and the final output is the corrected text.

### Key Designs

1. **Prompt-free Error Detection and Correction**:

    - **Function**: Detects and corrects errors solely using the token probability distribution of the LLM, without using any prompt.
    - **Mechanism**: For each character position in the original text, compute the conditional probability $P(c_i | c_1, ..., c_{i-1})$ predicted by the LLM. If the probability at a given position is significantly lower than a threshold, that position is flagged as a potential error. Then, candidate characters (homophones, visually similar characters) are enumerated, and the replacement with the highest probability is selected.
    - **Design Motivation**: Directly leverages the language modeling capability of the LLM as an "implicit knowledge base," avoiding the overhead of training specialized models for the CSC task. LLMs naturally contain vast amounts of Chinese language knowledge learned during pre-training, which is reflected through token probabilities.

2. **Variable-length Error Correction Based on Levenshtein Distance**:

    - **Function**: Extends the prompt-free method from equal-length substitutions to support insertion and deletion operations.
    - **Mechanism**: Uses Levenshtein edit distance to align the original text and candidate corrections. Let the original text be $S$ and the candidate correction be $S'$. Calculate $d(S, S')$ using dynamic programming, constraining the edit distance to not exceed a threshold $\delta$ to avoid over-correction. For each possible edit operation (substitution, insertion, deletion), compute the language model probability gain: $\Delta P = P(S') - P(S)$, and choose the correction that maximizes this gain.
    - **Design Motivation**: Traditional CSC methods assume equal-length inputs and outputs and cannot handle missing and redundant errors. Levenshtein distance is a classic string similarity metric that naturally supports all three edit operations, aligning perfectly with the three error types in C2EC.

3. **Prompt-based LLM Enhancement**:

    - **Function**: Leverages the instruction-following ability of the LLM to perform secondary verification and correction on the first-stage results.
    - **Mechanism**: Designs prompts containing task descriptions and exemplars. The original text and first-stage corrections are jointly fed into the LLM, prompting it to judge whether the corrections are reasonable and perform further edits. The prompt explicitly specifies error types (substitution/missing/redundancy) to guide the LLM's attention to all types of errors.
    - **Design Motivation**: Although highly efficient, prompt-free methods rely on local probability judgments and might miss errors that require global semantic understanding. Prompt-based methods exploit the LLM's reasoning abilities to conduct a global review, making the two approaches complementary.

### Loss & Training

The proposed method is **completely training-free** and does not involve loss function design. Key hyperparameters include:
- Probability threshold for error detection
- Levenshtein distance upper bound $\delta$
- Search scope for candidate characters (homophone table, visually similar character table)

## Key Experimental Results

### C2EC Benchmark Construction

The authors extracted and manually verified data from the CCTC and Lemon datasets to construct a high-quality C2EC benchmark. This benchmark covers all three error types, with each sample manually audited and verified.

### Main Results

| Dataset | Metric | Ours (14B) | GPT-4o | DeepSeek-V3 (671B) | Gain |
|--------|------|-----------|--------|-------------------|------|
| CSC (Traditional) | F1 | ~SOTA | Lower | High | 14B ≈ 671B |
| C2EC (General) | F1 | ~SOTA | Lower | High | 14B without fine-tuning aligns with a 50x larger model |
| CCTC Subset | Precision | High | Medium | High | - |
| Lemon Subset | Recall | High | Medium | High | - |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Prompt-free Only | Baseline | Can only handle substitution errors; variable-length errors cannot be corrected |
| + Levenshtein | Significant improvement | Supports missing and redundant errors, leading to a substantial boost in C2EC performance |
| + Prompt-based Enhancement | Optimal | Global semantic review further reduces false corrections and missed detections |
| Different LLM Backbones | Varies | The 14B model exhibits comparable performance to the 671B model |
| Different Edit Distance Thresholds | Sensitive | Too large leads to over-correction, while too small causes missed errors |

### Key Findings
- The Levenshtein distance extension is the **largest contributor** to C2EC performance, expanding the capabilities from handling only substitution errors to all three error types.
- The prompt-based secondary enhancement contributes significantly in complex contexts, particularly for errors that require semantic understanding to resolve.
- Under the proposed framework, a 14B LLM achieves performance comparable to a 671B LLM (e.g., DeepSeek-V3), demonstrating the vital importance of method design.
- The edit distance threshold is a critical hyperparameter that requires balancing coverage and precision.

## Highlights & Insights

- **Task Definition Novelty**: Expanding CSC to C2EC covers substitution, missing, and redundant errors, filling a critical gap in the field. This task formulation is much closer to real-world application demands.
- **Completely Training-free**: Intelligently combining LLM probability analysis and instruction-following abilities achieves high-quality error correction without any fine-tuning, significantly lowering deployment barriers.
- **Small Model Rivals Large Model**: Under this design, a 14B model matches the performance of a model 50 times larger, illustrating that algorithmic design can outweigh sheer parameter scale in specific tasks.
- **Ingenious Use of Levenshtein Distance**: Classic edit distance algorithms find a new application scene in the LLM era, elegantly scaling equal-length constraints to variable-length scenarios.

## Limitations & Future Work

- **Inference Speed**: The two-stage pipeline requires multiple LLM inference calls. Character-by-character probability calculations introduce significant computation overhead, which may cause latency issues in real-time scenarios.
- **Candidate Character Search Space**: The method relies on predefined homophones and visually similar character tables, which might miss highly unconventional errors.
- **Evaluation Limitations**: Although the C2EC benchmark covers three error types, its scale and domain diversity require further expansion.
- **Multi-error Interaction**: When a single sentence contains multiple types of errors, they may interfere with one another. The model's capacity to handle such scenarios warrants further validation.
- **Cross-lingual Generalization**: The core methodology relies heavily on Chinese homophone and visually similar character knowledge, making it difficult to directly transfer to spelling correction in other languages.

## Related Work & Insights

- **vs. SCOPE/ECSpell (Traditional CSC methods)**: Traditional methods utilize fine-tuned BERT models and assume equal-length inputs and outputs, limiting them to substitution errors. Ours is training-free and supports all three error types, albeit with higher inference costs.
- **vs. Direct GPT-4 Correction**: When prompting GPT-4 directly to correct text, LLMs are prone to "over-correction," mistakenly modifying grammatically correct texts. The proposed two-stage method precisely localizes errors via probability analysis, achieving higher Precision.
- **vs. GEC (Grammatical Error Correction) methods**: GEC focuses on grammar-level errors (e.g., word order, missing particles), whereas C2EC focuses on character-level errors with a finer granularity. The two approaches are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ Innovative task definition (C2EC), clever combination of methods, though components are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐ The 26-page paper contains extensive experiments and analysis, with rigorous comparisons across multiple datasets and models.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and systematic description of methodology.
- Value: ⭐⭐⭐⭐ Fills a critical gap with the C2EC task, offering a highly practical training-free solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] NeKo: Cross-Modality Post-Recognition Error Correction with Tasks-Guided Mixture-of-Experts Language Model](neko_cross-modality_post-recognition_error_correction_with_tasks-guided_mixture-.md)
- [\[ACL 2025\] Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning](character_level_understanding.md)
- [\[ACL 2026\] VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs](../../ACL2026/llm_nlp/voyager_a_training_free_approach_for_generating_diverse_datasets_using_llms.md)

</div>

<!-- RELATED:END -->
