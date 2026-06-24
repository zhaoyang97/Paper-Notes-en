---
title: >-
  [Paper Note] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models
description: >-
  [ACL 2025][LLM Pretraining][Code Simplification] This paper proposes LeanCode, a context-aware attention-score-based code simplification method. By leveraging CLS attention (for classification tasks) and encoder-decoder attention (for generation tasks) to measure token importance, LeanCode outperforms SOTA methods DietCode/SlimCode by up to 60% and 29% in code search and code summarization tasks respectively while reducing inference time by up to 40.9%.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Code Simplification"
  - "Attention Score"
  - "Pre-trained Models"
  - "Code Search"
  - "Code Summarization"
date: 2026-05-08
content_hash: 007624cbfdcdfd10
---

# LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.14759](https://arxiv.org/abs/2505.14759)  
**Code**: Yes  
**Area**: LLM Pre-training  
**Keywords**: Code Simplification, Attention Score, Pre-trained Models, Code Search, Code Summarization

## TL;DR

This paper proposes LeanCode, a context-aware attention-score-based code simplification method. By leveraging CLS attention (for classification tasks) and encoder-decoder attention (for generation tasks) to measure token importance, LeanCode outperforms SOTA methods DietCode/SlimCode by up to 60% and 29% in code search and code summarization tasks respectively while reducing inference time by up to 40.9%.

## Background & Motivation

**Background**: Pre-trained code language models (such as CodeBERT, CodeT5, and GPT-4) perform exceptionally well on downstream tasks like code search and summarization, but computational overhead increases significantly with input code length. For instance, CodeBERT restricts input to 512 tokens, leading to information loss when over-length code is truncated.

**Limitations of Prior Work**: Existing code simplification methods suffer from three key issues. DietCode uses the global average self-attention scores of all tokens to measure importance, but experiments show that the attention weights of the same token vary significantly across different contexts, making global averaging unreasonable. DietCode only utilizes encoder self-attention scores, ignoring CLS attention and encoder-decoder attention—two types of attention mechanisms directly related to downstream tasks. SlimCode categorizes tokens into 8 priority levels based on manual rules, which is too coarse-grained and might lead to misalignment between model and human cognition.

**Key Challenge**: Model attention scores should be averaged contextually (i.e., based on the statement type where the token resides) rather than globally. Furthermore, different downstream tasks should employ attention types directly related to the task (CLS for classification, encoder-decoder for generation) instead of general self-attention.

**Goal**: To design a context-aware code simplification method leveraging the model's own knowledge (rather than manual rules) to maintain downstream task performance to the maximum extent while reducing computation.

**Key Insight**: Through empirical studies, the authors find that: (1) high-variance tokens (such as tokens in method signatures) score far higher in CLS attention than other tokens; (2) encoder-decoder attention scores are directly related to generation tasks; (3) self-attention during the pre-training phase cannot replace these two downstream task attention mechanisms. These three findings directly guide the design.

**Core Idea**: Replace global averaging with context-aware, category-local attention average, and select the corresponding attention mechanism based on the downstream task type to guide token removal.

## Method

### Overall Architecture

LeanCode's overall workflow: first, fine-tune the model normally on the training set, collecting attention scores of each token during the last epoch. Then, group tokens in the training set by their statement categories to calculate the category-local attention average for each token in each category. During testing, retrieve the corresponding attention score based on the category of each token in the test code, remove a specified proportion of tokens with the lowest scores, and feed the simplified code into the model for the downstream task.

### Key Designs

1. **Category-Local Attention Average**:

    - **Function**: Calculate importance scores directly related to downstream tasks for each token within a specific statement context.
    - **Mechanism**: Categorize code statements into 21 categories (method signatures, return statements, variable declarations, etc.), and calculate the average attention score for each token $t$ in category $c$ across all its occurrences: $\mu_t^c = \frac{\sum_{j=1}^{m} \sum_{t \in p_k, L(p_k) \in c} s_t}{n_t^c}$. CLS attention score $s_t$ is used for classification tasks, while encoder-decoder attention score is used for generation tasks.
    - **Design Motivation**: Experiments show that the variance of a token's attention score across different statement types is extremely high (e.g., tokens in method signatures score much higher than those in loop bodies); global averaging would drown out these contextual differences. Category-local averaging reduces the variance by 0.55 to 844 times.

2. **Task-Aware Attention Selection Strategy**:

    - **Function**: Select the most relevant attention mechanism based on downstream task types.
    - **Mechanism**: For classification tasks like code search, use the attention score from the CLS token to each input token (Eq. 5), since the CLS vector is fed directly to the fully-connected layer for classification decisions. For sequence-to-sequence tasks like code summarization, use encoder-decoder attention scores (Eq. 6), as the decoder's attention on source tokens during token generation directly reflects their importance to the generation task.
    - **Design Motivation**: Self-attention in the pre-training phase serves pre-training objectives like MLM/RTD and cannot directly reflect token importance in downstream tasks; CLS and EnDe attention are directly involved in downstream decisions.

3. **Token-level Removal Algorithm**:

    - **Function**: Remove the least important tokens up to a target simplification ratio while ensuring important tokens are retained.
    - **Mechanism**: For each code snippet, retrieve the attention score $\mu_t^c$ for each token under its corresponding statement category, then remove tokens one by one from lowest to highest score until the target simplification ratio $\mathcal{X}$ is met. Unlike DietCode, LeanCode performs pure token-level deletion instead of state-level deletion first, preventing the loss of critical tokens.
    - **Design Motivation**: DietCode's two-stage (statement-level then token-level) deletion strategy leads to complex knapsack optimization and is highly time-consuming (e.g., 9 hours for 10% simplification). LeanCode's pure token-level strategy is dozens of times more efficient.

### Loss & Training

During the training phase, standard downstream task losses are used (cross-entropy for code search, sequence generation loss for code summarization). Attention scores are only collected during the last training epoch, with an extra training overhead of only about 5% (e.g., increasing standard training from 300 minutes to 315.5 minutes).

## Key Experimental Results

### Main Results (Code Search MRR)

| Ratio | DietCode (CodeBERT) | SlimCode (CodeBERT) | LeanCode (CodeBERT) | DietCode (CodeT5) | SlimCode (CodeT5) | LeanCode (CodeT5) |
|--------|-------|--------|--------|-------|--------|--------|
| Base | 0.726 | 0.726 | 0.726 | 0.747 | 0.747 | 0.747 |
| 10% | 0.663 (↓8.67%) | 0.731 (↑0.68%) | 0.728 (↑0.27%) | 0.699 (↓6.42%) | 0.738 (↓1.2%) | 0.743 (↓0.53%) |
| 30% | 0.529 (↓27.13%) | 0.700 (↓3.58%) | 0.716 (↓1.37%) | 0.624 (↓16.46%) | 0.723 (↓3.21%) | 0.724 (↓3.07%) |
| 50% | 0.429 (↓40.9%) | 0.594 (↓18.18%) | 0.688 (↓5.23%) | 0.561 (↓24.89%) | 0.641 (↓14.19%) | 0.706 (↓5.48%) |

### Ablation Study (Replacement study: LeanCode weights + DietCode algorithm)

| Ratio | LeanCode Weights + DietCode Algorithm (MRR) | Original DietCode (MRR) | Original LeanCode (MRR) |
|--------|------|------|------|
| 10% | 0.701 (↓3.44%) | 0.663 (↓8.67%) | 0.728 (↑0.27%) |
| 30% | 0.702 (↓3.31%) | 0.529 (↓27.13%) | 0.716 (↓1.37%) |
| 50% | 0.682 (↓6.06%) | 0.429 (↓40.9%) | 0.688 (↓5.23%) |

### Key Findings

- At a 50% simplification ratio, LeanCode's MRR drops by only 5.23%~5.48%, whereas DietCode drops by 40.9%~24.89%, demonstrating an extremely prominent advantage.
- Interestingly, LeanCode and SlimCode even outperform the unsimplified baseline at low simplification ratios. This is because removing low-quality tokens allows subsequent high-quality tokens to fit into the model's 512-token input window.
- SlimCode's performance drops sharply once the simplification ratio exceeds 30%, as its 8 priority levels cannot distinguish tokens of similar importance.
- LeanCode's pruning time is approximately 1/10 of DietCode's, falling into the same order of magnitude as SlimCode (minutes vs. hours).
- GPT-4o cross-model transfer experiments verify that the simplified code from LeanCode remains effective on other models; code search accuracy improves by 0.49% under a 30% simplification ratio.

## Highlights & Insights

- **From Model Knowledge Rather than Human Rules**: LeanCode lets the model "tell you" which tokens are important. This aligns better with the model's actual cognitive patterns than SlimCode's manually defined 8 priority levels. When human rules misalign with model cognition (e.g., operators being critical to the model in certain contexts), LeanCode handles them correctly whereas SlimCode fails.
- **Counter-Intuitive Finding of 'Removal Leading to Improvement'**: The performance improvement over baseline at low simplification ratios reveals that token quality is fundamentally more critical than quantity under input window limits. This insight can be transferred to any input length-constrained scenario (e.g., context compression in RAG).
- **Generality of Category-Local Attention**: The idea of grouping token importance by context can be generalized to NLP (e.g., categorizing by syntactic elements) and multimodal scenarios (e.g., categorizing by image region types).

## Limitations & Future Work

- Evaluated only on the Java language. Although literature reports similar trends in other languages, this has not been directly validated.
- Evaluated only using three models (CodeBERT, CodeT5, GPT-4o), without covering newer code LLMs.
- Code statement categories are predefined into 21 classes, which may not scale to all programming languages or coding styles.
- Attention score collection requires an extra training pass, which might be costly for ultra-large models.

## Related Work & Insights

- **vs DietCode**: DietCode uses global self-attention average, while LeanCode uses context-aware CLS/EnDe attention. LeanCode achieves over 60% MRR improvement at a 50% simplification ratio.
- **vs SlimCode**: SlimCode uses 8-level manual rules, while LeanCode uses the model's own attention. SlimCode's performance drops sharply at 30%+ ratios.
- **vs SIVAND/P2IM**: These methods simplify iteratively based on delta debugging, requiring repeated model runs, making them far less efficient than LeanCode's one-time attention collection.

## Rating

- Novelty: ⭐⭐⭐⭐ Context-aware + task-aware attention selection is a reasonable and effective innovation, though the overarching concept is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very comprehensive, spanning 3 models, 2 tasks, 10%-50% simplification ratios, pruning time comparison, replacement ablation, and cross-model transfer.
- Writing Quality: ⭐⭐⭐⭐ The empirical study section is clear and convincing, though some sections are highly repetitive.
- Value: ⭐⭐⭐⭐ Highly practical in engineering, directly applicable to reducing inference costs of code LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Chinese Grammatical Error Correction With Pre-trained Models and Linguistic Clues](chinese_grammatical_error_correction_with_pre-trained_models_and_linguistic_clue.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ICML 2025\] Large Language Models are Demonstration Pre-Selectors for Themselves](../../ICML2025/llm_pretraining/large_language_models_are_demonstration_pre-selectors_for_themselves.md)

</div>

<!-- RELATED:END -->
