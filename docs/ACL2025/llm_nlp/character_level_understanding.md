---
title: >-
  [Paper Note] Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning
description: >-
  [ACL 2025][LLM (Other)][Token Internal Structure] Proposes TIPA (Token Internal Position Awareness), a method that designes reverse character prediction training on the tokenizer vocabulary to enhance LLMs' perception of the internal character structure and positions within tokens, significantly improving performance on character-level tasks like Chinese Spelling Correction.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Token Internal Structure"
  - "Character-Level Understanding"
  - "BPE Tokenization"
  - "Chinese Spelling Correction"
  - "Position Awareness"
date: 2026-05-08
content_hash: 40ab8fc33a249a46
---

# Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning

**Conference**: ACL 2025  
**arXiv**: [2411.17679](https://arxiv.org/abs/2411.17679)  
**Code**: [https://github.com/FloatFrank/TIPA](https://github.com/FloatFrank/TIPA)  
**Area**: LLM NLP / Character-Level Understanding  
**Keywords**: Token Internal Structure, Character-Level Understanding, BPE Tokenization, Chinese Spelling Correction, Position Awareness  

## TL;DR

Proposes TIPA (Token Internal Position Awareness), a method that designes reverse character prediction training on the tokenizer vocabulary to enhance LLMs' perception of the internal character structure and positions within tokens, significantly improving performance on character-level tasks like Chinese Spelling Correction.

## Background & Motivation

**Background**:
   - Mainstream LLMs employ subword tokenization methods like BPE to segment text into subword units for computational efficiency.
   - LLM training centers on next-token prediction, focusing on sequential dependencies between tokens.
   - This design naturally conceals the internal character structure information of tokens.

**Limitations of Prior Work**:
   - LLMs cannot accurately hands-on perceive the composition and positions of characters inside a token; for example, they often answer incorrectly when asked "how many r's are in strawberry".
   - This issue is more severe in Chinese scenarios: a token may contain multiple Chinese characters, making it difficult for the model to locate specific characters in the original text.
   - This severely impacts tasks requiring precise character localization, such as Chinese Spelling Correction (CSC).
   - Existing byte-level models (e.g., ByT5) require architectural changes and cannot be adapted to existing subword LLMs at a low cost.

**Key Challenge**:
   - BPE tokenization improves computational efficiency but sacrifices the visibility of character-level information.
   - While models learn relationships between tokens, they remain largely "unaware" of the order and positions of characters inside tokens.

**Goal**
   - To enhance LLM understanding of internal character structures and positions within tokens without modifying the model architecture.

**Key Insight**:
   - Construct training data using the tokenizer's own vocabulary to teach the model token internal structures via a reverse character prediction task.

**Core Idea**:
   - Directing the model to output the position and content of each character inside a token in reverse order (e.g., "girl" → {4: "l", 3: "r", 2: "i", 1: "g"}), thereby implicitly learning the token's internal character structure.

## Method

### Overall Architecture

The TIPA system consists of two core components:
1. **TIPA**: Single-token reverse character position training based on the tokenizer vocabulary.
2. **MTIPA**: Character position training extended to the multi-token sentence level.

### Key Designs

1. **TIPA (Token Internal Position Awareness)**:

    - **Function**: Generates a reverse character-position mapping as training data for each token in the tokenizer vocabulary.
    - **Mechanism**: Given a token $t$, decompose it into a character sequence $[c_1, c_2, \dots, c_n]$, and then construct a reverse mapping $\{n: c_n, n-1: c_{n-1}, \dots, 1: c_1\}$.
    - Training prompt example: Input "girl", output {"4": "l", "3": "r", "2": "i", "1": "g"}
    - **Design Motivation**: Outputting in reverse order ensures that the first number represents the token length, cleverly unifying tokenization, length, and position information into a single task.
    - Only tokens from the tokenizer vocabulary that can be represented in UTF-8 are used, requiring no external data.

2. **MTIPA (Multi-Token Internal Position Awareness)**:

    - **Function**: Extends reverse character prediction to the full sentence level.
    - **Mechanism**: Samples sentences from the training data of target tasks, decomposing full sentences into characters, and mapping reverse positions.
    - The sampling ratio $r$ is set to a small value (e.g., 10%) to balance data quantity and training efficiency.
    - **Design Motivation**: MTIPA is specifically designed for tasks requiring precise character position prediction (e.g., CSC with position prediction).

3. **Redefining the CSC Task**:

    - **Function**: Shifting traditional "outputting the corrected full sentence" to "outputting the position of the incorrect character, the incorrect character, and the correct character".
   - **Mechanism**: e.g., "Industry insiders say..." -> [{"position": 4, "incorrect": "shi (event-character)", "correction": "shi (scholar-character)"}]
    - **Design Motivation**: Dramatically reduces the number of output tokens (the position method requires 36-51% fewer tokens than traditional methods), thereby improving efficiency.

4. **Extension Method: Full-Parameter SFT**:

    - Merges TIPA data with tulu-3-sft-mixture data to perform full-parameter fine-tuning on Llama-3.1-8B.
    - Produces Llama-3.1-Tulu-TIPA-8B, which enhances character-level processing while retaining the model's general capabilities.

### Loss & Training

- Standard SFT (supervised fine-tuning) training is used, supporting both LoRA and full-parameter fine-tuning.
- TIPA data is derived from the tokenizer vocabulary without requiring external annotated data.
- The training overhead is minimal, and no additional latency is introduced during inference.

## Key Experimental Results

### Main Results

**Experiment 1: CSC Task with Position Prediction**
- Introduces a new evaluation metric, PPA (Position Prediction Accuracy), to measure the model's ability to locate incorrect characters.
- TIPA + MTIPA significantly improves the model's character position prediction accuracy.

**Experiment 2: Traditional CSC Task**
- On the CSCD-Test and Lemon datasets, TIPA improves the model's spelling correction performance.
- Even without explicitly requiring position prediction, TIPA improves correction performance by enhancing character-level understanding.

**Comparison of Output Token Count**:

| Dataset | Traditional Method Token Count | Position Method Token Count |
|--------|-------------------|-------------------|
| Train | 8,905,800 | 8,016,111 |
| CSCD-Test | 188,310 | 54,897 |
| Lemon | 532,684 | 258,112 |

- The position-based method reduces output tokens by approximately 51-71%.

### Key Findings

1. **Reverse Order Superiority**: Reverse prediction implicitly encodes token length information, outperforming forward order prediction.
2. **Generality of TIPA**: TIPA remains effective even in downstream tasks that do not explicitly require precise position prediction.
3. **LoRA vs. Full-Parameter Fine-tuning**: Combining full-parameter fine-tuning with tulu-3 data enhances character-level understanding while preserving general capabilities.
4. **MTIPA Data Volume Control**: Excessively long MTIPA data may lead to prolonged training times, and training LoRA on large amounts of length reasoning information might degrade performance on specific tasks.
5. **GPT-4o Also Struggles with Character Localization**: Even the strongest closed-source models exhibit insufficient awareness of character positions within tokens.

## Highlights & Insights

- **Utilizing the Tokenizer's Own Vocabulary**: Cleverly uses existing resources to construct training data without requiring external annotations.
- **Ingenuity of the Reverse Design**: A simple reverse operation simultaneously addresses three sub-problems: segmentation, length, and position.
- **Redefining the CSC Task Paradigm**: Shifting from outputting the entire sentence to outputting positions + characters, dramatically reducing inference costs.
- **Revealing the Inherent Limitations of BPE**: Systematically demonstrates the negative impact of subword tokenization on character-level tasks.

## Limitations & Future Work

1. Primarily validated on Chinese CSC tasks, with insufficient validation on other languages and character-level tasks (e.g., English spelling correction, character-level understanding).
2. TIPA requires an additional training step; although the data is sourced from the vocabulary, fine-tuning is still necessary.
3. The optimal sampling ratio $r$ for MTIPA needs to be adjusted for different tasks.
4. Full-parameter fine-tuning is costly, and the effectiveness of LoRA on smaller models may be limited.
5. The applicability of TIPA to models using non-BPE tokenization (e.g., byte-level models) remains to be explored.

## Related Work & Insights

- **ByT5** (Xue et al., 2022): A byte-level model possessing character-level precision by nature but requiring architectural changes.
- **C-LLM** (Li et al., 2024): Uses character-level tokenization to enhance character-level understanding.
- **ReLM** (Liu et al., 2024): Redefining CSC as a sentence rewriting task.
- **The Reversal Curse** (Berglund et al., 2023): Models struggle to understand reversed relationships, echoing TIPA's reverse training.
- Inspiration for future work: Token internal structure is an overlooked yet critical dimension that warrants further investigation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of reverse character prediction is novel, though the core remains data augmentation and SFT.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Sufficient experiments on CSC tasks, but coverage of other character-level tasks is limited.
- Writing Quality: ⭐⭐⭐⭐ — Clear description of methods with well-defined new evaluation metrics.
- Value: ⭐⭐⭐⭐ — Exposes important character-level understanding issues and is highly practical for Chinese NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](table_lora_structure_understanding.md)
- [\[ACL 2025\] Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction](idea_enhancing_the_rule_learning_ability_of_large_language_model_agent_through_i.md)
- [\[ACL 2025\] Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility](meaning_beyond_truth_conditions_evaluating_discourse_level_understanding_via_ana.md)
- [\[ACL 2025\] ECLM: Entity Level Language Model for Spoken Language Understanding with Chain of Intent](eclm_entity_level_language_model_spoken_language_understanding.md)

</div>

<!-- RELATED:END -->
