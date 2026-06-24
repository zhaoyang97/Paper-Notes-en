---
title: >-
  [Paper Note] TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models
description: >-
  [ACL 2025][LLM (Other)][LoRA] TableLoRA proposes a specialized LoRA module for table tasks, improving table serialization through a special token encoder and encoding cell row/column positional information with 2D LoRA. Under parameter-efficient fine-tuning (PEFT) settings, it achieves a 5.9% improvement on HiTab compared to vanilla LoRA, bridging 40.56% of the performance gap between LoRA and full fine-tuning.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LoRA"
  - "table understanding"
  - "structured data"
  - "2D positional encoding"
  - "PEFT"
date: 2026-05-08
content_hash: 712d6b25e285a6db
---

# TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2503.04396](https://arxiv.org/abs/2503.04396)  
**Code**: [https://github.com/microsoft/TableLoRA](https://github.com/microsoft/TableLoRA)  
**Area**: LLM/NLP  
**Keywords**: LoRA, table understanding, structured data, 2D positional encoding, PEFT  

## TL;DR
TableLoRA proposes a specialized LoRA module for table tasks, improving table serialization through a special token encoder and encoding cell row/column positional information with 2D LoRA. Under parameter-efficient fine-tuning (PEFT) settings, it achieves a 5.9% improvement on HiTab compared to vanilla LoRA, bridging 40.56% of the performance gap between LoRA and full fine-tuning.

## Background & Motivation

**Background**: Table data is widely used in numerous domains, and processing table tasks under the PEFT paradigm has become increasingly important for LLMs.

**Limitations of Prior Work**: (a) Table serialization methods (e.g., markdown or HTML) heavily affect model comprehension, yet existing approaches still struggle to accurately identify table structures (such as alignment within the same column); (b) once two-dimensional table structures are flattened into one-dimensional sequences, row and column positional information can only be learned implicitly through attention mechanisms, which is insufficiently learned under low-parameter PEFT.

**Key Challenge**: The two-dimensional positional relationships of a table are critical to understanding its structure, but vanilla LoRA does not explicitly encode this structural information.

**Goal**: To enable LLMs to better understand table structures under a low-parameter PEFT setting.

**Key Insight**: Directly conveying table structural relationships to the model through architectural design, rather than relying on attention mechanisms to learn them implicitly.

**Core Idea**: Replacing markdown markers with special tokens to improve serialization, combined with injecting low-rank row and column positional encodings into each layer to explicitly inform the LLM of the table structure.

## Method

### Overall Architecture

Two components operate in parallel: (1) The Special Tokens Encoder introduces `[tab]`, `[row]`, and `[cell]` special token embeddings before the Transformer layers; (2) The 2D LoRA fuses low-rank embeddings of row and column indices with token embeddings in each layer.

### Key Designs

1. **Special Tokens Encoder**:

    - **Function**: Uses `[tab]`, `[row]`, and `[cell]` to replace markdown/HTML markers for table serialization.
    - **Mechanism**: Inspired by P-Tuning, these special tokens feature learnable embeddings, learning table structural semantics through gradient propagation during fine-tuning.
    - **Design Motivation**: Traditional punctuation/markers (such as `|` and `\n`) are not tailored for tables; specialized tokens can better represent structural boundaries.

2. **2D LoRA**:

    - **Function**: Encodes row and column index information into low-rank embeddings, injecting them into token representations at each layer.
    - **Mechanism**: Creates separate low-rank embeddings $E_{row} \in \mathbb{R}^{R \times r}$ and $E_{col} \in \mathbb{R}^{C \times r}$ for row and column indices respectively. These are scaled to the hidden dimension via an up-projection matrix and then added to the token representations. This operates in parallel with the original LoRA.
    - **Design Motivation**: The information density of 2D coordinates is relatively low compared to token semantics; thus, utilizing low-rank encoding is sufficient and parameter-efficient.

### Loss & Training
Standard task loss is optimized jointly with LoRA during fine-tuning. 2D LoRA runs in parallel with standard LoRA in each layer.

## Key Experimental Results

### Main Results

Three models (Llama-2-7B, Llama-3-8B, Qwen2-7B), four datasets (HiTab, WikiTableQuestions, TabFact, SQA).

| Method | HiTab ↑ | WTQ ↑ | TabFact ↑ |
|------|---------|-------|-----------|
| LoRA | 38.5 | 55.2 | 72.8 |
| **TableLoRA** | **44.4 (+5.9)** | **57.1** | **74.5** |
| Full Fine-tuning | 52.9 | 59.3 | 76.1 |

### Ablation Study

| Configuration | HiTab Acc | Description |
|------|----------|------|
| TableLoRA (Full) | **44.4** | Special Tokens + 2D LoRA |
| Special Tokens Only | 41.2 | Without 2D LoRA |
| 2D LoRA Only | 42.8 | Without Special Tokens |
| LoRA Baseline | 38.5 | No table-specific design |

### Key Findings
- **Largest improvement on HiTab (+5.9%)**: HiTab contains hierarchical headers, which require precise row/column positional comprehension the most.
- **Bridges 40.56% of the gap between LoRA and full fine-tuning**: Effectively enhances structure understanding with extreme parameter efficiency.
- **Complementary components**: Special Tokens improve serialized representations, while 2D LoRA provides positional information.

## Highlights & Insights
- **First table-specific LoRA**: The idea of directly encoding domain knowledge (2D structure) into the LoRA architecture can be generalized to other structured data (such as graphs or code ASTs).
- **Low-rank encoding of positional information**: The information density of row/column indices is significantly lower than that of semantic content, making low-rank encoding highly reasonable.

## Limitations & Future Work
- Only flat tables and simple hierarchical headers are covered; complex merged cells and nested tables are not addressed.
- No comparison with specialized table LLMs (e.g., TableGPT).
- The maximum number of rows and columns is limited by the preset embedding sizes.

## Related Work & Insights
- **vs Vanilla LoRA**: Vanilla LoRA lacks awareness of table structure; TableLoRA resolves this through specialized encoding.
- **vs TableGPT/TableLLM**: These models learn table understanding through large-scale training, whereas TableLoRA incurs only PEFT-level overhead.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First table-specific LoRA; the design of 2D positional encoding is elegant and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 3 models across 4 datasets, including control experiments and detailed analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition and intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ Offers high reference value to both the table processing and PEFT communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DenseLoRA: Dense Low-Rank Adaptation of Large Language Models](denselora_dense_low-rank_adaptation_of_large_language_models.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[ACL 2025\] RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates](rocoft_efficient_finetuning_of_large_language_models_with_row-column_updates.md)
- [\[ACL 2025\] Cultural Learning-Based Culture Adaptation of Language Models](cultural_learning-based_culture_adaptation_of_language_models.md)
- [\[ACL 2025\] Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning](character_level_understanding.md)

</div>

<!-- RELATED:END -->
