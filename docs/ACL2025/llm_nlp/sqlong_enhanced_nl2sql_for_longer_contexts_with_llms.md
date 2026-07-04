---
title: >-
  [Paper Note] SQLong: Enhanced NL2SQL for Longer Contexts with LLMs
description: >-
  [ACL2025][LLM (Other)][NL2SQL] This paper proposes SQLong, a data augmentation framework for NL2SQL in long-context scenarios. By injecting synthetic `CREATE TABLE` statements sampled from other databases into the training data to extend the context length, it enables fine-tuned LLMs to achieve significantly improved SQL generation accuracy in large-scale schema scenarios.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "NL2SQL"
  - "Long Context"
  - "Data Augmentation"
  - "Large-Scale Database Schema"
  - "Text-to-SQL"
date: 2026-05-08
content_hash: da9ecea84a1f6fed
---

# SQLong: Enhanced NL2SQL for Longer Contexts with LLMs

**Conference**: ACL2025  
**arXiv**: [2502.16747](https://arxiv.org/abs/2502.16747)  
**Code**: To be open-sourced (planned release mentioned in the paper)  
**Area**: LLM/NLP  
**Keywords**: NL2SQL, Long Context, Data Augmentation, Large-Scale Database Schema, Text-to-SQL

## TL;DR
This paper proposes SQLong, a data augmentation framework for NL2SQL in long-context scenarios. By injecting synthetic `CREATE TABLE` statements sampled from other databases into the training data to extend the context length, it enables fine-tuned LLMs to achieve significantly improved SQL generation accuracy in large-scale schema scenarios.

## Background & Motivation

1. **NL2SQL performance drops sharply on large schemas**: Existing LLMs perform exceptionally well on small schema benchmarks such as Spider and BIRD, but encounter significant performance drops in long-context scenarios under real-world large-scale databases (hundreds of tables).
2. **Small schema size in training data**: Spider features an average of only $5\pm3$ tables and BIRD has $7\pm3$ tables. The training data fails to cover the complex, large schemas of real-world scenarios, resulting in insufficient long-context generalization capabilities of models.
3. **Lack of long-context NL2SQL evaluation benchmarks**: Input query prompt lengths in existing benchmarks typically do not exceed 2000–2500 tokens, and there is no publicly available large-schema test set to evaluate model robustness in long contexts.
4. **Difficulty in collecting real large-schema data**: Enterprise-level database schemas involve privacy and permission issues, making them difficult to collect and release at scale, which hinders the development of long-context NL2SQL research.
5. **Immaturity of RAG solutions**: Although retrieval-augmented Schema Linking methods show potential, no mature approach has fully resolved the long-context issue. The proposed augmentation method can complement RAG.
6. **Opportunities from growing LLM context windows**: Llama-3.1 supports up to 128k context and CodeQwen supports up to 64k context, establishing a technical foundation for directly processing large schemas, which requires matching training strategies.

## Method

### SQLong Three-Step Pipeline

**Step 1: Schema Collection**
Collect all `CREATE TABLE` statements along with 3 rows of sample data for each table from all databases in the training set to compile a comprehensive schema pool.

**Step 2: Schema Augmentation**
For each training instance (input prompt + target SQL), a set of tables is randomly sampled from the schema pool (table names must not duplicate the original schema). These sampled tables are merged with the original schema, and the table order is randomly shuffled. Key designs:
- Cultivating diversity by ensuring the injected tables originate from **different databases**.
- Random shuffling leaves the position of the original tables unfixed, forcing the model to learn to locate relevant schemas in long contexts.
- **The target SQL remains unchanged**, and only the input schema context is lengthened.

**Step 3: Long-Context Prompt Generation**
Assemble the new prompt in the format of (task instructions, long-context db schema, question), ensuring that the total length of the prompt + SQL does not exceed a preset context threshold (e.g., 32k tokens). Context lengths are randomly sampled from 4,096 to 32,768 with a step size of 512.

### Loss & Training
- The augmented dataset is merged with the original training set.
- Base models: CodeQwen1.5-7B-Chat (64k), Llama-3.1-8B-Instruct (128k).
- SFT minimizes the standard log-likelihood loss.
- 8×H100 80GB GPUs, batch size=1, gradient accumulation=8, maximum of 5 epochs.

### Long-Context Test Set Construction
- Cross-construction is employed to prevent data leakage: Spider series long-context test sets are constructed using schemas from the BIRD training set, while BIRD long-context test sets are constructed using schemas from the Spider training set.
- 9 context length levels: 8k, 16k, 24k, 32k, 40k, 48k, 56k, 64k, 128k.
- A total of 45 long-context test sets are constructed.

## Key Experimental Results

### Main Results: Performance on Original Short-Context Datasets (Execution Accuracy %)

| Model | Spider-dev | Spider-realistic | Spider-syn | Spider-test | BIRD-dev | Average |
|------|-----------|-----------------|-----------|------------|---------|------|
| Qwen2-72B-Instruct | 82.7 | 80.7 | 73.0 | 82.9 | 53.7 | 74.6 |
| CodeQwen-7B (w/o SQLong) | 81.9 | 76.2 | 68.7 | 79.6 | 51.4 | 71.6 |
| CodeQwen-7B (SQLong) | **83.4** | **79.7** | **71.2** | **81.3** | **53.3** | **73.8** |
| Llama-70B-Instruct | 80.7 | 78.0 | 73.0 | 83.7 | 61.5 | 75.4 |
| Llama-8B (w/o SQLong) | 79.2 | 76.4 | 69.6 | 80.4 | 51.9 | 71.5 |
| Llama-8B (SQLong) | **83.2** | **78.0** | **73.1** | **81.8** | **53.3** | **73.9** |

**Key Findings**: SQLong fine-tuning brings an average increase of over 2.2%; after being fine-tuned with SQLong, the 7B/8B models approach or even outperform the unaugmented 72B/70B models on short contexts. Long-context augmentation training does not degrade short-context performance, but instead enhances it.

### Main Results: Performance on Long-Context Test Sets (Spider-test Execution Accuracy %, Selected Examples)

| Context Length | Llama-8B (w/o SQLong) | Llama-8B (SQLong) | Llama-70B | Gain |
|-----------|--------------------|--------------------|-----------|---------|
| 8k | 69.9% | 77.1% | — | +7.2% |
| 24k | 59.0% | 72.3% | — | +13.3% |
| 64k | — | — | — | — |

**Key Findings**:
- The SQLong-fine-tuned Llama-8B outperforms the unaugmented Llama-70B in **41 out of 45** long-context test sets.
- On average, SQLong fine-tuning yields an **11% absolute gain** (compared to without SQLong), outperforming the 70B model by **6%**.
- Position robustness experiments demonstrate that the SQLong-fine-tuned model is significantly more robust to the positions of schemas in prompts.

## Highlights & Insights

- **Framework is simple and efficient**: Data augmentation is achieved purely through sampling and concatenation without requiring extra annotation or model generation, making the implementation cost extremely low.
- **Small models outperform large models**: The 8B model fine-tuned with SQLong fully outperforms the unaugmented 70B model in long-context scenarios, offering exceptional cost-efficiency.
- **Simultaneous short and long-context capabilities**: Long-context augmentation training does not hurt short-context performance, but instead brings an average improvement of 2.2%.
- **Systematic evaluation framework**: For the first time, 45 long-context NL2SQL test sets (up to 128k tokens) are constructed, filling the void in evaluation benchmarks.
- **Validation of position robustness**: Experiments controlling schema positions prove that SQLong enhances the model's ability to locate information within long contexts.

## Limitations & Future Work

1. **Fine-tuning context length is restricted to 32k**: Due to computational resource constraints, fine-tuning was limited to 32k, failing to fully exploit the 128k capability window of Llama-3.1.
2. **Injected schemas lack semantic association with the original query**: Randomly sampled distractor tables might be too simple, whereas real-world scenarios present semantically similar distractor tables that pose greater difficulties.
3. **Absence of comparison with RAG Schema Linking**: The paper acknowledges that it does not directly compare against retrieval-augmented solutions, making it difficult to gauge the competitiveness of SQLong relative to RAG.
4. **Evaluation is restricted to two 7B/8B base models**: The effectiveness on larger models (such as fine-tuning a 70B model) or newer architectures remains unverified.
5. **Lack of deep analysis on augmented data quality**: There is a lack of ablation studies analyzing the impact of sampling strategies (e.g., number of sampled tables, diversity of sources).

## Related Work & Insights

### vs RAG-based Schema Linking
RAG-based approaches focus on "subtraction" by retrieving a relevant subset of schemas to shorten input lengths. In contrast, SQLong employs an "addition" strategy via data augmentation training to empower models to directly process long schemas. These strategies are complementary: SQLong improves the fundamental long-context capacity of the model, whereas RAG further scales down the input at inference time. The paper suggests that combining both could potentially yield greater benefits.

### vs Prompt Engineering Schemes (such as DIN-SQL / DAIL-SQL)
DIN-SQL and DAIL-SQL improve NL2SQL performance by craftily designing prompt formats (such as decomposing subproblems and selecting examples). However, both assume that the schema falls within the model's context window. SQLong is orthogonal to these methods, focusing on scenarios where schemas exceed typical lengths, allowing them to be combined seamlessly.

### vs Spider-Syn / Spider-Realistic
Spider-Syn and Spider-Realistic test robustness using synonym substitution and removing explicit column names, focusing on **linguistic variation**. SQLong, conversely, targets robustness across different **context lengths**. These are different dimensions of augmentation. In experiments, SQLong also achieved substantial improvements on these variant test sets.

## Rating
- **Novelty**: ⭐⭐⭐ (The idea of data augmentation is intuitive and simple, but the technical novelty is somewhat limited since the core process is "sampling and concatenation".)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (5 benchmarks × 9 length levels = 45 test sets, delivering a comprehensive evaluation across both short and long contexts, including position-robustness analysis.)
- **Writing Quality**: ⭐⭐⭐⭐ (Clearly structured, complete description of the pipeline, with rich figures and tables.)
- **Value**: ⭐⭐⭐⭐ (Systematically defines the long-context NL2SQL task for the first time, providing a practical augmentation scheme and evaluation benchmarks.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Towards Enhanced Immersion and Agency for LLM-based Interactive Drama](towards_enhanced_immersion_and_agency_for_llm-based_interactive_drama.md)
- [\[ACL 2025\] ReCall: Library-Like Behavior In Language Models is Enhanced by Self-Referencing Causal Cycles](library-like_behavior_in_language_models_is_enhanced_by_self-referencing_causal_.md)
- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)
- [\[ACL 2025\] Beyond Output Matching: Bidirectional Alignment for Enhanced In-Context Learning](beyond_output_matching_bidirectional_alignment_for_enhanced_in-context_learning.md)

</div>

<!-- RELATED:END -->
