---
title: >-
  [Paper Note] PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation
description: >-
  [ICML 2025][LLM Evaluation][benchmark] PhantomWiki is proposed—an evaluation framework that on-demand generates fictional world corpora and QA pairs. By controlling reasoning difficulty via context-free grammars (CFGs) and adjusting retrieval difficulty via universe scale, it achieves a decoupled evaluation of LLM reasoning and retrieval capabilities while naturally resisting data leakage.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "benchmark"
  - "reasoning evaluation"
  - "retrieval evaluation"
  - "synthetic data"
  - "data contamination"
date: 2026-05-08
content_hash: 94a691688d03a55a
---

# PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation

**Conference**: ICML 2025  
**arXiv**: [2502.20377](https://arxiv.org/abs/2502.20377)  
**Code**: [github.com/kilian-group/phantom-wiki](https://github.com/kilian-group/phantom-wiki)  
**Area**: LLM Evaluation  
**Keywords**: benchmark, reasoning evaluation, retrieval evaluation, synthetic data, data contamination

## TL;DR

PhantomWiki is proposed—an evaluation framework that on-demand generates fictional world corpora and QA pairs. By controlling reasoning difficulty via context-free grammars (CFGs) and adjusting retrieval difficulty via universe scale, it achieves a decoupled evaluation of LLM reasoning and retrieval capabilities while naturally resisting data leakage.

## Background & Motivation

Existing QA benchmarks (such as SQuAD, HotpotQA, DROP, MuSiQue, etc.) suffer from two fundamental problems:

**Data Leakage and Memorization**: Static datasets are inevitably crawled into training data, causing inflated model performance. Studies on GSM8K have shown that frontier models experience significant performance drops after minor question variations, suggesting they might simply be "memorizing answers."

**Difficulty in Decoupling Reasoning and Retrieval**: When questions involve real-world knowledge (such as Mozart's birth date), it is impossible to distinguish whether a model is reasoning, retrieving, or recalling memory from pre-training. Modifying Wikipedia facts to construct new QA pairs faces the challenge of cross-article factual consistency (e.g., if Mozart's birthday is changed, mentions of their meeting in Beethoven's biography must be synchronized).

Therefore, the core idea of the authors is: **to generate entirely new fictional worlds on demand for each evaluation, without relying on any existing data**, fundamentally eliminating data contamination and evaluating reasoning and retrieval capabilities separately by independently controlling the number of reasoning steps and the corpus scale.

## Method

### Overall Architecture

The pipeline of PhantomWiki consists of four stages:

1. **Character Generation**: Construct a fictional world with $n$ characters by first generating a family tree (iteratively creating parent-child relationships) and then generating a friendship graph using the Erdős–Rényi model.
2. **Fact Generation**: Assign attributes to each character, including names (based on gender and family surnames, totaling 15 million full-name combinations), birth dates (consistent with family relationships), professions (300+ types), and hobbies (600+ types).
3. **Article Generation**: Convert facts into wiki-like biography articles (approx. 160 tokens each) using predefined templates. These articles serve as the sole source of information provided to the models during evaluation.
4. **QA Pair Generation**: Generate question templates using a context-free grammar (CFG), and then solve for all correct answers using Prolog logic programming.

### Key Designs

#### CFG-based Question Generation

CFG is one of the most core designs of PhantomWiki. By recursively combining question templates, the CFG can systematically generate questions of varying complexity:

- **Recursion depth $d = 1$**: Simple questions like "Who is the friend of David?"
- **Increasing recursion depth $d$**: Multi-hop questions are generated, such as "Who is the nephew of the friend of the brother of David?"
- **Diversified question types**:
    - Relation queries: "Who is the `<<<relation>>>` of `<<<name>>>`?"
    - Attribute queries: "Who is the person whose hobby is birdwatching?"
    - Aggregation queries: "How many brothers does David have?"
    - Compositional queries: "How many friends does the brother of the person whose hobby is birdwatching have?"

The choice of relation types also influences the number of reasoning steps: `nephew` requires 2 reasoning steps (sibling + son), whereas `second cousin` requires 5 steps. The CFG generates the corresponding Prolog query templates in parallel with the questions.

#### Solving Answers with Prolog Logic Programming

Fictional worlds are represented as Prolog facts and rules:

- Facts: `hobby("David", "birdwatching")`
- Rules: `nephew(X, Y) :- sibling(X, A), son(A, Y)`

Prolog programs can exhaustively find all answers that satisfy the constraints, ensuring the **completeness and verifiability** of the answers. For example, the question "Who is the nephew of the friend of the person whose hobby is birdwatching?" corresponds to the joint evaluation of three Prolog query statements.

#### Difficulty Control Mechanism

- **Reasoning difficulty**: Controlled by CFG recursion depth $d$ and relation types to determine the number of reasoning steps (1 to 15+ steps).
- **Retrieval difficulty**: Controlled by the universe scale $n$, forcing the model to use retrieval when the corpus exceeds the model's context window.

#### Data Leakage Resistance Analysis

The number of possible universes is $\Theta(2^{n^2} \cdot c^n)$, where $n$ is the number of characters and $c$ is the total number of attribute options. Even for $n=100$, the universe space is astronomical, making memorization virtually impossible.

### Loss & Training

PhantomWiki itself is an evaluation tool rather than a training method. However, the paper explores fine-tuning LLMs on PhantomWiki:

- **SFT (Supervised Fine-Tuning)**: Fine-tuned Qwen2.5-0.5B and 3B on 10 PhantomWiki instances; performance was on par with the baseline.
- **GRPO (Group Relative Policy Optimization)**: Trained on the same data, resulting in a significant improvement in F1 (Qwen2.5-3B increased from 16.82 to 31.38), though still far from optimal.
- Even after fine-tuning, the performance of models on new datasets still decreased as the number of reasoning steps increased, proving the robustness of the PhantomWiki evaluation against memorization.

The evaluation metric is the **answer-level F1 score**: models must predict all answers (as a comma-separated list), with precision, recall, and F1 computed at the answer level.

## Key Experimental Results

### Main Results

Under three universe scales ($n=50/500/5000$), 4 frontier LLMs × 5 prompting strategies were evaluated:

| Model | Method | $n=50$ F1(%) | $n=500$ F1(%) | $n=5000$ F1(%) |
|------|------|-----------|------------|-------------|
| DeepSeek-R1-32B | CoT (In-Context) | **52.42** | 19.65 | — |
| GPT-4o | CoT (In-Context) | 50.66 | 41.02 | — |
| Llama-3.3-70B | CoT (In-Context) | 48.37 | 25.99 | — |
| GPT-4o | ReAct (Agentic) | 38.70 | 37.39 | **36.85** |
| Llama-3.3-70B | ReAct (Agentic) | 35.83 | **35.56** | 30.89 |
| Gemini-1.5-Flash | ReAct (Agentic) | 30.92 | 26.99 | 23.47 |
| GPT-4o | ZeroShot-RAG | 28.05 | 22.32 | 18.13 |
| DeepSeek-R1-32B | ReAct (Agentic) | 5.47 | 3.57 | 4.74 |

### Ablation Study

| Method | Qwen2.5-0.5B F1(%) | Qwen2.5-3B (LoRA) F1(%) | Notes |
|------|--------------------|-----------------------|------|
| ZeroShot | 11.78 ± 0.94 | 16.82 ± 2.37 | Baseline |
| CoT | 2.68 ± 0.22 | 13.71 ± 0.81 | Small models struggle to follow CoT format |
| SFT | 11.71 ± 1.10 | 16.89 ± 2.22 | Supervised fine-tuning shows no significant improvement |
| GRPO | **13.25 ± 0.93** | **31.38 ± 0.86** | RL fine-tuning significantly improves reasoning ability |

### Generation Efficiency

| Universe Scale $n$ | Total Time | Fact Gen | Article Gen | Question Gen |
|-----------|--------|---------|---------|---------|
| 100 | 0.97s | 0.46s | 0.07s | 0.44s |
| 1,000 | 2.86s | 0.90s | 0.59s | 1.37s |
| 10,000 | 20.91s | 5.38s | 5.87s | 9.66s |
| 100,000 | 5.57min | 0.81min | 0.97min | 3.79min |
| 1,000,000 | 3.86h | 9.47min | 11.77min | 3.51h |

### Key Findings

1. **Reasoning ability drops sharply with step count**: The F1 score of all models and prompting methods degrades significantly as the number of reasoning steps increases. CoT decays slower than ZeroShot, but each additional reasoning step remains highly challenging.
2. **RAG nearly fails on multi-hop reasoning**: ZeroShot-RAG and CoT-RAG achieve near-zero F1 scores for reasoning above 5 steps, because a single retrieval cannot fetch all documents required for multi-hop reasoning.
3. **Agentic prompting (ReAct) performs best in large-scale retrieval**: Dynamic, interactive retrieval significantly outperforms one-shot RAG, with ReAct being the only viable paradigm at $n=5000$.
4. **DeepSeek-R1 possesses strong reasoning but weak tool utilization**: It exhibits the best performance under CoT, but has extremely low F1 under ReAct (~5%), exposing its deficient tool-calling capabilities.
5. **Multi-branch reasoning represents an additional failure mode**: Models frequently miss branches when looking for all possible answers (e.g., missing some descendants of grandchildren when searching for great-grandchildren).

## Highlights & Insights

- **Extremely elegant evaluation philosophy**: Releasing a "dataset generation pipeline" rather than a fixed dataset fundamentally solves the issues of benchmark obsolescence and data leakage.
- **An ingenious combination of CFG and Prolog**: CFG ensures systematic coverage of questions and controllable difficulty, while Prolog guarantees the exhaustive correctness of answers. Together, they achieve an "fully automatic, verifiable, and difficulty-controllable" evaluation.
- **Decoupled reasoning and retrieval**: By independently adjusting the question complexity and the corpus scale—two orthogonal dimensions—this study is the first to systematically decouple and measure LLM reasoning and retrieval abilities.
- **Simple yet clever template articles**: Biography templates of approx. 160 tokens avoid the hallucination issues introduced by LLM-based rewriting while maintaining generation speed and cost efficiency.
- **Scalable to millions**: A universe of $n=1\text{M}$ (comparable to the scale of Wikipedia biographies) can be generated on a standard CPU in about 4 hours.

## Limitations & Future Work

1. **Highly templated text**: Current articles feature a monotonic style ("The job of David is a farmer"), which is far from real Wikipedia. While LLM rewriting risks introducing hallucinations, consistency-preserving rewriting methods are worth exploring.
2. **Limited relation types**: Currently only family and friendship relationships are included, lacking complex real-world social ties like professional or teacher-student relationships.
3. **Basic question types**: Lacks more complex question patterns, such as comparative ("Who is older?") or multi-constraint questions. Although the paper mentions CFGs are extensible, this is not implemented in the current version.
4. **Text-only modality**: Multimodal evaluation involving vision, audio, etc., is not covered.
5. **BM25 for RAG might be unfair**: The uniqueness of synthetic text makes neural retrievers less applicable, but using only BM25 might underestimate the true potential of RAG.

## Related Work & Insights

- **CLUTRR (Sinha et al., 2019)**: Tests relational reasoning via crowdsourced family stories. PhantomWiki can be seen as an extension in a large-scale, fully automated, multi-hop setting.
- **RepLiQA (Monteiro et al., 2024)**: Crowdsources corpora to resist data leakage, but remains a static dataset, which is less thorough than the on-demand generation in PhantomWiki.
- **ToolQA (Zhuang et al., 2023)**: Evaluates tool-augmented QA across multiple domains, but relies on real-world data and requires manual verification. PhantomWiki focuses on systematic stress-testing of logical reasoning.
- **RULER (Hsieh et al., 2024)**: A long-context evaluation framework, but with a narrower scope. PhantomWiki provides a richer evaluation along the retrieval dimension.

**Insight**: This benchmark paradigm of "releasing the generator instead of the dataset" is worth promoting across more research directions, especially for areas that prize fair evaluation. The fact that GRPO fine-tuning significantly outperforms SFT also hints at the potential of reinforcement learning for complex reasoning tasks.

## Rating

| Dimension | Score (1-5) | Description / Notes |
|------|-----------|------|
| Novelty | ⭐⭐⭐⭐ | The concept of "generator-as-a-benchmark" is relatively novel in the QA evaluation field |
| Technical Depth | ⭐⭐⭐⭐ | The CFG + Prolog design is elegant and accompanied by theoretical analysis |
| Quality of Experiments | ⭐⭐⭐⭐⭐ | Comprehensive evaluation across multiple models, strategies, and scales, yielding clear conclusions |
| Usability | ⭐⭐⭐⭐ | Available via `pip install`, with sample data on Hugging Face |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure with informative diagrams |
| Overall Rating | ⭐⭐⭐⭐ | Excellent evaluation infrastructure work; the idea of decoupling reasoning and retrieval is highly valuable |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](../../ACL2025/llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ECCV 2024\] Sync from the Sea: Retrieving Alignable Videos from Large-Scale Datasets](../../ECCV2024/llm_evaluation/sync_from_the_sea_retrieving_alignable_videos_from_large-scale_datasets.md)
- [\[ICML 2025\] Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation](leveraging_online_olympiad-level_math_problems_for_llms_training_and_contaminati.md)
- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](../../ACL2025/llm_evaluation/mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](../../ACL2025/llm_evaluation/physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)

</div>

<!-- RELATED:END -->
