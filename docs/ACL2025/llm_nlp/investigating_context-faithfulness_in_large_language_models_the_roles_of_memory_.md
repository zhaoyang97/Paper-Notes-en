---
title: >-
  [Paper Note] Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style
description: >-
  [ACL 2025][LLM (Other)][context faithfulness] The authors quantify "memory strength" by measuring the consistency of LLM responses to different paraphrases of the same question, finding that the model's acceptance of external evidence is highly negatively correlated with memory strength, and paraphrased evidence is more effective than repetitive or detailed evidence.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "context faithfulness"
  - "RAG"
  - "memory strength"
  - "evidence style"
  - "knowledge conflict"
date: 2026-05-08
content_hash: c85266fb0e180754
---

# Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style

**Conference**: ACL 2025  
**arXiv**: [2409.10955](https://arxiv.org/abs/2409.10955)  
**Code**: [https://github.com/liyp0095/ContextFaithful](https://github.com/liyp0095/ContextFaithful)  
**Area**: LLM/NLP  
**Keywords**: context faithfulness, RAG, memory strength, evidence style, knowledge conflict

## TL;DR
The authors quantify "memory strength" by measuring the consistency of LLM responses to different paraphrases of the same question, finding that the model's acceptance of external evidence is highly negatively correlated with memory strength, and paraphrased evidence is more effective than repetitive or detailed evidence.

## Background & Motivation

**Background**: RAG enhances LLMs by introducing external information. However, when external information conflicts with the model's internal memory knowledge, LLMs often ignore the external evidence.

**Limitations of Prior Work**: (1) Prior works often use long contexts for testing, which fails to distinguish between "ignoring evidence" and "incapacity to understand long context"; (2) different LLMs possess varying amounts of internal knowledge, making evaluation on a unified dataset unfair.

**Key Challenge**: How to fairly and accurately measure the context faithfulness of LLMs? This requires a model-aware evaluation framework.

**Goal**: (1) Quantify memory strength and analyze its impact on context faithfulness; (2) investigate how the evidence presentation style affects persuasiveness.

**Key Insight**: Consistency across paraphrased questions is leveraged to measure memory strength—more consistent answers to different expressions of the same question indicate stronger memory.

**Core Idea**: An LLM's context faithfulness is a function of its memory strength: stronger internal memories lead to greater rejection of external evidence. Moreover, paraphrased evidence is more persuasive than simple repetition or detailed extensions.

## Method

### Overall Architecture
Generate 7 paraphrases for each question -> LLM answers all paraphrases -> Cluster these responses to calculate memory strength (negative entropy) -> Generate Counter-Memory Answers (CMA) -> Construct evidence in different styles -> Evaluate LLM's choices among Memory Answer (MA), CMA, or Undecided.

### Key Designs

1. **Memory Strength Quantification**

    - For a question $Q$, generate $n=7$ paraphrases, and have the LLM answer each of them.
    - Cluster the answers and calculate $S(Q) = \sum_i \frac{N(c_i)}{n} \log \frac{N(c_i)}{n}$ (negative entropy).
    - $S(Q) = 0$ represents the strongest memory (the same answer for all paraphrases), while more negative values indicate weaker memory.
    - **Design Motivation**: Consistency directly reflects the model's confidence in its internal knowledge.

2. **Evidence Style Classification**

    - **Direct Evidence**: Semantically equivalent paraphrasing of the CMA (most concise and clear).
    - **Indirect Evidence**: CMA supplemented with additional supporting details (more detailed but potentially distracting).
    - Validation using NLI models: direct evidence must mutually entail the CMA, while indirect evidence must entail the CMA and not entail the MA.
    - **Design Motivation**: Distinguishing the impact of "what is said" versus "how it is said" on persuasiveness.

3. **CMA Generation and Filtering**

    - The CMA must not match any of the answers from the paraphrased questions.
    - Use an LLM for entity replacement (e.g., replacing dates or names with reasonable alternatives).
    - **Design Motivation**: Ensuring that the CMA strictly conflicts with all of the model's internal memories.

4. **Four-group Memory Strength Binning**

    - Four bins: low, mid-low, mid-high, and high.
    - Corresponding to intervals $[-2,-1]$, $(-1,-0.5]$, $(-0.5,-0.25]$, and $(-0.25,0]$ respectively.
    - **Design Motivation**: Granular analysis of behavioral differences across various levels of memory strength.

## Key Experimental Results

### Main Results — Memory Strength vs. Context Faithfulness (Direct Evidence, popQA Dataset)

| Memory Strength | GPT-4 MA% | GPT-4 CMA% | LLaMA2-7B MA% | LLaMA2-7B CMA% |
|---------|----------|-----------|-------------|---------------|
| Low | ~10% | ~80% | ~15% | ~70% |
| Mid-low | ~25% | ~60% | ~25% | ~55% |
| Mid-high | ~35% | ~50% | ~30% | ~45% |
| **High** | **~50%** | **~35%** | **~40%** | **~30%** |

### Evidence Style Comparison (High Memory Strength Group)

| Evidence Style | GPT-4 CMA Acceptance Rate | LLaMA2-70B CMA Acceptance Rate |
|---------|-----------------|---------------------|
| Direct (Single) | 35% | 40% |
| Direct (Multiple Paraphrases) | **55%** | **58%** |
| Indirect (Detailed) | 38% | 42% |
| Direct (Repetitive) | 30% | 35% |

### Cross-Model Memory Strength Distribution

| Model | Average Memory Strength | MA Ratio | Description |
|------|-----------|---------|------|
| GPT-4 | High | **~50%** | Vast knowledge but unfaithful |
| Claude3.5| High | ~20% (High UCT) | Vast knowledge but tends to be "Undecided" |
| LLaMA3.2-3B | **Low** | ~35% | Limited knowledge but remains stubborn |

### Key Findings
- **Memory strength is positively correlated with the MA ratio**: Consistent across all models and datasets; stronger memory leads to a higher rate of rejecting external evidence.
- **Paraphrased evidence is the most effective**: Presenting multiple different phrasings of the same evidence is more persuasive to LLMs than simple repetition or adding details (+20% CMA acceptance rate in the high-memory group).
- **Model scale is positively correlated with memory strength**: GPT-4 has more questions falling into the high-memory group, whereas LLaMA3.2-3B has more in the low-memory group.
- **Newer models are less faithful**: GPT-4 exhibits a higher MA ratio than ChatGPT, and LLaMA3.2 has a higher one than LLaMA2.
- **Low memory strength does not equate to high faithfulness**: LLaMA3.2-3B contains the least internal knowledge yet still maintains a relatively high MA ratio, indicating stubbornness even during knowledge conflicts.
- **Insufficiency of existing fairness metrics**: Different models exhibit distinct memory strength distributions, making direct comparisons of simple MA ratios unfair.

## Highlights & Insights
- **The methodology for quantifying memory strength** is simple yet effective, replacing the reliance on training data frequency with paraphrastic consistency, making it applicable to any LLM (including closed-source ones).
- **The persuasiveness hierarchy of "Paraphrase > Repetitive > Detailed"** offers direct guidance for RAG system design, demonstrating that retrieving multiple differently phrased evidences is more effective than repeating the exact same evidence.
- **The discovery that newer models are more stubborn** is critical, suggesting that post-training techniques like RLHF might boost model "confidence" at the cost of context faithfulness.

## Limitations & Future Work
- Paraphrases are generated by ChatGPT, which may limit quality.
- The MCQ (Multiple-Choice Question) format may not fully reflect behaviors in open-ended generation.
- Evaluated on only 2 datasets (popQA + NQ).
- Future directions: Memory-strength-aware RAG strategies, dynamic evidence augmentation, and enhancing context faithfulness during model training.

## Related Work & Insights
- **vs Xie et al. (2024)**: They found that LLMs are highly receptive to coherent evidence but overlooked the factor of memory strength. This paper demonstrates that the acceptance rate is significantly lower than expected under strong memory conditions.
- **vs Longpre et al. (2021)**: They tested faithfulness using entity replacement. This paper improves upon evidence generation and memory quantification methodologies.

## Rating
- Novelty: ⭐⭐⭐⭐ Both memory strength quantification and evidence style analysis represent novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 6 LLMs, 2 datasets, 4 memory bins, and multiple evidence styles.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology and high-quality visualizations.
- Value: ⭐⭐⭐⭐⭐ Highly practical value for both RAG system design and research into LLM trustworthiness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pitfalls of Scale: Investigating the Inverse Task of Redefinition in Large Language Models](pitfalls_of_scale_investigating_the_inverse_task_of_redefinition_in_large_langua.md)
- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)
- [\[ACL 2025\] Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization](improving_contextual_faithfulness_of_large_language_models_via_retrieval_heads-i.md)
- [\[ACL 2025\] CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models](cognibench_cognitive_faithfulness.md)
- [\[ACL 2025\] RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](retrollm_empowering_large_language_models_to_retrieve_fine-grained_evidence_with.md)

</div>

<!-- RELATED:END -->
