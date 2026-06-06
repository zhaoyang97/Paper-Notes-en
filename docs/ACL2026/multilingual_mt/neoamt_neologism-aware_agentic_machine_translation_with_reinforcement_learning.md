---
title: >-
  [Paper Note] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning
description: >-
  [ACL2026][Multilingual & Machine Translation][Neologism Translation] NeoAMT transforms neologism translation from a problem purely dependent on model parameter knowledge into an agentic MT task characterized by "reasonin…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "Neologism Translation"
  - "Machine Translation"
  - "Retrieval-Augmented Generation"
  - "Reinforcement Learning"
  - "Wiktionary"
date: 2026-05-08
content_hash: 717e3080fc007a80
---

# NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning

**Conference**: ACL2026  
**arXiv**: [2601.03790](https://arxiv.org/abs/2601.03790)  
**Code**: https://github.com/gpgg/neoamt  
**Area**: Multilingual Machine Translation / Agent / Reinforcement Learning  
**Keywords**: Neologism Translation, Machine Translation, Retrieval-Augmented Generation, Reinforcement Learning, Wiktionary  

## TL;DR
NeoAMT transforms neologism translation from a problem purely dependent on model parameter knowledge into an agentic MT task characterized by "reasoning, then dictionary look-up, then translation." By utilizing GRPO training focused on neologism hit rates, overall translation quality, and translation difficulty, the 8B model significantly outperforms SFT, retrieval-free RL, and various general/translation-specific LLMs on the Neko neologism translation benchmark.

## Background & Motivation
**Background**: Recently, LLM-based machine translation has increasingly adopted training paradigms from reasoning models and RL. Works like MT-R1-Zero, DeepTrans, and SSR-Zero primarily use reward design to encourage models to think before translating or utilize neural evaluators and format rewards to improve overall quality. This approach assumes that the model already possesses sufficient linguistic knowledge and focuses on how to activate it.

**Limitations of Prior Work**: Neologism translation challenges this assumption. Internet slang, subculture terms, technical vocabulary, and new political/community expressions emerge constantly, while the parametric knowledge of LLMs is essentially frozen after training. When source sentences contain terms like "给她爱" (Grand Theft Auto), "长草" (abandoned/desired), or "铁胶" (Iron Man fans), models may mistranslate based on literal meaning or treat neologisms as ordinary phrases even with reasoning. Furthermore, while the existing Neo-bench contains neologism tasks, its MT subset is small (240 entries), limited in language pairs, and not public, lacking a data foundation for systematic training.

**Key Challenge**: Neologism translation requires two types of capabilities: understanding the true meaning of a new word in the current context and naturally integrating that meaning into the target sentence. Pure SFT is limited by small-scale data; reasoning-only RL is restricted by parametric knowledge; and standard RAG often disrupts instruction following and fluency when dictionary entries are simply stuffed into the prompt.

**Goal**: The authors decompose the task into three sub-problems: 1) Constructing a multilingual, multi-direction neologism translation dataset with definitions and examples. 2) Developing a dictionary retrieval tool callable by an MT agent rather than treated as a one-time prompt augmentation. 3) Designing an RL training method that teaches the model when to search, what to search for, and how to utilize results, optimizing both neologism accuracy and overall quality.

**Key Insight**: Neologism translation resembles the human translation process of "looking up a dictionary for unfamiliar terms" rather than a general Q&A RAG task. a translation agent should not passively receive retrieval results but should identify potential neologisms and actively initiate queries to correct the translation based on dictionary information.

**Core Idea**: Build a neologism dataset and retrieval tool using Wiktionary, then train an agent with GRPO to alternate between reasoning and retrieval, shifting the model from "memorization-based translation" to "verification-informed translation."

## Method
The NeoAMT method consists of two layers: the data and tool layer solves "where to find knowledge," while the RL training layer solves "how the model learns to use it." The model generates trajectories including `<think>`, `<search>`, `<information>`, and `<translation>` tags.

### Overall Architecture
The input consists of a source sentence with a neologism and a target language direction. The system uses a unified prompt requiring the model to reason in `<think>` and use `<search>` for query terms when necessary. The search tool retrieves top-k entries from a dictionary built from English Wiktionary dumps and writes results into `<information>`. The model can iterate reasoning and searching up to 3 rounds before providing the final output in `<translation>`.

Training data comes from the Neko dataset, which processes approximately 10 million records from the 2025-08-23 Wiktionary dump into 3.3 million structured entries across 16 languages, including 3,606 entries labeled as neologisms. Data is categorized into: Type 1 (labeled neologisms with examples and human translations), Type 2 (neologisms with examples but no translations), and Type 3 (common words or neologisms without full examples).

The Neko retrieval tool encodes full dictionary entries (including part of speech, etymology, and glosses) using `bge-m3` to distinguish homographs. The backend uses FAISS for cosine similarity search.

The training phase uses Qwen3-4B and Qwen3-8B as base models, implemented via `verl/vLLM` for GRPO. Unlike standard GRPO, NeoAMT optimizes a composite reward (neologism hit, neural quality, format, and optional process rewards) and dynamically adjusts rollout numbers based on translation difficulty.

### Key Designs
1. **Wiktionary-driven Neko Dataset and Retrieval Dictionary**:
    - **Function**: Provides samples and external knowledge for training/testing to address the lack of public neologism MT benchmarks.
    - **Mechanism**: Structured entries from 16 languages were extracted. Type 1 provides human-verified pairs for testing. Type 2 uses GPT-5 to generate training translations under definition constraints with neologism spans labeled in the target language. The dictionary encodes sense and etymology into dense vectors to avoid surface-level confusion.
    - **Design Motivation**: The bottleneck is the lack of context-aware neologism definitions. Using Wiktionary ensures reproducible training/evaluation sources and task-relevant external knowledge.

2. **Neologism-oriented Agentic Translation Prompt and Search Loop**:
    - **Function**: Enables the model to identify unfamiliar expressions and actively query dictionaries.
    - **Mechanism**: Model outputs are organized into interactive trajectories. The retrieval system returns results that the model must then integrate. Loss masking is applied so the model only learns its own behavior, not the external text in `<information>`.
    - **Design Motivation**: To prevent the model from getting lost in large prompts or generating hallucinations, the retrieval action is placed within a learnable reasoning process guided by RL rewards.

3. **Neologism Rewards and Translation Difficulty-based Adaptive GRPO**:
    - **Function**: Balances neologism hit rates, semantic quality, and computational budget.
    - **Mechanism**: The outcome reward includes a format indicator, a neologism reward ($R_{neo}$, checking for lemmatized neologism spans), and a neural quality reward ($R_{neural}$, a mix of XCOMET-XL and CometKiwi-DA-XL). Total reward: $R=1_{format}(\lambda R_{neo}+\sigma R_q+(1-\lambda-\sigma)R_{neural})$, where $\lambda=0.1$ and process reward $\sigma=0.1$ if used.
    - **Design Motivation**: To prevent the model from prioritizing fluency over accuracy (or vice versa), adaptive sampling (RQE) allocates more rollouts to difficult samples defined by the quality variance $v=\Phi(x,y^{ref})-\Phi(x,\hat{y})$.

### Loss & Training
NeoAMT uses GRPO for policy optimization. Group size $g$ is dynamic: starting at 4, it increases when $v > 0$ (model performing worse than reference) according to $g=g_{initial}\exp(\alpha v+\psi)$ and decreases when $v < 0$. Parameters are set to $\alpha=10, \gamma=-5, \psi=0$. Training uses 8 A100 80GB GPUs with a prompt length of 1024 and response length of 4096.

## Key Experimental Results

### Main Results
The main experiment focuses on the other-language-to-English direction (743 samples). Metrics include neologism-specific hit rates (EXACT, LEM-FUZZY) and overall quality (GEMBA-GPT5, LJ-GPT5).

| Model | EXACT | LEM-FUZZY | GEMBA(GPT5) | LJ(GPT5) | Note |
|------|------:|----------:|------------:|---------:|----------|
| Qwen3-4B | 13.19 | 18.57 | 65.24 | 51.94 | Limited neologism knowledge |
| SFT-4B | 13.73 | 18.84 | 66.92 | 54.00 | SFT improves quality but lacks neologism accuracy |
| GRPO-4B | 13.73 | 18.98 | 71.29 | 55.34 | Reasoning improves MT but lacks external knowledge |
| NeoAMT-4B | 17.63 | 21.53 | 72.93 | 58.16 | Search loop improves both metrics |
| NeoAMT-8B | 22.34 | 28.67 | 78.28 | 66.40 | **Best overall performance** |

SFT offers negligible improvements for neologisms. NeoAMT-8B significantly increases EXACT and LJ scores, proving that search actions fill the parametric knowledge gap.

### Ablation Study
Comparing RAG vs. Agentic Search and the effect of RQE adaptive sampling:

| Configuration | Neologism Metrics | Overall Quality | Note |
|------|--------------|--------------|------|
| Qwen3-8B + RAG | EXACT 23.68 / LEM-FUZZY 25.43 | GEMBA 69.57 / LJ 56.14 | High hit rate, lower fluency |
| NeoAMT-8B | EXACT 22.34 / LEM-FUZZY 28.67 | GEMBA 78.28 / LJ 66.40 | Better balance |
| NeoAMT-8B w/o RQE | EXACT 20.19 / LEM-FUZZY 25.17 | GEMBA 77.65 / LJ 64.50 | Insufficient learning on hard samples |

### Key Findings
- **SFT is insufficient**: Knowledge injection is more critical than increasing examples.
- **RAG vs. Agent**: RAG causes the model to ignore instructions or hallucinate entries; agentic search allows for selective integration.
- **RQE is effective**: Removing difficulty-based sampling leads to a noticeable drop in all metrics.
- **Inference cost**: NeoAMT-8B takes ~0.77s/sentence vs. 0.33s for direct MT; the latency is higher but acceptable for deployment.

## Highlights & Insights
- Defining neologism translation as an agentic task aligns with human behavior (identify, look up, integrate).
- Cohesive data and tool design (Wiktionary for all stages) ensures consistency in training and retrieval.
- Reward design effectively penalizes "fluent but incorrect" translations while maintaining overall quality scores.

## Limitations & Future Work
- **Retriever Bottleneck**: Using general-purpose embeddings like `bge-m3` is not optimal for dictionary-specific retrieval.
- **Language Bias**: Wiktionary content is primarily curated by English-speaking communities, leading to gaps in non-English slang.
- **Synthetic Data Dependency**: Training relies on GPT-5 generated translations, which might contain undetected errors in low-resource directions.

## Related Work & Insights
Compared to **MT-R1-Zero** and **DeepTrans**, NeoAMT incorporates external tools to solve knowledge deficits. Unlike **Search-R1** (QA focused), NeoAMT tailors the search mechanism to translation-specific constraints like terminology hit rates and quality estimation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Excellent integration of agentic MT and GRPO).
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ (Solid comparisons, though human evaluation could be broader).
- **Writing Quality**: ⭐⭐⭐⭐☆ (Clear structure and detailed appendices).
- **Value**: ⭐⭐⭐⭐⭐ (Directly addresses real-world pain points in MT systems).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2026\] Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax](reinforcement_learning_with_semantic_rewards_enables_low-resource_language_expan.md)
- [\[ACL 2026\] PEAR: Pairwise Evaluation for Automatic Relative Scoring in Machine Translation](pear_pairwise_evaluation_for_automatic_relative_scoring_in_machine_translation.md)
- [\[ACL 2026\] Selective Contrastive Learning For Gloss Free Sign Language Translation](selective_contrastive_learning_for_gloss_free_sign_language_translation.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)

</div>

<!-- RELATED:END -->
