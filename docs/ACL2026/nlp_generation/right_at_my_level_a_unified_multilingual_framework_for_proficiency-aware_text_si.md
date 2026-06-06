---
title: >-
  [Paper Note] Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification
description: >-
  [ACL 2026][Text Generation][Text Simplification] The Re-RIGHT framework is proposed, utilizing Group Relative Policy Optimization (GRPO) training with a three-module reward (vocabulary coverage + semantic preservation +…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Text Simplification"
  - "Vocabulary Level Control"
  - "Multilingual Reinforcement Learning"
  - "GRPO"
  - "Language Learning"
date: 2026-05-08
content_hash: ec84336d5428d8bf
---

# Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification

**Conference**: ACL 2026  
**arXiv**: [2604.05302](https://arxiv.org/abs/2604.05302)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Text Simplification, Vocabulary Level Control, Multilingual Reinforcement Learning, GRPO, Language Learning

## TL;DR

The Re-RIGHT framework is proposed, utilizing Group Relative Policy Optimization (GRPO) training with a three-module reward (vocabulary coverage + semantic preservation + coherence). Using a 4B policy model, it achieves precise text simplification matching learner proficiency levels (CEFR/JLPT/TOPIK/HSK) across four languages (English, Japanese, Korean, Chinese), outperforming large models such as GPT-5.2 and Gemini 2.5.

## Background & Motivation

**Background**: Second Language Acquisition (SLA) theory (the Input Hypothesis) suggests that learners benefit most from "comprehensible input" situated between their current level ($i$) and a slightly higher level ($i+1$). Research indicates that learners need to recognize 95-98% of the vocabulary in a text to read fluently. Text simplification aims to control lexical complexity to match the target reader's vocabulary knowledge.

**Limitations of Prior Work**: (1) Even state-of-the-art LLMs (GPT-5.2, Gemini 2.5) cannot reliably generate text that precisely matches specific proficiency levels, particularly in lower levels (e.g., CEFR A1 achieves only 45.1% vocabulary coverage) and non-English languages; (2) Existing RL methods require pre-annotated corpora with level labels and are primarily restricted to English; (3) Constructing personalized parallel corpora is highly expensive.

**Key Challenge**: LLMs lack the ability to precisely control lexical levels—they can write "simple" text but cannot guarantee vocabulary coverage for a specific level. This issue becomes more severe as the level decreases or the language becomes less mainstream (English A1 average 42.6% vs. Korean TOPIK1 only 29.8%).

**Goal**: To train a unified multilingual policy model that precisely controls lexical simplification across four languages under their respective proficiency systems without requiring level-annotated parallel corpora.

**Key Insight**: Utilize official vocabulary level data for each language (CEFR/JLPT/TOPIK/HSK, totaling 43K entries) as evaluation signals rather than training labels, allowing the model to autonomously learn simplification strategies through GRPO training.

**Core Idea**: Train a 4B policy model using GRPO with three reward modules controlling vocabulary level coverage, semantic preservation, and text coherence, achieving precise cross-lingual and cross-level simplification without parallel corpora.

## Method

### Overall Architecture

In the preparation phase, selected Wikipedia articles are collected as training seed data (8,057 articles, 69,220 text blocks) alongside 43K multilingual vocabulary level data points. During the training phase, the GRPO algorithm optimizes the Qwen3-4B policy model. For each prompt, 8 candidate responses are sampled, and a weighted combination of three reward modules provides the feedback signal. A single policy model uniformly handles four languages and all proficiency levels.

### Key Designs

1.  **Vocabulary Coverage Reward**:

    - **Function**: Measures the proportion of vocabulary in the generated text that belongs to the target level or below.
    - **Mechanism**: Candidate text undergoes lemmatization, and functional words, stop words, and proper nouns are removed. The ratio of content words with level $\ell(w_i) \leq \ell_t$ is calculated: $\text{score}_{vocab} = |\{w_i \in M(C) \mid \ell(w_i) \leq \ell_t\}| / |M(C)|$. The reward is defined as the difference in vocabulary coverage between the simplified version and the original: $r_{vocab} = \text{score}_{rollout} - \text{score}_{original}$, encouraging improvement rather than absolute values. For Chinese, character-level decomposition matching is also supported.
    - **Design Motivation**: Achieves precise level assessment based on 43K vocabulary entries. Difference-based rewards allow the model to learn more effectively during GRPO intra-group comparisons.

2.  **Semantic Preservation Reward**:

    - **Function**: Ensures that simplification does not result in the loss of information from the source text.
    - **Mechanism**: A two-stage approach—first, BERTScore is used for greedy pairing of reference sentences and candidate sentences (supporting 1:N alignment to allow complex sentences to be split into multiple simple ones). Next, a multilingual NLI model performs bidirectional verification of entailment. Pairs with full bidirectional entailment receive 1.0, unidirectional receives 0.5, and others receive 0. The final score is the average across all pairs.
    - **Design Motivation**: Since sentences may be split or merged during simplification, traditional sentence-by-sentence comparison is unsuitable. 1:N greedy pairing combined with bidirectional entailment correctly handles structural changes.

3.  **Coherence Reward**:

    - **Function**: Ensures the naturalness and fluency of the generated text.
    - **Mechanism**: Employs an LLM-as-a-judge approach, where a 14B evaluation model compares the reference and candidate texts to provide a score from 0-100. This is normalized and subjected to a quadratic transformation to increase penalties for low-quality output: $r_{coherence} = \max(1 - (\frac{1-\text{score}}{1-\alpha})^2, 0) - \beta J(S_t(A), S_t(B))$. A Jaccard similarity penalty term prevents the model from directly copying the source text.
    - **Design Motivation**: RL training is prone to reward hacking—models may generate repetitive templates or copy the source text directly. Quadratic penalties and copy penalties work together to prevent these degenerative behaviors.

### Loss & Training

The GRPO algorithm is used, and the final reward is a weighted sum of the three modules. The policy model is Qwen3-4B with LoRA, sampling 8 candidates per prompt. The training set consists of Wikipedia text blocks (maximum 512 tokens). A single model handles all languages and levels. Evaluation utilizes different models (averaging scores from gemma-3-27b and Qwen3-32B) to avoid self-evaluation bias.

## Key Experimental Results

### Main Results

| Language | Method | Vocab Coverage (Total) | Vocab Coverage (Simple) | Semantic Preservation | Coherence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EN | Re-RIGHT | 81.6% | 66.9% | 80.8 | 82.9 |
| EN | GPT-5.2 | 71.0% | 52.4% | 76.1 | 84.6 |
| EN | Gemini 2.5 | 77.7% | 59.1% | 72.0 | 82.8 |
| JA | Re-RIGHT | 76.0% | 60.4% | 80.6 | 83.1 |
| JA | Gemini 2.5 | 65.8% | 51.1% | 61.7 | 85.2 |
| ZH | Re-RIGHT | 80.2% | 66.1% | 76.6 | 83.7 |
| ZH | Gemini 2.5 | 64.4% | 48.6% | 66.7 | 85.3 |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| FUDGE (Controlled Decoding) | EN 74.2% | Limited improvement due to rule-based discriminators |
| Base Qwen3-4B | EN 72.6% | Baseline for pure prompting methods |
| Reference (Original) | EN 58.3% | Original unsimplified text |

### Key Findings

- After Re-RIGHT training, the 4B model significantly outperforms GPT-5.2 (EN: 81.6% vs. 71.0%) and Gemini 2.5 (ZH: 80.2% vs. 64.4%) in vocabulary coverage, proving that precise control can be acquired via RL training.
- Improvements are most significant at simple levels: English Easy improved from 52.4% (GPT-5.2) to 66.9%, and Chinese from 48.6% (Gemini) to 66.1%.
- In terms of semantic preservation, Re-RIGHT generally performs better than large models, indicating that RL training enhances both control and simplification quality.
- Coherence is slightly lower than GPT-5.2 (a reasonable trade-off) but substantially better than simple constrained decoding methods.

## Highlights & Insights

- **Precise Level Simplification without Parallel Corpora**: Traditional methods require large quantities of level-annotated parallel sentence pairs; Re-RIGHT only needs a vocabulary list and seed texts, significantly lowering the barrier for expansion to new languages. This framework can easily scale to any language with an official vocabulary level system.
- **1:N Bidirectional Entailment Semantic Evaluation**: Elegantly handles sentence splitting caused by simplification—greedy pairing followed by bidirectional entailment is more robust than traditional sentence-wise BERTScore.
- **Insights from 4B Model > GPT-5.2**: Precise control is not a problem solvable by model scale alone; it requires targeted RL training. This provides further evidence for the "Small Model + Precise Training > Large Model + General Capability" paradigm.

## Limitations & Future Work

- Coherence rewards rely on a 14B evaluation model; training costs and stability are constrained by the quality of this evaluator.
- Currently limited to four languages; expansion to low-resource languages may be hindered by the lack of official vocabulary level data.
- Training and inference are restricted to 512 tokens; simplification of long documents requires further research.
- Future work: Combine reading ease metrics for fine-grained syntactic simplification, and conduct user studies with real L2 learners to verify actual pedagogical effectiveness.

## Related Work & Insights

- **vs. Li et al. (2025b) PPO Method**: They used PPO to train an English simplification model but required CEFR-level-labeled sentence data; Re-RIGHT uses GRPO + vocabulary lists and supports multiple languages.
- **vs. FUDGE Controlled Decoding**: FUDGE checks vocabulary constraints step-by-step during decoding, but its improvement is limited (74.2% vs. 81.6%) because decoding-level control cannot alter the internal lexical preference of the model.

## Rating

- Novelty: ⭐⭐⭐⭐ Innovative RL framework for multilingual level-precise simplification; the three-module reward design is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across four languages; comparisons with GPT-5.2/Gemini 2.5 are persuasive; intuitive examples provided.
- Writing Quality: ⭐⭐⭐⭐ Strong motivation, clear methodology, and solid linguistic background.
- Value: ⭐⭐⭐⭐ High practical value—directly serves L2 education; 4B model is locally deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](../../AAAI2026/nlp_generation/c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)
- [\[ACL 2026\] XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)
- [\[ACL 2026\] Frankentext: Stitching Random Text Fragments into Long-Form Narratives](frankentext_stitching_random_text_fragments_into_long-form_narratives.md)
- [\[ACL 2026\] Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style](can_you_make_it_sound_like_you_post-editing_llm-generated_text_for_personal_styl.md)

</div>

<!-- RELATED:END -->
