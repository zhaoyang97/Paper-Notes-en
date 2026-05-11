---
title: >-
  [Paper Note] Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification
description: >-
  [ACL 2026][Reinforcement Learning][Text Simplification] This paper proposes Re-RIGHT, a framework that trains a 4B policy model via GRPO with a three-module reward (vocabulary coverage + semantic preservation + coherence…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Text Simplification"
  - "Vocabulary-Level Control"
  - "Multilingual Reinforcement Learning"
  - "GRPO"
  - "Language Learning"
date: 2026-05-08
content_hash: 76386f9cfc27167b
---

# Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification

**Conference**: ACL 2026
**arXiv**: [2604.05302](https://arxiv.org/abs/2604.05302)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Text Simplification, Vocabulary-Level Control, Multilingual Reinforcement Learning, GRPO, Language Learning

## TL;DR

This paper proposes Re-RIGHT, a framework that trains a 4B policy model via GRPO with a three-module reward (vocabulary coverage + semantic preservation + coherence) to accurately simplify text in English, Japanese, Korean, and Chinese according to learner proficiency levels (CEFR/JLPT/TOPIK/HSK), outperforming large models such as GPT-5.2 and Gemini 2.5.

## Background & Motivation

**Background**: Second language acquisition theory (the Input Hypothesis) suggests that learners acquire language most effectively from "comprehensible input" slightly above their current level ($i$ to $i+1$). Research shows that learners need to recognize 95–98% of vocabulary in a text for fluent reading. Text simplification aims to control lexical complexity to match the target reader's vocabulary knowledge.

**Limitations of Prior Work**: (1) Even state-of-the-art LLMs (GPT-5.2, Gemini 2.5) cannot reliably generate text that precisely matches a specific proficiency level, especially at lower levels (e.g., CEFR A1 achieves only 45.1% vocabulary coverage) and in non-English languages; (2) existing RL methods require pre-annotated level-labeled sentence corpora and are largely limited to English; (3) constructing personalized parallel corpora is costly.

**Key Challenge**: LLMs lack fine-grained control over vocabulary proficiency levels — they can produce "simple" text but cannot guarantee vocabulary coverage at a specific level. The problem worsens at lower levels and for less-resourced languages (English A1 averages 42.6% vs. Korean TOPIK1 only 29.8%).

**Goal**: To train a unified multilingual policy model that precisely controls vocabulary simplification across four languages and their respective proficiency frameworks, without requiring level-annotated parallel corpora.

**Key Insight**: Official vocabulary-level data from each language (CEFR/JLPT/TOPIK/HSK, totaling 43K entries) are used as evaluation signals rather than training labels, enabling the model to learn simplification strategies autonomously via GRPO training.

**Core Idea**: A 4B policy model is trained with GRPO using three reward modules that separately control vocabulary-level coverage, semantic preservation, and textual coherence, achieving precise cross-lingual and cross-level simplification without parallel corpora.

## Method

### Overall Architecture

The preparation phase collects Wikipedia featured articles as training seed data (8,057 articles, 69,220 text chunks) along with 43K multilingual vocabulary-level entries. The training phase optimizes the Qwen3-4B policy model with the GRPO algorithm, sampling 8 candidate responses per prompt and using a weighted combination of three reward modules as the training signal. A single policy model handles all four languages and proficiency levels uniformly.

### Key Designs

1. **Vocabulary Coverage Reward**:

    - Function: Measures the proportion of target-level-or-below vocabulary in the generated text.
    - Mechanism: Candidate texts are lemmatized; function words, stop words, and proper nouns are removed. The proportion of content words at or below the target level is computed as $\text{score}_{vocab} = |\{w_i \in M(C) \mid \ell(w_i) \leq \ell_t\}| / |M(C)|$. The reward is the difference in vocabulary coverage between the simplified output and the original text: $r_{vocab} = \text{score}_{rollout} - \text{score}_{original}$, encouraging improvement rather than optimizing absolute values. Chinese additionally supports character-level decomposition matching.
    - Design Motivation: The 43K vocabulary-level data enable precise level assessment; the delta reward facilitates more effective learning through within-group comparisons in GRPO.

2. **Semantic Preservation Reward**:

    - Function: Ensures that simplification does not lose information from the source text.
    - Mechanism: A two-stage approach — BERTScore is first used to greedily align reference sentences with candidate sentences (supporting 1:N alignment to accommodate complex-to-simple sentence splits), followed by bidirectional entailment verification using a multilingual NLI model. Mutual entailment scores 1.0, one-directional entailment scores 0.5, and no entailment scores 0. The final score is the average across all aligned pairs.
    - Design Motivation: Sentences may be split or merged during simplification, making traditional sentence-by-sentence comparison inappropriate; 1:N greedy alignment combined with bidirectional entailment correctly handles structural changes.

3. **Coherence Reward**:

    - Function: Ensures the naturalness and fluency of generated text.
    - Mechanism: An LLM-as-a-judge approach is adopted, where a 14B evaluation model scores both the reference and candidate texts on a 0–100 scale. After normalization, a quadratic transformation amplifies penalties for low-quality outputs: $r_{coherence} = \max(1 - (\frac{1-\text{score}}{1-\alpha})^2, 0) - \beta J(S_t(A), S_t(B))$. A Jaccard similarity penalty term discourages the model from copying the source text directly.
    - Design Motivation: RL training is prone to reward hacking — the model may generate repetitive templates or copy the source verbatim. The quadratic penalty and copy penalty jointly prevent these degenerate behaviors.

### Loss & Training

The GRPO algorithm is used, with the final reward being a weighted sum of the three modules. The policy model is Qwen3-4B with LoRA, sampling 8 candidates per prompt. The training set consists of Wikipedia featured article text chunks (up to 512 tokens), and a single model handles all languages and levels. During evaluation, a separate evaluation model (average of gemma-3-27b and Qwen3-32B) is used to avoid self-evaluation bias.

## Key Experimental Results

### Main Results

| Language | Method | Vocab. Coverage (Overall) | Vocab. Coverage (Simple) | Semantic Preservation | Coherence |
|----------|--------|--------------------------|-------------------------|-----------------------|-----------|
| EN | Re-RIGHT | 81.6% | 66.9% | 80.8 | 82.9 |
| EN | GPT-5.2 | 71.0% | 52.4% | 76.1 | 84.6 |
| EN | Gemini 2.5 | 77.7% | 59.1% | 72.0 | 82.8 |
| JA | Re-RIGHT | 76.0% | 60.4% | 80.6 | 83.1 |
| JA | Gemini 2.5 | 65.8% | 51.1% | 61.7 | 85.2 |
| ZH | Re-RIGHT | 80.2% | 66.1% | 76.6 | 83.7 |
| ZH | Gemini 2.5 | 64.4% | 48.6% | 66.7 | 85.3 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| FUDGE (controlled decoding) | EN 74.2% | Limited by rule-based discriminator |
| Base Qwen3-4B | EN 72.6% | Prompting-only baseline |
| Reference (original text) | EN 58.3% | Unsimplified source text |

### Key Findings

- After Re-RIGHT training, the 4B model substantially outperforms GPT-5.2 (EN: 81.6% vs. 71.0%) and Gemini 2.5 (ZH: 80.2% vs. 64.4%) in vocabulary coverage, demonstrating that precise control can be acquired through RL training.
- Improvements are most pronounced at lower proficiency levels: English Easy increases from 52.4% (GPT-5.2) to 66.9%; Chinese from 48.6% (Gemini) to 66.1%.
- Re-RIGHT also generally outperforms larger models in semantic preservation, indicating that RL training not only improves controllability but also enhances simplification quality.
- Coherence is slightly lower than GPT-5.2 (a reasonable trade-off) but substantially better than simple constrained decoding methods.

## Highlights & Insights

- **Proficiency-precise simplification without parallel corpora**: Conventional methods require large quantities of level-annotated parallel sentence pairs, whereas Re-RIGHT requires only a vocabulary list and seed text for training, significantly lowering the barrier to extension across new languages. The framework can readily be applied to any language with an official vocabulary-level system.
- **1:N bidirectional entailment for semantic evaluation**: The approach elegantly handles sentence splitting caused by simplification — greedy alignment followed by bidirectional entailment is more robust than traditional sentence-by-sentence BERTScore comparison.
- **Insights from 4B model outperforming GPT-5.2**: Precise control capability is not a function of model scale alone; targeted RL training is essential. This provides further evidence for the "small model + precise training > large model + general capability" paradigm.

## Limitations & Future Work

- The coherence reward relies on a 14B evaluation model; training cost and stability are constrained by the quality of the judge model.
- The current framework covers only four languages; extension to low-resource languages may be hindered by the lack of vocabulary-level data.
- Training and inference are limited to 512 tokens; long-document simplification requires further investigation.
- Future directions include incorporating reading ease metrics for finer-grained syntactic simplification and conducting user studies with actual L2 learners to validate real-world pedagogical effectiveness.

## Related Work & Insights

- **vs. Li et al. (2025b) PPO approach**: Their method trains an English simplification model with PPO but requires CEFR level-annotated sentence data; Re-RIGHT requires only GRPO and a vocabulary list, and supports multiple languages.
- **vs. FUDGE controlled decoding**: FUDGE enforces vocabulary-level constraints token by token during decoding, but yields limited gains (74.2% vs. 81.6%), as decoding-level control cannot alter the model's internal vocabulary selection preferences.

## Rating

- Novelty: ⭐⭐⭐⭐ — The multilingual proficiency-precise simplification RL framework is novel, with an elegant three-module reward design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across four languages; comparisons with GPT-5.2/Gemini 2.5 are persuasive; qualitative examples are illustrative.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is well-argued, methodology is clearly described, and the linguistic background is well-grounded.
- Value: ⭐⭐⭐⭐ — High practical value — directly applicable to L2 education; the 4B model supports local deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[AAAI 2026\] Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction](../../AAAI2026/reinforcement_learning/learning_to_generate_and_extract_a_multi-agent_collaboration_framework_for_zero-.md)
- [\[ACL 2026\] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)
- [\[ACL 2026\] Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](reverse_constitutional_ai_a_framework_for_controllable_toxic_data_generation_via.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](../../AAAI2026/reinforcement_learning/mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)

</div>

<!-- RELATED:END -->
