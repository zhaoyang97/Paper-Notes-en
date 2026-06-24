---
title: >-
  [Paper Note] Real-time Factuality Assessment from Adversarial Feedback
description: >-
  [ACL 2025][LLM Safety][Factuality Assessment] This paper reveals the “data leakage” issue in existing factuality assessment datasets (where LLMs easily identify old misinformation due to pre-training memorization) and proposes an iterative rewriting pipeline based on adversarial feedback from a RAG detector to generate truly challenging real-time fake news variants, causing a 17.5% absolute drop in ROC-AUC for the GPT-4o RAG detector.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Factuality Assessment"
  - "Adversarial Feedback"
  - "Real-time News"
  - "Retrieval-Augmented Generation"
  - "Misinformation Detection"
date: 2026-05-08
content_hash: bebd7cf6627563ac
---

# Real-time Factuality Assessment from Adversarial Feedback

**Conference**: ACL 2025  
**arXiv**: [2410.14651](https://arxiv.org/abs/2410.14651)  
**Code**: None  
**Area**: LLM Safety / Factuality Assessment  
**Keywords**: Factuality Assessment, Adversarial Feedback, Real-time News, Retrieval-Augmented Generation, Misinformation Detection

## TL;DR

This paper reveals the “data leakage” issue in existing factuality assessment datasets (where LLMs easily identify old misinformation due to pre-training memorization) and proposes an iterative rewriting pipeline based on adversarial feedback from a RAG detector to generate truly challenging real-time fake news variants, causing a 17.5% absolute drop in ROC-AUC for the GPT-4o RAG detector.

## Background & Motivation

**Background**: Utilizing LLMs for misinformation detection is an increasingly important research direction. Existing assessments typically use historical claims from fact-checking websites (such as LIAR, PolitiFact) to let models judge the truthfulness of these claims.

**Limitations of Prior Work**: The authors identify a concerning phenomenon—even in tests after the knowledge cutoff date, the accuracy of LLM-based detectors on these historical datasets continues to rise over time. This is not because the reasoning ability of the models has improved, but because these popular pieces of misinformation are highly likely to have appeared in the pre-training corpora of newer models, or because exploitable shallow patterns exist in the datasets (e.g., claims from specific sources are always false).

**Key Challenge**: Existing factuality assessments do not truly test the reasoning and evidence-analysis capabilities of models, but rather test their "memorization" and "pattern matching" abilities. This leads to overly optimistic evaluation results, obscuring the vulnerability of models when facing genuinely novel misinformation.

**Goal**: Build a dynamic assessment framework that can truly challenge LLM-based detectors—evaluation data should be based on real-time events and adversarially optimized to be as difficult to detect as possible.

**Key Insight**: The authors propose a counter-intuitive strategy—utilizing natural language feedback from a RAG-based detector to conversely help generate more deceptive fake news. The stronger the detector, the more its feedback reveals "where it is not natural enough," thereby guiding the attacker to iteratively optimize the fake content.

**Core Idea**: Design an adversarial feedback loop where the RAG detector analyzes the text to be detected and provides reasons for rejection, and the attacker iteratively rewrites it accordingly to evade detection. This forms a closed loop of "detection-feedback-rewriting," where the generated final fake news variants constitute high-quality evaluation data.

## Method

### Overall Architecture

The entire pipeline consists of three core roles and an iterative process: (1) **News Collector**—continuously collects real-time news as raw materials; (2) **RAG-based Detector**—performs factuality assessment on the text and outputs structured feedback (judgment + reasoning); (3) **Adversarial Rewriter**—iteratively modifies the text based on the detector's feedback to evade detection. The process is: collect real news → perform initial fake rewriting → detector provides feedback → rewrite based on feedback → detect again... until the detector fails to identify it or the maximum number of iterations is reached.

### Key Designs

1. **RAG-based Factuality Detector**:

    - **Function**: Performs factuality judgments on input text and provides structured feedback that can be exploited by the attacker.
    - **Mechanism**: Upon receiving the text to be detected, the detector first retrieves relevant real-time evidence documents via a search engine (e.g., Google Search API). Then, an LLM (e.g., GPT-4o) synthesizes the input text and the retrieved evidence to make a judgment, outputting three parts: (a) a true/false judgment; (b) a summary of key evidence supporting the judgment; (c) specific reasons identifying the falsehood (e.g., "the claim X contradicts evidence Y"). This structured feedback is not only used for detection but also provides a precise optimization direction for adversarial rewriting.
    - **Design Motivation**: Natural language feedback contains much more information than binary labels—it tells the attacker "where you were exposed," making adversarial rewriting targeted.

2. **Feedback-based Iterative Adversarial Rewriting**:

    - **Function**: Progressively optimizes fake news text based on the detector's feedback to make it increasingly difficult to detect.
    - **Mechanism**: The rewriter (also driven by an LLM) receives feedback from the detector, analyzes the "flaws" pointed out within, and modifies the text targetedly. For example, if the feedback states that "the claimed time contradicts the evidence," the rewriter will adjust the time details to make it harder to disprove. Each round of rewriting makes only the minimum necessary modifications to maintain the overall coherence and journalistic style of the text. The iterative process continues until the detector judges it as "true" or the maximum number of rounds is reached (set to 5 rounds in experiments). The final products are a series of fake news variants with different difficulty levels.
    - **Design Motivation**: Single-round rewriting is often not refined enough, and many flaws require multi-round iterations to fix. Progressive rewriting also generates evaluation data with different difficulty gradients.

3. **Real-time News Collection and Diversity Assurance**:

    - **Function**: Ensures that evaluation data is based on real current events to avoid pre-training data leakage.
    - **Mechanism**: Continually crawls newly published news articles (within 24-48 hours) from multiple news sources. To ensure diversity, it covers various categories like politics, economics, technology, and sports, while sampling from different sources (mainstream media, local news, specialized media). Each piece of news is rewritten into fake news immediately after collection, ensuring the rewriting timestamp is earlier than any possible model updates.
    - **Design Motivation**: Only when based on completely fresh events can the possibility of "the model having seen this news" be ruled out, truly testing reasoning rather than memory capability.

### Loss & Training

Ours is an evaluation framework and does *not* involve model training. All components (detector, rewriter) are based on the reasoning capabilities of existing LLMs (GPT-4o, Claude-3, etc.), implemented via prompt engineering. Key hyperparameters include the maximum number of iterations (5), the number of retrieved documents (top-10), and the rewriting amplitude control.

## Key Experimental Results

### Main Results

| Detector | Traditional Dataset ROC-AUC | No-rewrite News ROC-AUC | 1-round Rewrite ROC-AUC | Iterative Rewrite ROC-AUC | Gain |
|--------|-------------------|------------------|-----------------|-----------------|---------|
| GPT-4o (RAG) | 94.2 | 91.5 | 82.3 | **74.0** | **-17.5** |
| GPT-4o (No RAG) | 88.6 | 72.1 | 65.8 | 58.4 | -30.2 |
| Claude-3 (RAG) | 92.8 | 89.7 | 80.1 | 72.6 | -20.2 |
| Llama3-70B (RAG) | 87.3 | 83.2 | 73.5 | 66.8 | -20.5 |
| Llama3-70B (No RAG) | 79.5 | 61.4 | 53.2 | 48.7 | -30.8 |

### RAG vs No RAG Detection Comparison

| Evaluation Scenario | RAG Detector Acc | No RAG Detector Acc | Gap |
|---------|---------------|------------------|------|
| Traditional Benchmark (Old Data) | 91.2 | 85.4 | +5.8 |
| Real-time News (No Rewrite) | 87.3 | 68.5 | +18.8 |
| Real-time News (Adversarial Rewrite) | 72.1 | 52.3 | +19.8 |
| Unseen Event News | 79.6 | 54.1 | +25.5 |

### Key Findings

- **Traditional datasets severely overestimate detector capability**: The GPT-4o RAG detector achieves 94.2% ROC-AUC on traditional datasets, but only 74.0% when facing adversarially rewritten real-time news.
- **RAG is a critical capability when facing new events**: The No RAG detector performs much worse on real-time news compared to the RAG detector (a gap of 18-25%), confirming the necessity of retrieved evidence for timely factual judgments.
- **Iterative rewriting is significantly more effective than single-round rewriting**: Transitioning from single-round to iterative rewriting additionally reduces ROC-AUC by about 8-10%, proving the value of multi-round adversarial feedback.
- **All non-RAG detectors perform near random guessing on unseen events** (~50% accuracy), indicating that they essentially rely on pre-training memory rather than reasoning.
- GPT-4o RAG is the most robust detector, but still experiences a significant performance drop under targeted adversarial attacks.

## Highlights & Insights

- **Reveals the industry-wide blind spot of "data leakage"**: Seemingly year-on-year improvements in detection accuracy may just be because old misinformation has been integrated into training data, rather than any real improvement in model capability. This finding serves as an important warning to the entire factuality assessment field.
- **"Using detector feedback to attack the detector" is an ingenious adversarial design**: The rejection reasons given by the detector provide exactly the direction of improvement for the attacker, forming a natural "arms race" style evaluation. This concept can be transferred to any "detection-evasion" adversarial scenario.
- **The dual value of RAG**: RAG is not only a necessary tool for detection, but its feedback is also key to generating high-quality adversarial samples—a single component serving both the offense and defense sides.

## Limitations & Future Work

- The quality of adversarial rewriting depends on the capability of the helper LLM used for rewriting, which will also change as stronger LLMs emerge.
- Experiments are mainly based on English news, and the effectiveness of misinformation detection in other languages and cultural contexts remains unknown.
- The pipeline requires real-time search engine APIs, making deployment costs relatively high.
- It has not explored how to use adversarial data to inversely enhance the detector—currently, it is only used for evaluation, not training.
- Adversarial rewriting could be maliciously exploited to generate more realistic fake news at scale, posing ethical risks (although discussed in the paper, concrete preventive measures are lacking).

## Related Work & Insights

- **vs LIAR/PolitiFact benchmarks**: These traditional benchmarks use historical claims, whereas this paper directly points out their data leakage issues, which makes them unsuitable for evaluating detection capabilities on future events.
- **vs FActScore**: FActScore decomposes generated text into atomic facts for validation one by one. This work focuses on news article-level truthfulness judgment, which is closer to practical scenarios but coarser in granularity.
- **vs Adversarial NLI/Adversarial QA**: These studies use human adversarial annotation on NLU tasks, whereas this paper automates this process through LLM adversarial feedback, which offers much stronger scalability.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Revealing the data leakage problem and proposing the adversarial feedback loop are highly novel in both problem definition and method design.
- Experimental Thoroughness: ⭐⭐⭐⭐ It covers multiple detectors, RAG/no-RAG comparisons, and iteration round analysis, but lacks human evaluations.
- Writing Quality: ⭐⭐⭐⭐ The motivation is powerfully explained, and the logical chain is clear.
- Value: ⭐⭐⭐⭐⭐ It holds important insights for the research direction of factuality detection, and the revealed evaluation bias is worthy of attention from the entire community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reinforcement Learning with Backtracking Feedback](../../NeurIPS2025/llm_safety/reinforcement_learning_with_backtracking_feedback.md)
- [\[ACL 2026\] CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge](../../ACL2026/llm_safety/curate_continual_unlearning_in_real_time_with_ensured_preservation_of_llm_knowle.md)
- [\[ACL 2025\] Ewe: Improving Factuality with Explicit Working Memory](improving_factuality_with_explicit_working_memory.md)
- [\[ACL 2025\] Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models](chinese_simpleqa_a_chinese_factuality_evaluation_for_large_language_models.md)
- [\[ACL 2025\] Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs](are_the_hidden_states_hiding_something_testing_the_limits_of_factuality-encoding.md)

</div>

<!-- RELATED:END -->
