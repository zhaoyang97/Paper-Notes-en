---
title: >-
  [Paper Note] Not All Explanations for Deep Learning Phenomena Are Equally Valuable
description: >-
  [ICML 2025 Oral][Recommender Systems][Deep Learning Phenomena] This is a position paper arguing that "counter-intuitive phenomena" in deep learning (such as double descent, grokking, and the lottery ticket hypothesis) rarely occur in practical settings. Instead of pursuing isolated explanations for these phenomena, researchers should treat them as empirical testbeds to evaluate and refine broader deep learning theories.
tags:
  - "ICML 2025 Oral"
  - "Recommender Systems"
  - "Deep Learning Phenomena"
  - "Scientific Methodology"
  - "Double Descent"
  - "Grokking"
  - "Lottery Ticket Hypothesis"
date: 2026-05-08
content_hash: ea67c60f9c3d22e0
---

# Not All Explanations for Deep Learning Phenomena Are Equally Valuable

**Conference**: ICML 2025 Oral  
**arXiv**: [2506.23286](https://arxiv.org/abs/2506.23286)  
**Code**: None  
**Area**: Recommender Systems  
**Keywords**: Deep Learning Phenomena, Scientific Methodology, Double Descent, Grokking, Lottery Ticket Hypothesis

## TL;DR

This is a position paper arguing that "counter-intuitive phenomena" in deep learning (such as double descent, grokking, and the lottery ticket hypothesis) rarely occur in practical settings. Instead of pursuing isolated explanations for these phenomena, researchers should treat them as empirical testbeds to evaluate and refine broader deep learning theories.

## Background & Motivation

In recent years, the field of deep learning has witnessed a series of surprising empirical phenomena, including double descent, grokking, and the lottery ticket hypothesis. These phenomena seemingly contradict our traditional understanding of neural network behaviors and have thus garnered significant research attention. Collectively, the three original papers have accumulated over 7,200 citations, with hundreds of papers referencing them annually at top-tier conferences.

However, the authors highlight a **Key Challenge**: **these phenomena rarely occur in practical deep learning applications**. Double descent disappears when proper regularization is applied; grokking is observed only on small algorithmic datasets and fails to replicate in large-scale tasks; the lottery ticket hypothesis, while theoretically valid, cannot be efficiently identified prior to training. This implies that a massive amount of research resources is devoted to isolated explanations of "edge cases," which often result in **narrow ad hoc hypotheses** that lack generalizability.

The **Core Idea** of this paper is that the true value of deep learning phenomena lies not in "solving" or "explaining" them in isolation, but in using them as **extreme testbeds** to refine our **broad explanatory theories** regarding core deep learning principles (e.g., generalization, optimization, and sparsity).

## Method

### Overall Architecture

This work is a methodological position paper rather than a proposal for a new algorithm. Its core argumentative framework is structured into three levels:

1. **Empirical Analysis**: Scrutinizing the practical relevance of double descent, grokking, and the lottery ticket hypothesis.
2. **Theoretical Distinction**: Distinguishing between two research paradigms: "narrow ad hoc hypotheses" and "broad explanatory theories".
3. **Practical Recommendations**: Formulating concrete action guidelines for future research.

### Key Designs

1. **Distinction between "Narrow Ad Hoc Hypotheses" vs. "Broad Explanatory Theories"**: The authors construct a deliberately absurd counterexample to illustrate this distinction—using the "number of prime numbers among network parameters" to "explain" double descent and grokking. While this "theory" indeed tracks test performance empirically, it clearly lacks any generalizable value. This demonstrates that **an explanation that is accurate for a specific phenomenon is not necessarily useful for the broader field**. The authors argue that genuinely valuable research should leverage these phenomena to revise or test our understanding of core concepts such as the bias-variance tradeoff, optimization dynamics, and model sparsity.

2. **Sociotechnical Pragmatism Framework**: The authors introduce the framework of Watson et al. (2024), suggesting that the value of deep learning research should be measured by its **downstream impact**—where "impact" encompasses both technical advancement and societal considerations. Under this framework, the value of knowledge depends on its utility; "a theory with no practical impact is merely a formal exercise." The authors find that 96% of ML papers claim performance and generalization as their goals, with over 50% explicitly expressing concern for real-world applications.

3. **Advocacy for Scientific Methodology**: The authors argue that research on deep learning phenomena should adhere more rigorously to the scientific method, including hypothesis-driven research, reporting of negative results, falsifiability, preregistration, and meta-studies/replication. These practices have long-standing traditions in the natural sciences but remain underutilized in deep learning.

### 三大现象的具体分析

| Phenomenon | Practical Irrelevance | Broad Theoretical Value |
|------|-------------|-------------|
| Double Descent | Disappears with proper regularization; does not appear in LLM/ViT scaling analysis | Driven the re-examination and understanding of bias-variance tradeoff, benign overfitting, and memorization |
| Grokking | Confined to small algorithmic datasets; effects diminish on large datasets; can be induced by artificially boosting initialization | Propelled research into learning dynamics, lazy-to-feature learning, and numerical instability of Softmax |
| Lottery Ticket | Cannot be efficiently identified prior to training; sparsity benefits are difficult to realize on modern hardware | Influenced the understanding of pruning, quantization, and parameter-efficient fine-tuning |

## Key Experimental Results

### Main Results

As a position paper, this work does not include traditional experiments. However, it provides the following quantitative evidence:

| Metric | Data |
|------|------|
| Total citations of the three original papers | 7,272 (as of June 2025) |
| Related papers in NeurIPS 2024 main track | 149 |
| Related papers in ICML 2024 main track | 132 |
| Related papers in ICLR 2024 main track | 108 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| "Prime Parameter" Hypothesis | Highly correlated with test performance | Demonstrates that "accurate explanation" does not equate to "utility" |
| Double descent + Regularization | Phenomenon disappears | Indicates it does not surface in practical training |
| Grokking + Large Datasets | Significant reduction in effect | Shows it is restricted to edge cases |

### Key Findings

- Deep learning phenomena do not pose substantial challenges in practical applications and should not be studied under a "problem-solution" paradigm.
- There is a plethora of research offering "narrow ad hoc hypotheses," which, although correct under specific settings, contribute marginally to advancing the field.
- The true value of these phenomena lies in providing extreme settings with low computational costs and low academic entry barriers, which can be utilized to test and refine our understanding of core deep learning principles.

## Highlights & Insights

- The formulation of the **"prime parameter" counterexample** is highly creative; a seemingly absurd yet empirically "valid" theory intuitively visualizes the gap between an "accurate explanation" and a "useful explanation".
- Research on deep learning phenomena possesses unique advantages: low computational resource demands, low entry barriers, alignment with scientific exploration rather than chasing SOTA, and serving as an excellent testing ground for the intersection of theory and experimentation.
- It provides reflective value for the research orientation of the entire community: when a phenomenon is not an actual problem, striving to "solve" it might be exhausting efforts in the wrong direction.

## Limitations & Future Work

- As a position paper, the evaluation of its core arguments is largely subjective—how to quantify the "broad theoretical value of an explanation" remains ambiguous.
- Although the analyses of the three phenomena are representative, they are not fully comprehensive (e.g., newer phenomena such as neural scaling laws and emergence are not discussed).
- The paper acknowledges the difficulty in assessing which theories have "expected utility," yet the actionable guidelines provided remain somewhat abstract.
- The possibility that "edge phenomena" might become "no longer marginal" in the future with the advent of new architectures or tasks is under-discussed.

## Related Work & Insights

- Aligns with the critical analysis perspective of Schaeffer et al. (2024) regarding LLM "emergent abilities".
- Complements the call by Karl et al. (2024) for reporting negative results in deep learning research.
- Inspires reflection: In application fields like recommender systems, does research on many "counter-intuitive phenomena" suffer from similar issues—explaining behavior that only occurs in extreme settings while failing to contribute to real-world system improvements?

## Rating

- Novelty: ⭐⭐⭐⭐ Clear and insightful perspectives, though the position paper itself does not propose a new methodology.
- Experimental Thoroughness: ⭐⭐⭐ As a position paper, quantitative evidence is primarily bibliometric, though the "prime parameter" counterexample is cleverly designed.
- Writing Quality: ⭐⭐⭐⭐⭐ The argumentation is logically rigorous, progressive, and features clear illustrations.
- Value: ⭐⭐⭐⭐ Offers important reflective value for the research orientation of the community, though practical implementation remains distant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] LCRON: Learning Cascade Ranking as One Network](learning_cascade_ranking_as_one_network.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[ICLR 2026\] On the Mechanisms of Collaborative Learning in VAE Recommenders](../../ICLR2026/recommender/on_the_mechanisms_of_collaborative_learning_in_vae_recommenders.md)
- [\[ICML 2025\] SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [\[NeurIPS 2025\] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](../../NeurIPS2025/recommender/transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)

</div>

<!-- RELATED:END -->
