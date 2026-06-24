---
title: >-
  [Paper Note] Beyond English-Centric Training: How Reinforcement Learning Improves Cross-Lingual Reasoning in LLMs
description: >-
  [ICLR2026][Reasoning][Cross-lingual reasoning] Using Qwen2.5-3B-Base for controlled comparisons, the authors systematically demonstrate for the first time that RL (GRPO) possesses significantly stronger cross-lingual generalization for multilingual reasoning than SFT. Counter-intuitively, RL using **non-English** (German/Chinese) data outperforms English data. The study provides mechanistic explanations from three perspectives: "reasoning-time language inconsistency…
tags:
  - "ICLR2026"
  - "Reasoning"
  - "Cross-lingual reasoning"
  - "RL vs SFT"
  - "GRPO"
  - "Multilingual generalization"
  - "Language inconsistency"
date: 2026-05-08
content_hash: 61efdf2f0d6ed76f
---

# Beyond English-Centric Training: How Reinforcement Learning Improves Cross-Lingual Reasoning in LLMs

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=hdrG6SaTcA](https://openreview.net/forum?id=hdrG6SaTcA)  
**Code**: To be confirmed  
**Area**: LLM Reasoning / Cross-lingual Generalization / Reinforcement Learning  
**Keywords**: Cross-lingual reasoning, RL vs SFT, GRPO, Multilingual generalization, Language inconsistency

## TL;DR
Using Qwen2.5-3B-Base for controlled comparisons, the authors systematically demonstrate for the first time that RL (GRPO) possesses significantly stronger cross-lingual generalization for multilingual reasoning than SFT. Counter-intuitively, RL using **non-English** (German/Chinese) data outperforms English data. The study provides mechanistic explanations from three perspectives: "reasoning-time language inconsistency, sampling exploration, and semantic space drift."

## Background & Motivation
**Background**: The complex reasoning capabilities of large models have leaped forward through RL (especially methods like GRPO that reward final answer correctness). Multilingual reasoning requires models to understand semantics across languages and perform logical inference in various linguistic contexts. Most existing enhancements for multilingual reasoning remain at the prompting level (e.g., "translate-then-solve"), and few research works have investigated: **How do different training paradigms (SFT vs RL) themselves affect a model's intrinsic cross-lingual generalization capability?**

**Limitations of Prior Work**: SFT improves reasoning by imitating high-quality CoT trajectories. However, its essence is "memorizing and repeating given expert trajectories," making it prone to overfitting the **language and patterns** of the training data—reasoning skills learned in one language might not generalize to another. While RL has shown stronger cross-task generalization in English mathematical/logical reasoning, its performance and mechanisms in the **cross-lingual** dimension remained a gap.

**Key Challenge**: Most LLMs are pre-trained on English-centric corpora. Conventional intuition suggests that "RL with English data best exploits model potential." However, if RL learns a robust reasoning strategy **independent of specific languages**, the choice of training language might defy this intuition—this is the tension this paper aims to verify.

**Goal**: Decomposition into two verifiable sub-questions: (1) Given the same base model and training data, which paradigm—RL or SFT—exhibits stronger cross-lingual generalization? (2) Which language is optimal for training data—is it English? Furthermore, the study seeks to understand "why."

**Key Insight**: The authors fix the base model (Qwen2.5-3B-Base) and data sources (multilingual translations of GSM8K/LUFFY), varying only "training paradigm" and "training language" to conduct a clean controlled experiment. Three sets of mechanism probes are then used to explain the observed phenomena.

**Core Idea**: This is an **empirical study**. Its contribution lies not in proposing a new method, but in using rigorous controlled experiments to provide two counter-intuitive conclusions (RL $\gg$ SFT in cross-lingual generalization, and non-English RL $\gt$ English RL) and attributing the underlying mechanisms to "reasoning-time language inconsistency + sampling exploration + minimal semantic space drift."

## Method

### Overall Architecture
This paper is a research design of "training-evaluation-attribution." It consists of three layers: **First**, fixing the base model and data source, it compares SFT and RL (GRPO) paradigms across three training languages (En/Zh/De) to obtain accuracy across multilingual reasoning benchmarks. **Second**, a normalized generalization score $Gen$ is used to quantify "how much of the inherent potential in each language the model has extracted," leading to two core findings. **Third**, three sets of mechanism probes explain the reasons behind these findings.

Specifically, training data consists of multilingual versions of GSM8K (8K samples per language) and LUFFY (45K samples per language) translated by Qwen3-30B-A3B and verified by DeepSeek-V3. SFT uses full-parameter fine-tuning on LlamaFactory, and RL uses the verl platform for GRPO (answers wrapped in `\boxed{}`, rewards based on correctness). Evaluation covers four reasoning categories: Math (MGSM / MMath500 / MAIME2024), Commonsense (MMLU-ProX-Lite), Science (MGPQA-D), and Logic (Multilingual LogiQA), plus Instruction Following (M-ifEval), spanning 10 languages (En, Zh, De, Es, Fr, Ja, Ru, Th, Sw, Bn). Conclusions are replicated on SmolLM3-3B-Base and Qwen2.5-7B-Base to rule out model-specific accidents.

### Key Designs

**1. Controlled Contrast Design: Separating "Paradigm" and "Training Language"**

To answer "where RL outperforms SFT" and "which training language to choose," the biggest pitfall is confounding variables. This paper fixes all other factors: the same Qwen2.5-3B-Base, the same batch of quality-checked parallel corpora from GSM8K/LUFFY, the same 3 epochs, and full-parameter fine-tuning. The only variables are "SFT vs RL" and "Training Language (En/Zh/De)." This allows differences to be cleanly attributed. To ensure fairness in RL, GRPO uses unified hyperparameters: LR $1\times10^{-6}$, rollout batch 512, sampling temperature 1.0, and KL coefficient 0.001. This design makes the counter-intuitive conclusion "non-English RL is better" robust—since SFT shows almost no such variance across the same languages (Avg fluctuates between 46.3% and 47.6%, within statistical noise), ruling out "data quality differences" as an explanation.

**2. Gen Score: Normalizing by "Remaining Improvement Space"**

Base accuracy varies wildly across languages (English 63.4%, Bengali 1.2%). Looking directly at "absolute point improvement" would favor low-baseline languages. The authors define a generalization score:

$$\text{Gen}(M_{\text{tuned}}) = \frac{1}{|L|}\sum_{l\in L} \frac{\text{Acc}(M_{\text{tuned}}, l) - \text{Acc}(M_{\text{base}}, l)}{1 - \text{Acc}(M_{\text{base}}, l)}$$

The denominator $1-\text{Acc}(M_{\text{base}}, l)$ is the upper bound of "how much more can be gained," and the numerator is the actual gain. The ratio represents the percentage of remaining space filled, averaged across all evaluation languages. This metric provides a fair scale for comparisons like "RL(De) Gen=60.4 far exceeds SFT(De) 19.3."

**3. Mechanism Probe 1—Language Consistency: Constraints on Language to Observe Performance Decay**

The authors observed a key phenomenon: after RL with German data, the model **does not strictly use German** to solve German problems; it spontaneously switches to English or a mix of Zh/En/De for reasoning (Table 6 shows language consistency for RL(Zh) and RL(De) is 0.0%). They hypothesize: **this "language inconsistency" is the source of cross-lingual generalization**. Verified via: (a) prompting the model to use only one language; (b) adding a consistency reward $r_{\text{overall}} = 0.5\,r_{\text{acc}} + 0.5\,r_{\text{consistency}}$ (using langid to detect output language). Result: the more consistency is enforced, the worse the performance—RL(De) drops from 61.4% to 60.5% (prompt) and 52.0% (prompt+reward). This suggests locking the model into a single language cuts off its access to more robust multilingual reasoning modules established during pre-training. Notably, allowing the model to **freely choose** (pure RL) is better than "encouraging inconsistency," indicating that while inconsistency is key, "unconstrained exploration" is equally vital.

**4. Mechanism Probes 2 & 3—Sampling Exploration and Semantic Drift: Evidence from Training Dynamics and Geometry**

The second probe introduces RFT (Rejection Fine-Tuning) as an intermediate state: it samples multiple times from an RL model and fine-tunes on correct samples, making it more on-policy than SFT. Performance follows a clear hierarchy: SFT 46.3% $\rightarrow$ RFT 66.8% $\rightarrow$ RL 71.5%, proving "self-exploration of solution paths" is the key. RFT data fits the model distribution better, while full RL adds online, continuous sampling with positive and negative samples, surpassing pure imitation. Perplexity (PPL) and self-similarity (BLEU among sampled responses) explain "why German is better": German problems have the highest PPL (1.414) and lowest self-similarity (0.425), meaning the model faces **greater uncertainty** with German. This uncertainty forces the model to jump out of single-language constraints during RL exploration, inadvertently activating stronger cross-lingual generalization. The third probe looks at representation geometry: using PCA on final layer hidden states to calculate drift vectors $h_{\text{diff}} = h_{\text{RL}} - h_{\text{Base}}$. RL-De shows the most concentrated distribution (minimal deviation from base), while RL-En is more dispersed. The "smaller drift, stronger generalization" ranking matches the accuracy ranking. Language consistency constraints increase drift and decrease performance. Conclusion: Pre-training established a general multilingual reasoning structure; **smaller drift better preserves this structure**. Thus, RL's "language inconsistency" paradoxically enhances cross-lingual transfer by retaining pre-trained structures.

## Key Experimental Results

### Main Results
Comparison of Qwen2.5-3B-Base on MGSM (Avg = average accuracy of 10 languages, Gen = generalization score):

| Training Setup | En | Zh | De | Avg | Gen |
|----------|------|------|------|------|------|
| Base | 63.4 | 48.3 | 33.5 | 31.8 | 0.0 |
| SFT (En) | 64.7 | 54.5 | 50.7 | 45.2 | 18.1 |
| **RL (En)** | 85.8 | 72.1 | 70.8 | **62.7** | **49.1** |
| SFT (Zh) | 65.7 | 58.7 | 48.4 | 46.9 | 20.4 |
| **RL (Zh)** | 86.1 | 76.3 | 74.2 | **66.0** | **52.6** |
| SFT (De) | 63.9 | 54.2 | 57.5 | 46.3 | 19.3 |
| **RL (De)** | 91.0 | 77.6 | 80.5 | **71.5** | **60.4** |

Observations: ① Under the same training language, RL significantly outperforms SFT by +17.5 to +25.2 points (e.g., RL(De) 71.5 vs SFT(De) 46.3). ② **Non-English RL outperforms English RL**—RL(De) 71.5 > RL(En) 62.7, an +8.8 point advantage for German, while SFT results across training languages are nearly static (46.3~47.6). This contrast (variance in RL, none in SFT) rules out "data quality" as an explanation.

### Ablation Study

| Configuration | Key Metrics (MGSM Avg) | Description |
|------|------|------|
| RL (De) Full | 71.5 | Free language choice, strongest |
| RL (De) + Consistency prompt | 60.5 | Forced German, performance drop |
| RL (De) + Prompt + Reward | 52.0 | Further forced, sharper drop |
| SFT $\rightarrow$ RFT $\rightarrow$ RL (De) | 46.3 $\rightarrow$ 66.8 $\rightarrow$ 71.5 | Increasing exploration = rising performance |
| SFT + RL (De) Cold-start | 52.6 | Worse than direct RL (71.5) |
| RL (Mix of three languages) | 68.1 | Worse than single-language RL (De) (71.5) |

### Key Findings
- **Forcing language consistency kills generalization**: Locking the RL model into the training language significantly reduces accuracy (from 61.4 to 52.0). Unconstrained RL has 0.0% consistency with the training language—spontaneous cross-lingual mixed thinking is its strength.
- **Sampling exploration is the root of RL's superiority**: The hierarchy Base < SFT < RFT < RL shows that "on-policy sampling and online optimization" drives generalization; pure imitation (SFT) has the lowest ceiling.
- **Why German is best**: German problems present the highest uncertainty (PPL 1.414, self-similarity 0.425), forcing diverse exploration paths; meanwhile, RL-De has minimal representation drift, preserving universal pre-trained structures.
- **Cold-start is harmful**: Pre-SFT before RL (cold-start) is inferior to direct RL. The authors speculate SFT causes premature convergence to language-specific patterns, limiting RL exploration.
- **Robust across scales/models**: Findings replicated on SmolLM3-3B and Qwen2.5-7B, with RL(De) remaining superior.

## Highlights & Insights
- **"Language inconsistency is a feature, not a bug"**: Intuition suggests matching the question's language, but the negative ablation using consistency rewards proves that forcing consistency cuts off the model from stronger pre-trained reasoning circuits—an elegant "proving by reversing" experimental design.
- **The three-probe loop**: Language inconsistency (behavior), sampling exploration (dynamics), and minimal semantic drift (geometry) provide a convincing closed loop explaining why RL preserves pre-trained generalization.
- **Actionable insights**: For post-training small models in multilingual reasoning, prioritize RL over SFT. Do not default to English; pick a language with higher uncertainty/distance (like German). Avoid language consistency constraints and cold-start SFT.
- **Gen score utility**: The normalized $Gen$ score is suitable for any multilingual/multi-task evaluation where base accuracies vary widely, preventing low-baseline bias.

## Limitations & Future Work
- **Preliminary mechanisms**: The mechanistic analyses provide **correlational evidence** rather than strict causality; the relationship between drift and performance remains an observation.
- **Scale and model family**: Primary conclusions are based on 3B models. While replicated at 7B, whether this holds for much larger models or models with different pre-training ratios (especially non-English $\gt$ English) requires more validation.
- **"German is best" might be specific**: Attributing German's advantage to complexity/uncertainty is a plausible post-hoc narrative. It lacks a quantitative criterion to predict the optimal source language for a given base model beforehand.
- **Task scope**: Limited to reasoning tasks. Whether "language inconsistency dividends" exist for non-reasoning tasks (e.g., creative writing, dialogue) with soft rewards is unknown.
- **Potential improvements**: Use "source language uncertainty (PPL/self-similarity)" as a selection metric for training. Expore "controlled language mixing" rewards to maintain generalization while preventing total loss of control.

## Related Work & Insights
- **vs translate-then-solve (Qin et al. 2023 / Huang et al. 2023)**: Those methods improve **inference** via external prompting; this paper studies how **training paradigms** shape **intrinsic** cross-lingual capability, finding RL enables language-agnostic reasoning.
- **vs SFT-based reasoning distillation (MAmmoTH/MetaMath)**: Those rely on imitating expert trajectories. This paper notes that imitation is the bottleneck for cross-lingual generalization, as it traps the model in the training language's patterns.
- **vs English-centric RL (most RL-for-reasoning defaults to English)**: This is the first work to systematically prove non-English RL can be superior, challenging the English-centric default.
- **vs Cross-task generalization (Huan et al. 2025)**: This work extends the conclusion "RL generalizes better than SFT" from the cross-task dimension to the **cross-lingual** dimension, adding multilingual-specific mechanisms like language inconsistency.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic contrast of RL vs SFT for cross-lingual reasoning; "non-English RL is better" is a high-value counter-intuitive discovery.
- Experimental Thoroughness: ⭐⭐⭐⭐ 10 languages × multiple tasks × 3 base models + three mechanism probes. Minus points for preliminary mechanism analysis and focus on small/medium models.
- Writing Quality: ⭐⭐⭐⭐ Clear logic chain (Findings $\rightarrow$ Mechanisms); impressive "proof through negative ablation" design.
- Value: ⭐⭐⭐⭐ Provides actionable guidance for multilingual reasoning post-training: use RL, choose non-English data, and don't force consistency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning](curriculum_reinforcement_learning_from_easy_to_hard_tasks_improves_llm_reasoning.md)
- [\[ICLR 2026\] Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning](emergent_hierarchical_reasoning_in_llms_through_reinforcement_learning.md)
- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](../../NeurIPS2025/llm_reasoning/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[ICLR 2026\] NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning](nft_bridging_supervised_learning_and_reinforcement_learning_in_math_reasoning.md)
- [\[ICLR 2026\] Learning to Reason over Continuous Tokens with Reinforcement Learning (HyRea)](learning_to_reason_over_continuous_tokens_with_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
