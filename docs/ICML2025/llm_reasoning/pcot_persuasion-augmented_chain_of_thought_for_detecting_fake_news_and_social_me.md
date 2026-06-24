---
title: >-
  [Paper Note] PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation
description: >-
  [ICML 2025][Reasoning][Disinformation Detection] Proposes PCoT (Persuasion-Augmented Chain of Thought), which utilizes a two-stage reasoning framework—first requiring the LLM to identify persuasive strategies in the text, and then injecting the persuasion analysis results into the disinformation detection reasoning. In zero-shot settings, it achieves an average F1 improvement of approximately 15% across 5 LLMs and 5 datasets.
tags:
  - "ICML 2025"
  - "Reasoning"
  - "Disinformation Detection"
  - "Chain of Thought"
  - "Persuasive Techniques"
  - "Zero-Shot Classification"
  - "Prompt Engineering"
date: 2026-05-08
content_hash: ca92902b0530f6d4
---

# PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation

**Conference**: ICML 2025  
**arXiv**: [2506.06842](https://arxiv.org/abs/2506.06842)  
**Code**: [Yes](https://github.com/ArkadiusDS/PCoT)  
**Area**: LLM Reasoning  
**Keywords**: Disinformation Detection, Chain of Thought, Persuasive Techniques, Zero-Shot Classification, Prompt Engineering

## TL;DR

Proposes PCoT (Persuasion-Augmented Chain of Thought), which utilizes a two-stage reasoning framework—first requiring the LLM to identify persuasive strategies in the text, and then injecting the persuasion analysis results into the disinformation detection reasoning. In zero-shot settings, it achieves an average F1 improvement of approximately 15% across 5 LLMs and 5 datasets.

## Background & Motivation

1. **Background**: Disinformation is widely spread on social media and news platforms, threatening democracy and public trust. Traditional detection methods rely on supervised learning and human-annotated data, which suffer from high annotation costs and poor generalization. Recently, zero-shot detection using LLMs such as GPT-4 has been proven to outperform supervised models like BERT.
2. **Limitations of Prior Work**: Although zero-shot methods do not require annotated data, when directly using LLMs for binary classification ("is/is not disinformation"), LLMs lack the ability to systematically analyze **manipulation and persuasive techniques** within the text. This limits detection accuracy, yielding poor performance particularly on long texts (news articles) and content containing complex rhetorical devices.
3. **Key Challenge**: Psychological research indicates that humans can better distinguish between real and fake news if they learn to identify persuasive fallacies. However, existing zero-shot LLM schemes do not leverage this cognitive mechanism—LLMs "possess" knowledge of persuasive techniques but are not guided to employ it.
4. **Goal**: How to systematically integrate persuasion knowledge into the reasoning process of LLMs to enhance zero-shot disinformation detection?
5. **Key Insight**: Inspired by psychological findings—namely, that understanding persuasion strategies helps humans identify disinformation—this work injects persuasion knowledge into the Chain of Thought reasoning chain of LLMs.
6. **Core Idea**: Use a two-stage CoT—first analyzing the persuasive strategies and explanations in the text, and then using this as an augmented context to make the disinformation decision—thereby achieving persuasion-knowledge-augmented zero-shot detection.

## Method

### Overall Architecture

PCoT is a **two-stage** zero-shot reasoning framework:

- **Input**: A raw text $T$ of a news article or a social media post.
- **Stage 1 (Persuasion Detection Step)**: The LLM receives the text $T$, persona $I_P$, persuasion knowledge $K_P$ (containing definitions of 6 high-level strategies and their underlying techniques), and task guidelines $G_P$, and outputs binary labels and explanations for each persuasive strategy.
- **Stage 2 (Disinformation Detection Step)**: The LLM receives the text $T$, persona $I_D$, the persuasion analysis output $A_T$ from the first stage, and task guidelines $G_D$, and outputs the binary classification result for disinformation.
- **Output**: $Y_T \in \{\text{Yes}, \text{No}\}$—indicating whether the text is disinformation.

The key idea is that the first stage forces the LLM to "think slow" first—analyzing step-by-step whether various persuasion strategies are utilized in the text and providing justifications. These intermediate reasoning products are then injected into the second stage, assisting the LLM in making a more accurate final judgment.

### Key Designs

#### 1. **Persuasion Knowledge Infusion**

- **Function**: Inject a comprehensive taxonomy of persuasive strategies into the prompt of the first stage, containing 6 broad categories of high-level strategies and the definitions of their specific underlying techniques.
- **Mechanism**: Adopt the persuasive technology taxonomy proposed by Piskorski et al. (2023) (developed by the Joint Research Centre of the European Commission), categorized into 6 major strategies:
    - **Attack on reputation [AR]**: Attacking the reputation/credibility of the opponent rather than discussing the topic itself.
    - **Justification [J]**: Supporting claims with explanations or appeals.
    - **Simplification [S]**: Oversimplifying causal relationships or choices.
    - **Distraction [D]**: Diverting attention away from the core argument.
    - **Call [C]**: Calling for specific actions or ways of thinking.
    - **Manipulative wording [MW]**: Using emotionally charged, exaggerated, or misleading phrasing.
- **Design Motivation**: Simply listing strategy names (Base MT) yields poor results. Injecting detailed definitions and sub-technique descriptions (DMT) enables the LLM to identify persuasive strategies more accurately. Experiments demonstrate that DMT improves performance by 9% over Base MT (F1 micro 0.722 vs 0.664).

#### 2. **Stage 1: Detailed Multitask (DMT) Persuasive Strategy Detection**

- **Function**: Use a single prompt to concurrently detect all 6 persuasive strategies and generate explanations for each strategy.
- **Mechanism**: Formally denoted as $A_T = \{p_i: (y_{p_i}, E_{p_i}) \mid p_i \in P\}$, where $y_{p_i}$ is the binary label for strategy $p_i$, and $E_{p_i}$ is the explanation generated by the LLM. The model generation process is formulated as $A_T \sim M(T, I_P, K_P, G_P)$.
- **Design Motivation**: Three prompt variants were tested—(1) DMT (single prompt multi-task, with detailed knowledge), (2) DTAT (independent prompt per strategy), and (3) Base MT (single prompt without knowledge infusion). DMT performs the best because the joint analysis of multiple strategies can exploit inter-strategy correlations; the introduction of explanations enhances final prediction robustness.

#### 3. **Stage 2: Persuasion-Augmented Disinformation Detection**

- **Function**: Treat the persuasion analysis results generated in the first stage as extra context to augment the reasoning for disinformation detection.
- **Mechanism**: $Y_T \sim M(T, I_D, A_T, G_D)$, where $A_T$ contains the labels and explanations for each strategy, based on which the LLM makes the final binary classification decision.
- **Design Motivation**: The advantage of the two-stage scheme lies in breaking down complex reasoning into manageable sub-tasks. Compared to the single-step scheme (performing both persuasion analysis and disinformation detection in a single prompt), the two-stage scheme improves the average F1 by 7% (0.815 vs 0.765). Although the single-step scheme outperforms the baseline (+8%), it does not utilize intermediate reasoning as effectively as the two-stage approach.

#### 4. **Prompt Adaptation Strategy**

- **Function**: Combine PCoT's persuasion-augmentation idea with three existing competitive prompting methods.
- **Mechanism**: Select the three best-performing zero-shot methods from the evaluation by Lucas et al. (2023) as baselines—VaN (vanilla prompt), Z-CoT (zero-shot CoT with "step by step"), and DeF-SpeC (emphasizing contextual and abductive reasoning)—and then append PCoT's persuasion analysis stage to each.
- **Design Motivation**: Verify whether the improvements of PCoT are consistent across different prompt styles, rather than being effective only under a specific prompt. Experiments show that all three baselines achieve significant improvements, demonstrating that PCoT is a general enhancement framework.

### Loss & Training

PCoT is an **inference-only method** that does not involve any training or fine-tuning. All experiments are conducted in a zero-shot setting, with the temperature set to 0 to obtain the most deterministic outputs. The evaluation metric is F1 score, and statistical significance is verified via McNemar's test ($p < 0.01$).

## Key Experimental Results

### Main Results

Overall F1 comparison (Overall Average) across 5 LLMs, 3 prompting methods, and 5 datasets:

| Model | VaN Base → PCoT | Z-CoT Base → PCoT | DeF-SpeC Base → PCoT |
|------|-----------------|--------------------|-----------------------|
| GPT 4o Mini | 0.759 → **0.845** (+11%) | 0.765 → **0.846** (+11%) | 0.772 → **0.834** (+8%) |
| Gemini 1.5 Flash | 0.681 → **0.810** (+19%) | 0.689 → **0.808** (+17%) | 0.744 → **0.834** (+12%) |
| Claude 3 Haiku | 0.710 → **0.797** (+12%) | 0.588 → **0.774** (+32%) | 0.780 → **0.795** (+2%) |
| Llama 3.3 70B | 0.740 → **0.845** (+14%) | 0.722 → **0.843** (+17%) | 0.732 → **0.832** (+14%) |
| Llama 3.1 8B | 0.627 → **0.792** (+26%) | 0.660 → **0.791** (+20%) | 0.697 → **0.773** (+11%) |
| **Average** | 0.711 → **0.815** (+15%) | — | — |

PCoT vs other prompting methods (Overall F1):

| Model | Z-CoT | RaR | CoVe | **PCoT** |
|------|-------|-----|------|----------|
| GPT 4o Mini | 0.765 | 0.698 | 0.790 | **0.846** |
| Gemini 1.5 Flash | 0.689 | 0.573 | 0.736 | **0.808** |
| Claude 3 Haiku | 0.588 | 0.768 | 0.441 | **0.774** |
| Llama 3.3 70B | 0.722 | 0.657 | 0.835 | **0.843** |
| Llama 3.1 8B | 0.660 | 0.566 | 0.764 | **0.791** |

### Ablation Study

| Configuration | F1 (Average) | Description |
|------|-----------|------|
| PCoT (Two-stage, with persuasion knowledge) | **0.815** ±0.027 | Full method |
| PCoT Single Step (Single-stage) | 0.765 ±0.072 | Single prompt for persuasion + detection, +8% over Base |
| PCoT Base Version (No strategy details) | ~0.791 | Simple definitions of persuasion without the 6 strategies |
| Base (No persuasion augmentation) | 0.711 ±0.055 | Vanilla prompt without PCoT |
| DMT (Stage 1 persuasion detection) | 0.722 F1-micro | Multi-task persuasion detection with knowledge infusion |
| DTAT (Independent strategy detection) | 0.689 F1-micro| 6 separate independent prompts |
| Base MT (No knowledge infusion) | 0.664 F1-micro | Strategy names only |

PCoT vs reasoning models (o1-mini / o3-mini):

| Model | Overall F1 |
|------|-----------|
| GPT 4o Mini + PCoT | **0.846** |
| Llama 3.1 8B + PCoT | 0.791 |
| o3-mini | 0.770 |
| o1-mini | 0.634 |

### Key Findings

- **PCoT benefits smaller models the most**: Llama 3.1 8B achieves the largest average improvement of 18%, indicating that persuasion knowledge infusion provides more pronounced compensation when model capacity is relatively weak.
- **Longer texts benefit more**: PCoT improves by 18% on news articles and 8% on social media posts—persuasive strategies are richer in long texts, making PCoT's analysis more helpful.
- **Disparate distribution of persuasive strategies**: 92% of disinformation texts contain at least one persuasive strategy, compared to only 72% for credible texts. Four strategies—Attack on reputation, Simplification, Distraction, and Manipulative wording—are highly correlated with disinformation; Justification and Call appear with similar frequencies in both real and fake news.
- **Even the "lightweight" version of PCoT** (without strategy details, using only the general definition of persuasion) outperforms the baseline, showing that the cognitive path of "guiding LLM to think about persuasion" itself is valuable.
- **PCoT-augmented smaller models outperform reasoning models**: Llama 3.1 8B + PCoT (0.791) outperforms o3-mini (0.770) and o1-mini (0.634), suggesting that structured domain knowledge infusion is more effective than pure reasoning capabilities.

## Highlights & Insights

- **Knowledge transfer from psychology to AI**: Guiding prompt design with psychological findings ("understanding persuasive techniques helps identify disinformation") is an effective paradigm borrowed from cognitive science. This interdisciplinary approach can be transferred to other NLP tasks requiring analysis of rhetoric or argumentative quality.
- **Two-stage > Single-stage**: Decomposing complex reasoning into "analyzing intermediate signals first, then making the final judgment" is more effective than a single step. This aligns with the philosophy of CoT, but PCoT's novelty lies in **specifying what the CoT should contemplate**—not just a generic "step-by-step," but a structured analysis guided by domain knowledge.
- **Marginal effects of knowledge infusion**: From Base MT (0.664) → DTAT (0.689) → DMT (0.722), it is apparent that the more detailed the injected knowledge and the more multi-faceted the analysis, the better the results. However, even a generic concept of "persuasion" yields improvements, showing that the direction itself is correct.
- **Dataset contribution**: Releases MultiDis (expert-annotated, with three rounds of annotations) and EUDisinfo (based on the EU Disinfo database), both containing content from after 2024 to ensure they are not part of the LLM training sets, providing high-quality evaluation resources for the disinformation domain.

## Limitations & Future Work

- **English only**: All datasets and experiments only cover English; multi-lingual scenarios are not validated. Persuasive techniques might manifest differently across various languages or cultures.
- **Fixed persuasive strategies**: The 6 tactics come from a fixed taxonomy and are not dynamically selected. Different topics/domains might require different subsets of strategies. The authors also highlight dynamic strategy selection as a future direction.
- **Two API call overheads**: PCoT requires two-stage reasoning, doubling the inference cost compared to the baseline. This could be a bottleneck for large-scale, real-time detection scenarios.
- **Lack of verification with external knowledge**: PCoT relies entirely on rhetoric/persuasion analysis at the textual level and does not perform fact-checking. It performs less effectively on "bland disinformation" without persuasive techniques (improving by only 7% on the subset without persuasion).
- **Future Work**: (1) Combine with Retrieval-Augmented Generation (RAG) to implement dual-channel detection that couples fact-checking with persuasion analysis; (2) Dynamic strategy selection—automatically filtering the most relevant subset of persuasive techniques based on the text topic; (3) Multimodal expansion—applying PCoT to disinformation detection involving images or videos.

## Related Work & Insights

- **vs Zero-Shot Baselines from Lucas et al. (2023)**: Lucas et al. systematically evaluated multiple zero-shot prompt methods (VaN, Z-CoT, DeF-SpeC), and PCoT achieves significant improvements over all of their best-performing methods. The key difference is that PCoT introduces intermediate reasoning steps guided by domain knowledge.
- **vs Chain-of-Verification (CoVe)**: CoVe lets LLMs self-verify their reasoning processes, which is unstable and underperforms even the baseline on Claude (0.441). The advantage of PCoT is providing **structured reasoning anchors** through persuasion analysis rather than generalized self-verification.
- **vs OpenAI Reasoning Models (o1/o3-mini)**: PCoT-augmented standard models (even 8B) outperform specialized reasoning models, showing that structured domain knowledge infusion is more effective than general reasoning capability enhancement for specific tasks.
- **Relation to Kamali et al. (2022)**: That study first introduced persuasion as intermediate labels for health-domain disinformation detection in few-shot scenarios, but was confined to a specific domain and few-shot settings. PCoT generalizes this approach into a zero-shot, multi-domain, multi-model general framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The core idea is clean and psychologically substantiated; two-stage persuasion-augmented CoT is novel in disinformation detection; however, technically it remains within prompt engineering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 5 LLMs × 5 datasets × 3 prompting methods, incorporating new datasets, ablations, comparison against reasoning models, and statistical significance tests; extremely robust.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete motivational chain, standard experimental explanations; the dataset section is slightly redundant but highly informative.
- **Value**: ⭐⭐⭐⭐ — Proposes a simple and effective general framework that any LLM can use plug-and-play; the two new datasets offer actual value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness](towards_better_chain-of-thought_a_reflection_on_effectiveness_and_faithfulness.md)
- [\[ICML 2025\] Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models](emergent_symbolic_mechanisms_support_abstract_reasoning_in_large_language_models.md)
- [\[ICML 2025\] DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination](dynamic_benchmarking_of_reasoning_capabilities_in_code_large_language_models_und.md)
- [\[ICML 2025\] Self-Consistency Preference Optimization](self-consistency_preference_optimization.md)
- [\[ICML 2025\] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game](improving_rationality_in_the_reasoning_process_of_language_models_through_self-p.md)

</div>

<!-- RELATED:END -->
