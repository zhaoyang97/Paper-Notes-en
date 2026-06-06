---
title: >-
  [Paper Note] Preference Leakage: A Contamination Problem in LLM-as-a-judge
description: >-
  [ICLR2026][LLM Evaluation][LLM-as-a-Judge] This paper is the first to formally define and systematically investigate **Preference Leakage** in LLM-as-a-Judge — when the synthetic data generator $M_G$ and the judge $M_J$…
tags:
  - "ICLR2026"
  - "LLM Evaluation"
  - "LLM-as-a-Judge"
  - "Preference Leakage"
  - "Data Contamination"
  - "Evaluation Bias"
  - "Synthetic Data"
date: 2026-05-08
content_hash: f10d53fe525020a2
---

# Preference Leakage: A Contamination Problem in LLM-as-a-judge

**Conference**: ICLR2026
**arXiv**: [2502.01534](https://arxiv.org/abs/2502.01534)  
**Code**: [David-Li0406/Preference-Leakage](https://github.com/David-Li0406/Preference-Leakage)  
**Area**: LLM Evaluation
**Keywords**: LLM-as-a-Judge, Preference Leakage, Data Contamination, Evaluation Bias, Synthetic Data

## TL;DR

This paper is the first to formally define and systematically investigate **Preference Leakage** in LLM-as-a-Judge — when the synthetic data generator $M_G$ and the judge $M_J$ are related (same model / inheritance / same family), the judge exhibits systematic preference toward the "associated student model." Under the same-model scenario, PLS reaches 28.7% on Arena-Hard, and this bias is more subtle and harder to detect than egocentric bias.

---

## Background & Motivation

**LLM-as-a-Judge has become the dominant evaluation paradigm**: Traditional n-gram metrics (BLEU/ROUGE) fail to adequately evaluate open-ended long-form generation by LLMs, prompting the community to adopt powerful LLMs as judges. Leaderboards such as AlpacaEval 2.0 and Arena-Hard widely adopt this approach.

**Synthetic data training has become mainstream**: To improve training efficiency, researchers extensively use LLM-generated synthetic data to fine-tune student models (e.g., using GPT-4o-generated instruction data to train student models).

**Heavy overlap between data generators and evaluators**: Given the limited number of frontier models, the community frequently uses GPT-4 both as the data generator and as the judge. This overlap resembles data leakage in traditional machine learning, but occurs on the evaluation side and is far more covert.

**Known biases do not cover this problem**: Prior work has identified position bias, length bias, and egocentric bias in LLM-based evaluation; however, preference leakage is a novel, systematic contamination induced by the coupling of the data generation–evaluation pipeline, and has not been systematically studied.

**Detection is extremely difficult**: Most LLMs do not disclose their training data, and distillation relationships are hard to quantify, making preference leakage more difficult to detect than conventional data contamination.

**Core research questions**: The paper is organized around three RQs — (RQ1) Does preference leakage introduce systematic bias? (RQ2) How severe is preference leakage under different relationship types? (RQ3) What are the underlying mechanisms of preference leakage?

---

## Method

### Problem Formalization

Three types of entities are defined:

- **Data generator $M_G$**: Generates synthetic dataset $D_{syn}$ for training student models, with conditional distribution $P_{M_G}(y|x)$
- **Student model $M_S$**: Trained on $D_{syn}$, with output distribution $P_{M_S}(y|x)$
- **Judge model $M_J$**: Provides a scoring function $S_{M_J}(y|x)$

**Preference leakage** occurs when $M_G$ and $M_J$ are related: $M_J$ assigns inflated scores to $M_S$'s outputs — not because of higher quality, but because $M_S$ inherits spurious features (style, format, wording) from $M_G$, toward which $M_J$ has a natural preference:

$$\mathbb{E}_{x, y_S \sim P_{M_S}}[S_{M_J}(y_S|x) \mid M_G \sim_{rel} M_J] > \mathbb{E}_{x, y_S \sim P_{M_S}}[S_{M_{J'}}(y_S|x) \mid M_G \not\sim_{rel} M_{J'}]$$

### Three Relationship Types

| Type | Definition | Typical Scenario |
|------|------|---------|
| **Same Model** | $M_G \equiv M_J$ | GPT-4o generates data; GPT-4o serves as judge |
| **Inheritance** | $M_J \leftarrow \text{FineTune}(M_G, D)$ or vice versa | GPT-4o generates data → fine-tuned model serves as judge |
| **Same Family** | $M_G, M_J \in \text{Family}(A_X, D_X)$ | GPT-4o generates data; GPT-4-turbo serves as judge |

### Preference Leakage Score (PLS)

To quantify the degree of bias introduced by preference leakage, the Preference Leakage Score is defined as:

$$\text{PLS}(i,j) = \frac{\left(\frac{\text{WR}(i,i) - \text{AVG}(i,j)}{\text{AVG}(i,j)}\right) + \left(\frac{\text{WR}(j,j) - \text{AVG}(j,i)}{\text{AVG}(j,i)}\right)}{2}$$

where $\text{WR}(i,j)$ is the win rate assigned by judge $j$ to student model $i$, and $\text{AVG}(i,j) = \frac{\text{WR}(i,i) + \text{WR}(i,j)}{2}$. PLS > 0 indicates that the judge favors its associated student model; larger values indicate more severe bias.

### Experimental Design

- **Data generation**: 30,000 prompts sampled from UltraFeedback; responses generated by GPT-4o, Gemini-1.5-flash, and LLaMA-3.3-70B respectively
- **Student models**: Mistral-7B-v0.1 and Qwen-2.5-14B (both pretrained, not instruct versions, to avoid interference from existing distillation data)
- **Evaluation benchmarks**: Arena-Hard (500 questions) and AlpacaEval 2.0 (805 questions)
- **Comparison setup**: 3 generators × 2 student models × 3 judges × 2 benchmarks

---

## Key Experimental Results

### Main Results: Preference Leakage Is Pervasive (Table 1)

| Student Model | Generator/Judge Pair | Arena-Hard PLS | AlpacaEval PLS | Average |
|---------|-------------|---------------|---------------|------|
| Mistral-7B | GPT-4o & Gemini-1.5 | 28.7% | 18.4% | **23.6%** |
| Mistral-7B | GPT-4o & LLaMA-3.3 | -1.5% | 1.4% | -0.1% |
| Mistral-7B | LLaMA-3.3 & Gemini-1.5 | 13.1% | 19.8% | 16.4% |
| Qwen-14B | GPT-4o & Gemini-1.5 | 37.1% | 18.6% | **27.9%** |
| Qwen-14B | GPT-4o & LLaMA-3.3 | 1.0% | 2.3% | 1.7% |
| Qwen-14B | LLaMA-3.3 & Gemini-1.5 | 25.4% | 18.4% | 21.9% |

**Key finding**: The vast majority of model pairs exhibit significantly positive PLS, indicating that judges clearly favor their associated student models.

### Relationship Type Analysis (Table 2)

| Relationship Type | Arena-Hard | AlpacaEval 2.0 | Average |
|---------|-----------|---------------|------|
| Same Model | 28.7% | 18.4% | **23.6%** |
| Inheritance + Same Instructions | 17.8% | 20.7% | 19.3% |
| Inheritance + Different Instructions | 18.3% | 26.3% | 22.3% |
| Same Family + Same Series | 10.1% | 7.6% | 8.9% |
| Same Family + Different Series | 3.3% | 2.2% | 2.8% |

**Conclusion**: The severity of preference leakage is strongly positively correlated with the degree of relatedness: Same Model > Inheritance > Same Family (Same Series) > Same Family (Different Series).

### Comparison of Learning Methods (Table 3)

| Learning Method | Arena-Hard | AlpacaEval 2.0 | Average |
|---------|-----------|---------------|------|
| SFT | 28.7% | 18.4% | **23.6%** |
| DPO | 7.7% | 2.7% | 5.2% |
| ICL | -4.2% | -1.1% | -2.7% |

**Finding**: SFT exhibits the most severe leakage; DPO's pairwise optimization mechanism substantially reduces leakage; ICL does not update parameters and is thus largely unaffected.

### Spurious Feature Ablation (Table 6)

| Setting | GPT & Gemini | GPT & LLaMA | LLaMA & Gemini |
|------|-------------|-------------|---------------|
| Baseline | 17.5% | 2.3% | 18.8% |
| − Remove style | 9.0% | 3.3% | 14.6% |
| − Remove format | 9.8% | 1.9% | 14.5% |
| − Remove wording | 11.2% | 2.4% | 18.2% |

**Finding**: Style and format are the primary carriers of preference leakage; removing them leads to significant PLS reduction. Lexical-level substitution has limited effect, indicating that preference leakage is driven by surface stylistic features rather than semantic similarity.

### Mitigation Methods (Table 7)

| Method | Error Bias ↓ |
|------|-------------|
| Baseline | 17.8 |
| + Prompting | 18.3 |
| + Chain-of-Thought | 15.6 |
| + Paraphrase | 18.7 |
| + Auto Calibration | 20.7 |
| + Contextual Calibration | **7.3** |

**Finding**: Only Contextual Calibration (post-hoc calibration on a held-out set) effectively mitigates preference leakage, reducing Error Bias from 17.8 to 7.3. Simple prompting and paraphrase are largely ineffective.

### Other Key Findings

- **Smaller models suffer more severe leakage**: Smaller models such as LLaMA-3-1B and Qwen-2.5-3B exhibit higher PLS than larger models. The authors hypothesize that smaller models, having limited learning capacity, rely more heavily on repeatedly occurring surface features (format/style), which are precisely the carriers of preference leakage.
- **Judges cannot identify their associated student models**: All three judge models achieve near-random accuracy (~41–53%) on the task of identifying outputs generated by "their own" student model, demonstrating that preference leakage is an unconscious, implicit bias. However, a BERT classifier can distinguish outputs from different student models with 82.4% accuracy, confirming that synthetic data does embed detectable features.
- **Subjective tasks exhibit more severe leakage**: Open-ended subjective tasks such as coding and creative writing show substantially higher PLS than objective tasks with standard answers such as mathematics; subjective evaluation dimensions such as fairness yield higher PLS than objective dimensions such as completeness.
- **Linear correlation with data mixing ratio**: Even 10% synthetic data introduces measurable preference leakage, and PLS grows linearly with the proportion of synthetic data, with no apparent threshold effect.
- **Impact on real leaderboards**: On the AlpacaEval 2.0 leaderboard, ranking shifts attributable to preference leakage (average +1.33 ranks for the Vicuna series) exceed those due to egocentric bias (GPT-4 Preview +1.00 rank).

---

## Highlights & Insights

- **First formal definition**: The paper conceptualizes the coupling of data generation and evaluation in the LLM evaluation pipeline as "preference leakage," analogous to traditional data leakage but more covert
- **Systematic experimental design**: Three relationship types × three learning methods × multiple data mixing ratios × two benchmarks × multiple model scales, achieving broad coverage
- **In-depth mechanistic analysis**: Identification experiments demonstrate that leakage is implicit; spurious feature ablation pinpoints style and format as the primary carriers
- **PLS metric**: A standardized metric for quantifying preference leakage is proposed to facilitate future research
- **Practical recommendations**: The community is alerted to avoid using related models as both generator and judge in LLM-as-a-Judge pipelines

---

## Limitations & Future Work

1. **Preliminary mitigation methods**: Only five mitigation strategies are explored, of which only contextual calibration is effective; however, it requires an additional held-out dataset, limiting its practicality
2. **Limited coverage of real-world scenarios**: The main experiments are conducted under controlled SFT settings; complex real-world training pipelines (multi-round distillation, multi-source data mixing, RLHF, etc.) are not fully addressed
3. **Limited leaderboard analysis**: Only AlpacaEval and LMArena are analyzed; most leaderboards lack traceable distillation relationship metadata
4. **English only**: All experiments are conducted on English benchmarks; cross-lingual settings are not explored
5. **Coarse-grained relationship type definitions**: In practice, inter-model relationships are far more complex than the three defined types (e.g., indirect distillation chains, multi-hop inheritance)

---

## Related Work & Insights

- **LLM-as-a-Judge**: Zheng et al. (2023) pioneered the use of LLMs for automatic evaluation; the Prometheus series (Kim et al., 2023/2024) subsequently developed open-source judge models. Prior work has identified position bias and length bias, among others.
- **Egocentric Bias**: Koo et al. (2024) and Panickssery et al. (2024) find that LLM judges tend to favor their own generated content. Preference leakage generalizes this scenario — it does not require the judge and generator to be identical, only that they are "related."
- **Data Leakage/Contamination**: Deng et al. (2024) and others study the overlap between training data and evaluation sets. Preference leakage represents a new variant of data contamination manifesting on the evaluation side.

---

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic definition of preference leakage, with a unique perspective and broad implications
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three relationship types × three learning methods × multiple mixing ratios × multiple model scales × feature ablation × mitigation methods
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear problem definition, tightly organized around three RQs, rigorous formalization
- **Value**: ⭐⭐⭐⭐⭐ — Profound implications for LLM evaluation paradigms, directly bearing on leaderboard fairness
- **Overall**: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ICLR 2026\] Subliminal Signals in Preference Labels](subliminal_signals_in_preference_labels.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](../../ACL2026/llm_evaluation/contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ACL 2026\] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases](../../ACL2026/llm_evaluation/reasoning_model_is_superior_llm-judge_yet_suffers_from_biases.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](../../AAAI2026/llm_evaluation/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)

</div>

<!-- RELATED:END -->
