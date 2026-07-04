---
title: >-
  [Paper Note] On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures
description: >-
  [ACL 2025][Reasoning][Measurement system generalization] This paper systematically investigates the generalization capability of LLMs across measurement systems (currency, length, weight). It reveals that models default to dominant measurements from their training data (e.g., USD, metric system), resulting in a significant accuracy drop for queries using non-dominant systems. While Chain-of-Thought (CoT) reasoning can alleviate this performance drop…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Measurement system generalization"
  - "cultural bias"
  - "test-time compute"
  - "inference cost inequality"
  - "cross-cultural NLP"
date: 2026-05-08
content_hash: e025ebc9bcb5c405
---

# On Generalization across Measurement Systems: LLMs Entail More Test-Time Compute for Underrepresented Cultures

**Conference**: ACL 2025  
**arXiv**: [2506.02591](https://arxiv.org/abs/2506.02591)  
**Code**: [GitHub](https://github.com/MinhDucBui/MeasurementSystemBias)  
**Area**: LLM Inference  
**Keywords**: Measurement system generalization, cultural bias, test-time compute, inference cost inequality, cross-cultural NLP

## TL;DR
This paper systematically investigates the generalization capability of LLMs across measurement systems (currency, length, weight). It reveals that models default to dominant measurements from their training data (e.g., USD, metric system), resulting in a significant accuracy drop for queries using non-dominant systems. While Chain-of-Thought (CoT) reasoning can alleviate this performance drop, it increases inference costs by up to 300%, creating a systemic inequality for users from underrepresented cultures.

## Background & Motivation

**Background**: LLMs encode vast amounts of factual knowledge in their parameters (Petroni et al. 2019; Allen-Zhu & Li 2024), including various measurement values, but their training data is heavily dominated by English and Western cultures.

**Limitations of Prior Work**: Users from diverse cultural backgrounds employ different measurement systems (metric vs. imperial, different currencies, and local weight units like "jin" or "geun"). However, whether LLMs can fairly provide accurate information to all users has not been systematically investigated.

**Key Challenge**: The relationships between measurement systems are deterministic (e.g., $1\text{ mile} = 1.609\text{ km}$), allowing humans to state facts using any measurement system. However, LLMs suffer substantial degradation in performance when queried in non-dominant measurements. This is not a lack of knowledge, but a processing equity issue.

**Key Insight**: Three classes of cross-cultural evaluation datasets (fiscal data, city distances, food prices) were compiled, covering 112 currencies, 10 length units, and 10 weight units. The study examines three research questions: (RQ1) What measurements do LLMs default to? (RQ2) How does accuracy change when switching measurements? (RQ3) Can reasoning strategies bridge the gap, and at what cost?

**Core Idea**: LLMs require additional reasoning steps to correctly answer queries in non-dominant measurement systems. Users from underrepresented cultures are essentially charged an "inference tax" (with inference costs increasing by up to 300%).

## Method

### Overall Architecture
This work constructs three classes of cross-cultural measurement datasets, where each query can be posed using various units. It evaluates accuracy and inference costs across seven open-source LLMs under three strategies: direct answering, sequential single hops, and Chain-of-Thought (CoT) reasoning.

### Dataset Design

- **Fiscal Data (Currency System)**: GDP per capita of 148 countries (year 2021), converted into 112 currencies using IMF exchange rates, totaling 16,576 samples.
- **City Distances (Length System)**: Inter-city distances across Germany, Russia, China, Japan (using kilometers) and the US (using miles), calculated with the Haversine formula, testing 10 length units (km, mile, Cape Foot, Thai wa, Chinese li, etc.), totaling 38,000+ samples.
- **Food Prices per Weight (Weight System)**: Food commodity prices in 76 countries (in kilograms) from WFP and food prices from US BLS (in pounds), testing 10 weight units (kg, lb, jin, geun, etc.), totaling 27,820 samples.

### Three Core Research Questions

1. **RQ1: Default Measurement Identification**: What do LLMs default to when no unit is specified? Unconstrained prompts are used to analyze the distribution of measurement systems in the models' default outputs.
2. **RQ2: Cross-Measurement Stability**: How much does accuracy change when specific non-default measurements are requested? Evaluation is conducted using MAPD (inverse Mean Absolute Percentage Deviation), and Wilcoxon signed-rank tests are used to determine statistical significance ($p=0.001$).
3. **RQ3: Mitigation via Reasoning and Its Cost**: Cross-measurement conversion is modeled as a multi-hop reasoning problem consisting of three steps: (i) retrieving the fact in the default measurement $\rightarrow$ (ii) acquiring the conversion rate $\rightarrow$ (iii) arithmetic reasoning. Two strategies are tested: Sequential Single Hops and CoT.

### Evaluation Metric: MAPD

$$MAPD_k = 100 \times \left(1 - \frac{1}{N} \times \sum \min\left(\frac{|p(x_i, f_k) - y_i|}{|y_i|}, 1\right)\right)$$

Higher values indicate better performance, with the deviation capped at 100% to reduce the impact of extreme predictions.

### Evaluated Models
Seven open-source instruction-tuned LLMs: Qwen2.5 (72B/7B), Llama 3.3 (70B), Llama 3.1 (70B/8B), and Aya Expanse (32B/8B), categorized by scale into Large ($\ge 70\text{B}$) and Small ($<70\text{B}$) groups.

## Key Experimental Results

### Table 1: Default Measurement System Statistics of LLMs

| Data / Measurement Type | Large Models | Small Models | Average |
|---|---|---|---|
| **Currency (Fiscal Data)** | | | |
| → Local Currency | 18% | 16% | 17% |
| → USD | 76% | 82% | 80% |
| → Others | 6% | 2% | 3% |
| **Weight (Kilogram Data)** | | | |
| → Kilogram | 100% | 100% | 100% |
| → Pound/Others | 0% | 0% | 0% |
| **Length (Kilometer Data)** | | | |
| → Kilometer | 100% | 100% | 100% |
| → Mile/Others | 0% | 0% | 0% |
| **Length (Mile Data / US Cities)** | | | |
| → Mile | 63% | 78% | 73% |
| → Kilometer | 37% | 22% | 27% |

**Findings**: LLMs default to dominant measurements from their training data—defaulting to USD in 80% of fiscal queries, while weight and length default units strictly follow the cultural context of the data source.

### Table 2: Performance Changes and Inference Cost Increase under Reasoning Strategies (relative to the default measurement baseline without reasoning)

| Default Measurement | Small Direct | Small Seq | Small CoT | Large Direct | Large Seq | Large CoT | Cost Seq | Cost CoT |
|---|---|---|---|---|---|---|---|---|
| USD → Non-default currency | -77.5% | -31.3% | -44.2% | -72.5% | -0.1% | +0.8% | +189% | **+302%** |
| kg → Non-default weight | -33.1% | -17.9% | -13.9% | -59.1% | -4.8% | -8.1% | +191% | +206% |
| lb → Non-default weight | -34.0% | -19.1% | -33.3% | -29.3% | -4.7% | -9.4% | +199% | +187% |
| km → Non-default length | -63.6% | -31.8% | -41.7% | -57.9% | -7.9% | -4.0% | +180% | +210% |
| mile → Non-default length | -74.4% | -34.6% | -50.0% | -73.1% | -7.2% | -7.5% | +197% | +185% |

**Key Findings**: Large models using Seq/CoT can reduce performance losses from 50–70% to within 5–10%, but at the expense of an increase in inference cost by 180–302% (up to **3x** for currency conversion). For Small models, the benefit of reasoning strategies is limited.

### Table 3: Performance Gaps under Multilingual Alignment Settings (* indicates $p < 0.001$ significance)

| Setting | Llama 3.3 70B | Qwen2.5 72B | Aya 32B |
|---|---|---|---|
| Korean + KRW vs. USD | -37.89* | -27.43* | -36.36* |
| Turkish + TRY vs. USD | -57.01* | -42.22* | -49.13* |
| Korean + Geun vs. kg | -21.99* | -31.12* | -22.11* |
| Chinese + Jin vs. kg | -25.39* | -26.79* | -11.33* |
| Japanese + Ken vs. km | -80.09* | -75.71* | -73.90* |
| Chinese + Li vs. km | -20.19* | -13.18* | -15.03* |

Even when language and measurement systems are aligned, performance degradation remains statistically significant in almost all cases—language alignment does not resolve measurement bias.

### Table 4: Currency Bias and Country Income Levels

| Country Income Level | Average MAPD |
|---|---|
| High-income countries | 42.80% |
| Upper-middle-income countries | 19.79% |
| Lower-middle-income countries | 14.31% |
| Low-income countries | 9.03% |

The accuracy for currencies of high-income countries is **34 percentage points** higher than that of low-income countries, demonstrating a clear bias toward wealthy nations.

## Highlights & Insights

1. **Precise Problem Definition**: This work is the first to model "cross-measurement factual queries" as multi-hop reasoning problems with a clear theoretical framework requiring three steps: fact retrieval, conversion rate retrieval, and arithmetic reasoning.
2. **The concept of 'Inference Tax'**: It uncovers a hidden inequity mechanism where users from non-dominant cultures require models to perform extra reasoning (increasing cost by up to 300%), with users from economically developing regions being the least able to afford these extra costs.
3. **Comprehensive Evaluation**: The study evaluates 112 currencies, 10 length units, and 10 weight units across 7 models in multilingual settings, presenting a rare and extensive scale.
4. **Rigorous Experimental Design**: It utilizes the most favorable exchange rates to mitigate volatility impacts, ensures statistical reliability with Wilcoxon tests ($p=0.001$), and guarantees robustness using three prompt variants.
5. **CoT Anchor Analysis**: 91% of CoT traces start from USD for currency conversion, revealing a clear home-system bias in model reasoning paths.

## Limitations & Future Work

1. **Predominantly English Prompts**: Since English is the strongest language of LLMs, these results represent a "best-case scenario"; the issues are expected to be more severe in other languages.
2. **Limited Measurement Type Coverage**: The study only covers currency, length, and weight, leaving other types like temperature, area, and volume unexplored.
3. **Open-source Models Only**: Proprietary models such as GPT-4 and Claude were not evaluated.
4. **Tool-use in Practical Scenarios**: Although using external tools for unit conversion is more optimal, the paper notes that users intuitively tend to query LLMs in direct conversation.
5. **Time-varying Exchange Rates**: The study relies strictly on 2021 data, without considering temporal variances over a wider span.

## Related Work & Insights

* **Bias in Factual Retrieval**: Mallen et al. 2023 and Manvi et al. 2024 show that LLMs exhibit lower accuracy for low-popularity entities and less-developed regions.
* **Cultural Bias**: Kirk et al. 2024 and Cao et al. 2024 reveal that LLMs are heavily biased toward Western values.
* **Multi-hop Reasoning**: Press et al. 2023 and Biran et al. 2024 study implicit multi-hop queries, which remain challenging for LLMs.
* **Processing of Measurement Systems**: Park et al. 2022 observe that small models struggle with cross-measurement comparison, while Dinu et al. 2020 integrate measurement conversion inside translation models.
* **Our Distinction**: This work focuses on factual retrieval within decoder-only LLMs, systematically quantifying the discrepancy in reasoning costs during cross-measurement generalization for the first time.

## Rating

| Dimension | Score (1-10) | Description |
|---|---|---|
| Problem Significance | 8 | Reveals hidden inequities in LLMs against underrepresented cultures, yielding strong social impact |
| Novelty | 7 | Novel problem setting (cross-measurement generalization + inference cost inequity), though the methodology is primarily evaluation-driven |
| Experimental Thoroughness | 9 | 112 currencies x 7 models x 3 reasoning strategies x multilingual; extremely comprehensive coverage with rigorous statistical testing |
| Reproducibility | 9 | Datasets and code are open-sourced; all data sources are publicly available |
| Writing Quality | 8 | Clear logic, well-structured RQ-driven narrative, and effective visualizations |
| Overall Rating | **8** | An exemplary evaluation-focused study with insightful problem definition, outstanding dataset scale, and rigorous experimentation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Retrieval Systems with Inference-Time Logical Reasoning](enhancing_retrieval_systems_with_inference-time_logical_reasoning.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](../../NeurIPS2025/llm_reasoning/provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)
- [\[ICLR 2026\] e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs](../../ICLR2026/llm_reasoning/e3_learning_to_explore_enables_extrapolation_of_test-time_compute_for_llms.md)
- [\[ICLR 2026\] Strategic Scaling of Test-Time Compute: A Bandit Learning Approach](../../ICLR2026/llm_reasoning/strategic_scaling_of_test-time_compute_a_bandit_learning_approach.md)

</div>

<!-- RELATED:END -->
