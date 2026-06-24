---
title: >-
  [Paper Note] Atomic Calibration of LLMs in Long-Form Generations
description: >-
  [LLM Evaluation] This work systematically studies atomic calibration in long-form generation, categorizing confidence elicitation methods into discriminative and generative approaches. It finds these two types to be complementary and proposes a fusion strategy based on confidence consistency, revealing interesting patterns in how model confidence changes during the generation process.
tags:
  - "LLM Evaluation"
date: 2026-05-08
content_hash: cc70c40005635658
---

# Atomic Calibration of LLMs in Long-Form Generations

- **Conference**: ACL 2025
- **arXiv**: [2410.13246](https://arxiv.org/abs/2410.13246)
- **Code**: Not provided
- **Area**: LLM Evaluation / Uncertainty Estimation / Factuality Calibration
- **Keywords**: Atomic Calibration, Confidence Elicitation, Long-Form Generation, Hallucination, Confidence Fusion

## TL;DR

This work systematically studies atomic calibration in long-form generation, categorizing confidence elicitation methods into discriminative and generative approaches. It finds these two types to be complementary and proposes a fusion strategy based on confidence consistency, revealing interesting patterns in how model confidence changes during the generation process.

## Background & Motivation

- **Problem**: LLM confidence calibration is crucial for hallucination detection, but existing studies mainly focus on **response-level calibration (macro calibration)** in short-text QA tasks—assigning a single confidence score to the entire answer. In long-form generation, an answer may contain both accurate and inaccurate claims, and a single score cannot reflect fine-grained factuality.
- **Key Questions**: (1) Why is it necessary to evaluate calibration at the atomic claim level? (2) What factors influence atomic-level calibration? (3) What patterns can atomic-level analysis reveal that macro-level analysis cannot?
- **Core Definition**: **Atomic-level calibration** = Decomposing a long-form response into atomic claims (each containing a single fact), assigning confidence to each claim, and evaluating how well the confidence aligns with actual factuality.

## Method

### Overall Architecture

1. Given a query $q$, the LLM generates a long-form response $x$.
2. Use GPT-4o to decompose $x$ into $N$ atomic claims $\{c_1, ..., c_N\}$.
3. Use GPT-4o combined with Wikipedia/Google Search to verify the factuality label $y_i \in \{0, 1\}$ for each claim.
4. Use different confidence elicitation methods to estimate the confidence $f(c_i)$ for each claim.
5. Use ECE, Brier Score, and AUROC to evaluate the quality of atomic calibration.

### Key Designs

1. **Discriminative confidence methods** — let the model self-evaluate:
    - **Dis-Single**: Directly ask the model whether a single claim is true, taking $P(\text{True})$ as the confidence.
    - **Dis-Context**: Same as above, but providing the original context to assist the judgment.
    - **Dis-Rating**: Prompt the model to directly output a numerical confidence score from 0 to 10.

2. **Generative confidence methods** — based on sampling consistency:
    - **Gen-Binary**: Sample an additional $K$ responses, use an NLI model to determine if the atomic claim is supported, and compute confidence as $\frac{|K_s|}{|K|}$.
    - **Gen-Multi**: Distinguish between "contradicted" and "not mentioned", computing confidence as $\frac{|K_s|}{|K_s| + |K_c|}$.

3. **Confidence Fusion Strategies** (proposed in this paper):
    - **AdjustedAlpha**: Dynamically adjust the fusion weight $\alpha' = \alpha + \gamma_a \cdot d$ based on the difference between the two confidence values $d = B - A$.
    - **DampedFusion**: Apply damping based on consistency, defined as $\gamma(d) = 1 - k \cdot |d|$, to reduce the overall confidence when inconsistencies exist.
    - **Core Idea**: Traditional weighted averaging cannot distinguish between cases like $(0, 1)$ and $(0.4, 0.6)$. The former possesses higher inconsistency and should result in a lower final confidence.

### Loss & Training

No training process is involved. Evaluation is conducted using Expected Calibration Error (ECE), Brier Score (BS), and AUROC.

## Key Experimental Results

### Main Results — Atomic Calibration (ECE ↓, BS ↓, AUROC ↑)

| Method | Llama3-8B ECE | Mistral-7B ECE | Qwen2-7B ECE |
|------|-------------|---------------|-------------|
| Dis-Context | 35.5 / 11.9 / 12.5 | 24.8 / 15.7 / 20.6 | 26.5 / 13.9 / 17.2 |
| Dis-Single | 32.6 / 14.3 / 19.2 | 30.2 / 20.4 / 24.0 | 29.3 / 16.1 / 18.7 |
| Gen-Binary | **10.0 / 8.5 / 11.1** | **13.7 / 8.4 / 12.7** | **10.9 / 6.3 / 9.5** |
| Gen-Multi | 37.4 / 12.6 / 21.9 | 42.2 / 13.4 / 26.6 | 41.7 / 11.6 / 21.0 |

(The three columns correspond to Bios / LongFact / WildHallu datasets respectively)

### Ablation Study — Impact of Model Size on Calibration

| Method | Qwen2-7B ECE | Qwen2-57B ECE | Qwen2-72B ECE |
|------|-------------|---------------|-------------|
| Gen-Binary | 10.9 | 10.5 | 11.2 |
| Dis-Rating | 41.5 | 23.2 | **11.4** |

(Bios dataset)

### Confidence Fusion Results (Cross-category: Gen-Binary + Dis-Context)

| Fusion Method | Llama3-8B ECE | Mistral-7B ECE |
|---------|-------------|---------------|
| WAvg (Weighted Average) | 15.2 | 12.8 |
| MinConf | 14.0 | 11.5 |
| **AdjustedAlpha** | **9.3** | **10.2** |
| **DampedFusion** | **9.5** | **10.4** |

### Key Findings

1. **Atomic calibration is significantly worse than response-level calibration**: All models show a much higher ECE at the atomic level compared to the response level (data points consistently lie above the identity line). Even models that appear well-calibrated at the response level still perform poorly at the atomic level.
2. **Gen-Binary is the most reliable single method**: It achieves the lowest ECE across almost all models and datasets, but does not yield the highest AUROC. This demonstrates that calibration and discriminative ability are distinct dimensions.
3. **Discriminative and generative methods are complementary**: Cross-category fusion (Gen + Dis) significantly improves calibration, while intra-category fusion (Dis + Dis) yields limited gains.
4. **Larger models are not necessarily better calibrated**: Generative methods are insensitive to model size. However, in discriminative methods, larger models perform significantly better (Qwen2-7B Dis-Rating ECE drops from 41.5 to 11.4 on Qwen2-72B).
5. **Confidence changes during the generation process**: In discriminative methods, model confidence decreases as generation progresses. In generative methods, confidence is lowest during the middle of the generation. This suggests that different methods capture different types of uncertainty.

## Highlights & Insights

- **Systematic Framework**: For the first time, a formal definition of atomic-level calibration (Definition 2) is presented, clearly distinguishing between macro vs. atomic calibration and proving that the two are not interchangeable.
- **Methodology Taxonomy**: Categorizes confidence elicitation methods into discriminative and generative approaches, revealing their complementarity and providing clear methodological guidance for future research.
- **Profound Analytical Insights**: Analyzing the patterns of confidence variation across generation positions and alignment across different methods provides a fresh understanding of the intrinsic uncertainty in LLMs.

## Limitations & Future Work

- Atomic claim decomposition and factuality verification rely on GPT-4o, introducing pipeline errors.
- Post-processing calibration methods (e.g., temperature scaling, Platt scaling) are not considered, evaluating only "raw" calibration.
- Only 7 models from 3 families were evaluated, lacking analysis on larger-scale models (e.g., GPT-4, Claude).
- The hyperparameters $\gamma_a$ and $k$ in the confidence fusion strategies (AdjustedAlpha, DampedFusion) require tuning on a validation set.
- Focus is limited to the factuality dimension; atomic-level calibration for other quality dimensions, such as coherence or creativity, remains unexplored.

## Related Work & Insights

- **Atomic Fact Decomposition**: FActScore (Min et al., 2023), VeriScore (Song et al., 2024), D-FActScore (Chiang & Lee, 2024)
- **Uncertainty Estimation**: Semantic Entropy (Kuhn et al., 2022), P(true) (Kadavath et al., 2022), Self-Rating (Tian et al., 2023)
- **Long-Form Calibration**: Luq (Zhang et al., 2024b), Linguistic Calibration (Band et al., 2024)
- **Confidence Fusion**: Weighted averaging by Rivera et al. (2024)

## Rating

- **Novelty**: 8/10 — The formal definition of atomic-level calibration and the discovery of the complementarity between discriminative/generative methods carry significant value.
- **Technical Depth**: 7/10 — The fusion strategies are simple and effective though lacking in high technical complexity; the formal definition is rigorous.
- **Experimental Thoroughness**: 8/10 — Incorporates 7 models, 3 datasets, 5 methods, and multiple fusion strategies, offering rich analysis dimensions (model size, generation position, method alignment).
- **Writing Quality**: 8/10 — Concept definitions are clear, experimental results are presented systematically, and diagrams aid understanding.
- **Overall Score**: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)
- [\[ACL 2025\] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](androidlab_autonomous_agent.md)
- [\[ACL 2025\] Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning](browsing_lost_unformed_recollections_a_benchmark_for_tip-of-the-tongue_search_an.md)

</div>

<!-- RELATED:END -->
