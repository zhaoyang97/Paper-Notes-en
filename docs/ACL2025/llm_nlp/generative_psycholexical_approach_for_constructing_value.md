---
title: >-
  [Paper Note] Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models
description: >-
  [ACL 2025][LLM (Other)][value system] Proposed the Generative Psycho-Lexical Approach (GPLA) to automatically construct a five-factor value system for LLMs (Social Responsibility, Risk-Taking, Rule-Following, Self-Competence, and Rationality), outperforming the classical Schwartz human value system in structural validity, safety prediction, and value alignment.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "value system"
  - "psycho-lexical approach"
  - "LLM alignment"
  - "safety prediction"
  - "generative psychometrics"
date: 2026-05-08
content_hash: 548458b12576ba27
---

# Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.02444](https://arxiv.org/abs/2502.02444)  
**Code**: None  
**Area**: LLM Values / AI Safety  
**Keywords**: value system, psycho-lexical approach, LLM alignment, safety prediction, generative psychometrics

## TL;DR

Proposed the Generative Psycho-Lexical Approach (GPLA) to automatically construct a five-factor value system for LLMs (Social Responsibility, Risk-Taking, Rule-Following, Self-Competence, and Rationality), outperforming the classical Schwartz human value system in structural validity, safety prediction, and value alignment.

## Background & Motivation
**Background**: Values are core beliefs that drive individual and collective behaviors. LLM value evaluation, comprehension, and alignment have become research hotspots in the field of AI safety.

**Limitations of Prior Work**: Existing research primarily relies on the Schwartz value system designed for humans, lacking a psychologically well-grounded value system tailor-made for LLMs. The 10 dimensions of the human value system exhibit poor CFA fit on LLMs.

**Key Challenge**: Human value systems may not adequately capture the unique value dimensions of LLMs, and traditional psycho-lexical approaches rely heavily on extensive human labeling and self-reporting, which suffer from high bias and lack scalability.

**Goal**: To construct a data-driven, psychologically grounded value system tailored for LLMs and provide standardized evaluation tasks.

**Key Insight**: Combine the traditional psycho-lexical approach with the generative capabilities of LLMs to achieve a fully automated value system construction pipeline, bypassing manual annotation.

**Core Idea**: Automatically extract perceptions, recognize values, filter redundancy, perform non-reactive profiling, and conduct statistical modeling using LLMs to construct an LLM-specific value system.

## Method

### Overall Architecture
GPLA adopts an agent-based framework consisting of three LLM agents (Perception Parser $M_P$, Value Generator $M_G$, and Value Evaluator $M_E$) and five steps: corpus-based perception extraction $\to$ value recognition $\to$ value filtering $\to$ non-reactive profiling $\to$ PCA modeling. Corpus sources include ValueBench, GPV, BeaverTails, and ValueLex, covering diverse value-rich LLM outputs. The entire pipeline is automated without human intervention.

### Key Designs
1. **Perception Extraction and Value Recognition**: Use $M_P$ to extract value-rich expressions (perceptions) from corpora such as ValueBench and BeaverTails, then map them to underlying values using $M_G$ (the Kaleido model) and record frequencies.
2. **Value Filtering**: Use ROUGE scores and embedding similarity for deduplication, prioritizing high-frequency values to ensure a concise and representative vocabulary.
3. **Non-reactive Value Profiling**: Adopt the GPV method to profile 693 LLM agents (33 LLMs $\times$ 21 profiling prompts) to avoid self-reporting bias, followed by PCA to extract latent factors.

### Loss & Training
- The alignment task uses PPO, with the objective of minimizing $|\mathbf{x}_V^* - M_E(p, r, V)|$
- Safety prediction uses a linear probe with the Bradley-Terry model to optimize pairwise cross-entropy
- Confirmatory Factor Analysis (CFA) uses standard psychometric validation processes

## Key Experimental Results

### Main Results (CFA Structural Validity)

| Value System | #Values | CFI↑ | GFI↑ | AIC↓ | BIC↓ |
|---------|---------|------|------|------|------|
| Schwartz (H) | 4 | 0.56 | 0.52 | 340 | 1484 |
| Schwartz (L) | 10 | 0.23 | 0.22 | 324 | 1464 |
| **Ours** | **5** | **0.68** | **0.65** | **265** | **1145** |

### Ablation Study (Safety Prediction and Alignment)

| Value System | Safety Prediction Accuracy | Alignment-Harmlessness↓ | Alignment-Helpfulness↑ |
|---------|-------------|-----------------|-----------------|
| Schwartz (H) | 81±15% | -1.52 | 2.15 |
| Schwartz (L) | 74±16% | -1.40 | 2.13 |
| **Ours** | **87±9%** | **-1.26** | **2.16** |

### Key Findings
- The proposed value system significantly outperforms Schwartz in CFI (0.68 vs 0.56), GFI (0.65 vs 0.52), and BIC (1145 vs 1484).
- The standard deviation in safety prediction is smaller (9% vs 15%), indicating that the proposed system is more stable and reliable.

- The five-factor system consists of: Social Responsibility ($\alpha=0.957$), Risk-Taking ($\alpha=0.919$), Rule-Following ($\alpha=0.842$), Self-Competence ($\alpha=0.761$), and Rationality ($\alpha=0.722$), all exceeding the standard psychometric threshold of 0.7.
- Social Responsibility, Rule-Following, and Rationality promote safety, whereas Risk-Taking and Self-Competence undermine safety.
- LLM value consistency is highly correlated with safety scores ($r=0.73$).
- Consistency of value profiling across different datasets reaches 0.87.

## Highlights & Insights
- First systematically proposed methodology for constructing a value system tailored specifically for LLMs, with a solid theoretical foundation (the lexical hypothesis).
- The fully automated pipeline of GPLA addresses the issues of manual labor costs and biases inherent in traditional methods.
- Three benchmark tasks (CFA, safety prediction, and value alignment) constitute a comprehensive evaluation framework.
- Significant improvements are achieved across all tasks compared to the Schwartz value system.
- The five-factor structure is clear and highly interpretable: Social Responsibility vs. Risk-Taking forms opposing axes (validated by circumplex analysis).
- Non-reactive profiling avoids self-reporting bias, yielding more reliable measurement results.
- Found that LLM value consistency is positively correlated with safety ($r=0.73$), providing a new perspective for safety evaluation.
- Large-scale profiling across 693 LLM agents (33 models $\times$ 21 profiling prompts) ensures statistical reliability.

## Limitations & Future Work
- The Cronbach's Alpha of certain factors (Rationality = 0.722) is close to the threshold, suggesting room for further optimizing the selection of atomic values.
- The corpus sources are limited (e.g., ValueBench, BeaverTails), and can be extended to more diverse LLM output scenarios.
- The dynamic evolution of the value system (such as updates across model iterations) has not been considered.
- Insufficient cross-cultural validation—the value system may lean toward Western values due to cultural biases in the training data.
- GPLA relies heavily on the quality of the three LLM agents; alternative model choices may affect the resulting value system.
- The manifestation of value conflicts (e.g., Social Responsibility vs. Self-Competence) in specific tasks has not been explored.

## Related Work & Insights
- A modern adaptation of the traditional psycho-lexical approach (Allport & Odbert, 1936), replacing manual annotation pipelines with LLMs.
- GPV (Ye et al., 2025b) provides the foundation for non-reactive value profiling, acting as a core component of GPLA.
- Comparisons with the Schwartz value theory validate that LLMs indeed require a dedicated value system.
- Safety prediction results have practical implications for evaluating LLM deployment risks.
- ValueLex (Biedma et al., 2024) was a prior attempt but suffered from flaws in psychological grounding; this work provides a detailed comparison with it.
- The value alignment framework of BaseAlign (Yao et al., 2024a) was extended to arbitrary value systems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic proposal for an LLM value system construction methodology, integrating psychology and AI.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across three benchmarks, though experiments with larger-scale models are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, close integration of theory and experiments, and intuitive figures.
- Value: ⭐⭐⭐⭐⭐ Significant theoretical and practical contributions to the field of LLM safety and alignment.
- Overall Rating: Pioneering work that provides new infrastructure (methodology + value system + evaluation tasks) for LLM value research.
- Practicality: The value system can be directly applied to safety evaluations prior to LLM deployment and can predict safety when combined with a linear probe.
- Reproducibility: The methodology pipeline is clear, but depends on multiple specialized models (e.g., Kaleido, ValueLlama).
- Extensibility: GPLA can be applied to other psychological constructs (e.g., AI personality systems, attitude frameworks).
- Open Questions: Should the value system be dynamically updated with model iterations? How should emerging values (value outliers) be handled?
- Interdisciplinary Value: Bridges psychometric theory and AI safety practices, opening up new directions for interdisciplinary research.
- Key Numbers: 693 LLM agents, 33 models, 21 prompts, 5 value factors, 25 atomic values.
- Methodological Contribution: The five-step pipeline of GPLA can be reused to construct psychological construct systems in other domains.
- Practical Impact: Helps model developers evaluate and adjust the intrinsic value orientations of LLMs prior to training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MEraser: An Effective Fingerprint Erasure Approach for Large Language Models](meraser_fingerprint_erasure.md)
- [\[ACL 2025\] Value Portrait: Assessing Language Models' Values through Psychometrically and Ecologically Valid Items](value_portrait_assessing_language_models_values_through_psychometrically_and_eco.md)
- [\[ACL 2025\] Do Language Models Understand Honorific Systems in Javanese?](do_language_models_understand_honorific_systems_in_javanese.md)
- [\[ACL 2025\] MExGen: Multi-Level Explanations for Generative Language Models](mexgen_multi_level_explanations.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)

</div>

<!-- RELATED:END -->
