---
title: >-
  [Paper Note] Towards Effective and Efficient Continual Pre-training of Large Language Models
description: >-
  [ACL 2025][LLM Pretraining][Continual Pre-training] This work systematically investigates data strategies for the continual pre-training of Llama-3 (8B). By employing three primary strategies—topic-based data mixture, perplexity-based curriculum learning, and high-quality synthetic scientific QA data—the proposed approach significantly enhances Chinese capabilities (C-Eval +8.81) and scientific reasoning (MATH +12.00) using only 100B tokens, while effectively maintaining the…
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Continual Pre-training"
  - "Synthetic Data"
  - "Catastrophic Forgetting"
  - "Curriculum Learning"
  - "Bilingual Adaptation"
date: 2026-05-08
content_hash: 9c2cbd2d22cdfbcb
---

# Towards Effective and Efficient Continual Pre-training of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2407.18743](https://arxiv.org/abs/2407.18743)  
**Code**: [https://github.com/RUC-GSAI/Llama-3-SynE](https://github.com/RUC-GSAI/Llama-3-SynE)  
**Area**: LLM Pre-training  
**Keywords**: Continual Pre-training, Synthetic Data, Catastrophic Forgetting, Curriculum Learning, Bilingual Adaptation

## TL;DR

This work systematically investigates data strategies for the continual pre-training of Llama-3 (8B). By employing three primary strategies—topic-based data mixture, perplexity-based curriculum learning, and high-quality synthetic scientific QA data—the proposed approach significantly enhances Chinese capabilities (C-Eval +8.81) and scientific reasoning (MATH +12.00) using only 100B tokens, while effectively maintaining the original English capabilities.

## Background & Motivation

**Background**: Large language models acquire strong general capabilities through massive pre-training but still exhibit deficiencies in specific scenarios. For instance, Llama-3 is primarily pre-trained on English data, resulting in suboptimal performance on Chinese tasks. As a general model, it may also lack multi-disciplinary scientific knowledge. Continual pre-training (CPT) is a dominant method to enhance specific abilities.

**Limitations of Prior Work**: Catastrophic forgetting in CPT remains a core technical challenge—as new capabilities are enhanced, original capabilities may drop significantly. While CPT is widely used, critical training details regarding data selection, mixture, and curriculum scheduling are insufficiently discussed, particularly on how to simultaneously develop new capabilities and maintain existing ones under a limited training budget.

**Key Challenge**: Developing new capabilities requires a large volume of domain data, but excessive domain data can overwrite the prior knowledge distribution, leading to forgetting. How to achieve a precise balance between "learning the new" and "preserving the old"?

**Goal**: Systematically explore the data recipe of continual pre-training under a limited computational budget (~100B tokens)—specifically, how to efficiently enhance Llama-3's Chinese and scientific reasoning capabilities while maximally retaining general English capabilities.

**Key Insight**: Deconstruct the CPT process into two stages: a bilingual adaptation stage (to enhance Chinese) and a synthetic data enhancement stage (to enhance scientific reasoning). Targeted data strategies are designed for each stage, and a smaller model (TinyLlama) is leveraged for proxy experiments to reduce exploration costs.

**Core Idea**: Achieve stable and efficient continual pre-training through topic-based dynamic data mixture, perplexity-based curriculum learning, and large-scale synthetic multi-disciplinary scientific QA data.

## Method

### Overall Architecture

The training is divided into two stages: (1) Bilingual adaptation stage: mixed Chinese-English training on 92.5B tokens with a Chinese-to-English ratio of 2:8, incorporating topic-based data mixture and perplexity-based curriculum strategies; (2) Synthetic enhancement stage: integrating synthetic multi-disciplinary scientific QA data and code QA data on 7.5B tokens, with a ratio of Chinese:English:Synthetic = 1:7:2. Total training budget is approximately 100B tokens.

### Key Designs

1. **Topic-based Data Mixture**:

    - **Function**: Dynamically adjust data mixtures at a more granular "topic level" rather than the "dataset level".
    - **Mechanism**: Based on MMLU/CMMLU, 11 English and 11 Chinese topics (e.g., math/physics, computer science, law/policy) are defined. A TinyBERT classifier is trained to automatically annotate topic labels for web pages. During training, changes in perplexity (PPL) for each topic on the validation set are periodically monitored: $\Delta p_i = p_i^{(t)} - p_i^{(t-1)}$. After normalization, an adjustment coefficient is calculated as $f_i = 1 + \alpha \cdot \delta_{p_i} \cdot w_i$, dynamically updating the sampling ratio of each topic as $r_i^{(t)} = \frac{r_i^{(t-1)} \cdot f_i}{\sum_j r_j^{(t-1)} \cdot f_j}$.
    - **Design Motivation**: Dataset-level mixing is too coarse, as different topics within the same dataset may have varying speeds of knowledge acquisition. Topic-level monitoring enables fine-grained detection of which topics are degrading or have been sufficiently learned.

2. **Perplexity-based Data Curriculum (PPL-based Data Curriculum)**:

    - **Function**: Organize Chinese training data in an easy-to-hard curriculum.
    - **Mechanism**: Measure the difficulty of training data using the model's own PPL score, sorting Chinese data from low to high PPL, and gradually increasing training difficulty.
    - **Design Motivation**: Since Llama-3 contains almost no Chinese pre-training data, training directly on complex Chinese materials would cause a severe distribution conflict. Starting with "simple" Chinese data provides a smooth transition, helping to mitigate catastrophic forgetting of English performance.

3. **Scientific QA Synthesis**:

    - **Function**: Generate high-quality scientific QA pairs to enhance multi-disciplinary reasoning capabilities.
    - **Mechanism**: Covering 9 scientific disciplines (mathematics, physics, chemistry, biology, astronomy, earth sciences, medicine, computer science, and liberal arts), text snippets are extracted from scientific web pages in Dolma and C4, and Mistral-7B-Instruct is used to generate corresponding QA pairs. The QA pairs are concatenated as plain text and added to the CPT corpus. Similarly, code QA data is generated based on LeetCode using in-context learning to preserve programming capabilities.
    - **Design Motivation**: Authentic scientific QA data is scarce. Synthetic data can better extract key knowledge from web pages, reducing noise from irrelevant information, and the QA format aligns closer with downstream tasks.

### Loss & Training

Standard next-token prediction loss for language modeling is used. The optimizer is AdamW ($\beta_1=0.9$, $\beta_2=0.95$). The learning rate scheduler adopts WSD (Warmup-Stable-Decay), with a warmup phase of 10B tokens linearly scaling from $1\times10^{-7}$ to $1\times10^{-5}$, followed by a stable phase. BFloat16 mixed-precision training is used with gradient clipping at 1.0 and a maximum context length of 8192 tokens. Flash Attention and DeepSpeed ZeRO Stage 2 are utilized to optimize efficiency.

## Key Experimental Results

### Main Results

| Model | MMLU | C-Eval | CMMLU | MATH | GSM8K | HumanEval | MBPP |
|--------|------|------|------|------|------|------|------|
| Llama-3-8B | 66.60 | 49.43 | 51.03 | 16.20 | 54.40 | 36.59 | 47.00 |
| Llama-3-SynE (Ours) | 65.19 | **58.24** | **57.34** | **28.20** | **60.80** | **42.07** | 45.60 |
| MAmmoTH2-8B | 64.89 | 46.56 | 45.90 | 34.10 | 61.70 | 17.68 | 38.80 |
| Llama-3-Chinese-8B | 64.10 | 50.14 | 51.20 | 3.60 | 0.80 | 9.76 | 14.80 |

### Proxy Experiment Findings (TinyLlama)

| Experiment | Key Findings |
|------|---------|
| Synthetic Data Effectiveness | 1B synthetic data + 4B raw data > 5B pure raw data (+2.5% on average for science benchmarks) |
| Synthetic Data Quality | Error rates below 30% have minimal impact on performance, dropping significantly only when >50% |
| Synthetic Data Ratio | 20% is the optimal mixing ratio; excessively high ratios (40%) lead to degradation |
| Data Curriculum | Easy-to-hard (LH) outperforms hard-to-easy (HL) and random sampling |
| Discipline Separation | Deliberate isolation of training by discipline degrades performance |

### Key Findings

- C-Eval improves by 8.81 points and CMMLU by 6.31 points, validating the effectiveness of the bilingual adaptation strategy.
- MATH improves by 12.00 points and SciEval by 4.13 points, proving the value of synthetic scientific QA data.
- MMLU decreases by only 1.41 points, indicating that forgetting is effectively controlled.
- The Gaokao Biology sub-task improves by 25.71 points (43.81 $\to$ 69.52), indicating the most significant improvement in Chinese scientific reasoning.
- Even with a 30% error rate in synthetic data, the impact on model performance remains small, demonstrating that LLMs have high tolerance for noise in synthetic data.
- Programming ability (HumanEval +5.48) also shows improvement, benefiting from the protection of synthetic code QA data.

## Highlights & Insights

- **Proxy model strategy is highly practical**: First systematically exploring dimensions of the data strategy (effectiveness, quality, ratio, curriculum) on TinyLlama, then transferring the optimal recipe to Llama-3 substantially reduces experimental costs. This methodology is transferable to any LLM CPT scenario.
- **Topic-based dynamic data mixture** is more fine-grained than traditional dataset-level mixing. Achieving automated adjustments through PPL monitoring is a practical engineering contribution.
- **Synthetic data does not need to be perfect**: Performance is hardly affected under a 30% error rate, providing confidence in utilizing large-scale, low-cost synthetic data.
- **Deliberate isolation by discipline in curriculum learning is counterproductive**, demonstrating that cross-disciplinary knowledge integration is more effective than isolated learning.

## Limitations & Future Work

- English MMLU still exhibits a slight drop (-1.41), showing that completely eliminating forgetting remains a challenge.
- Synthetic data is generated by Mistral-7B, limiting its quality ceiling to the generator's capacity. Utilizing stronger models or introducing human-in-the-loop quality control may yield further improvements.
- The topic classifier is based on a simple TinyBERT, which possesses limited classification accuracy and might affect the precision of data mixing.
- Training with 100B tokens still demands substantial computational resources; more extreme low-budget scenarios were not explored.

## Related Work & Insights

- **vs Llama-3-Chinese-8B**: This model also adapts Llama-3 for Chinese, but severely degrades math and coding capabilities (MATH 3.60, HumanEval 9.76), showing that a lack of synthetic data protection leads to catastrophic forgetting.
- **vs MAmmoTH2-8B**: Slightly weaker on English scientific benchmarks but significantly leading in Chinese and coding; furthermore, MAmmoTH2 degrades on Chinese benchmarks (C-Eval 46.56 vs Llama-3's 49.43), verifying the importance of balancing multiple capabilities.
- **vs Galactica-6.7B**: The scientific-domain model performs extremely poorly on general tasks (MMLU 37.13), whereas the proposed method balances general and scientific capabilities.

## Rating

- **Novelty**: ⭐⭐⭐ Individual technical components (curriculum learning, data mixture, synthetic data) are not entirely novel on their own, but the systematic integration and detailed exploration are commendable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The system of proxy experiments plus main experiments is highly comprehensive. Controlled experiments on synthetic data quality, ratio, and curriculum are highly valuable.
- **Writing Quality**: ⭐⭐⭐⭐ Technical report style, well-structured and detailed experimental descriptions, though the innovative narrative is slightly weak.
- **Value**: ⭐⭐⭐⭐ Provides a reproducible best-practice guide for CPT, offering direct reference value for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AsyncLM: Efficient and Adaptive Async Pre-training of Language Models](asynclm_efficient_and_adaptive_async_pre-training_of_language_models.md)
- [\[ACL 2025\] Improving Continual Pre-training Through Seamless Data Packing](improving_continual_pre-training_through_seamless_data_packing.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ACL 2025\] Velocitune: A Velocity-based Dynamic Domain Reweighting Method for Continual Pre-training](velocitune_a_velocity-based_dynamic_domain_reweighting_method_for_continual_pre-.md)
- [\[ACL 2025\] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)

</div>

<!-- RELATED:END -->
