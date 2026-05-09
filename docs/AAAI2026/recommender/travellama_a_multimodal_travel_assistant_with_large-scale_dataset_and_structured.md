---
title: >-
  [Paper Note] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning
description: >-
  [AAAI 2026][Recommender Systems][Multimodal Travel Assistant] This paper presents TraveLLaMA, a multimodal language model system for travel assistance. By constructing the TravelQA dataset with 265K QA pairs and the Travel-CoT structured reasoning framework, the system achieves a 10.8% accuracy improvement on travel-related question answering and obtains a SUS usability score of 82.5 in a 500-participant user study.
tags:
  - AAAI 2026
  - Recommender Systems
  - Multimodal Travel Assistant
  - Vision-Language Model
  - Chain-of-Thought Reasoning
  - Travel Planning
  - Dataset
date: 2026-05-08
content_hash: 16047a5fb5ea119b
---

# TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning

**Conference**: AAAI 2026
**arXiv**: [2504.16505](https://arxiv.org/abs/2504.16505)
**Code**: [https://travellama-best.github.io/](https://travellama-best.github.io/)
**Area**: Recommender Systems
**Keywords**: Multimodal Travel Assistant, Vision-Language Model, Chain-of-Thought Reasoning, Travel Planning, Dataset

## TL;DR

This paper presents TraveLLaMA, a multimodal language model system for travel assistance. By constructing the TravelQA dataset with 265K QA pairs and the Travel-CoT structured reasoning framework, the system achieves a 10.8% accuracy improvement on travel-related question answering and obtains a SUS usability score of 82.5 in a 500-participant user study.

## Background & Motivation

Travel planning represents a quintessential complex real-world AI application, requiring simultaneous understanding of visual scenes, geographic context, and practical constraints. Although large language models (LLMs) have achieved remarkable success across many domains, a critical bottleneck remains in travel assistance: **the lack of multimodal datasets capable of capturing the inherently visual and contextual nature of travel planning**.

Specifically, effective travel assistance demands integration across multiple modalities:
- **Recommending a restaurant** requires understanding its location on a map, recognizing ambient atmosphere from photos, interpreting user reviews, and accounting for operational constraints.
- **Planning a day itinerary** requires analyzing inter-attraction distances, understanding transportation options from visual maps, recognizing architectural styles, and integrating temporal constraints such as opening hours and peak periods.

Furthermore, the challenge extends well beyond simple multimodal understanding to **culturally aware and contextually appropriate reasoning**. For instance, visiting temples in Kyoto versus spending a day on a beach in Bali requires entirely different preparations, yet current AI systems tend to provide generic advice. Even models capable of processing individual modalities often fail to coherently synthesize information, yielding recommendations that may be factually correct but practically infeasible.

## Method

### Overall Architecture

The TraveLLaMA system comprises three core contributions: the TravelQA dataset, the Travel-CoT reasoning framework, and an interactive agent system. The overall pipeline fine-tunes a vision-language model on the dataset to inject domain knowledge, then employs Travel-CoT to decompose complex queries, and finally delivers real-time interactive travel planning through the agent system.

### Key Designs

#### 1. **TravelQA Dataset**: The First Large-Scale Multimodal Travel QA Dataset

TravelQA contains 265K QA pairs spanning 35+ cities worldwide (North America, Asia, and Europe), covering six major categories: attractions (70K), dining (52K), accommodation (39K), transportation (26K), culture (39K), and practical information (34K).

Data composition:
- **160K Text QA**: Expanded from 26K factual units into five distinct questions each (130K), supplemented by 30K augmented QA pairs focusing on practical constraints such as safety, cost, and accessibility.
- **100K Visual-Language QA**: Derived from 20K points of interest (POIs), each with an average of 4–5 street-view or map images; GPT-4 generates three categories of questions per image (identification, experience, and practical), yielding approximately $20K \times 4\text{-}5 \times 3 \approx 100K$ pairs.
- **5K CoT Reasoning Examples**: Expert-annotated chain-of-thought samples, each containing reasoning along three dimensions: spatial, temporal, and practical.

Quality assurance: All splits are constructed with POI-disjoint partitioning (each POI and its associated images and metadata appear in only one split), preventing data leakage. Text answers average 45.6 words, visual answers average 25–28 words, and all QA pairs undergo multi-stage validation.

#### 2. **Travel-CoT Structured Reasoning Framework**: Decomposing Travel Queries into Spatial, Temporal, and Practical Dimensions

Travel-CoT adopts a **two-stage formulation**. Given multimodal input $(x, Q)$, the model first generates a reasoning chain:

$$r = f_\theta(x, Q), \quad r = \{r_s, r_t, r_p\}$$

where $r_s$ encodes spatial understanding (location, distance, routes), $r_t$ encodes temporal scheduling (operating hours, time allocation), and $r_p$ captures practical constraints (budget, accessibility, safety).

The final answer is generated conditioned on both the input and the reasoning chain:

$$y \sim P_\phi(y \mid x, Q, r)$$

Both components are jointly trained using 5,000 expert-annotated Travel-CoT examples:

$$\mathcal{L} = \lambda \mathcal{L}_{CoT}(r^*, r) + \mathcal{L}_{ans}(y^*, y)$$

**Design Motivation**: Pre-trained VLMs handle factual queries reasonably well but underperform on planning queries requiring multi-factor reasoning. Travel-CoT explicitly decomposes queries along three dimensions, improving both answer accuracy and interpretability of decision paths.

#### 3. **ReAct-Style Agent System**: Interactive Planning with Real-Time Service Integration

The agent system processes multimodal travel requests through four stages:
- **Query Analysis**: Extracts textual constraints (destination, duration, budget, group size) and interprets visual inputs (e.g., landmark recognition from uploaded photos).
- **Reasoning**: Applies Travel-CoT to organize spatial, temporal, and practical requirements.
- **Tool Invocation**: Calls APIs to retrieve schedules, prices, reviews, and transportation information; the internal state evolves as:

$$\text{Plan}_t = \text{Update}(\text{Plan}_{t-1}, \text{Tool}(\pi(s_t, \text{Plan}_{t-1})), r)$$

- **Result Integration**: Generates detailed itineraries incorporating timetables, budgets, and constraint verification.

### Loss & Training

- Training conducted on 8 A100 GPUs.
- Training set: 213K QA pairs; test set: 52K QA pairs.
- Visual inputs normalized to $336 \times 336$ resolution; text inputs truncated to a maximum of 512 tokens.
- The model is first fine-tuned on standard QA, then post-trained on CoT QA.
- Learning rate scheduling, gradient clipping, and early stopping based on validation performance are employed.

## Key Experimental Results

### Main Results

| Model | LLM Backbone | Pure Text | VQA | Full Score | Gain |
|-------|-------------|-----------|-----|------------|------|
| LLaVA-1.5 (pretrained) | Vicuna-13B | 74.3 | 63.3 | 70.0 | — |
| LLaVA-1.5 (fine-tuned) | Vicuna-13B | 80.4 | 68.9 | 76.0 | +8.6% |
| Qwen-VL (fine-tuned) | Qwen-7B | 78.7 | 67.7 | 74.5 | +9.4% |
| Shikra (fine-tuned) | Vicuna-13B | 77.7 | 66.7 | 73.5 | +8.9% |
| **TraveLLaMA (Ours)** | **Vicuna-13B** | **82.5** | **70.5** | **77.8** | **+10.8%** |

Full Score is computed as a weighted average (61.5% text + 38.5% visual-language). Domain-specific fine-tuning yields baseline gains of 6.2–9.4%, with Travel-CoT reasoning providing further significant improvement.

### User Study

| System | SUS Score | Rating |
|--------|-----------|--------|
| **TraveLLaMA** | **82.5** | **Excellent** |
| Claude 3.5 | 76.3 | Good |

A total of 500 participants (250 per system), aged 18–62. TraveLLaMA significantly outperforms Claude 3.5 in learnability, ease of use, and complexity reduction; the 6.2-point SUS gap is primarily attributed to domain-optimized design.

### Ablation Study

| Configuration | Pure Text | VQA | Full | Note |
|--------------|-----------|-----|------|------|
| Pretrained only | 74.3 | 63.3 | 70.0 | Baseline |
| + TravelQA fine-tuning | 80.4 | 68.9 | 76.0 | Domain knowledge injection |
| + Travel-CoT | 82.5 | 70.5 | 77.8 | Structured reasoning gain |

Key findings:
- Travel-CoT reasoning contributes an additional +1.8% gain on top of fine-tuning.
- Qwen-VL achieves the largest relative improvement (+9.4%), suggesting that higher image resolution ($448^2$) benefits visual understanding.
- The effect of domain-specific data is universal and consistent across all architectures.

### Key Findings

1. **Domain fine-tuning is highly effective**: All model architectures achieve consistent performance gains after TravelQA fine-tuning, validating the value of domain-specific training data.
2. **Structured reasoning provides further gains**: Travel-CoT improves accuracy beyond fine-tuning alone, particularly on complex queries requiring multi-step reasoning.
3. **Practical usability validated**: A SUS score of 82.5 reaches the "Excellent" tier, confirming the system's usability in real travel scenarios.
4. **Multimodal capability**: Performance gains are especially pronounced on visual tasks such as map interpretation and scene understanding.

## Highlights & Insights

1. **Clever data construction strategy**: GPT-4 is leveraged to transform fragmented web-based travel information into high-quality multimodal training data, enabling cost-effective large-scale dataset creation.
2. **Three-dimensional reasoning decomposition** (spatial + temporal + practical) is intuitively grounded and empirically validated — travel planning inherently demands simultaneous consideration of all three dimensions.
3. **POI-disjoint data splitting** prevents cross-modal leakage and ensures evaluation rigor.
4. **The complete pipeline from data to reasoning to system** demonstrates a systematic methodology for building vertical-domain AI assistants.

## Limitations & Future Work

1. **Temporal validity**: Travel information (e.g., operating hours, prices) is highly time-sensitive, and knowledge encoded during training may become outdated.
2. **Geographic coverage**: The dataset covers only 35+ cities, concentrated primarily in North America, Asia, and Europe, with insufficient coverage of other regions.
3. **Cultural depth**: Although cultural awareness is mentioned, 5K CoT examples may be insufficient to cover the long-tail distribution of global cultural variation.
4. **Evaluation limitations**: MCQ-format evaluation measures factual accuracy, but qualities of travel recommendations such as creativity and personalization are difficult to quantify with automatic metrics.
5. **Scalability**: The ReAct-style agent relies on external APIs, posing latency and availability challenges in real-world deployment.

## Related Work & Insights

- Compared to TravelPlanner (text-only, 1,225 queries), TravelQA offers substantially larger scale and multimodal support.
- The work draws on geospatial LLM research such as GeoLLM and GeoReasoner, while focusing specifically on practical travel assistance.
- The multi-dimensional decomposition approach of Travel-CoT is generalizable to other domains requiring multi-factor decision-making, such as healthcare and finance.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Dataset and reasoning framework design are creative, though the technical pipeline (fine-tuning + CoT) is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Combines quantitative experiments, a large-scale user study, and qualitative analysis for comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Paper structure is clear, with detailed descriptions of data construction.
- **Value**: ⭐⭐⭐⭐ — Provides a valuable dataset and a methodological reference for building vertical-domain multimodal assistants.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](../../NeurIPS2025/recommender/r2ec_towards_large_recommender_models_with_reasoning.md)
- [\[ICLR 2026\] C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](../../ICLR2026/recommender/c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)
- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)
- [\[NeurIPS 2025\] ASAP: An Agentic Solution to Auto-Optimize Performance of Large-Scale LLM Training](../../NeurIPS2025/recommender/asap_an_agentic_solution_to_auto-optimize_performance_of_large-scale_llm_trainin.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)

<!-- RELATED:END -->
