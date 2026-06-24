---
title: >-
  [Paper Note] PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data
description: >-
  [ACL 2025][Information Retrieval & RAG][Personalization] Proposes a synthetic data generation pipeline to create the PersonaBench benchmark, which contains diverse user personas and simulated private documents (chat logs, AI interactions, purchase history). It is designed to evaluate AI models' ability to extract personal information from noisy user data. Experimental results show that current RAG formulations are far from sufficient for this task.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Personalization"
  - "RAG"
  - "Synthetic Data"
  - "User Persona"
  - "Private Data Understanding"
date: 2026-05-08
content_hash: db452b94dec2a683
---

# PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data

**Conference**: ACL 2025  
**arXiv**: [2502.20616](https://arxiv.org/abs/2502.20616)  
**Code**: None (Dataset scheduled for release without ground-truth profiles)  
**Area**: NLP / Personalized AI  
**Keywords**: Personalization, RAG, Synthetic Data, User Persona, Private Data Understanding

## TL;DR

Proposes a synthetic data generation pipeline to create the PersonaBench benchmark, which contains diverse user personas and simulated private documents (chat logs, AI interactions, purchase history). It is designed to evaluate AI models' ability to extract personal information from noisy user data. Experimental results show that current RAG formulations are far from sufficient for this task.

## Background & Motivation

Personalization is a core capability of AI assistants. When a user asks for vacation recommendations, the model should consider their preferred climate, travel budget, and past travel experiences. The mainstream paradigm to achieve personalization is Retrieval-Augmented Generation (RAG): retrieving relevant information from the user's private data, appending it to the query, and feeding it to the LLM to generate personalized answers.

However, evaluating this capability faces fundamental obstacles:

- **No public datasets**: Due to privacy concerns, there are no publicly available datasets pairing private user documents with corresponding profile information.
- **Noisy real-world data**: Personal information in real-world user data is fragmented, scattered across large volumes of irrelevant content, and dynamic over time.
- **Narrow coverage of existing benchmarks**: Tau-bench and AppWorld focus on API calls, with overly simplified user personas; LaMP focuses on writing style mimicry rather than information extraction.

This work bypasses privacy constraints through a synthetic data generation pipeline, creating realistic user data for standardized evaluation.

## Method

### Overall Architecture

The data generation pipeline consists of two stages:

- **Stage 1: User Persona Synthesis** — Creating diverse, socially connected virtual users.
- **Stage 2: Private Document Synthesis** — Generating simulated daily activity documents based on the synthesized personas.

Subsequently, personal questions are designed to evaluate the RAG system's ability to extract personal information from these documents.

### Key Designs

1. **User Persona Synthesis (Stage 1)**:

    - **Persona Template Definition**: Three meta-categories — demographics (age, occupation, etc.), psychological profiles (preferences), and social information (relationships).
    - **Persona Sampling & Social Graph**: Short descriptions are sampled from the public persona dataset of Chan et al. to increase diversity and avoid repetitive biases inherent in LLM generation. First, 3 personas are randomly sampled to establish an initial social graph via LLMs, which is then expanded to include more individuals. Post-processing is applied to ensure symmetric and consistent relationship edges.
    - **Persona Completion**: Socially anchored attributes are generated first (e.g., colleagues must work at the same company), followed by the independent completion of remaining attributes (hobbies, dietary preferences, etc.) to ensure internal consistency.
    - Design Motivation: Directly prompting LLMs to fill templates often leads to severe repetitions; incorporating a social graph ensures that personas are not isolated individuals but part of an interconnected community, which is closer to reality.

2. **Private Document Synthesis (Stage 2)**:

    - Three document types:
        - **Dialogue History**: Multi-turn conversations between connected users in the social graph.
        - **User-AI Interactions**: Dialogue or chitchat records between users and AI assistants.
        - **Purchase History**: Item purchase records generated based on user preferences.
    - **Four generation strategies**:
        - Personal data generation: Randomly selecting an attribute and prompting the LLM to generate a session that implicitly discloses this attribute.
        - Noise data generation: Irrelevant conversations (e.g., weather, daily issues) that do not reveal personal info, used to increase difficulty by adjusting the noise ratio.
        - Real-world news integration: Injecting real-world news events into dialogues with a 20% probability.
        - Information update: Updating existing preferences in subsequent dialogues with a <1% probability, simulating preference shifts.

3. **Evaluation Design**:

    - Generating three types of personal questions for each user: basic information (269), preferences (186), and social (127, including multi-hop), totaling 582 questions.
    - Social questions require multi-hop reasoning (e.g., "What is my sister's favorite movie?" requires identifying the sister first and then retrieving her preferences).
    - The evaluation consists of two levels: retrieval evaluation (Recall/NDCG) and end-to-end evaluation (Recall/F1).

### Loss & Training

This work does not involve model training; instead, it evaluates the performance of existing RAG pipelines on this benchmark.

## Key Experimental Results

### Retrieval Evaluation (Noise Ratio 0.5)

| Retriever | Parameters | Recall | NDCG |
|-----------|------------|--------|------|
| all-MiniLM-L6-v2 | 23M | 0.236 | 0.186 |
| all-mpnet-base-v2 | 110M | 0.267 | 0.229 |
| bge-m3 | 567M | 0.325 | 0.280 |

### End-to-End Evaluation (Noise Ratio 0.5, bge-m3 Retriever)

| Base LLM | Recall (Overall) | F1 (Overall) |
|----------|------------------|--------------|
| GPT-3.5-turbo | 0.224 | 0.222 |
| GPT-4 | 0.228 | 0.223 |
| GPT-4o | 0.237 | 0.241 |
| **GPT-4o-mini** | **0.277** | **0.281** |
| GPT-4o-mini (Ground Truth Context) | 0.502 | 0.521 |

### Key Findings

1. **Current RAG systems are severely deficient**: The Recall of the best retriever is only 0.325, indicating that over half of the critical information fails to be retrieved.
2. **GPT-4o-mini unexpectedly outperforms**: GPT-4o-mini outperforms GPT-4o in end-to-end evaluations, demonstrating that "generally more capable" does not equate to "stronger personal information understanding."
3. **Recall remains only around 50% even with Ground Truth context**: This indicates that the implicit nature of personal information makes it difficult for LLMs to extract completely even when presented with the correct documents.
4. **Sensitivity to noise**: Performance in both retrieval and end-to-end evaluations consistently drops as the noise ratio increases from 0.0 to 0.7, highlighting noise robustness as a key bottleneck.
5. **GPT-4o outperforms GPT-4o-mini in info updates and noise robustness**: Different models have different strengths and weaknesses, further indicating that personalization understanding is a multi-dimensional capability.
6. **Social questions (multi-hop) are the most challenging**: Social questions requiring reasoning across documents pose dual difficulties in both retrieval and generation.

## Highlights & Insights

- **Sophisticated synthesis pipeline**: The multi-layered design incorporating social graphs, persona completion, diverse document types, and noise/update strategies allows synthetic data to remain highly realistic while maintaining privacy safety.
- **Revealing fundamental limitations of RAG**: It is not just retrieval that is challenging; even with the correct context, LLMs can only extract about half of the information, suggesting that personal info understanding requires paradigms beyond conventional RAG.
- **Noise ratio as a controllable difficulty knob**: Different noise levels provide a clean experimental dimension for analyzing the robustness of model information extraction.
- **Multi-dimensional radar chart analysis**: Comparing models across multiple dimensions, such as noise robustness, awareness of information updates, and social understanding, provides deeper insights than single metrics.

## Limitations & Future Work

- All data are synthesized purely by GPT-4o, which may inherit model biases and suffer from distribution shifts compared to real-world user data.
- Ground-truth profiles are not publicly released (to prevent cheating), which limits replication and extension by the community.
- Purchase history format is relatively simplified (only title/description/brand/category), lacking multimodal information such as images.
- Documents are partitioned at the session level without exploring the impact of finer-grained chunking strategies on retrieval performance.
- Only standard RAG pipelines are tested; more advanced retrieval strategies, such as Graph RAG or iterative retrieval, have not been evaluated.

## Related Work & Insights

This work is closely related to synthetic data generation (such as generative agent simulations by Park et al. 2023) and personalization evaluation (LaMP, Tau-bench). The core insight is: in scenarios where private data is inaccessible, high-quality synthetic data combined with carefully designed evaluation protocols can serve as a substitute for real-world data to analyze system performance. This methodology can be extended to other privacy-sensitive domains such as healthcare and finance.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The multi-layered design of the synthetic data pipeline (social graph, noise control, information updates) is highly original, and the benchmark precisely fills a crucial gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive with 12 RAG configurations, 4 noise levels, multi-dimensional analysis, and extensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The description of the pipeline is clear with rich diagrams, though the method section is slightly lengthy and could be streamlined.
- **Value**: ⭐⭐⭐⭐ — Uncovers core bottlenecks in personalized AI (difficulty of comprehension even with correct context) and provides clear directions for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On Synthetic Data Strategies for Domain-Specific Generative Retrieval](on_synthetic_data_strategies_for_domain-specific_generative_retrieval.md)
- [\[ACL 2025\] HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation](hoh_a_dynamic_benchmark_for_evaluating_the_impact_of_outdated_information_on_ret.md)
- [\[ICML 2025\] Understanding Synthetic Context Extension via Retrieval Heads](../../ICML2025/information_retrieval/understanding_synthetic_context_extension_via_retrieval_heads.md)
- [\[ACL 2025\] The Distracting Effect: Understanding Irrelevant Passages in RAG](the_distracting_effect_understanding_irrelevant_passages_in_rag.md)
- [\[ACL 2025\] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models](coir_a_comprehensive_benchmark_for_code_information_retrieval_models.md)

</div>

<!-- RELATED:END -->
