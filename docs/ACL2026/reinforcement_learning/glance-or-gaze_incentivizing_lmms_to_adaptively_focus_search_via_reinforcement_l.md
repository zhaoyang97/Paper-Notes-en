---
title: >-
  [Paper Note] Glance-or-Gaze: Incentivizing LMMs to Adaptively Focus Search via Reinforcement Learning
description: >-
  [ACL2026][Reinforcement Learning][Multimodal Large Language Models (MLLMs)] This paper proposes Glance-or-Gaze (GoG), enabling Multimodal Large Language Models (MLLMs) to first glance at the full image and then adaptivel…
tags:
  - "ACL2026"
  - "Reinforcement Learning"
  - "Multimodal Large Language Models (MLLMs)"
  - "Visual Search"
  - "Selective Gaze"
  - "GRPO"
  - "Complexity-Adaptive RL"
date: 2026-05-08
content_hash: 4c645f8d97a4d4cb
---

# Glance-or-Gaze: Incentivizing LMMs to Adaptively Focus Search via Reinforcement Learning

**Conference**: ACL2026  
**arXiv**: [2601.13942](https://arxiv.org/abs/2601.13942)  
**Code**: https://github.com/TOM-ZHOUch/Glance-or-Gaze  
**Area**: Reinforcement Learning / Multimodal Retrieval-Augmented Generation / Visual Question Answering  
**Keywords**: Multimodal Large Language Models (MLLMs), Visual Search, Selective Gaze, GRPO, Complexity-Adaptive RL  

## TL;DR
This paper proposes Glance-or-Gaze (GoG), enabling Multimodal Large Language Models (MLLMs) to first glance at the full image and then adaptively gaze at high-value regions when answering knowledge-intensive visual questions. Through SFT and complexity-adaptive GRPO, GoG significantly outperforms baselines such as direct answering, full-scale search, and MMSearch-R1 across 6 VQA and search benchmarks.

## Background & Motivation
**Background**: MLLMs have demonstrated the capability to perform numerous general visual understanding tasks. However, when questions involve long-tail entities, recent facts, or fine-grained visual regions, the parametric knowledge of the model often becomes outdated or insufficient. Search-augmented LMMs have emerged as a natural direction, where models invoke text search, image search, or webpage reading tools to supplement the reasoning process with external information.

**Limitations of Prior Work**: Existing methods often send the entire image or all candidate regions for retrieval, which is equivalent to "search upon sight." This introduces two issues: first, high visual redundancy and noise, where the retrieval system may be misled by irrelevant regions; second, many methods perform only single-turn tool calls, lacking the ability to reflect, switch regions, or re-verify when visual evidence is insufficient.

**Key Challenge**: Knowledge-intensive VQA requires external search, yet search should not be an indiscriminate and expensive process. The model must perform planning regarding "whether to search," "search globally or locally," and "what to do if the initial focus is wrong," rather than passively following a fixed workflow.

**Goal**: The authors aim to transform MLLMs from passive perceivers into active visual planners: using the full image for context first, then selecting whether to invoke text search, full-image search, or local-crop search based on the problem, while correcting focused regions via multi-step reflection for complex samples.

**Key Insight**: The paper observes that humans do not look at an entire image uniformly when solving visual search problems; they first "glance" to obtain context and then "gaze" at high-value regions. The authors abstract this behavior as Selective Gaze and treat the choice of "where to look" as a trainable strategy.

**Core Idea**: Utilize Selective Gaze to filter irrelevant visual regions, followed by SFT to teach basic behaviors and GRPO on complex samples to reinforce multi-step search planning, thereby transforming external retrieval into a demand-driven and reflective process of visual evidence acquisition.

## Method
The GoG method consists of two layers: the first is the behavioral format and tool-use paradigm, defining the "glance-then-gaze" sequence; the second is the reinforcement learning strategy, teaching the model how to combine various search actions and correct erroneous focus in complex problems.

### Overall Architecture

The input is an image and a knowledge-intensive question, and the output is the final answer with search/cropping/reflection trajectories. The process is divided into two training stages.

The first stage is Reflective GoG Behavior Alignment. The authors construct GoG-Instruct data from FVQA and InfoSeek, using uncertainty filtering to remove simple samples that the model can already answer directly. They then synthesize tool trajectories containing Glance, Decision, and Gaze, followed by manual checks for answer correctness and region rationality. This stage uses supervised fine-tuning to let the model master the GoG behavioral format.

The second stage is Complexity-Adaptive Reinforcement Learning. The authors evaluate the initial queries using the SFT-tuned GoG model and categorize samples by pass rates. The final RL uses the more challenging Level 2 samples. Training utilizes GRPO, allowing the model to learn multi-step tool-invocation strategies under rewards for answer accuracy and format compliance.

### Key Designs

1.  **Selective Gaze Mechanism**:
    - **Function**: Filters image regions truly relevant to the question before executing a search, avoiding noise introduced by global retrieval or blind cropping.
    - **Mechanism**: The model first understands the global context through a Glance, proposes candidate regions, and decides whether to answer directly, perform text search, perform full-image search, or execute a Gaze search on selected cropped regions.
    - **Design Motivation**: Key evidence for many visual knowledge questions resides only in small areas; full-image search pushes backgrounds, irrelevant objects, and erroneous OCR results into retrieval. Selective Gaze makes "where to look" an explicit action, serving as a physical anchor for the reasoning process.

2.  **Reflective GoG Behavior Alignment**:
    - **Function**: Uses SFT to teach the base MLLM structured active search behaviors rather than relying on transient prompt induction.
    - **Mechanism**: The authors generate $N=4$ answers for each query using Qwen3-VL-235B-Instruct; if the model is correct $n=4$ times, the sample is considered too simple and filtered. After retaining samples requiring external verification, they construct multi-turn trajectories of "Global Observation - Tool Decision - Local Search," verified by humans for correctness. The final GoG-Instruct contains 5,750 samples, where 43.5% require no search and 56.5% require various search forms.
    - **Design Motivation**: If RL is applied directly, the model may not know valid tool formats or effective search trajectories; using SFT to establish behavior priors reduces the cost of subsequent RL exploration.

3.  **Complexity-Adaptive GRPO**:
    - **Function**: Enhances multi-step search, hybrid search, and error reflection capabilities on complex samples beyond the SFT baseline.
    - **Mechanism**: Samples with an SFT pass rate of approximately 50% are categorized as Level 1, while samples where SFT frequently fails are expanded into Level 2 for training. GRPO samples a group of trajectories for each input and normalizes advantages using group mean and variance. The reward consists of answer accuracy $r_{acc}$ and format score $r_{fmt}$, formulated as $r_i=(1-\lambda)r_{acc}+\lambda r_{fmt}$.
    - **Design Motivation**: Simple samples do not provide sufficient planning signals; difficult samples force the model to learn to combine text, image, and crop searches and to reflect actively after a Gaze error.

### Loss & Training

The SFT stage utilizes a standard causal language modeling objective to maximize $\pi_\theta(y_t^*|x,I,y_{<t}^*)$ on multi-turn dialogue trajectories $y^*$. Implementation uses Qwen2.5-VL-7B-Instruct and Qwen3-VL-8B-Think as backbones, adding LoRA to all transformer blocks with a rank of 8.

The RL stage utilizes GRPO in the veRL framework. Key training settings include: SFT for 3 epochs with a learning rate of $1e^{-4}$ and global batch size of 8; RL actor learning rate of $2e^{-6}$, KL coefficient $\beta=0.001$, rollout number $N=4$, and a maximum response length of 8,192 tokens. RL training spans 15 epochs on 8 NVIDIA H800 GPUs.

## Key Experimental Results

### Main Results

| Model/Paradigm | Avg. | FVQA | InfoSeek | SimpleVQA | MMSearch | LiveVQA | DynVQA |
|--------|------|------|----------|-----------|----------|---------|--------|
| GPT-4o Direct Answer | 30.68 | 42.00 | 30.60 | 43.44 | 21.64 | 14.80 | 31.59 |
| Qwen3-VL-8B-Thinking Direct Answer | 23.84 | 24.56 | 16.05 | 41.76 | 15.20 | 15.15 | 30.31 |
| Qwen3-VL-8B-Think Prompt-based GoG | 41.70 | 51.33 | 32.00 | 62.69 | 36.84 | 25.95 | 41.36 |
| Qwen3-VL-8B-Thinking Full-Search | 46.99 | 57.33 | 32.25 | 61.90 | 63.84 | 23.55 | 39.09 |
| MMSearch-R1* | 36.91 | 42.39 | 24.65 | 54.79 | 40.94 | 23.85 | 34.84 |
| GoG-3-8B-Think-SFT | 50.17 | 62.17 | 40.55 | 65.65 | 53.80 | 32.40 | 46.46 |
| GoG-3-8B-Think-RL | 56.88 | 68.44 | 49.05 | 66.44 | 65.50 | 43.85 | 48.02 |

GoG-3-8B-Think-RL scores 9.89 points higher than the strongest Full-Search Workflow on average, 15.18 points higher than Prompt-based GoG, and 19.97 points higher than the reproduced MMSearch-R1. These results indicate that gains do not stem from "always searching" but from learning when to search, where to search, and how to reflect.

### Ablation Study

| Configuration | Avg. | FVQA | Info | Simple | MM | Live | Dyn | Description |
|------|------|------|------|--------|----|------|-----|------|
| Qwen2.5 SFT w/o SG | 41.76 | 53.39 | 40.10 | 59.53 | 40.94 | 22.30 | 34.28 | Remove Selective Gaze |
| Qwen2.5 Full SFT | 43.28 | 53.72 | 41.90 | 60.61 | 40.35 | 24.55 | 38.53 | Avg +1.52, DynVQA +4.25 |
| Qwen3 SFT w/o SG | 48.34 | 60.17 | 40.55 | 64.86 | 46.20 | 32.80 | 45.47 | Remove Selective Gaze |
| Qwen3 Full SFT | 50.17 | 62.17 | 40.55 | 65.65 | 53.80 | 32.40 | 46.46 | Avg +1.83, MMSearch +7.60 |

| RL Data Construction | Avg. | FVQA | Info | Simple | MM | Live | Dyn | Description |
|------|------|------|------|--------|----|------|-----|------|
| Qwen2.5 RL w/ Level 1 Data | 47.28 | 64.00 | 50.35 | 64.66 | 50.88 | 32.95 | 39.94 | Easier samples |
| Qwen2.5 RL w/ Level 2 Data | 53.22 | 66.78 | 51.05 | 64.86 | 56.73 | 37.70 | 42.21 | Avg +5.94 |
| Qwen3 RL w/ Level 1 Data | 48.89 | 66.39 | 44.95 | 64.86 | 61.99 | 39.55 | 45.61 | Easier samples |
| Qwen3 RL w/ Level 2 Data | 52.38 | 68.44 | 49.05 | 66.44 | 65.50 | 43.85 | 48.02 | Avg +3.49 |

### Key Findings

- SFT primarily teaches basic tool calling: After SFT, 62.3% of Qwen3-VL-Think samples used only one type of search, and 28.7% used hybrid search; after RL, hybrid search increased to 76.7%, and the no-search ratio dropped below 3%.
- Selective Gaze improves focus quality rather than just increasing tool calls: Crop selection accuracy for Qwen3-VL-Think improved from 42.1% to 48.9%, and for Qwen2.5-VL from 46.7% to 51.3%.
- In a manual check of 100 Gaze samples from GoG-3-8B-Think, RL improved Gaze correctness from 59% to 75% and increased the reflection rate after erroneous selections from 30% to 70%.

## Highlights & Insights
- **Modeling Visual Search as Active Planning**: The paper does not treat retrieval augmentation as an external module but allows the model to learn "whether to look, where to look, and what to do if wrong." This is closer to real human VQA processes than full-scale search and is better suited for long-tail entities and fine-grained visual evidence.
- **Clear Division of Labor between SFT and RL**: SFT establishes valid behaviors and tool formats, while RL learns strategies on difficult samples. This division avoids cold-start issues for pure RL and explains the significant increase in hybrid search and reflection behaviors post-RL.
- **Difficult Samples are More Valuable Than Marginal Ones**: Level 2 data outperforms Level 1 data, suggesting that in tool-use reasoning, strategy is strengthened not by "almost correct" samples but by samples that expose incorrect focus, faulty searches, and broken reflection chains.
- **Transferable Insights**: The Selective Gaze concept can be transferred to web agents, medical imaging VQA, or remote sensing: first learn local evidence selection, then bind external tool calls to the selected evidence region, rather than processing the entire input indiscriminately.

## Limitations & Future Work

- **Instability of Search Infrastructure**: The authors report sporadic failures (1-5%) in Jina Reader/search pipelines due to network instability, API timeouts, or abnormal web content formats. Such failures directly lead to missing evidence and affect the final answer.
- **Limited Language Coverage**: Experiments focused on English benchmarks; the generalization of GoG in multilingual or cross-lingual VQA remains unverified. Selective Gaze and search strategies may need re-adaptation for culture-specific entities and non-English content.
- **Dependence on External Judges**: Answer correctness in RL rewards is judged by GPT-oss-120b, which might introduce judge bias. Future work could investigate more verifiable task rewards or multi-judge consistency.
- **Tool Cost Issues**: The ratio of hybrid searches rose significantly after RL, indicating the model prefers multi-step verification; while this improves accuracy, it increases latency and call costs. Deployment requires budget-aware rewards or termination strategies.

## Related Work & Insights
- **vs RAG-style Multimodal Retrieval**: Traditional RAG often treats retrieval as fixed pre-processing; this work turns retrieval into an optional, combinable, and reflective action for the model. The advantage is adaptivity, but at the cost of more complex training and tool environments.
- **vs Prompt-based GoG Agent**: Prompt agents rely on prompt induction for search workflows; this work embeds behavior into the model's strategy via SFT/RL. Qwen3 prompt GoG averaged 41.70, whereas GoG-3-8B-Think-RL reached 56.88.
- **vs Full-Search Workflow**: Full-scale search forces retrieval for every question, providing knowledge but adding significant noise; GoG learns to filter unnecessary or erroneous regions, outperforming the strongest Full-Search by an average of 9.89 points.
- **vs MMSearch-R1**: MMSearch-R1 incorporates search into training but lacks the Selective Gaze and multi-step visual reflection mechanism proposed here. GoG's average improvement over MMSearch-R1 reaches 19.97, identifying visual focus strategy as the key source of gain.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Combines glance/gaze active visual focus with complexity-adaptive RL; provides comprehensive problem definition and training strategy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 6 benchmarks, multiple search paradigms, SG ablation, RL data difficulty ablation, and manual behavioral analysis with a solid chain of evidence.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly structured with sufficient explanation of methods and results; however, there are minor typos in Level descriptions in Table 5 requiring context for understanding.
- **Value**: ⭐⭐⭐⭐⭐ Offers direct inspiration for search-augmented MLLMs and multimodal agents, particularly for open-world VQA requiring fine-grained visual evidence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Surprising Difficulty of Search in Model-Based Reinforcement Learning](../../ICML2026/reinforcement_learning/the_surprising_difficulty_of_search_in_model-based_reinforcement_learning.md)
- [\[NeurIPS 2025\] Note 5: ReSearch — Learning to Reason with Search](../../NeurIPS2025/reinforcement_learning/research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)
- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](../../NeurIPS2025/reinforcement_learning/incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[NeurIPS 2025\] Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/learning_to_focus_prioritizing_informative_histories_with_structured_attention_m.md)

</div>

<!-- RELATED:END -->
