---
title: >-
  [Paper Note] Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework
description: >-
  [ACL2025][Interpretability][Temporal Reasoning] Proposes the GETER framework, which injects temporal knowledge graph structural information into LLMs via a lightweight Structure-Text Adapter, enabling the model to deliver both accurate predictions and explainable reasoning explanations in temporal reasoning tasks.
tags:
  - "ACL2025"
  - "Interpretability"
  - "Temporal Reasoning"
  - "Explainability"
  - "Temporal Knowledge Graphs"
  - "Graph Structure and Text Alignment"
  - "Instruction Tuning"
date: 2026-05-08
content_hash: 89c2f436317277a8
---

# Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework

**Conference**: ACL2025  
**arXiv**: [2505.15245](https://arxiv.org/abs/2505.15245)  
**Code**: [carryTatum/GETER](https://github.com/carryTatum/GETER)  
**Area**: Explainability  
**Keywords**: Temporal Reasoning, Explainability, Temporal Knowledge Graphs, Graph Structure and Text Alignment, Instruction Tuning

## TL;DR

Proposes the GETER framework, which injects temporal knowledge graph structural information into LLMs via a lightweight Structure-Text Adapter, enabling the model to deliver both accurate predictions and explainable reasoning explanations in temporal reasoning tasks.

## Background & Motivation

**Background**: Temporal Reasoning is a core capability in NLP, which is crucial in scenarios such as search recommendation, news aggregation, and other applications. In recent years, LLMs have achieved significant progress in temporal reasoning, with various works improving accuracy through ICL, CoT, finetuning, and other methods.

**Limitations of Prior Work**: Almost all existing works "focus only on performance and ignore explainability"—LLMs provide prediction results but fail to explain their reasoning process, lacking transparency and trustworthiness.

**Key Challenge**: When relying solely on textual information, LLMs often suffer from hallucination, making it difficult to generate convincing temporal reasoning explanations. Conversely, traditional explainable methods (such as logical rules and reinforcement learning paths) offer limited explanation capability and poor generalization.

**Goal**: How to enable LLMs to make both accurate predictions and clearly present the reasoning process in complex temporal reasoning scenarios?

**Key Insight**: Leveraging the structured information of Temporal Knowledge Graphs (TKGs) to compensate for the limitations of pure text reasoning, enhancing the explainable temporal reasoning capability of LLMs through graph structure-text alignment.

**Core Idea**: Utilizing a temporal encoder to encode TKG structural information into soft graph tokens, which are projected into the text token space of the LLM via a lightweight adapter, and then concatenated with instruction-tuning prompts to generate explanation texts.

## Method

### Overall Architecture

GETER (Graph structures with text for Explainable TEmporal Reasoning) consists of three core modules:
1. **Temporal Encoder**: Employs TKG models such as RE-GCN to learn structural representations of entities and relations on temporal knowledge graphs.
2. **Structure-Text Prefix Adapter**: Projects graph structural representations into the text embedding space of the LLM.
3. **Instruction Tuning**: Fine-tunes the LLM using LoRA to generate explanation texts via the combination of soft graph tokens and prompt tokens.

### Key Designs

### Key Design 1: ETR Benchmark Construction

- **Function**: Constructs an explainable temporal reasoning benchmark covering multiple temporal granularities (minutes/days/years).
- **Design Motivation**: Existing benchmarks do not evaluate explanation quality, and lack a comprehensive consideration of positive, negative, and neutral samples.
- **Mechanism**:
    - Extracts reasoning chains from TKGs using Breadth-First Search (BFS) and converts them into natural language;
    - Uses GPT-4o to generate high-quality explanation texts based on the query and the reasoning chains;
    - Constructs negative samples (counterfactuals) via entity substitution, and filters semantic neutral relations using Natural Language Inference (NLI) models to construct neutral samples;
    - Covers five datasets: ICEWS14, ICEWS05-15, ICEWS18, GDELT, and WIKI, totaling ~60k training samples and ~9k test samples.

### Key Design 2: Structure-Text Adapter

- **Function**: Integrates query and reasoning chain graph structural representations and projects them into the LLM embedding space.
- **Design Motivation**: Relying solely on text prevents the LLM from capturing structural temporal patterns between events in the TKG.
- **Mechanism**:
    - Concatenates and sums the structural embeddings of all triples $(e_s', r', e_o')$ in the reasoning chain: $S_{C} = \sum (e_s' \| r' \| e_o')$;
    - Averages this with the query structural representation $S_q$, then projects it into the LLM space via a linear projection matrix $W_p \in \mathbb{R}^{3d_s \times d_x}$;
    - Obtains a single soft graph token $S_{graph}$, which is pre-pended to the text embeddings as a prefix.

### Key Design 3: Instruction Tuning

- **Function**: Trains the LLM to generate explanations guided jointly by the soft graph token and text information.
- **Design Motivation**: It is necessary to organically combine structural information with semantic information while controlling fine-tuning overhead.
- **Mechanism**: The final input is $X' = S_{graph} \| X$, with the optimization objective to maximize the likelihood of the explanation text $Y_A$. Parameter-efficient fine-tuning is performed using LoRA.

### Loss & Training

- Standard autoregressive language modeling loss: $P(Y_A | X', X_I) = \prod_{j=1}^{L} P_\theta(y_j | X', X_I, Y_{<j})$
- Parameter-efficient fine-tuning via LoRA, with the temporal encoder being frozen after pre-training.
- Utilizes DeepSpeed to accelerate training and inference.

## Key Experimental Results

### Main Results: Prediction F1 (%)

| Model | ICEWS14 Overall | GDELT Overall | ICEWS05-15 Overall |
|------|:-:|:-:|:-:|
| GPT-4o zero-shot | 39.95 | 36.83 | 40.58 |
| Llama3-8B LoRA | 65.59 | 56.44 | 65.86 |
| **GETER (Llama3)** | **74.25** | **72.51** | **81.84** |
| Qwen2.5-7B LoRA | 71.90 | 46.95 | 73.72 |
| **GETER (Qwen2.5)** | **78.12** | **73.27** | **80.23** |
| Mistral-7B LoRA | 71.18 | 65.05 | 76.07 |
| **GETER (Mistral)** | **79.08** | **72.02** | **81.80** |

GETER improves Overall F1 by 7~28% compared to LoRA-only, and by approximately 100% compared to GPT-4o zero-shot.

### Ablation Study (Mistral, Overall F1 %)

| Variant | ICEWS14 | GDELT | ICEWS05-15 |
|------|:-:|:-:|:-:|
| GETER (full) | 79.08 | 72.02 | 81.80 |
| w/o Structure-Text Adapter | 71.18 (↓7.90) | 65.05 (↓6.97) | 76.07 (↓5.73) |
| w/o Reasoning Chains Text | 72.05 (↓7.03) | 68.89 (↓3.13) | 77.82 (↓3.98) |
| w/o Both | 66.79 (↓12.29) | 47.80 (↓24.22) | 61.95 (↓19.85) |

### Key Findings

1. **Structural information is crucial**: Removing the Structure-Text Adapter leads to a 5.7~7.9% drop in F1, indicating that graph structural features have a unique value for temporal reasoning modeling.
2. **Reasoning chain text is also indispensable**: Reasoning chain texts provide temporal contextual background, and removing them results in a distinct performance drop.
3. **The two are complementary**: Removing both components simultaneously (Line 4) causes a sharp plunge in performance (-24.22% on GDELT), proving that the integration of structure + text is key to the success of GETER.
4. **Robust to temporal encoders**: Replacing the temporal encoder with different encoders like CEN/CENET/SiMFy, GETER still significantly outperforms LoRA-only.
5. **Little impact of MLP depth**: A 1-layer MLP is sufficient, with deeper networks yielding no additional benefits.
6. **Reasoning chain order**: Descending order achieves the best performance (80.68%), but even random ordering maintains 77.57%, demonstrating the robustness of the framework.

## Highlights & Insights

1. **First systematic study on explainable temporal reasoning in LLMs**: It not only requires correct predictions but also demands reasonable explanations, filling a gap in this research direction.
2. **Delicate design of the ETR benchmark**: Covers positive, negative, and neutral samples across multiple temporal granularities, providing a comprehensive evaluation dimension.
3. **Lightweight alignment scheme**: Bridges the graph structure to the text space using only a single linear projection, proving that simple methods remain effective in cross-modal alignment.
4. **Practical significance**: Explainable temporal reasoning has direct application values in scenarios such as news analysis, financial forecasting, and event early-warning.

## Limitations & Future Work

1. **Computational overhead**: Fine-tuning and inference of LLMs remain resource-intensive, despite using LoRA and DeepSpeed.
2. **Noise in reasoning chains**: Some reasoning chains extracted via BFS may contain noise, which affects explanation quality.
3. **GPT-4o generated ground-truth explanations**: The "gold standard" explanations of the benchmark are generated by GPT-4o rather than human annotation, which may introduce systematic bias.
4. **Limitations in temporal granularity**: Although to-date coverage includes minutes/days/years, it does not involve finer granularities (seconds) or more complex temporal reasoning (interval reasoning, periodic reasoning).
5. **Soft graph token as a single vector**: Compressing all graph structural information into a single token might cause an information bottleneck for complex reasoning chains.

## Related Work & Insights

### vs. Logical Rule Methods (e.g., TLogic)
Logical rule methods guarantee explainability through explicit rule templates, but suffer from poor generalization and struggle with complex scenarios. Leveraging the generative capability of LLMs, GETER produces more flexible and natural reasoning explanations while retaining the precision of graph structures.

### vs. Reinforcement Learning Path Methods (e.g., CluSTeR, TITer)
RL-based methods construct reasoning paths through predefined reward mechanisms, but their decision processes are implicit, leading to limited explainability. GETER directly generates natural language explanations, which are far more readable than RL path representations.

### vs. Pure Text LLM Methods (e.g., CoT, ICL)
Pure-text methods ignore the structured information of TKGs, making them prone to hallucinations. By introducing a structural prior, GETER effectively reduces hallucinations and improves explanation quality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — For the first time injecting graph structural information into LLMs to enhance explainable temporal reasoning, presenting dual contributions in both benchmark and framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experiments covering 5 datasets, 4 LLM backbones, and detailed ablation and discussion.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich diagrams, and sufficient motivation elicitation.
- **Value**: ⭐⭐⭐⭐ — Explainable temporal reasoning is an important and under-explored direction, and both the benchmark and the framework hold value for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](../../ACL2026/interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](../../ACL2026/interpretability/preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)
- [\[ACL 2025\] Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](../../NeurIPS2025/interpretability/table_as_a_modality_for_large_language_models.md)

</div>

<!-- RELATED:END -->
