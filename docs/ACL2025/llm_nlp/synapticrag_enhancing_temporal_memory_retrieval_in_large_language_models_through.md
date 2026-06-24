---
title: >-
  [Paper Note] SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms
description: >-
  [ACL2025][LLM (Other)][Memory Retrieval] Proposes SynapticRAG, which draws inspiration from synaptic transmission and the Leaky Integrate-and-Fire (LIF) model in neuroscience. By fusing temporal association triggers with semantic similarity, it achieves up to a 14.66% improvement over the state-of-the-art (SOTA) on conversational memory retrieval tasks.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Memory Retrieval"
  - "Temporal Association"
  - "Synaptic Mechanism"
  - "Conversational Systems"
  - "RAG"
date: 2026-05-08
content_hash: c4f6e86fd921b76c
---

# SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms

**Conference**: ACL2025  
**arXiv**: [2410.13553](https://arxiv.org/abs/2410.13553)  
**Code**: [tamoharu/SynapticRAG-Benchmark](https://github.com/tamoharu/SynapticRAG-Benchmark)  
**Area**: LLM/NLP  
**Keywords**: Memory Retrieval, Temporal Association, Synaptic Mechanism, Conversational Systems, RAG

## TL;DR

Proposes SynapticRAG, which draws inspiration from synaptic transmission and the Leaky Integrate-and-Fire (LIF) model in neuroscience. By fusing temporal association triggers with semantic similarity, it achieves up to a 14.66% improvement over the state-of-the-art (SOTA) on conversational memory retrieval tasks.

## Background & Motivation

**Insufficient Long-Term Memory in LLMs**: Large language models struggle to maintain coherent long-term memory in multi-turn conversation scenarios. Especially when the conversational context spans multiple sessions, the response relevance of current SOTA memory retrieval methods drops by approximately 17%.

**Existing RAG Relies Solely on Semantic Similarity**: Traditional RAG and its variants primarily retrieve knowledge based on vector cosine similarity, ignoring the temporal distribution dependencies in conversations, making it difficult to process complex contexts across different time periods.

**Inspiration from Cognitive Science**: Human memory retrieval relies on temporal triggering (Tulving, 1985) and synaptic connections (Hebb, 1949), rather than memory strength alone. Temporally close memories often constitute a coherent conversation chain.

**Limitations of Existing Temporal Enhancement Methods**: MemoryBank utilizes the Ebbinghaus forgetting curve, and MyAgent employs a memory consolidation model, but both fail to effectively capture complex temporal associations within dialogues.

**Lack of Suitable Evaluation Benchmarks**: Existing datasets (e.g., PerLTQA) only partially meet the requirements for evaluating temporal-aware dialogues, lacking specialized datasets that contain complex temporal dependencies and explicit annotations.

**Need for Cross-Lingual Generalization**: Conversational systems need to work effectively across multiple languages such as English, Chinese, and Japanese, requiring language-agnostic memory retrieval mechanisms.

## Method

### Overall Architecture

SynapticRAG comprises four core components: (1) memory construction and storage—encoding dialogue turns as vector nodes to build an attributed graph; (2) memory node selection—dual filtering based on semantic and temporal information; (3) stimulus propagation mechanism—simulating synaptic signal propagation on a hierarchical graph; and (4) stimulus activation—deciding whether nodes "fire" to be selected as retrieval results based on a dynamic LIF model.

### Key Designs

#### Key Design 1: Memory Construction and Temporal Association Calculation

- **Function**: Encodes each dialogue turn into a vector, storing them as memory nodes in a graph with edges weighted by cosine similarity. Each node maintains a weighted spike train $[(t_i, s_i)]_{i=1}^n$ to record its stimulus history.
- **Design Motivation**: Relying solely on semantic similarity fails to capture temporal dependencies in dialogues, necessitating supplementary information from the temporal dimension.
- **Mechanism**: Aligns the spike trains of two nodes using Dynamic Time Warping (DTW) to construct a distance matrix, which is then converted into an association intensity matrix $W(i,j) = \exp(-D(i,j)/\bar{\tau})$ via exponential decay. The temporal association score $T_{\text{score}}$ is obtained through DTW accumulation, which is finally multiplied by the cosine similarity $C_{\text{score}}$ to yield the propagation score $P_{\text{score}} = T_{\text{score}} \cdot C_{\text{score}}$.

#### Key Design 2: Stimulus Propagation Mechanism

- **Function**: Propagates stimulus signals layer by layer in a hierarchical graph to construct activation paths from input nodes to deep memory nodes.
- **Design Motivation**: Simulates the transmission process of synaptic signals in biological neural networks, allowing semantically and temporally associated memory nodes to be activated in a "chain reaction."
- **Mechanism**: Starting from the initial node ($S_{X_0}=1$), the stimulus intensity decays along the path as $S_{X_{i+1}} = P_{\text{score}}(X_i, X_{i+1}) \cdot S_{X_i}$. Each layer filters eligible parent nodes using a threshold $\text{stim}_{\text{th}}$ and determines candidate child nodes using $\cos_{\text{th}}$. When a child node receives stimuli from multiple parent nodes, the parent node providing the maximum stimulus intensity is selected. Propagation terminates when there are no eligible parent nodes.

#### Key Design 3: Dynamic Leaky Integrate-and-Fire (LIF) Activation

- **Function**: Introduces an enhanced LIF model that converts discrete stimuli into a continuous input current, determining which nodes are activated through the dynamic evolution of membrane potential.
- **Design Motivation**: The fixed time constant $\tau$ of the standard LIF model fails to balance recent inputs with long-term memory, requiring an adaptive mechanism.
- **Mechanism**: Input current generation: $\tau_X \frac{dI_X}{dt} = -I_X(t) + \sum_{t_s \in \Gamma} S_X(t)\delta(t-t_s)$; membrane potential evolution: $\tau_X \frac{dV_X}{dt} = -V_X(t) + I_X(t)$; dynamic time constant update: $\tau_X(t+\Delta t) = \tau_X(t) + \frac{1-\exp(-\Delta t)}{1+\exp(-\Delta t)}$ (approaching 0 under frequent inputs to maintain sensitivity, and approaching 1 under sparse inputs to retain long-term memory). When the membrane potential exceeds the threshold $V_{\text{th}}$, the node "fires" and is reset to the resting state post-firing.

### Loss & Training

This work does not involve end-to-end training. Instead, Optuna is used to optimize hyper-parameters ($\cos_{\text{th}}=0.262$, $\tau_{\text{init}}=43.07$, $V_{\text{th}}=0.099$, $\text{stim}_{\text{th}}=0.037$, $V_{\text{rest}}=2.903$, $I_{\text{rest}}=-7.128$). Text vectorization uses text-embedding-3-large, and generation uses GPT-4o.

## Key Experimental Results

### Main Results: Dynamic Retrieval Accuracy (ERC / ERC-MG)

| Method | SMRCs-EN | SMRCs-JA | PerLTQA-EN | PerLTQA-CN | Average |
|------|----------|----------|------------|------------|------|
| RAG | 71.55 | 81.58 | 82.14 | 84.95 | 80.15 |
| MemoryBank | 32.60 | 49.34 | 33.62 | 36.75 | 38.11 |
| MemoryBank(Adt) | 62.36 | 72.15 | 67.24 | 71.31 | 68.37 |
| MyAgent | 62.14 | 70.61 | 61.99 | 65.14 | 65.05 |
| MyAgent(Adt) | 63.24 | 77.19 | 66.49 | 69.54 | 69.19 |
| **SynapticRAG** | **86.21** | **92.32** | **90.47** | **93.12** | **90.53** |
| Gain% | +14.66 | +10.74 | +8.33 | +8.17 | +10.38 |

SynapticRAG substantially outperforms the strongest baselines across all four datasets, with an average improvement of 10.38% and a maximum improvement of 14.66% (SMRCs-EN). It maintains a 7.76%–14.66% advantage under the strictest ERC-MG metric as well.

### Ablation Study

| Ablation Variant | SMRCs-EN | SMRCs-JA | PerLTQA-EN | PerLTQA-CN | Average |
|---------|----------|----------|------------|------------|------|
| Full SynapticRAG | +14.66 | +10.74 | +8.33 | +8.17 | +10.41 |
| W/o Temporal Integration | -11.82 | -8.55 | -9.68 | -9.54 | -9.90 |
| W/o Stimulus Propagation | -8.31 | -2.41 | -3.60 | -3.36 | -4.35 |
| W/o Dynamic Time Constant | -14.66 | -10.96 | -8.29 | -8.24 | -10.54 |

Among the three components, the dynamic time constant has the greatest impact (average -10.54%), followed by temporal integration (-9.90%) and stimulus propagation (-4.35%).

## Highlights & Insights

1. **Interdisciplinary Innovation**: Systematically introduces concepts from neuroscience, such as synaptic propagation, the LIF model, and dynamic time constants, into dialogue memory retrieval. Unlike simple forgetting curves, it provides a richer, biologically inspired mechanism.
2. **Self-constructed High-Quality Dataset SMRCs**: Specifically builds a bilingual (English/Japanese) dialogue dataset containing complex temporal dependencies, filling a gap in evaluation benchmarks for this field, with a Krippendorff's Alpha of 0.782.
3. **Proposed Dynamic Evaluation Metrics ERC/ERC-MG**: Addresses real-world scenarios where different queries require varying amounts of memory by designing evaluation metrics with adaptive retrieval quantities, making them more aligned with practical applications.
4. **Cross-Lingual Consistency**: Achieves robust improvements across English, Chinese, and Japanese, verifying the language-agnostic nature of the method.

## Limitations & Future Work

1. **Risk of Over-Retrieval**: Synaptic propagation may lead to the activation of too many nodes, increasing the generation prompt length and computational overhead. Early stopping mechanisms and pruning strategies could be introduced.
2. **Computational Complexity of $O(n^2)$**: The cosine similarity filtering process leads to quadratic complexity, which is unfavorable for ultra-large-scale memory stores. The paper proposes an $O(n)$ centroid vector optimization scheme, but details are in the appendix.
3. **Sensitivity of Parameters**: The $\tau_{\text{scale}}$ parameter, in particular, heavily impacts performance, which drops sharply beyond the optimal value; different tasks may require independent hyperparameter tuning.
4. **Lack of Interpretability**: Synthetic propagation and firing mechanisms reduce interpretability compared to simple similarity-based methods.
5. **Limited Dataset Scale**: SMRCs contains only 101 dialogues / 456 tasks. It could be scaled up further to verify the robustness of the method.

## Related Work & Insights

- **RAG Series** (Lewis 2020, Borgeaud 2022): Provides a foundational retrieval paradigm based on semantic similarity, serving as the baseline starting point for this study.
- **MemoryBank** (Zhong 2023): Introduces the Ebbinghaus forgetting curve to adjust memory weights, but has limited temporal modeling capabilities.
- **MyAgent** (Hou 2024): Adheres to a cognitive-inspired memory recall model considering relevance, time, and recall probability.
- **HippoRAG** (Gutiérrez 2025): Also designs long-term memory for LLMs inspired by neuroscience (the hippocampus), complementing the focus of this work.
- **Insights**: The hierarchical graph propagation + LIF activation mechanism in this work can be migrated to other scenarios requiring temporal-aware retrieval (e.g., event extraction, timeline reasoning).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Systematically introduces synaptic propagation and dynamic LIF models into conversational memory retrieval, exhibiting a high level of interdisciplinary innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated on four datasets, five baselines, and three languages, along with complete ablation and parameter analyses, though the dataset scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ — Derivations of formulas are complete and clear, and the framework diagram is intuitive, though some symbol definitions could be more compact.
- **Value**: ⭐⭐⭐⭐ — Provides a new biologically inspired paradigm for conversational system memory management, offering practical reference value for temporal enhancement in RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)
- [\[ACL 2025\] TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues](tremu_towards_neuro-symbolic_temporal_reasoning_for_llm-agents_with_memory_in_mu.md)
- [\[ACL 2025\] ChronoSense: Exploring Temporal Understanding in Large Language Models with Time Intervals of Events](chronosense_exploring_temporal_understanding_in_large_language_models_with_time_.md)
- [\[ACL 2025\] Investigating Context-Faithfulness in Large Language Models: The Roles of Memory Strength and Evidence Style](investigating_context-faithfulness_in_large_language_models_the_roles_of_memory_.md)
- [\[ACL 2025\] Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction](idea_enhancing_the_rule_learning_ability_of_large_language_model_agent_through_i.md)

</div>

<!-- RELATED:END -->
