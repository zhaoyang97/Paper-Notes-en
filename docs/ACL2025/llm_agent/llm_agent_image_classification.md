---
title: >-
  [Paper Note] Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models
description: >-
  [ACL 2025][LLM Agent][concept bottleneck model] This paper proposes Conditional Concept Bottleneck Models (CoCoBMs) and an LLM-driven Concept Agent framework. By introducing a class-conditional concept scoring mechanism and dynamically refining the concept bank based on environmental feedback, the framework enhances classification accuracy by 6% while improving interpretability by approximately 30% across six datasets.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "concept bottleneck model"
  - "Interpretable Classification"
  - "Dynamic Concept Bank"
  - "CLIP"
date: 2026-05-08
content_hash: 0ff5b7702534a34d
---

# Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models

**Conference**: ACL 2025  
**arXiv**: [2506.01334](https://arxiv.org/abs/2506.01334)  
**Code**: None  
**Area**: LLM Agent  
**Keywords**: concept bottleneck model, LLM Agent, Interpretable Classification, Dynamic Concept Bank, CLIP

## TL;DR

This paper proposes Conditional Concept Bottleneck Models (CoCoBMs) and an LLM-driven Concept Agent framework. By introducing a class-conditional concept scoring mechanism and dynamically refining the concept bank based on environmental feedback, the framework enhances classification accuracy by 6% while improving interpretability by approximately 30% across six datasets.

## Background & Motivation

Concept Bottleneck Models (CBMs) decompose image classification into a decision-making process based on interpretable, human-readable concepts, serving as a representative approach in interpretable deep learning. Recently, CLIP-based CBMs have leveraged LLMs to generate candidate concepts, eliminating the need for manual concept bank construction and labeling. However, three critical issues remain unresolved:

**Uncertainty in the Number of Concepts**: What is the optimal number of concepts? LaBo uses 10,000 concepts for the CUB dataset, while LM4CV achieves similar performance with only 32. Too many concepts lead to redundancy, whereas too few result in insufficient coverage. This two-orders-of-magnitude difference highlights the lack of a systematic mechanism for optimizing the concept count in current methods.

**Limitations of the Shared Scoring Mechanism**: Traditional CBMs apply the same concept scoring (shared scoring) across all classes. However, the contribution of a single concept can vary dramatically across different classes (e.g., the concept "red feathers" has far higher discriminative value for "cardinal" than for "pigeon").

**Limitations of Concept Editing**: Existing methods only allow humans to manually edit concept scores at test time to correct errors, which is not scalable and fails to utilize the factual knowledge of LLMs to automatically correct inappropriately activated concepts.

## Method

### Overall Architecture

The framework consists of two core innovations:

1. **CoCoBMs**: Employs class-conditional concept scoring and weighting mechanisms.
2. **Concept Agent**: An LLM agent equipped with memory, planning, and action modules that dynamically optimizes the concept bank using environmental feedback.

### Key Designs

#### 1. Conditional Concept Bottleneck Models (CoCoBMs)

**Problem Analysis**: Traditional CBMs calculate concept scores as $\vec{s_c} = P(\vec{s_c} | x_i, \mathcal{C})$, where all classes share the exact same set of concept scores. CoCoBMs introduce class-conditional scoring:

$$\vec{s_c^j} = \|_{k=1}^{M} P(s_{c_k}^j | x_i, y_j, c_k)$$

Here, the score of each concept $c_k$ is conditioned on the hypothetical label $y_j$, forming a concept matrix of size $R^{N \times M}$ instead of a shared concept vector.

**Condition Learning**: A prompt learning strategy is adopted, where learnable conditional prompts are appended to the text input:

$$p_k^j = [t_1][t_2] \ldots [t_q][y_j][c_k]$$

where $[y_j]$ and $[c_k]$ are class name and concept name tokens, respectively. $t_i$ represents learnable vectors with the same dimension as CLIP word embeddings, which are shared across all labels and concepts to prevent information leakage.

**Editable Matrix**: An editable matrix $E$ is introduced to constrain concept activations that conflict with factual knowledge:

$$E_{jk} = \begin{cases} 1, & \text{if } c_k \notin y_j \\ 0, & \text{if } c_k \in y_j \end{cases}$$

When $E_{jk}=1$ (indicating that the concept is incompatible with the class), the score is forced to be truncated to non-positive values: $s_{c_k}^j = \min(s_{c_k}^j, 0)$.

#### 2. Concept Agent

The Agent contains three modules:

**Memory Module**:
- Maintains a generated concept list $M_g$, a deleted concept list $M_d$, and fact-verified concept-label pairs $M_f$.
- Stores the concept bank updated after each iteration.

**Action Module**:
- **Concept Generation**: Employs an LLM to generate candidate concepts for each class using the prompt template: "What are the helpful visual features to distinguish [CLS] from other [S-CLS]?"
- **Concept Selection**: Utilizes a learning-to-search method to select a fixed number of concepts from the candidate pool.
- **Fact Verification**: Uses an LLM to judge the relevance of each concept-label pair via multiple-choice questions (critical/occasionally/unrelated).
- **Instance Selection**: Selects representative samples using K-Means clustering to serve as a few-shot environment.
- **Environment Perception**: Interacts with the environment using CoCoBMs as a tool to obtain validation set feedback.

**Planning Module**:
- Analyzes the Score Activation Pattern of concepts: computes normalized contribution scores for each concept on the validation set.
- Calculates the binary activation pattern $P_{act}^c = [a_1^c, \ldots, a_N^c]$, where $a_j^c = 1$ when $\bar{s}_c^j > t_a$.
- **Redundant Concept Detection**: Identifies (1) concepts that do not contribute to any label, and (2) concepts that share the exact same activation pattern as other concepts but are less effective.
- **Insufficient Concept Detection**: Identifies (1) labels for which no concepts are activated, and (2) labels that share an identical concept set with other labels.
- Feeds back insufficiency information to the Action Module to guide new concept generation.

### Loss & Training

CoCoBMs use a weighted binary cross-entropy loss:

$$-\frac{1}{N} \sum_{j=1}^{N} \left[W_p y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)\right]$$

where the positive weight $W_p = N$ is utilized to compensate for label imbalance within each sample.

Training Strategy: The Agent iteratively refines the concept bank until all labels can be reliably identified and redundancies are eliminated. Finally, the CoCoBMs are trained on the full dataset to acquire the final performance.

## Key Experimental Results

### Main Results

**Classification accuracy comparison across 6 datasets (CLIP ViT-B/32 backbone)**:

The number of concepts determined by this method is approximately equal to the number of labels, allowing for a fair comparison against LaBo-n and LM4CV-n:

- Compared to LaBo-n: Average improvement of **6.15%**
- Compared to LaBo-3n (with 3x concepts): Still **0.51%** higher
- Compared to LM4CV-n: Improvement of **5.97%**
- Compared to LM4CV-2n: Improvement of **3.21%**
- Compared to LF-CBM (which uses 16x concepts on CIFAR-10, for example): **1.36%** higher
- Gap with black-box models: Only 3.45% lower than linear probing, and 2.44% lower than prompt learning

**Interpretability Evaluation**:
- Overall interpretability score: **77.46%**
- Truthfulness: **81.59%**
- Distinguishability: **73.34%**
- An improvement of roughly **30%** over LM4CV-2n.

### Key Findings

1. **Dynamic vs. Static Grounding** (Table 1):
    - CIFAR-100: Accuracy 72.67% $\rightarrow$ 74.63%; Interpretability 67.90% $\rightarrow$ **79.30%**
    - Flower: Accuracy 87.45% $\rightarrow$ 89.51%; Interpretability 70.59% $\rightarrow$ **82.35%**
    - Dynamic grounding improves average interpretability by **10.76%**.

2. **Role of the Editable Matrix** (Table 2):
    - Without $E$, the interpretability of CIFAR-100 plummets from 79.30% to **39.60%**.
    - For Flower, it drops from 82.35% to **35.59%**.
    - While $E$ slightly sacrifices accuracy (~0.1-2%), it drastically improves interpretability, highlighting that factual constraints are crucial for interpretability.

3. **Robustness in Few-shot Environments**: As the number of samples increases, accuracy improves while interpretability remains stable, demonstrating that the model maintains strong interpretability even with limited data.

4. **Case Study (CIFAR-10)**: The Agent refines the concept bank to only 9 concepts after 4 iterations. During this process, it successfully identifies invalid concepts caused by dataset bias or CLIP pre-training bias, as well as redundant concepts merged due to identical activation patterns.

## Highlights & Insights

1. **Class-Conditional Scoring**: This breaks the assumption in traditional CBMs that concept scores must be shared. Through prompt learning, class-specific concept evaluation is elegantly realized, while the original CBM remains a special case (concept vectors can be recovered by folding along the label dimension).
2. **Agent-driven Dynamic Concept Bank**: It transforms concept bank construction from a static, one-off process into an iterative, feedback-based optimization process, marking the first such attempt in the field of concept bottleneck models.
3. **Quantitative Evaluation of Interpretability**: LLM-based metrics for truthfulness and distinguishability are proposed, filling a gap in the CBM literature regarding the lack of quantitative interpretability assessments.
4. **Random Words Can Also Classify**: An important experimental finding reveals that a concept bank consisting of 512 random words can still achieve decent classification accuracy. This underscores the necessity of interpretability evaluations rather than relying solely on accuracy.

## Limitations & Future Work

1. **Scalability of Fact Verification**: Since all possible concept-class pairs require verification, the computational cost escalates quickly as the number of classes and concepts grows.
2. **Uncertainty of LLM Internal Knowledge**: Concept generation relies on the knowledge base of LLMs, where inherent biases and randomness may affect concept quality.
3. **CLIP Dependency**: Visual perception depends entirely on the alignment capability of CLIP, and pre-training biases in CLIP may prevent certain visual features from being effectively detected.
4. **Evaluation Limited to Classification**: Future work is needed to explore extensions with more complex visual tasks like object detection and segmentation.

## Related Work & Insights

- **Evolution of CBMs**: Progressing from the original CBM by Koh et al. (2020) to Label-free CBMs (Oikarinen et al., 2023), LaBo (Yang et al., 2023), and LM4CV (Yan et al., 2023a), concept bank construction has steadily advanced from manual labeling to LLM generation.
- **LLM Agent Architectures**: Adhering to the classical Memory-Planning-Action framework (Yao et al., 2023), this work introduces the Agent paradigm into visual understanding tasks.
- **Prompt Learning**: Leverages learnable prompt strategies from CoOp (Zhou et al., 2022) to implement conditional scoring.
- **Insights**: Coupling the planning capabilities of LLM Agents with domain-specific model tooling is a highly generalizable paradigm — where the Agent does not solve visual tasks directly, but instead optimizes the tools used to solve them.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4.5 |
| Technical Depth | 4.5 |
| Experimental Thoroughness | 4.5 |
| Practical Value | 3.5 |
| Writing Quality | 4 |
| Overall Rating | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)
- [\[ACL 2025\] Enhancing LLM Agent Safety via Causal Influence Prompting](enhancing_llm_agent_safety_via_causal_influence_prompting.md)
- [\[ICCV 2025\] Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions](../../ICCV2025/llm_agent/embodied_image_captioning_self-supervised_learning_agents_for_spatially_coherent.md)
- [\[ICLR 2026\] GUI-Shift: Enhancing VLM-Based GUI Agents through Self-supervised Reinforcement Learning](../../ICLR2026/llm_agent/gui-shift_enhancing_vlm-based_gui_agents_through_self-supervised_reinforcement_l.md)
- [\[NeurIPS 2025\] Crucible: Quantifying the Potential of Control Algorithms through LLM Agents](../../NeurIPS2025/llm_agent/crucible_quantifying_the_potential_of_control_algorithms_through_llm_agents.md)

</div>

<!-- RELATED:END -->
