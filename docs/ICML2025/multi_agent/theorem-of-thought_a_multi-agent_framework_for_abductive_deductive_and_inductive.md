---
title: >-
  [Paper Note] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models
description: >-
  [ICML 2025][Multi-Agent][Multi-Agent Reasoning] The Theorem-of-Thought (ToTh) framework is proposed, where three agents simulating abductive, deductive, and inductive reasoning independently generate reasoning trajectories. These trajectories are constructed into a Formal Reasoning Graph (FRG) and consistently scored using NLI-calibrated Bayesian belief propagation. The terminal node of the highest-scoring graph is selected as the final answer, consistently outperforming CoT…
tags:
  - "ICML 2025"
  - "Multi-Agent"
  - "Multi-Agent Reasoning"
  - "Abductive Reasoning"
  - "Deductive Reasoning"
  - "Inductive Reasoning"
  - "Bayesian Belief Propagation"
  - "Reasoning Graph"
date: 2026-05-08
content_hash: 6c45348c111ceb3d
---

# Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models

**Conference**: ICML 2025  
**arXiv**: [2506.07106](https://arxiv.org/abs/2506.07106)  
**Code**: [https://github.com/KurbanIntelligenceLab/theorem-of-thought](https://github.com/KurbanIntelligenceLab/theorem-of-thought)  
**Area**: LLM Agent  
**Keywords**: Multi-Agent Reasoning, Abductive Reasoning, Deductive Reasoning, Inductive Reasoning, Bayesian Belief Propagation, Reasoning Graph

## TL;DR

The Theorem-of-Thought (ToTh) framework is proposed, where three agents simulating abductive, deductive, and inductive reasoning independently generate reasoning trajectories. These trajectories are constructed into a Formal Reasoning Graph (FRG) and consistently scored using NLI-calibrated Bayesian belief propagation. The terminal node of the highest-scoring graph is selected as the final answer, consistently outperforming CoT, Self-Consistency, and CoT-Decoding on symbolic and numerical reasoning tasks.

## Background & Motivation

1. **Background**: LLMs have made significant progress in reasoning tasks through technologies such as CoT prompting, Self-Consistency, and CoT-Decoding. CoT guides models to output intermediate reasoning steps, Self-Consistency improves robustness via voting over multiple samples, and CoT-Decoding utilizes diverse decoding paths to activate latent reasoning capabilities.
2. **Limitations of Prior Work**: These methods inherently operate along a linear reasoning path and lack mechanisms to **verify internal logical consistency**. Although the model outputs appear fluent and plausible, they can be logically self-inconsistent, leading to hallucinations, logical contradictions, and weak generalization capabilities. Simple majority voting in Self-Consistency performs particularly poorly on logic-intensive tasks because random sampling fails to capture structured dependencies.
3. **Key Challenge**: Human reasoning naturally integrates three complementary modes—abduction (explaining observations), deduction (deriving from premises), and induction (generalizing from samples). However, LLMs conflate these distinct reasoning processes into an undifferentiated stream, limiting interpretability and reliability.
4. **Goal**: How can LLMs be enabled to distinguish between different reasoning paradigms like humans and select the most reliable reasoning chain through formal consistency checking?
5. **Key Insight**: Drawing inspiration from the triarchic reasoning framework of abduction-deduction-induction in cognitive science, reasoning is decomposed into three independent agents. Each agent generates a structured reasoning graph instead of linear text, followed by verifiable consistency evaluation using belief propagation in probabilistic graphical models.
6. **Core Idea**: Replace a single linear CoT with three cognitive paradigm agents and NLI-calibrated Bayesian graph propagation to construct verifiable structured reasoning.

## Method

### Overall Architecture

The reasoning workflow of ToTh is divided into five phases:

1. **Multi-Paradigm Reasoning Agent Generation**: Given a query $q$, three independent agents (abductive, deductive, and inductive) each generate a reasoning trajectory.
2. **Formal Reasoning Graph Construction (FRG)**: Convert each reasoning trajectory into a directed graph, where nodes represent reasoning steps and edges represent logical dependencies between those steps.
3. **Bayesian Belief Propagation**: Perform belief propagation on the graph to propagate and aggregate the logical credibility of each node.
4. **Graph Scoring**: Score each graph comprehensively based on average confidence and logical entropy.
5. **Answer Extraction**: Extract the final answer from the terminal node of the highest-scoring graph.

### Key Designs

1. **Multi-Paradigm Reasoning Agents**
    - **Function**: Three agents independently generate reasoning trajectories using distinct cognitive reasoning paradigms.
    - **Mechanism**:
     - Abductive Agent $a_1$: Given observation $O$ and background knowledge $K$, infers the most likely hypothesis $H$, i.e., $\arg\max_H P(H|O,K)$.
     - Deductive Agent $a_2$: Derives conclusion $C$ from a set of premises $\{P_1,...,P_n\}$, i.e., $\{P_i\} \vdash C$.
     - Inductive Agent $a_3$: Generalizes rule $R$ from observed samples $\{x_1,...,x_n\}$, i.e., $\{x_i\} \Rightarrow R$.
    - Each agent independently produces a reasoning trajectory $\mathbf{r}^{(i)} = [r_1^{(i)}, r_2^{(i)}, ..., r_{s_i}^{(i)}]$.
    - **Design Motivation**: Human reasoning inherently blends these three modes, and a single reasoning mode is prone to failure on certain question types. By specializing the three paradigms, the framework can locate the most suitable reasoning path across diverse questions.

2. **Formal Reasoning Graph Construction**
    - **Function**: Converts each reasoning trajectory $\mathbf{r}^{(i)}$ into a directed graph $G^{(i)} = (V^{(i)}, E^{(i)})$.
    - **Mechanism**: Nodes $V$ represent reasoning steps. A pre-trained NLI model (RoBERTa-MNLI) evaluates the semantic relations between steps to generate weighted edges. The confidence score $\theta_{uv}$ of each edge is assigned based on the NLI predicted labels:
     - Entailment $\rightarrow \theta = 0.95$
     - Neutral $\rightarrow \theta = 0.60$
     - Contradiction $\rightarrow \theta = 0.10$
    - **Design Motivation**: Traditional CoT only outputs linear text, failing to quantify the logical strength between steps. Introducing an NLI model provides an external, calibrated assessment of logical consistency, converting "subjective plausibility" into computable confidence scores.

3. **Bayesian Confidence Propagation**
    - **Function**: Propagates confidence across the reasoning graph, amplifying beliefs along consistent paths and dampening them at contradictions.
    - **Mechanism**: The initial prior for each node is $P(v) = 0.5$ (maximum uncertainty). For a child node $v_c$ with a single parent $v_p$, the Bayesian update is:
     $$P(v_c) = \frac{P(v_p) \cdot \theta_{pc}}{P(v_p) \cdot \theta_{pc} + (1 - P(v_p)) \cdot (1 - \theta_{pc})}$$
     For a child node with multiple parents $\{v_{p_1},...,v_{p_m}\}$, the average of the updates from each parent is taken:
     $$P(v_c) = \frac{1}{m} \sum_{j=1}^m f(P(v_{p_j}), \theta_{p_j c})$$
    - **Design Motivation**: Drawing inspiration from Pearl's classic belief propagation in probabilistic graphical models, the consistency verification of reasoning chains is formalized. Consistent reasoning paths accumulate high confidence, whereas inconsistent paths are naturally dampened, without requiring manually set thresholds.

4. **Graph Scoring & Answer Extraction**
    - **Function**: Computes a comprehensive score for each reasoning graph and selects the best graph to extract the answer.
    - **Mechanism**:
     - Average confidence: $\mu^{(i)} = \frac{1}{|V^{(i)}|} \sum_{v} P(v)$
     - Normalized binary entropy: $H^{(i)} = -\frac{1}{|V^{(i)}|} \sum_v [p \log p + (1-p) \log(1-p)]$
     - Comprehensive score: $\text{Score}(G^{(i)}) = \mu^{(i)} - H^{(i)}$
     - Selecting the highest-scoring graph: $G^* = \arg\max_i \text{Score}(G^{(i)})$
    - **Design Motivation**: Relying solely on high confidence might select overconfident but incorrect chains; incorporating an entropy penalty prioritizes reasoning graphs that are both highly credible and have low uncertainty.

### Computational Complexity

The end-to-end complexity of ToTh is $O(k \cdot s)$ ($k=3$ agents, where $s$ is the number of reasoning steps per agent), which is linear with respect to the number of agents and steps. Compared to the $O(n)$ ($n=20$ samples) of Self-Consistency, ToTh requires only a single reasoning pass per agent along with lightweight NLI verification and scoring, making it much more efficient and scalable.

## Key Experimental Results

### Main Results

Evaluated on WebOfLies (symbolic logical reasoning) and MultiArith (numerical reasoning) benchmarks using three open-source models (Mistral-7B-v0.3, DeepSeek-7B, Phi-3.5-mini):

| Method | Mistral-7B WebOfLies | Mistral-7B MultiArith | DeepSeek-7B WebOfLies | DeepSeek-7B MultiArith | Phi-3.5 WebOfLies | Phi-3.5 MultiArith |
|------|---------------------|----------------------|----------------------|----------------------|-------------------|-------------------|
| CoT-Greedy | Low (~40%) | Mid | Low | Low | High | High |
| Self-Consistency | Very Low (~21%) | Low | Very Low (~14%) | Low | Mid | Mid |
| CoT-Decoding | Mid | Mid | Mid | Mid | **~99%** | High |
| **ToTh** | **~69%** (+29%) | **Highest** | **Highest** (+14%) | **Highest** | ~96% | High |

Key findings: ToTh **consistently outperforms all baselines** on Mistral-7B and DeepSeek-7B. On Phi-3.5 Mini, CoT-Decoding performs marginally better on WebOfLies (99% vs 96%), but ToTh surpasses CoT-Decoding by 4–5 percentage points on MultiArith.

### Robustness Experiment (Mistral-7B, Different Difficulty Levels)

| Method | WoL 3 sent. | WoL 4 sent. | WoL 5 sent. | MA d0/l3 | MA d0/l4 | MA d2/l3 |
|------|---------|---------|---------|----------|----------|----------|
| CoT-Greedy | 41 | 32 | 19 | 57 | 26 | 14 |
| Self-Consistency | 48 | 47 | 38 | 21 | 6 | 17 |
| CoT-Decoding | 54 | 48 | 46 | 55 | 41 | 24 |
| **ToTh** | **70** | **56** | **43** | **59** | **45** | 21 |

ToTh achieves the best results in 5 out of 6 settings. In the most challenging 5-sentence symbolic reasoning setup, ToTh (43%) significantly outperforms CoT-Greedy (19%) and comes close to CoT-Decoding (46%).

### Key Findings

- **Self-Consistency Fails Comprehensively**: It performs the worst across all settings, especially on symbolic reasoning tasks, achieving only 14-21%. Majority voting fails to capture structured logical dependencies, and random sampling is counterproductive in tasks requiring exact logic.
- **Discrepancies in Model Capabilities**: Phi-3.5 Mini achieves the highest absolute score (benefiting from target training towards educational scenarios), but the performance gain of ToTh is more pronounced in weaker models, indicating that the framework compensates for deficiencies in the models' inherent reasoning capabilities.
- **Mistral-7B Outperforms DeepSeek-7B**: Despite similar parameter scales, Mistral performs better in structured reasoning, which can likely be attributed to cleaner, reasoning-oriented pre-training data.
- **More Pronounced Advantage at Higher Difficulties**: While the gap among different methods is small on simple problems, ToTh's advantage in structured reasoning becomes prominent on highly complex problems (e.g., 5-sentence logic, deep arithmetic).

## Highlights & Insights

- **Elegant Integration of Cognitive Science and Reasoning Framework**: Materializing abduction, deduction, and induction—three classic cognitive reasoning modes—into three distinct agents is both theoretically grounded and highly implementable. This approach can be transferred to other tasks requiring multi-perspective analysis (e.g., debate generation, legal reasoning).
- **Clever Application of NLI as 'Logical Glue'**: Utilizing an off-the-shelf NLI model (RoBERTa-MNLI) to quantify the logical consistency between reasoning steps requires no additional training and works out-of-the-box. This trick can be directly applied to any scenario that requires verifying the quality of reasoning chains.
- **Comprehensive Confidence-Entropy Scoring**: Factoring in uncertainty rather than inspecting average confidence alone is highly systematic compared to simple majority voting, preventing the selection of "overconfident but incorrect" reasoning chains.
- **Training-Free Inference Enhancement**: The entire framework operates during inference, requiring no extra training data or fine-tuning, making it applicable to any existing LLM.

## Limitations & Future Work

- **Fixed Reasoning Types**: Uniformly applying the abductive, deductive, and inductive agents to all inputs, whereas certain tasks might only require one or a hybrid mixture of reasoning styles. Future work could dynamically route and activate corresponding agents based on the query type.
- **Sensitivity of Propagation**: Bayesian belief propagation is sensitive to noise on low-confidence nodes; errors in earlier reasoning steps propagate and amplify disproportionately in deeper graphs. Edge dropout or confidence smoothing could be introduced to mitigate this.
- **Limited Evaluation Scope**: Only evaluated on two benchmarks (WebOfLies and MultiArith), missing broader reasoning types such as commonsense reasoning and multi-hop QA.
- **Hardcoded Confidence Thresholds for NLI** ($0.95$/$0.60$/$0.10$), lacking end-to-end learning or adaptive tuning mechanisms.
- **Lack of Interaction Between Agents**: The three agents reason completely independently without any collaboration or information exchange. Introducing discussion/refinement loops among agents could further boost quality.

## Related Work & Insights

- **vs. Chain-of-Thought (CoT)**: CoT is a linear, single-chain reasoning method lacking logical consistency verification; ToTh utilizes graph structures and Bayesian propagation to explicitly verify the logical strength of each step, providing explainable quality assurance.
- **vs. Self-Consistency**: Self-Consistency relies on multiple sampling and majority voting, essentially being a statistical approach; ToTh replaces voting with structured scoring, demonstrating clear advantages on logic-intensive tasks ($14\%$ vs. $69\%$ on WebOfLies).
- **vs. Tree-of-Thought / Graph-of-Thought**: ToT/GoT manipulate reasoning structures but do not distinguish reasoning types, and they lack formal consistency scoring; ToTh simultaneously introduces both diverse reasoning paradigms and Bayesian evaluation.
- **vs. CoT-Decoding**: CoT-Decoding activates reasoning via diverse decoding paths, showing strong performance on well-aligned models (such as Phi-3.5); ToTh demonstrates a larger advantage on weaker models, suggesting that structured reasoning provides a stronger compensatory effect for model capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ Integrating cognitive triarchic reasoning with NLI-calibrated Bayesian graph propagation is novel, though individual components (multi-agent, NLI, Bayesian propagation) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐ Evaluated only on two benchmarks and three models; lacks ablation studies and broader evaluation across general reasoning tasks.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with complete formal definitions and rigorous descriptions of the methodology.
- Value: ⭐⭐⭐⭐ Provides a training-free solution for structured reasoning enhancement with highly transferable ideas, though the limited evaluation scale weakens its persuasiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](../../AAAI2026/multi_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](../../NeurIPS2025/multi_agent/large_language_models_miss_the_multi-agent_mark.md)
- [\[AAAI 2026\] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](../../AAAI2026/multi_agent/liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)
- [\[ICLR 2026\] Emergent Coordination in Multi-Agent Language Models](../../ICLR2026/multi_agent/emergent_coordination_in_multi-agent_language_models.md)
- [\[NeurIPS 2025\] Thought Communication in Multiagent Collaboration](../../NeurIPS2025/multi_agent/thought_communication_in_multiagent_collaboration.md)

</div>

<!-- RELATED:END -->
