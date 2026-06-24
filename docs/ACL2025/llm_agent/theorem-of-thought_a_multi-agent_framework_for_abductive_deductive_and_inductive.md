---
title: >-
  [Paper Note] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models
description: >-
  [ACL 2025][LLM Agent][Multi-Agent Reasoning] This paper proposes the Theorem-of-Thought (ToTh) framework, which models abductive, deductive, and inductive reasoning using three parallel agents. It constructs reasoning trajectories as Formal Reasoning Graphs and employs NLI-calibrated Bayesian belief propagation to select the most coherent reasoning chain, consistently outperforming CoT, Self-Consistency, and CoT-Decoding on symbolic and numerical reasoning.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Multi-Agent Reasoning"
  - "Abductive Reasoning"
  - "Deductive Reasoning"
  - "Inductive Reasoning"
  - "Bayesian Belief Propagation"
date: 2026-05-08
content_hash: d17cf9e2b1a0e6fd
---

# Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.07106](https://arxiv.org/abs/2506.07106)  
**Code**: [Available](https://github.com/KurbanIntelligenceLab/theorem-of-thought)  
**Area**: LLM Agent / Reasoning  
**Keywords**: Multi-Agent Reasoning, Abductive Reasoning, Deductive Reasoning, Inductive Reasoning, Bayesian Belief Propagation

## TL;DR

This paper proposes the Theorem-of-Thought (ToTh) framework, which models abductive, deductive, and inductive reasoning using three parallel agents. It constructs reasoning trajectories as Formal Reasoning Graphs and employs NLI-calibrated Bayesian belief propagation to select the most coherent reasoning chain, consistently outperforming CoT, Self-Consistency, and CoT-Decoding on symbolic and numerical reasoning.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Although LLM reasoning has progressed due to techniques like CoT, it still faces three core challenges:

**Reasoning Vulnerability**: Reasoning along a linear path is prone to hallucinations and logical inconsistencies.

**Lack of Logical Structural Verification**: CoT and Self-Consistency only encourage intermediate steps or majority voting, lacking mechanisms to verify internal coherence.

**Gap with Human Reasoning**: Human reasoning naturally integrates three modes—abductive (explanation), deductive (derivation), and inductive (generalization)—whereas LLMs conflate them into a single indistinguishable process.

Inspired by cognitive science, the authors argue that decomposing reasoning into three classical paradigms, followed by selecting the optimal path via structured verification, can improve both accuracy and explainability. This motivation is theoretically compelling—existing methods either perform single-path reasoning (CoT) or unstructured multi-path sampling (Self-Consistency), whereas ToTh systematically integrates "diversity of reasoning paradigms" with "logical verification."

## Method

### Overall Architecture

ToTh consists of four stages: (1) Three independent agents reason using abductive, deductive, and inductive approaches, respectively; (2) Reasoning trajectories are converted into a Formal Reasoning Graph (FRG); (3) Coherence is evaluated via NLI-calibrated Bayesian belief propagation; (4) The optimal graph is selected based on a comprehensive score, and the final answer is extracted from the terminal node.

### Key Designs

1. **Three Reasoning Agents**:

    - **Abductive Agent $a_1$**: Given observation $O$ and background knowledge $K$, it infers the most likely hypothesis $H = \arg\max_H P(H|O,K)$. The Mechanism is "finding causes from effects," suitable for tasks requiring explanations of observed phenomena.
    - **Deductive Agent $a_2$**: Derives conclusion $C$ from a set of premises $\{P_1,...,P_n\} \vdash C$. It follows strict logical derivation rules.
    - **Inductive Agent $a_3$**: Generalizes rules from multiple instances $\{x_1,...,x_n\} \Rightarrow R$. It operates on a specific-to-general reasoning pattern.
    - Each agent is guided by specific prompts tailored to their reasoning style, requiring step-by-step reasoning and rule citation.
    - Design Motivation: Different reasoning paradigms suit different problems; this three-pronged approach ensures at least one high-quality reasoning path is generated.

2. **Formal Reasoning Graph (FRG) Construction**:

    - Reasoning trajectories are converted into a directed graph $G^{(i)} = (V^{(i)}, E^{(i)})$.
    - RoBERTa-MNLI is used to evaluate the semantic relations between steps.
    - Trust scores are assigned based on NLI predictions: entailment = 0.95, neutral = 0.60, contradiction = 0.10.
    - This maps ambiguous natural language reasoning steps to quantifiable logical dependency relations.

3. **Bayesian Belief Propagation**:

    - Initializes priors for all nodes as $P(v)=0.5$.
    - Recursive update: $P(v_c) = \frac{P(v_p) \cdot \theta_{pc}}{P(v_p) \cdot \theta_{pc} + (1-P(v_p))(1-\theta_{pc})}$.
    - For nodes with multiple parents, the average of their respective updates is taken.
    - Consistent paths are amplified, while contradictions are attenuated, naturally achieving a "stronger good reasoning, weaker poor reasoning" effect.

4. **Graph Scoring & Answer Extraction**:

    - Score = Average confidence $\mu^{(i)}$ − Normalized entropy $H^{(i)}$.
    - Prefers graphs with high confidence and low uncertainty.
    - The terminal node of the highest-scoring graph yields the final answer.

### Complexity Analysis

- Total complexity is $\mathcal{O}(k \cdot s)$ ($k=3$ agents, $s$ reasoning steps), which is linear.
- Significantly superior to the $\mathcal{O}(n)$ full decodings required by Self-Consistency (where $n$ typically equals 20).

## Key Experimental Results

### Main Results (Accuracy %, Three Models × Two Datasets)

| Method | Mistral-7B WoL | Mistral-7B MA | DeepSeek-7B WoL | DeepSeek-7B MA | Phi-3.5 WoL | Phi-3.5 MA |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| CoT-Greedy | ~41 | ~26 | ~32 | ~21 | ~90 | ~55 |
| Self-Consistency | ~48 | ~21 | ~14 | ~14 | ~72 | ~40 |
| CoT-Decoding | ~54 | ~41 | ~48 | ~46 | ~99 | ~55 |
| **ToTh** | **~70** | **~45** | **~56** | **~43** | **96** | **~59** |

### Robustness Experiments (Mistral-7B, Different Difficulties)

| Method | WoL-3 Sents | WoL-4 Sents | WoL-5 Sents | d0/l3 | d0/l4 | d2/l3 |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| CoT-Greedy | 41 | 32 | 19 | 57 | 26 | 14 |
| Self-Consistency | 48 | 47 | 38 | 21 | 6 | 17 |
| CoT-Decoding | 54 | 48 | 46 | 55 | 41 | 24 |
| **ToTh** | **70** | **56** | **43** | **59** | **45** | 21 |

### Key Findings

1. ToTh achieves the best performance in 5 out of 6 difficulty settings, with d2/l3 being slightly lower than CoT-Decoding.
2. Self-Consistency fails severely in symbolic reasoning (DeepSeek scored only 14%), indicating that majority voting cannot handle structured logic.
3. Even in the most challenging 5-sentence symbolic reasoning setup, ToTh (43%) significantly outperforms CoT-Greedy (19%).
4. While Phi-3.5 achieves the highest absolute scores, the incremental gain of ToTh is more pronounced on weaker models.

## Highlights & Insights

1. **Systematic Integration of Cognitive Science**: Rather than merely borrowing terminology, the three reasoning paradigms are formally defined and independently implemented.
2. **NLI as a Logical Calibrator**: It cleverly leverages an NLI model to quantify textual logical relations into trust scores.
3. **Efficiency Advantage**: 3 decodings + lightweight post-processing vs. 20 decodings in Self-Consistency.
4. **Explainability**: The reasoning graph provides a complete, traceable reasoning path with quantified confidence for each step.

## Limitations & Future Work

1. **Fixed Reasoning Patterns**: Dynamic input-driven agent routing could be explored.
2. **Propagation Noise Amplification**: In long reasoning chains, early errors have a disproportionate impact, necessitating the introduction of confidence smoothing.
3. **Limited Evaluation Benchmarks**: The evaluation only uses WebOfLies and MultiArith, failing to cover commonsense or causal reasoning.
4. **NLI Bottleneck**: Relationships between complex reasoning steps may exceed the judgment capabilities of the NLI model.

## Related Work & Insights

- Positioned along the evolutionary trajectory of CoT $\rightarrow$ Self-Consistency $\rightarrow$ CoT-Decoding $\rightarrow$ ToT $\rightarrow$ GoT.
- Core difference from ToT/GoT: The latter focus on the diversity of search paths, whereas ToTh focuses on the diversity of reasoning paradigms and logical verification.
- The application of Bayesian propagation on reasoning graphs can be adapted to other scenarios that require logical validation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The integration of cognitive reasoning paradigms with Bayesian belief propagation is relatively novel.
- **Experimental Thoroughness**: ⭐⭐⭐ — Only two datasets and three models are used, representing a relatively small scale.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rigorous mathematical formulations.
- **Value**: ⭐⭐⭐⭐ — Provides a valuable new paradigm for structured reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)
- [\[ACL 2025\] FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models](fact_audit_factcheck.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)

</div>

<!-- RELATED:END -->
