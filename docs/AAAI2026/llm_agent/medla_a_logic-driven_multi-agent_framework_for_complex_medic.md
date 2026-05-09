---
title: >-
  [Paper Note] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models
description: >-
  [AAAI 2026][LLM Agent][Syllogistic Reasoning] This paper proposes MedLA, the first multi-agent medical reasoning framework based on syllogistic logic trees. Each agent organizes its reasoning as an explicit logic tree composed of syllogistic nodes (major premise–minor premise–conclusion). Multiple agents align and revise their logic trees at the premise level through graph-guided multi-round discussions. MedLA outperforms all baselines by 7.4% on MedDDx (8B model) and achieves an average accuracy of 69.9% on medical QA benchmarks with an 8B model, surpassing 70B RAG-based models.
tags:
  - AAAI 2026
  - LLM Agent
  - Syllogistic Reasoning
  - Logic Tree
  - Multi-Agent Discussion
  - Medical QA
  - Premise-Level Alignment
date: 2026-05-08
content_hash: c777b812c4cf6df0
---

# MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2509.23725](https://arxiv.org/abs/2509.23725)
**Code**: [https://github.com/alexander2618/MedLA](https://github.com/alexander2618/MedLA)
**Area**: LLM Agent / Medical Reasoning / Multi-Agent Collaboration
**Keywords**: Syllogistic Reasoning, Logic Tree, Multi-Agent Discussion, Medical QA, Premise-Level Alignment

## TL;DR
This paper proposes MedLA, the first multi-agent medical reasoning framework based on syllogistic logic trees. Each agent organizes its reasoning as an explicit logic tree composed of syllogistic nodes (major premise–minor premise–conclusion). Multiple agents align and revise their logic trees at the premise level through graph-guided multi-round discussions. MedLA outperforms all baselines by 7.4% on MedDDx (8B model) and achieves an average accuracy of 69.9% on medical QA benchmarks with an 8B model, surpassing 70B RAG-based models.

## Background & Motivation

**Background**: LLMs have been widely applied to medical reasoning. Existing approaches fall into two broad categories: knowledge fine-tuning (e.g., Med-PaLM) and reasoning stimulation (e.g., multi-agent role-playing discussions). Multi-agent methods (MedAgents, MDAgents) achieve collaboration by assigning fixed roles, offering low cost and high flexibility.

**Limitations of Prior Work**: Existing multi-agent systems only discuss at the **conclusion level** — each agent produces a conclusion, followed by voting or debate — and cannot examine the underlying logical details to identify the root cause of disagreements. When two agents reach different diagnoses, the system cannot determine whether the discrepancy stems from a wrong premise or a flawed reasoning step.

**Key Challenge**: Medical reasoning demands traceability and auditability — every conclusion should be traceable to specific medical rules and patient facts. However, natural language reasoning in LLMs is implicit and cannot be structurally inspected.

**Key Insight**: The classical syllogism (major premise–minor premise–conclusion) is the minimal unit of logical inference. Connecting multiple syllogisms in series or parallel forms a logic tree that explicitly represents the complete reasoning process, enabling precise premise-level alignment and error correction across agents.

**Core Idea**: Structure each agent's reasoning as a syllogistic logic tree, and align logic trees at the premise level through multi-round graph-guided discussions to achieve traceable and collaborative error correction.

## Method

### Overall Architecture
A three-phase pipeline:
- **Phase A**: A P-Agent extracts major/minor premises; a D-Agent decomposes the question into sub-problems.
- **Phase B**: Multiple M-Agents generate logic trees in parallel → a C-Agent evaluates node credibility → multi-round discussions revise the trees.
- **Phase C**: Aggregated logic trees are used to generate the final answer.

### Key Designs

1. **Syllogistic Logic Tree**:

    - **Function**: Structures the reasoning process as a DAG $\mathcal{T} = (V, E)$, where each node $v_i = (p_{\text{maj}}, p_{\text{min}}, C)$ represents a syllogism.
    - **Mechanism**: Leaf nodes store empirical observations or domain rules; internal nodes store intermediate inferences; the root node yields the final clinical decision. Every reasoning chain is traceable to its constituent premises.
    - **Design Motivation**: Two key advantages — (a) **Traceability**: any conclusion can be traced back to its supporting premises; (b) **Comparability**: logic trees from different agents can be aligned, enabling precise localization of conflicts at the premise level.

2. **P-Agent (Premise Extraction) + D-Agent (Question Decomposition)**:

    - The P-Agent extracts a major premise set $\mathcal{P}_{\text{maj}}$ (medical rules) and a minor premise set $\mathcal{P}_{\text{min}}$ (patient facts) from the input question.
    - The D-Agent recursively decomposes the question into atomic sub-questions using an elimination strategy that evaluates the plausibility of each candidate answer in turn.
    - **Design Motivation**: Transforms unstructured clinical questions into structured premises and sub-questions, providing inputs for subsequent logic tree construction.

3. **M-Agent (Logic Tree Generation) + C-Agent (Credibility Evaluation)**:

    - Multiple M-Agents independently generate logic trees $\mathcal{T}_{M^{(j)}}$ in parallel, organizing syllogistic nodes in TSV format.
    - The C-Agent evaluates the credibility of each node (High/Medium/Low); nodes rated Low are flagged as discussion material.
    - **Design Motivation**: Parallel generation ensures diversity; credibility evaluation pre-filters nodes before discussion, focusing attention on those most likely to contain errors.

4. **Graph-Guided Multi-Round Discussion**:

    - **Function**: Agents exchange their logic trees and perform comparative revision at the premise level.
    - **Mechanism**: After each M-Agent inspects the logic trees of other agents, it reviews low-credibility nodes — verifying whether premises are correct, determining whether premises should be added or removed, and reassigning scores. Discussions continue until convergence.
    - **Design Motivation**: Unlike conventional "conclusion debate," this constitutes "logical structure debate" — an agent can explicitly state "the major premise 'all X causes Y' in your third syllogism is incorrect," enabling precise identification of the source of disagreement.
    - **Theoretical Guarantees**: Property 1 proves that each revision round monotonically reduces inter-agent variance ($S_{t+1}^2 < S_t^2$); Property 2 proves convergence within a finite number of rounds.

## Key Experimental Results

### Main Results — MedDDx (Differential Diagnosis)

| Method | Basic | Intermediate | Expert | Avg. |
|--------|-------|--------------|--------|------|
| MDAgents (NeurIPS2024) | 42.1 | 37.5 | 33.4 | 37.7 |
| CoT-LLaMA3.1(8B) | 43.9 | 39.3 | 32.2 | 38.5 |
| MedRAG (70B) | 36.5 | 34.8 | 32.7 | 34.7 |
| **MedLA + LLaMA3.1(8B)** | **48.2** | **43.0** | **41.7** | **44.3** |

### Main Results — Medical QA

| Method | MMLU-Med | MedQA-US | BioASQ | Avg. |
|--------|----------|----------|--------|------|
| MDAgents (NeurIPS2024) | 65.0 | 53.4 | 64.0 | 60.8 |
| LLaMA3.1(8B) baseline | 67.7 | 56.3 | 68.7 | 64.2 |
| MedRAG (70B) | 57.9 | 48.7 | 71.9 | 59.5 |
| **MedLA + LLaMA3.1(8B)** | **70.7** | **62.6** | **76.5** | **69.9** |

### Ablation Study (MedDDx)

| Configuration | Basic | Intermediate | Expert |
|---------------|-------|--------------|--------|
| MedLA (full) | 48.2 | 43.0 | 41.7 |
| − Revision loop | 44.2 | 38.6 | (−3.1) |
| − Credibility | 41.8 | 37.2 | (−4.5) |
| − LogicTree (CoT Only) | 38.7 | 34.9 | (−6.8) |
| Majority Voting | 37.5 | 30.2 | (−11.5) |

### Key Findings
- **8B model surpasses 70B RAG model**: MedLA+LLaMA3.1(8B) achieves an average of 69.9% on QA benchmarks vs. 59.5% for MedRAG(70B), demonstrating that structured logic is more important than model scale or external retrieval.
- **Larger gains on harder cases**: MedLA achieves an 11.1 pp improvement on MedDDx Expert (30.6% → 41.7%) but only 4.6 pp on Basic, indicating that logic trees are most beneficial for complex reasoning.
- **Logic tree is the core component**: Removing the logic tree and falling back to CoT causes a 6.8 pp drop on Expert, substantially larger than removing the revision loop (−3.1 pp) or credibility evaluation (−4.5 pp).
- **Effective on DeepSeek-R1**: Performance on MedXpertQA improves from 21.3% to 36.0% (+14.7 pp), confirming effectiveness on commercial LLMs as well.
- **Benefits scale with model size**: LLaMA3.1-70B improves from 41.8% to 51.9% (+10.1 pp), indicating that the logic tree approach is orthogonal to model scale.
- **Acceptable computational cost**: With 17 sub-agents, total inference time is approximately 2× that of simple voting, far less than methods requiring additional fine-tuning.

## Highlights & Insights
- **"Premise-level alignment" vs. "conclusion-level debate"** is the most fundamental innovation. Existing multi-agent systems debate "whose answer is correct," whereas MedLA debates "which reasoning step of which agent is problematic," directly improving the precision of error correction.
- The choice of **the syllogism as the minimal unit of reasoning** is highly compelling — it is the most basic valid inference form in logic and naturally fits medical diagnosis (major premise = medical knowledge, minor premise = patient symptoms, conclusion = diagnosis).
- Surpassing RAG+70B models **without fine-tuning or external retrieval** demonstrates that the value of "reasoning structure" may be underestimated — better organization of existing knowledge can outweigh acquiring more knowledge.
- **Formal proofs (monotonically decreasing variance + finite-round convergence)** provide theoretical guarantees for the convergence of multi-agent discussions.

## Limitations & Future Work
- LLMs cannot natively output structured trees; in practice, prompts guide the model to produce syllogisms in TSV format, and format compliance may be unstable.
- Syllogisms require premises to be explicit propositions, whereas much medical reasoning is vague and probabilistic (e.g., "possibly," "tends toward"), making binary logic an imperfect fit.
- The elimination strategy (ruling out answer choices one by one) is suited to multiple-choice questions but not to open-ended diagnosis.
- The 17-agent reasoning pipeline introduces non-trivial latency, approximately 2× that of simpler approaches.
- Evaluation is conducted solely on multiple-choice formats; validation on free-text diagnostic generation is absent.

## Related Work & Insights
- **vs. MedAgents/MDAgents**: These methods conduct conclusion-level discussions using fixed specialist roles (radiology, internal medicine, etc.). MedLA conducts premise-level discussions using logic trees, outperforming them by 6.6–7.8 pp on MedDDx.
- **vs. CoT**: CoT produces implicit reasoning chains; MedLA's syllogistic structure yields explicit, traceable, and auditable logical representations.
- **vs. RAG**: MedRAG with a 70B model and external retrieval achieves only 59.5%, while MedLA with an 8B model and structured reasoning achieves 69.9% — intrinsic reasoning structure outweighs external augmentation.
- **Implications for medical AI**: Interpretability need not be an post-hoc add-on; it can be a core component of the reasoning process itself — structured reasoning constitutes a form of "interpretability-enhanced inference."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The combination of syllogistic logic trees and premise-level multi-agent discussion is a highly distinctive design, supported by complete theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three benchmarks, 20+ baselines, four categories of comparison methods, ablation studies, difficulty-level analysis, cross-scale validation, and timing analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Method descriptions are clear, but the appendix is extremely long (7 sections); the main text could be more concise.
- **Value**: ⭐⭐⭐⭐⭐ — An 8B model surpassing a 70B RAG model is a striking result; the structured reasoning paradigm carries important implications for medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)
- [\[AAAI 2026\] Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](beyond_react_a_planner-centric_framework_for_complex_tool-au.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](../../ACL2026/llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[ACL 2026\] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](../../ACL2026/llm_agent/implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)
- [\[NeurIPS 2025\] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](../../NeurIPS2025/llm_agent/debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)

</div>

<!-- RELATED:END -->
