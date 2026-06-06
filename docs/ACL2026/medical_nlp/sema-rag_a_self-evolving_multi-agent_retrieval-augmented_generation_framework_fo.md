---
title: >-
  [Paper Note] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning
description: >-
  [ACL 2026][Medical NLP][Medical QA] The paper proposes SEMA-RAG, a self-evolving multi-agent RAG framework that simulates phased clinical reasoning workflows through three specialized agents (Interpreter, Explorer…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Medical QA"
  - "Multi-agent RAG"
  - "Self-evolving retrieval"
  - "Evidence chain construction"
  - "Clinical reasoning"
date: 2026-05-08
content_hash: 2fd76bb040c4d4ba
---

<!-- Generated automatically by src/gen_stubs.py -->
# SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning

**Conference**: ACL 2026  
**arXiv**: [2605.17101](https://arxiv.org/abs/2605.17101)  
**Code**: None  
**Area**: Medical NLP  
**Keywords**: Medical QA, Multi-agent RAG, Self-evolving retrieval, Evidence chain construction, Clinical reasoning

## TL;DR
The paper proposes SEMA-RAG, a self-evolving multi-agent RAG framework that simulates phased clinical reasoning workflows through three specialized agents (Interpreter, Explorer, Arbiter), outperforming the strongest baselines by an average of +6.46 accuracy points across 5 medical QA benchmarks.

## Background & Motivation
**Background**: RAG is widely used to mitigate hallucinations and knowledge obsolescence in medical QA, but existing RAG methods primarily follow a single-turn static retrieval paradigm.

**Limitations of Prior Work**: (1) Problem-to-query transformation lacks clinical semantic interpretation, leaving implicit constraints unexposed; (2) Retrieval lacks a sufficiency feedback mechanism, making it difficult to form reliable evidence chains; (3) Coupling interpretation, exploration, and adjudication into a single reasoning chain imposes high cognitive load.

**Key Challenge**: Single-turn static RAG requires clinicians to simultaneously analyze, retrieve, evaluate, and diagnose immediately after receiving an initial record, failing to adjust reasoning as new evidence emerges—a severe mismatch with the multi-stage clinical reasoning process.

**Goal**: Reconstruct the RAG workflow to match clinical phased reasoning: expand single-turn queries into multi-turn iterative exploration, evaluating evidence sufficiency after each retrieval to determine the next action.

**Key Insight**: Task decoupling and role specialization—assigning interpretation, exploration, and adjudication to three collaborating specialized agents.

**Core Idea**: A three-agent division of labor (I-Agent interprets $\rightarrow$ E-Agent performs sufficiency-driven self-evolving retrieval $\rightarrow$ A-Agent conducts evidence arbitration), enhancing medical RAG reliability through closed-loop evidence chain construction.

## Method

### Overall Architecture
SEMA-RAG consists of three role-based agents sharing the same underlying LLM, distinguished only by role prompts: (1) I-Agent maps the raw question to a structured clinical schema; (2) E-Agent accumulates evidence turn-by-turn via a sufficiency-driven self-evolving retrieval loop; (3) A-Agent arbitrates the converged evidence set to output the final answer.

### Key Designs
1. **I-Agent (Problem Interpreter)**:
    - **Function**: Maps unstructured medical questions to structured clinical schema tuples.
    - **Mechanism**: Generates a schema $Q' = \langle o_{\text{int}}, o_{\text{ent}}, o_{\text{cons}}, q_{\text{init}} \rangle$ containing four components: clinical intent, medical entities, clinical constraints, and initial retrieval query; linearized concatenation serves as the retrieval entry point.
    - **Design Motivation**: Implicit constraints in the original question (e.g., "7th day of hospitalization" implying hospital-acquired infection) are easily ignored in direct retrieval; structured schemas make them explicit.

2. **E-Agent (Knowledge Explorer, Self-evolving Retrieval Loop)**:
    - **Function**: Performs sufficiency-driven multi-turn iterative retrieval to construct a closed evidence set $C^*$.
    - **Mechanism**: Evaluates an evidence sufficiency flag $s_t \in \{0,1\}$ after each retrieval; if $s_t=0$ (insufficient), it identifies evidence gaps $g_t$ and generates $m$ follow-up queries $\mathcal{Q}_{t+1}$; the loop terminates when $s_t=1$ or reaches $T_{\max}$ turns. MedCPT is used as the dense retriever.
    - **Design Motivation**: Single-turn retrieval cannot guarantee evidence coverage of all key constraints; sufficiency-driven closed-loop iteration avoids premature decisions based on incomplete evidence.

3. **A-Agent (Evidence Arbiter)**:
    - **Function**: Arbitrates the converged evidence set and generates a traceable evidence report.
    - **Mechanism**: Denoises and deduplicates redundant/contradictory evidence, identifies consistencies and conflicts, and organizes supporting/refuting clues into a structured report $R$; performs discrete answer selection $\tilde{y} = \text{Agent}_A(\text{Pmt}_{\text{ans}}, [Q, R])$ based on the report.
    - **Design Motivation**: Evidence in medical reasoning is often redundant or contradictory, necessitating a dedicated arbitration step to integrate evidence into a stable judgment base.

### Loss & Training
- Training-free: Three agents share the underlying LLM, distinguished only by role prompts.
- Default Hyperparameters: $T_{\max}=2$, $k=16$ (Top-k retrieval), $m=3$ (follow-up queries per turn).
- I/E-Agent temperature set to 1.0, A-Agent temperature set to 0.0 (deterministic output).

## Key Experimental Results

### Main Results (5 Benchmarks × 5 LLM Backbones, Accuracy %)

| Method | MMLU-Med | MedQA-US | MedMCQA | PubMedQA* | BioASQ | Average |
|------|---------|---------|--------|----------|--------|------|
| deepseek-v3.1 + CoT | 88.15 | 77.53 | 71.69 | 38.40 | 80.10 | 71.17 |
| deepseek-v3.1 + MedRAG | 88.61 | 77.14 | 67.99 | 44.60 | 78.48 | 71.36 |
| deepseek-v3.1 + i-MedRAG | 85.86 | 74.78 | 65.65 | 50.60 | 80.58 | 71.49 |
| **deepseek-v3.1 + SEMA-RAG** | **91.46** | **89.95** | **75.09** | **59.20** | **82.85** | **79.71** |
| gemini-2.0-flash + CoT | 58.22 | 65.12 | 41.33 | 40.20 | 68.45 | 54.66 |
| **gemini-2.0-flash + SEMA-RAG** | **80.99** | **90.42** | **71.60** | **59.20** | **88.19** | **78.08** |

### Ablation Study (deepseek-v3.1, MedQA-US / PubMedQA*)

| Configuration | MedQA-US | PubMedQA* |
|------|---------|----------|
| w/o I-Agent | 85.47 | 54.20 |
| w/o E-Agent | 83.58 | 50.80 |
| w/o A-Agent | 86.49 | 53.60 |
| **Full SEMA-RAG** | **89.95** | **59.20** |

### Key Findings
- Removing E-Agent leads to the largest performance drop (6.37 on MedQA-US), confirming self-evolving retrieval as the core source of gain.
- Query width $m$ ablation: $m=1 \rightarrow$ 86.72%, $m=2 \rightarrow$ 89.00%, $m=3 \rightarrow$ 89.95%, showing diminishing returns.
- Exploration depth $T_{\max}$ performs best at 2-3 turns; exceeding this may introduce noise.
- Efficiency comparison: SEMA-RAG averages 4.8 LLM calls / 3.4 retrievals / 9.5s latency, with a token consumption of 19,488 (vs. 21,517 for i-MedRAG), yet achieves 15.17% higher accuracy.

## Highlights & Insights
- The three-agent architecture precisely simulates the phased workflow of clinical reasoning (Interpretation $\rightarrow$ Exploration $\rightarrow$ Arbitration), with task decoupling being a universal principle.
- The sufficiency-driven early stopping mechanism is more efficient than fixed-turn iterations (like i-MedRAG's 3 turns)—achieving higher accuracy with fewer tokens.
- The most significant improvement occurs on gemini-2.0-flash (average +23.42), indicating the framework's strong enhancement effect on weaker models.
- Case studies vividly demonstrate how to form a reliable evidence chain through structured interpretation $\rightarrow$ gap identification $\rightarrow$ targeted retrieval.

## Limitations & Future Work
- Evaluation is limited to benchmark environments and has not been validated in real clinical workflows (e.g., longitudinal EHR reasoning).
- The framework depends on the quality and coverage of the retrieval corpus: when key evidence is missing or outdated, the self-evolving loop may still converge on incomplete evidence.
- Sufficiency judgment criteria have not yet been optimized for option-level separability or generative completeness.
- The additional overhead of multi-turn reasoning is better than fixed-step baselines but still higher than single-turn methods.

## Related Work & Insights
- MedRAG / MedCPT provide the retrieval foundation in the medical domain; SEMA-RAG builds a multi-turn closed loop upon them.
- i-MedRAG pioneered iterative medical RAG but lacked sufficiency feedback; SEMA-RAG's self-evolving mechanism is a key improvement.
- Multi-agent collaboration (CAMEL / MetaGPT / MedAgents) ideas can be extended to other high-risk domains requiring multi-stage reasoning.
- The gap detection + targeted follow-up pattern of self-evolving retrieval can inspire complex RAG system design in non-medical fields.

## Rating
- Novelty: ⭐⭐⭐⭐ Task decoupling + sufficiency-driven self-evolving retrieval are clear innovative points.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 benchmarks × 5 LLM backbones × complete ablation + efficiency analysis + case studies.
- Writing Quality: ⭐⭐⭐⭐ Rigorous formalized expression with intuitive clinical reasoning analogies.
- Value: ⭐⭐⭐⭐⭐ Achieves significant and consistent improvements in medical QA; the framework's philosophy has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)
- [\[ACL 2026\] Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)
- [\[ACL 2026\] IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)
- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](../../AAAI2026/medical_nlp/pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [\[ACL 2026\] Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction](efficient_and_effective_internal_memory_retrieval_for_llm-based_healthcare_predi.md)

</div>

<!-- RELATED:END -->
