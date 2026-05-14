---
title: >-
  [Paper Note] An LLM-Based Simulation Framework for Embodied Conversational Agents in Psychological Counseling
description: >-
  [AAAI 2026][Medical Imaging][Embodied Conversational Agents] This paper proposes the ECAs framework, which grounds psychological counseling simulation in established theories such as Cognitive Behavioral Therapy (CBT). B…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Embodied Conversational Agents"
  - "Psychological Counseling Simulation"
  - "Cognitive Behavioral Therapy"
  - "LLM Data Synthesis"
  - "Memory-Driven Dialogue"
date: 2026-05-08
content_hash: 41afc4c76964c224
---

# An LLM-Based Simulation Framework for Embodied Conversational Agents in Psychological Counseling

**Conference**: AAAI 2026
**arXiv**: [2410.22041v3](https://arxiv.org/abs/2410.22041v3)
**Code**: [https://github.com/AIR-DISCOVER/ECAs-Dataset](https://github.com/AIR-DISCOVER/ECAs-Dataset)
**Area**: Medical Imaging
**Keywords**: Embodied Conversational Agents, Psychological Counseling Simulation, Cognitive Behavioral Therapy, LLM Data Synthesis, Memory-Driven Dialogue

## TL;DR
This paper proposes the ECAs framework, which grounds psychological counseling simulation in established theories such as Cognitive Behavioral Therapy (CBT). By leveraging LLMs to expand real counseling cases into embodied cognitive memory spaces, the framework simulates the complete cognitive processes of clients in counseling sessions and generates high-fidelity dialogue data. ECAs significantly outperforms baselines in both expert and automated evaluations.

## Background & Motivation
Psychological counseling data is critical for training novice counselors, developing AI-assisted systems, and enabling automated diagnosis. However, the sensitivity of such data makes real-world corpora extremely scarce. Existing approaches fall into two categories: (1) human-simulated dialogues—authentic but costly and limited in coverage; (2) AI-synthesized data—scalable but capable of capturing only surface-level linguistic patterns, failing to represent the implicit complexity of psychological processes (e.g., core beliefs, automatic thoughts). Moreover, self-training loops are prone to distributional shift and mode collapse. Recent work such as Patient-Psi and Roleplay-doh has made preliminary progress but lacks deep integration with psychological theory.

## Core Problem
How can an LLM-based simulation framework grounded in psychological and counseling theory be developed to generate counseling dialogue data that is realistic, rich, and clinically consistent? The central challenge lies in the fact that psychological processes—especially the client's cognitive chain—are implicit and multi-layered, and cannot be reproduced through simple role-playing.

## Method

### Overall Architecture
The ECAs framework operates in three steps: **Step 1 (REAL)** extracts client basic information from the real-world dataset D4 (a Chinese depression counseling dataset); **Step 2 (SIM)** uses an LLM to expand the basic information into a comprehensive personal profile and social profile, then generates an embodied cognitive memory space comprising beliefs, cognitive processes, factual memories, and perceptual memories; **Step 3 (REAL)** generates authentic responses during dialogue via context-driven dynamic memory retrieval.

### Key Designs

1. **Six Simulation Principles (SP1–SP6)**: Constraints derived from psychological theory—(SP1) represent experiences across the full life span, constructing factual memories spanning childhood, adolescence, youth, middle age, and recent periods using embodied cognition theory; (SP2) simulate cognitive processes, modeling the cognitive chain from core beliefs → intermediate beliefs → automatic thoughts based on the CBT framework; (SP3) integrate perceptual memories encompassing not only facts but also emotional, behavioral, and physiological responses; (SP4) model social relationship networks through a five-stage interpersonal dynamics model; (SP5) maintain consistency throughout data synthesis; (SP6) enable context-driven memory retrieval.

2. **Dual-Profile Generation (Client Profile)**: Part 1, the personal profile—starting from real D4 data, the LLM generates the client's personality, appearance, hobbies, aspirations, daily habits, and recent experiences, tracing back developmental environments and past experiences to construct a coherent psychological trajectory. Part 2, the social profile—based on the personal profile, the LLM simulates changes in the client's social network across life stages, reflecting how social contexts trigger and sustain depressive symptoms.

3. **Four-Phase Embodied Memory Generation Paradigm**: An LLM generation pipeline grounded in CBT theory—Phase 1: generate core beliefs $B_c$ and intermediate beliefs $B_i$; Phase 2: identify key events at each life stage and establish causal relationships; Phase 3: enrich key events into factual memories using the 4W1H format (Who/What/When/Where/How); Phase 4: derive emotional responses $\xi_e$, behavioral responses $\xi_b$, and physiological responses $\xi_p$ from automatic thoughts $A$ (linked to $B_c$ and $B_i$), forming perceptual memories $V(c)$.

4. **Context-Driven Dynamic Memory Retrieval**: During dialogue, the Client Agent first uses an LLM to analyze the dialogue history $H_t$ and the counselor's query $q_t$ to determine the required memory type, then filters candidate memories via keyword matching, and finally ranks them by cosine similarity to select the top-3 most relevant memories to guide response generation. Retrieval is restricted to factual memories, perceptual memories, and automatic thoughts—core and intermediate beliefs are excluded, as real clients do not directly articulate these during sessions.

5. **Pilot Study and Iterative Refinement**: The framework underwent three rounds of iteration—Round 1: licensed counselors reviewed whether memory construction adhered to psychological principles; Round 2: consistency between memory scripts and profiles was evaluated (revealing that script-only evaluation failed to capture functional properties in dialogue); Round 3: dialogue data were generated to assess the clinical utility of the memories. Ultimately, 14 high-frequency counseling topics drawn from D4 were selected, covering five key clinical domains: depression risk, diet, sleep, suicide risk, and social life.

### Loss & Training
No training is required. The framework is implemented entirely through prompt engineering with GPT-4o across all generation phases, following a zero-shot paradigm.

## Key Experimental Results

### Dataset Scale
The ECAs-dataset contains detailed profiles for 451 Client Agents, of which 100 possess complete embodied memory spaces (400–1,500 individual memory entries each, with an average of 134.6 memory nodes), far exceeding CharacterDial (1 node) and PATIENT-ψ (1 node).

### Expert Evaluation (5 Licensed Counselors, 6 Client Agents)

| Dimension | ECAs vs GPT-4o | ECAs vs D4 (Human) | Statistical Significance |
|-----------|---------------|-------------------|--------------------------|
| Necessity | +0.57 (p=0.074) | +1.40 (p<0.001) | F(2,87)=16.17, η²=0.271 |
| Sufficiency | +1.07 (p=0.001) | +1.50 (p<0.001) | F(2,87)=15.80, η²=0.266 |
| Fidelity | +1.00 (p=0.005) | — | F(2,87)=5.19, η²=0.107 |
| Consistency | — | +0.57 (p=0.079) | F(2,87)=3.22, η²=0.069 |

### Automated Evaluation 1: Depression/Suicide Risk Classification (Macro F1)

| Task | GPT-4o | D4 (Human) | ECAs (Ours) |
|------|--------|-----------|------------|
| Depression Risk | 0.35 | 0.40 | **0.42** |
| Suicide Risk | 0.59 | 0.59 | **0.67** |

### Automated Evaluation 2: Dialogue Quality Total Score (/20)

| Method | Fidelity | Comprehensiveness | Consistency | Rationality | Specificity | Total |
|--------|----------|-------------------|-------------|-------------|-------------|-------|
| D4 (Human) | 2.40/5 | 2.91/7 | 1.03/2 | 1.98/3 | 0.96/2 | 9.28/20 |
| ECAs (Ours) | **4.66/5** | **6.26/7** | **1.99/2** | **2.98/3** | **1.97/2** | **17.90/20** |

### Ablation Study
The paper does not provide module-level ablations; however, the three-round pilot study demonstrates the necessity of iterative refinement: script-only evaluation cannot substitute for dialogue-level assessment, and the final question set required careful selection to cover five key clinical domains. The most substantial improvement is observed in suicide risk classification (+8 F1 vs. GPT-4o), indicating that embodied memory is especially critical for expressing high-risk topics.

## Highlights & Insights
- **Deep Integration of Psychological Theory**: Rather than simply prompting LLMs for role-playing, the framework systematically models the CBT belief–thought–response chain as a memory space—a fundamental distinction from works such as Patient-Psi.
- **Four-Phase Memory Generation Paradigm**: The cascaded generation from beliefs → events → factual memories → perceptual memories is grounded in psychological theory at every step, ensuring that the generated memories simulate cognition rather than merely fabricate narratives.
- **Exclusion of Belief Layers During Retrieval**: Real clients do not spontaneously articulate core beliefs during counseling; these manifest only as automatic thoughts and emotional reactions. This design detail reflects a nuanced understanding of the counseling process.
- **Three-Round Expert Iteration**: The framework was not designed in a single pass; iterative feedback from licensed counselors progressively refined the approach, strengthening its clinical credibility.

## Limitations & Future Work
- **Limited to Depression**: The framework is grounded in the D4 dataset (depression diagnoses); its applicability to other psychological disorders (anxiety, PTSD, etc.) remains unvalidated.
- **Dependence on Chinese Data**: D4 is a Chinese-language dataset; cross-lingual and cross-cultural generalizability has not been explored.
- **Absence of Explicit Ablations**: Independent contribution analyses of individual modules (personal profile vs. social profile vs. each memory phase vs. retrieval mechanism) are lacking.
- **Simplified Counselor Agent**: The paper focuses on the Client Agent; the counselor side does not model therapeutic strategies or intervention logic.
- **Evaluation Limitations**: Automated evaluation relies on GPT-4o to assess GPT-4o-generated data, introducing potential bias; expert evaluation sample sizes are modest (6 agents, 5 experts).

## Related Work & Insights
- **vs. Patient-Psi (EMNLP 2024)**: Patient-Psi also uses LLMs to simulate counseling clients, but its cognitive modeling is shallow (only 1 memory node) and does not include factual or life-stage memories. ECAs averages 134.6 memory nodes and incorporates a complete CBT cognitive chain.
- **vs. Roleplay-doh (EMNLP 2024)**: Roleplay-doh focuses on constraining LLM simulation via domain-expert-defined principles but lacks explicit memory spaces and cognitive process modeling.
- **vs. CharacterDial (EMNLP 2024)**: Designed for social role-playing rather than psychological counseling, with only attribute profiles and no belief or memory system.

The paradigm of systematically translating a psychological theory (CBT) into an LLM Agent memory architecture is transferable to other domains requiring deep persona simulation, such as educational and medical consultation simulations. The context-driven, type-aware memory retrieval strategy—keyword filtering followed by semantic ranking—offers a practical and more controllable memory management approach compared to pure vector retrieval.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Systematically instantiates CBT theory as an embodied memory generation framework for LLM Agents, achieving a high degree of theoretical integration.
- **Experimental Thoroughness**: ⭐⭐⭐ Combines expert and automated evaluation, but lacks ablation studies and has limited sample sizes.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with tight correspondence between theoretical derivation and implementation, though some notations are redundant.
- **Value**: ⭐⭐⭐⭐ Provides a high-quality data generation solution and a publicly available dataset for psychological counseling AI, with clear practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MCTSr-Zero: Self-Reflective Psychological Counseling Dialogues Generation via Principles and Adaptive Exploration](mctsr-zero_self-reflective_psychological_counseling_dialogues_generation_via_pri.md)
- [\[NeurIPS 2025\] LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation](../../NeurIPS2025/medical_imaging/llm-assisted_emergency_triage_benchmark_bridging_hospital-rich_and_mci-like_fiel.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)
- [\[AAAI 2026\] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization](spa_achieving_consensus_in_llm_alignment_via_self-priority_optimization.md)
- [\[AAAI 2026\] Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework](error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)

</div>

<!-- RELATED:END -->
