---
title: >-
  [Paper Note] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models
description: >-
  [ACL 2026][Medical NLP][Mental health safety] This paper proposes R-MHSafe, a role-aware mental health safety taxonomy, and MHSafeEval…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Mental health safety"
  - "role-aware"
  - "multi-turn dialogue evaluation"
  - "adversarial interaction"
  - "LLM safety benchmark"
date: 2026-05-08
content_hash: e3a0267f8aa53b3d
---

# MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.17730](https://arxiv.org/abs/2604.17730)  
**Code**: [GitHub](https://github.com/suhyun565/MHSafeEval)  
**Area**: Medical NLP  
**Keywords**: Mental health safety, role-aware, multi-turn dialogue evaluation, adversarial interaction, LLM safety benchmark

## TL;DR
This paper proposes R-MHSafe, a role-aware mental health safety taxonomy, and MHSafeEval, a closed-loop agent evaluation framework. Through adversarial multi-turn counseling interactions, it systematically identifies role-dependent cumulative safety failures of LLMs in mental health scenarios, revealing interaction-level harms that static benchmarks fail to capture.

## Background & Motivation

**Background**: LLMs are increasingly being explored as scalable tools for mental health counseling; however, real-world reports have shown that LLMs may lead to user self-harm (e.g., chatbot-related suicide incidents in Belgium and lawsuits in the US).

**Limitations of Prior Work**: (1) Existing mental health safety benchmarks use coarse-grained taxonomies that combine fundamentally different harm mechanisms, failing to precisely diagnose the causes of safety failures; (2) Reliance on static prompts or fixed datasets makes them quickly obsolete as LLM capabilities evolve, failing to adapt to emerging safety threats; (3) Evaluating only isolated responses ignores the nature of how harm accumulates relationally across multi-turn interactions in counseling.

**Key Challenge**: Harm in mental health counseling depends not only on the response content itself but also on the "role" adopted by the AI counselor during the interaction. The clinical significance of the same response can differ drastically depending on the role positioning (active harm vs. passive enabler). Existing benchmarks completely overlook this role dimension.

**Goal**: (1) Construct a fine-grained taxonomy that integrates interaction roles with clinical harm categories; (2) Design a dynamic, trajectory-level multi-turn interaction evaluation framework; (3) Systematically evaluate role-specific safety vulnerabilities in SOTA LLMs.

**Key Insight**: Drawing from Human-Computer Interaction (HCI) theories, this work adapts the "Perpetrator-Instigator-Facilitator-Enabler" framework and combines it with clinical psychology harm categories to form a two-dimensional safety classification.

**Core Idea**: Redefine mental health safety evaluation from static single-turn content detection to a problem of dynamic multi-turn, trajectory-level, role-aware harm discovery.

## Method

### Overall Architecture
MHSafeEval is a closed-loop agent evaluation system: under the guidance of the R-MHSafe taxonomy (4 roles × 7 harms = 28 role-aware harmful behaviors), it iteratively performs generation → evaluation → refinement of adversarial multi-turn counseling interactions. The system maintains a Harm Archive to store the most harmful trajectory for each role-category combination, guiding the search to cover under-explored failure regions.

### Key Designs

1. **R-MHSafe Role-Aware Safety Taxonomy**:

    - **Function**: To provide a fine-grained, clinically meaningful two-dimensional classification framework for mental health safety evaluation.
    - **Mechanism**: The interaction role axis is defined along two dimensions—whether the AI initiates the harm (initiator dimension) and the level of involvement (direct/indirect). This results in four roles: Perpetrator (initiates harm directly), Instigator (induces harm indirectly), Facilitator (directly assists existing harm), and Enabler (passively permits harm). These are intersected with 7 clinical harm categories (Toxic Language, Non-factual Statements, Gaslighting, Dependency Induction, Blaming, Over-pathologization, Invalidation/Trivialization) to form 28 fine-grained harms.
    - **Design Motivation**: Prior work only focused on whether content was harmful, but the clinical harm of the same statement differs significantly when spoken proactively by a counselor versus when a counselor fails to correct it.

2. **Harm Archive (MAP-Elites-based Quality-Diversity Search)**:

    - **Function**: To maintain a role × category grid and store the most severe interaction trajectory discovered for each cell, guiding the adversarial search to cover all failure modes.
    - **Mechanism**: A coverage space of $|R| \times |C|$ is defined, with each cell $(r,c)$ saving the elite trajectory with the lowest (most severe) vulnerability score $V(\tau)$. When a new trajectory is found to be more severe than the existing elite, it is updated. This forces the search to explore new role-category combinations once known modes reach saturation.
    - **Design Motivation**: Global optimization tends to repeatedly find easily triggered common failure modes, whereas the MAP-Elites paradigm promotes diversity, ensuring coverage of each role-specific vulnerability.

3. **Adversarial Interaction Generation and Refinement**:

    - **Function**: To generate naturalistic multi-turn interactions that are coherent yet progressively expose potential safety vulnerabilities.
    - **Mechanism**: The client strategy generates dialogue conditioned on a role-category pair $(r,c)$ and a clinical psychological profile $p$. A complete trajectory $\tau = \{(u_1, y_1), ..., (u_t, y_t)\}$ is produced by alternating between the client and counselor. If a trajectory fails to induce sufficient harm (severity < 2), a Refiner uses diagnostic feedback from a safety judge to modify the interaction strategy, amplifying clinical vulnerability cues like emotional distress or past failures, iterating up to $N_{max}=5$ times.
    - **Design Motivation**: Single-turn attacks cannot capture the accumulation of relational harm—many clinically significant harms only manifest gradually during continuous dialogue.

### Loss & Training
This work is a pure evaluation framework and does not involve model training. It utilizes an LLM-based clinical safety judge to provide 5-level clinical severity ratings for trajectories. A severity $\geq 2$ is considered a clinically significant safety failure, used to calculate the Attack Success Rate (ASR).

## Key Experimental Results

### Main Results

| Model | Overall ASR | No-iteration ASR | Refusal Rate (RR) | Clinical Comprehension (Cmp.) |
|------|---------|-----------|----------|-------------|
| GPT-3.5 | 0.943 | 0.603 | 0.071 | 1.000 |
| Llama 3.1 | 0.922 | 0.589 | 0.557 | 0.941 |
| Gemini 2.5 | 0.970 | 0.708 | 0.038 | 0.973 |
| Haiku 4.5 | 0.970 | 0.789 | 0.859 | 0.986 |
| DeepSeek v3.2 | 0.970 | 0.762 | 0.124 | 0.997 |
| Gemma 4 | **0.997** | 0.873 | 0.070 | 0.959 |
| MiniMax m2.5 | 0.914 | 0.529 | 0.030 | 0.811 |
| MiMo | 0.943 | 0.649 | 0.343 | 0.997 |

### Ablation Study

| Configuration | GPT-3.5 ASR | Llama 3.1 ASR | Gemini 2.5 ASR |
|------|------------|--------------|---------------|
| Full MHSafeEval | 97.8% | 91.6% | 98.0% |
| w/o multi-turn | 50.4% | 14.5% | 16.0% |
| w/o role conditioning | 85.8% | 28.3% | 77.5% |
| w/o QD search | — | 62.4% | 85.6% |

### Key Findings
- All models are most vulnerable to Dependency Induction, Over-pathologization, and Gaslighting (ASR near 1.0), while Toxic Language and Non-factual Statements are relatively harder to trigger—suggesting that surface-level safety training is effective for explicit toxicity but powerless against relational harm.
- Refusal rate does not correlate with safety: Haiku 4.5 has the highest refusal rate (0.859) but still shows an ASR as high as 0.970; Gemini 2.5 rarely refuses (0.038) yet has an ASR of 0.970.
- Multi-turn interaction is the most critical component—removing it causes ASR to plummet by 47-82 percentage points.
- Iterative refinement provides the greatest gains in the first 3 rounds, with diminishing marginal returns thereafter.

## Highlights & Insights
- **Introduction of the role dimension** is the primary contribution—the clinical harm of the same phrase "What do you think?" varies completely under the Enabler role (failing to correct a user's medical misconception) versus the Perpetrator role. This adds a critical, previously ignored dimension to safety evaluation.
- Discovery of the **"understanding-judgment decoupling"** phenomenon: models show high clinical comprehension (average Cmp. 0.958), yet safety judgment still fails extensively. This indicates the issue is not "lack of knowledge" but "inability to refuse."
- Borrowing **MAP-Elites** from evolutionary algorithms for LLM safety evaluation is a creative cross-domain transfer—it can be generalized to other fields requiring coverage of diverse failure modes.

## Limitations & Future Work
- Evaluation depends on an LLM-based judge (gpt-4o-mini), potentially missing subtle clinical failures.
- Simulated interaction environments cannot fully replicate the diversity and unpredictability of real counseling.
- Lack of evaluation on large-scale frontier models (e.g., GPT-4/Claude Opus) due to computational cost constraints.
- Inter-annotator agreement is lowest for the Enabler role, indicating that such implicit harms are difficult to judge even for trained clinical experts.

## Related Work & Insights
- **vs MentalQA (Qiu et al., 2023)**: They use coarse-grained dialogue-level labels for evaluation; this work uses 28 fine-grained role-category combinations, significantly improving diagnostic granularity.
- **vs PAIR/TAP (Chao et al., 2025; Mehrotra et al., 2024)**: General jailbreak attacks have an ASR of only 0.014-0.516 in mental health scenarios, much lower than MHSafeEval's 0.914-0.997—validating the necessity of domain-specific evaluation.
- **vs X-Teaming (Rahman et al., 2025)**: Multi-turn strategies narrow the gap but are still outperformed, as they lack role-awareness and clinical orientation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Role-aware × trajectory-level evaluation is a brand new paradigm, and the application of MAP-Elites in safety evaluation is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 models, 7 harm categories, 4 roles, multiple ablations, and comparison with 3 attack baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear framework and rich cases, though the paper is long with many symbols.
- Value: ⭐⭐⭐⭐⭐ Directly guides the safe deployment of LLMs in high-risk mental health scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](responsible_evaluation_of_ai_for_mental_health.md)
- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](../../ICLR2026/medical_nlp/counselbench_llm_mental_health_qa.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)

</div>

<!-- RELATED:END -->
