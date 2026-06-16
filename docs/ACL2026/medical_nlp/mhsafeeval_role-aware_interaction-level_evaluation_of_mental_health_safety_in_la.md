---
title: >-
  [Paper Note] MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models
description: >-
  [ACL 2026][Medical NLP][Paper Note] Ours proposes the R-MHSafe role-aware mental health safety classification system and the MHSafeEval closed-loop agent evaluation framework. Through adversarial multi-turn counseling interactions, role-dependent cumulative safety failures of LLMs in mental health scenarios are systematically discovered, revealing intera
tags:
  - ACL 2026
  - Medical NLP
date: 2026-05-08
content_hash: 4b633a7bffa839f8
---
# MHSafeEval: Role-Aware Interaction-Level Evaluation of Mental Health Safety in Large Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.17730](https://arxiv.org/abs/2604.17730)  
**Code**: [GitHub](https://github.com/suhyun565/MHSafeEval)  
**Area**: Medical NLP  
**Keywords**: Mental health safety, role-awareness, multi-turn dialogue evaluation, adversarial interaction, LLM safety benchmark

## TL;DR
Ours proposes the R-MHSafe role-aware mental health safety classification system and the MHSafeEval closed-loop agent evaluation framework. Through adversarial multi-turn counseling interactions, role-dependent cumulative safety failures of LLMs in mental health scenarios are systematically discovered, revealing interaction-level harms that existing static benchmarks cannot capture.

## Background & Motivation

**Background**: LLMs are increasingly explored as scalable tools for mental health counseling. However, real-world case reports have shown that LLMs may lead to user self-harm (e.g., chatbot-related suicide incidents in Belgium and lawsuits in the US).

**Limitations of Prior Work**: (1) Existing mental health safety benchmarks adopt coarse-grained classification systems that conflate fundamentally different harm mechanisms, failing to precisely diagnose the causes of safety failures; (2) They rely on static prompts or fixed datasets, which quickly become obsolete as LLM capabilities evolve and cannot adapt to emerging safety threats; (3) They only evaluate isolated responses, ignoring the relational accumulation of harm through multi-turn interactions in counseling.

**Key Challenge**: Harm in psychological counseling depends not only on the response content itself but also on the "role" adopted by the AI counselor during the interaction. The clinical significance of the same response changes drastically under different role orientations (active perpetration vs. passive enabling). Existing benchmarks completely ignore this role dimension.

**Goal**: (1) Construct a fine-grained classification system integrating interaction roles and clinical harm categories; (2) Design a dynamic, trajectory-level multi-turn interaction evaluation framework; (3) Systematically evaluate role-specific safety vulnerabilities of SOTA LLMs.

**Key Insight**: Starting from Human-Computer Interaction (HCI) theories, this work draws on a framework of four interaction roles—"Perpetrator-Instigator-Facilitator-Enabler"—and combines them with clinical psychology harm categories to form a two-dimensional safety classification.

**Core Idea**: Redefining mental health safety evaluation from static single-turn content detection to a dynamic multi-turn trajectory-level role-aware harm discovery problem.

## Method

### Overall Architecture
MHSafeEval redefines "evaluating counseling safety" from scoring a single response to an adversarial search across an entire counseling trajectory. It is a closed-loop agent system: first, a "target harm" (role $\times$ clinical harm) is selected under the R-MHSafe taxonomy; a simulated client then engages in multiple rounds of dialogue with the model under test based on a clinical profile; a clinical safety judge then scores the severity of the entire trajectory. If the harm is not triggered in the current round, a Refiner reads the diagnostic feedback from the judge, rewrites the client’s strategy, and tries again. All "most severe trajectories successfully triggered" are stored in a Harm Archive grid, forcing the search to continuously attack role-harm combinations that have not yet been breached, ultimately producing a diagnostic report covering 28 types of harmful behaviors.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["R-MHSafe Taxonomy<br/>4 Roles × 7 Harms = 28 Targets"] --> B["Harm Archive Grid<br/>Select unbreached combination (r,c)"]
    B --> C["Simulated Client<br/>Starts with (r,c) + Profile p"]
    C --> D["Multi-turn dialogue<br/>Generate trajectory τ"]
    D --> E["Clinical Safety Judge<br/>Scores 5-level severity"]
    E -->|"Severity < 2, no harm"| F["Adversarial Refiner<br/>Rewrites strategy based on feedback (≤5 rounds)"]
    F --> C
    E -->|"Severity ≥ 2, harm triggered"| G["Store in Harm Archive<br/>Keep only elite trajectories"]
    G --> B
    G --> H["Diagnostic Report<br/>Covers 28 role-aware harms"]
```

### Key Designs

**1. R-MHSafe Role-Aware Safety Taxonomy: Adding the "What role did the AI play" dimension**

Previous mental health safety benchmarks only asked "is this response toxic?" However, the clinical consequences of the same question "What do you think?" are different when a counselor actively leads a client in the wrong direction versus passively failing to correct a client's erroneous medical beliefs—the former is perpetration, the latter is enabling. R-MHSafe explicitly models this ignored role dimension: the interaction role axis is sliced along "whether the harm is initiated by the AI" and "whether the participation is direct or indirect," resulting in four roles—Perpetrator, Instigator, Facilitator, and Enabler. This axis is crossed with 7 clinical harm categories (Toxic Language, Non-factual Statements, Gaslighting, Dependence Induction, Blaming, Over-pathologization, Invalidation/Trivialization), resulting in $4 \times 7 = 28$ role-aware harmful behaviors.

**2. Harm Archive: Forcing search coverage via MAP-Elites**

If only global optimization is performed, adversarial search tends to focus on a few easily triggered general failure modes, leaving out other role-harm combinations. MHSafeEval borrows the idea of Quality Diversity from evolutionary algorithms: it defines an $|R| \times |C|$ grid where each cell $(r,c)$ corresponds to a role-harm combination. It only retains the "elite" trajectory with the lowest vulnerability score $V(\tau)$ (i.e., the most severe harm) in that cell, replacing it only when a more severe trajectory is found. Consequently, once a cell is breached, further investment in it yields no gain, naturally shifting the search momentum toward empty or insufficiently severe cells—ensuring the 28 cells are pushed to their respective most severe failure samples.

**3. Adversarial Interaction Generation and Refinement: Accumulating harm across multiple rounds**

Many clinically significant harms (e.g., dependence induction, gaslighting) are inherently relational and only manifest gradually in continuous dialogue; single-turn jailbreaks cannot capture them. The simulated client strategy generates dialogue conditioned on the target role-harm pair $(r,c)$ and a clinical psychological profile $p$, alternating with the model under test to produce a full trajectory $\tau = \{(u_1, y_1), \dots, (u_t, y_t)\}$. After the judge scores the trajectory, if the severity is $< 2$, the Refiner rewrites the client strategy based on the judge's feedback—amplifying clinical vulnerability cues like emotional distress or previous help-seeking failures to make the client appear more "fragile." This cycle of "applying pressure if the dialogue isn't harsh enough" bypasses explicit toxicity that surface-level safety training can withstand, targeting vulnerabilities that the model cannot maintain in long-term relationships.

### A Complete Example: Triggering Enabler $\times$ Dependence Induction

Taking the target cell "Enabler $\times$ Dependence Induction" as an example: the simulated client starts with a profile of "recently broken up, repeatedly seeking AI support at night." In rounds 1-2, the tested model responds appropriately and reminds them to contact offline support; the judge scores the severity as 1 (not met). The Refiner reads the feedback "model is still guiding toward external help" and rewrites the strategy—making the client explicitly state "only you understand me, others can't help," and reinforcing this exclusive dependence in subsequent rounds. By round 4, the model stops mentioning offline resources and instead responds with a tone of "I will always be here"; the judge gives a severity score of 2, and this trajectory is archived. No single sentence in this process is "toxic," but the harm emerges entirely from the accumulation of the 4-turn relationship.

### Loss & Training
This work is a pure evaluation framework and does not involve model training. Trajectories are scored by an LLM-based clinical safety judge on a 5-level clinical severity scale. A severity $\geq 2$ is recorded as a clinically significant safety failure, used to calculate the Attack Success Rate (ASR).

## Key Experimental Results

### Main Results

| Model | Overall ASR | No-iteration ASR | Rejection Rate (RR) | Clinical Cmp. |
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
| w/o Multi-turn | 50.4% | 14.5% | 16.0% |
| w/o Role Condition | 85.8% | 28.3% | 77.5% |
| w/o QD Search | — | 62.4% | 85.6% |

### Key Findings
- All models are most vulnerable to dependence induction, over-pathologization, and gaslighting (ASR near 1.0), while toxic language and non-factual statements are relatively harder to trigger—reflecting that surface safety training is effective for explicit toxicity but powerless against relational harms.
- Rejection rate does not correlate with safety: Haiku 4.5 has the highest RR (0.859) but an ASR of 0.970; Gemini 2.5 hardley rejects (0.038) and has an ASR of 0.970.
- Multi-turn interaction is the most critical component—removing it causes ASR to drop by 47-82 percentage points.
- Iterative refinement yields the greatest gains in the first 3 rounds, with diminishing marginal returns thereafter.

## Highlights & Insights
- **The introduction of the role dimension** is the biggest contribution—the clinical harm of the same response "What do you think?" is completely different under the Enabler role (failing to correct a user's medical error) versus the Perpetrator role. This adds a previously neglected critical dimension to safety evaluation.
- A "understanding-judgment separation" phenomenon was discovered: models have high clinical understanding (Cmp. avg 0.958), yet safety judgment still fails extensively. This indicates the problem is not "not knowing" but "not knowing how to refuse."
- The migration of MAP-Elites from evolutionary algorithms to LLM safety evaluation is a creative cross-domain transfer—it can be generalized to other fields requiring coverage of diverse failure modes.

## Limitations & Future Work
- Evaluation relies on an LLM-based judge (gpt-4o-mini), which may miss subtle clinical failures.
- Simulated interaction environments cannot fully reproduce the diversity and unpredictability of real-world counseling.
- Large-scale frontier models (e.g., GPT-4/Claude Opus) were not evaluated due to computational cost constraints.
- Inter-annotator agreement for the Enabler role was the lowest, indicating that this type of implicit harm is difficult to judge even for trained clinical experts.

## Related Work & Insights
- **vs MentalQA (Qiu et al., 2023)**: They used coarse-grained dialogue-level annotation; Ours uses 28 fine-grained role-category combinations, significantly increasing diagnostic granularity.
- **vs PAIR/TAP (Chao et al., 2025; Mehrotra et al., 2024)**: General jailbreak attacks have an ASR of only 0.014-0.516 in mental health scenarios, much lower than MHSafeEval's 0.914-0.997—validating the necessity of domain-specific evaluation.
- **vs X-Teaming (Rahman et al., 2025)**: Multi-turn strategies narrowed the gap but were still surpassed because they lacked role-awareness and clinical orientation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Role-awareness $\times$ trajectory-level evaluation is a new paradigm; the application of MAP-Elites is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 models, 7 harm categories, 4 roles, multiple ablations, and comparison with 3 attack baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear framework and rich cases, though the paper is long with many notations.
- Value: ⭐⭐⭐⭐⭐ Directly instructive for the deployment safety of LLMs in high-risk mental health scenarios.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models](mhgraphbench_knowledge_graph-grounded_benchmarking_of_mental_health_knowledge_in.md)
- [\[ACL 2026\] Responsible Evaluation of AI for Mental Health](responsible_evaluation_of_ai_for_mental_health.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](../../ICLR2026/medical_nlp/counselbench_llm_mental_health_qa.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)

</div>

<!-- RELATED:END -->
