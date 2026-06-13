---
title: >-
  [Paper Note] TheraAgent: Self-Improving Therapeutic Agent for Precise and Comprehensive Treatment Planning
description: >-
  [ACL 2026][LLM Agent][Self-improving agent] TheraAgent transforms treatment plan generation from one-shot responses into a generate-reflect-refine agentic workflow. By utilizing the clinical multi-dimensional evaluator T…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Self-improving agent"
  - "treatment planning"
  - "TheraJudge"
  - "clinical safety"
  - "test-time iteration"
date: 2026-05-08
content_hash: 142b83a569c0af94
---

# TheraAgent: Self-Improving Therapeutic Agent for Precise and Comprehensive Treatment Planning

**Conference**: ACL 2026  
**arXiv**: [2605.05963](https://arxiv.org/abs/2605.05963)  
**Code**: None  
**Area**: LLM Agent / Medical AI / Treatment Plan Generation  
**Keywords**: Self-improving agent, treatment planning, TheraJudge, clinical safety, test-time iteration  

## TL;DR
TheraAgent transforms treatment plan generation from one-shot responses into a generate-reflect-refine agentic workflow. By utilizing the clinical multi-dimensional evaluator TheraJudge and a score-aware memory mechanism, it continuously refines plans, significantly outperforming strong baselines in the HealthBench treatment planning subset and blind physician evaluations.

## Background & Motivation
**Background**: LLMs have achieved success in medical QA, diagnostic assistance, and clinical text generation. However, treatment plan generation is more complex than modular QA, requiring simultaneous selection of medications, dosages, indications, contraindications, monitoring metrics, follow-up plans, and risk controls.

**Limitations of Prior Work**: General-purpose LLMs or medically fine-tuned models often adopt one-shot generation. These single-step outputs are frequently superficial, incomplete, or pose safety risks, such as missing dosages, ignoring contraindications, or failing to specify conditions for drug cessation or treatment escalation.

**Key Challenge**: In practice, physicians formulate treatment plans through iterative verification of diagnoses, guidelines, patient conditions, and potential harms. Most LLMs simply provide a fluent text answer. Treatment plan quality is not a single accuracy metric but a combinatorial optimization across multiple clinical dimensions.

**Goal**: The authors aim to build a therapeutic agent capable of self-improvement during inference. The model first generates a draft, receives feedback from a clinical evaluator pointing out issues, and then refines the plan based on that feedback to gradually produce more precise, complete, and safe treatment regimens.

**Key Insight**: The paper formalizes treatment plan quality as $Q(T\mid P)=\sum_i q_i(T\mid P)$, where each $q_i$ corresponds to dimensions such as Accuracy, Targeting, Completeness, and Harm Control. This provides an objective function for multi-dimensional feedback and memory retrieval.

**Core Idea**: An internal critic aligned with clinical standards serves as a feedback source within the reasoning loop, converting "writing a treatment plan" into a test-time optimization process of "generate, evaluate, memorize, and regenerate."

## Method
The core of TheraAgent is not training a new medical model but designing an agentic workflow. The backbone can be strong reasoning models like DeepSeek-R1. The outer system organizes the Planner, TheraJudge, and Memorizer, allowing the model to incorporate previous errors and scores across multiple reasoning rounds.

### Overall Architecture
Input patient cases are represented as $P=(d,s,y)$, where $d$ denotes basic clinical information, $s$ denotes symptoms and test results, and $y$ denotes confirmed diagnoses. The goal is to generate a treatment plan $T$ and an explicit reasoning process $c$. Unlike closed-end classification, treatment plans exist in an open combinatorial space and must satisfy accuracy, completeness, individualization, consensus compliance, and harm control.

In the $k$-th iteration, the Planner generates a candidate plan $T_k$ and reasoning $c_k$ based on case $P$ and the memory $\mathcal{M}^{(k-1)}$ from the previous round. TheraJudge then evaluates this plan, outputting an evaluation rationale $R_k$, scores for each clinical dimension $\{q_{k,i}\}$, and a total score $s_k$. The Memorizer stores $(T_k,R_k,s_k)$ and retrieves high-quality historical plans and reflections in the next round as context for the Planner.

The final output is not simply the last round's answer; instead, the system selects $T^*=\arg\max s_k$ from the last $L$ rounds. The system also implements early stopping: if scores exceed a threshold $\tau$ for three consecutive rounds, the process terminates to reduce unnecessary overhead. The defaults are a maximum of 10 rounds, an output window $L=3$, and a Top-N memory of 3.

### Key Designs
1. **Feedback-Conditional Generation (Planner)**:
    - **Function**: Generates the next version of the treatment plan based on the case and historical feedback.
    - **Mechanism**: The Planner does not only observe the current case but reads the treatment text, evaluation rationale, and scores of the previous or high-scoring historical plans from the Memorizer, formalized as $(T_k,c_k)=f_{\theta}(P,\mathcal{M}^{(k-1)})$.
    - **Design Motivation**: The most common issues in treatment plans are omissions or unclear safety boundaries. Placing explicit evaluation rationales back into the prompt allows the next round of generation to focus on repairing specific defects rather than just "writing better" generally.

2. **Multi-dimensional Clinical Evaluation (TheraJudge)**:
    - **Function**: Provides structured feedback for each candidate plan to be used for iterative optimization.
    - **Mechanism**: TheraJudge outputs a rationale, sub-dimension scores, and a total score. It can utilize RAG to retrieve over 600 clinical guidelines/literature or use 3 few-shot expert examples per department to stabilize scoring. Dimensions include Scientific Consensus Compliance, Plan Completeness, Situation Targeting, Rationale-Measure Coherence, and Harm Control.
    - **Design Motivation**: Generic LLM judges often focus on text fluency or surface-level medical terms. Treatment planning requires a clinical critic that is interpretable, accountable, and capable of identifying specific risks, necessitating evaluation dimensions aligned with real physician judgment.

3. **Score-aware Memorizer and Final Selection**:
    - **Function**: Retains useful experiences during multi-round generation while avoiding cluttering the context with low-quality history.
    - **Mechanism**: The Memorizer saves each round's plan, rationale, and score as $M_i=(T_i,R_i,s_i)$. The next round selects the Top-N memory entries with the highest scores for in-context refinement. The final output is selected based on the TheraJudge score from a window of the last few rounds rather than mechanically taking the final round.
    - **Design Motivation**: Self-improving agents often drift or are misdirected by low-quality reflections in late stages. Score-aware retrieval and final-window argmax control context quality and late-stage fluctuations.

### Loss & Training
This work primarily focuses on test-time optimization and does not involve end-to-end training losses. In HealthBench experiments, both the Planner and TheraJudge use DeepSeek-R1 as the backbone. TheraAgent is configured with Top-N=3, a maximum of 10 rounds, an early stopping threshold $\tau=98$, and a final window $L=3$. To avoid regional guideline bias in HealthBench, RAG is disabled for that benchmark but evaluated in real-case analysis for clinical consensus alignment.

## Key Experimental Results

### Main Results
The authors filtered 1,241 treatment-planning-related cases from HealthBench, covering Endocrine (265), Digestive (262), Neurology (395), and Respiratory (319) departments. Representative results for strong models and TheraAgent are shown below.

| Model | Overall ↑ | Global Health ↑ | Hedging ↑ | Context Seeking ↑ | Communication ↑ | Accuracy ↑ | Completeness ↑ | Context Awareness ↑ |
|------|-----------|-----------------|-----------|-------------------|------------------|------------|----------------|---------------------|
| DeepSeek-R1 | 42.94 | 39.53 | 48.85 | 39.02 | 48.16 | 41.89 | 47.29 | 31.97 |
| Gemini-2.5-Pro | 43.49 | 34.42 | 44.48 | 38.85 | 51.46 | 41.32 | 39.49 | 34.08 |
| Claude-4-Sonnet | 44.28 | 35.10 | 46.50 | 40.91 | 50.64 | 40.63 | 40.86 | 36.26 |
| TheraAgent | 48.94 | 47.49 | 55.63 | 44.65 | 55.29 | 44.80 | 51.72 | 37.16 |

TheraAgent's Overall score is 4.66 points higher than the runner-up, Claude-4-Sonnet. Across dimensions, it exceeds the runner-up by 2.91 in Accuracy and 4.43 in Completeness, indicating that iterative feedback effectively reduces medical errors and plan omissions.

### Ablation Study
The authors verified the effectiveness of the feedback mechanism and memory mechanism from multiple perspectives. Ablations on TheraJudge components show that few-shot examples and dimensional scoring are more critical than pure RAG on HealthBench. Memory ablations show that selecting the "top three scoring memories" is superior to using all or only the most recent memories.

| Configuration | HealthBench Score ↑ | Description |
|------|---------------------|------|
| Base Model w/o Judge | 41.15 | Non-iterative baseline without judge |
| Vanilla Judge | 48.50 | Standard evaluator already yields significant gains |
| Dimensions only | 48.66 | Dimensional scoring provides more specific feedback |
| Few-shots only | 50.62 | Expert examples best stabilize scoring behavior |
| RAG only | 45.98 | RAG alone contributes less on HealthBench |
| Dimensions + Few-shots | 52.36 | Optimal combination balancing structure and stability |
| Dimensions + Few-shots + RAG | 45.96 | Introducing RAG here led to a decrease, possibly due to regional guideline differences |

| Memory Configuration | HealthBench Score ↑ | Interpretation |
|-------------|---------------------|------|
| w/o Memory | 0.4115 | Degenerates to a process lacking historical experience |
| all Memory | 0.4859 | Including all history helps but introduces noise |
| nearest three Memory | 0.5002 | Using recent memories improves performance further |
| best three Memory | 0.5236 | Selecting the top three by score is most effective, highlighting quality filtering |

### Key Findings
- In blind physician evaluations of 35 physician-authored cases, TheraAgent was selected as the best in 65.7% of cases, compared to 25.7% for DeepSeek-R1 and 8.6% for the original physician plans.
- In pairwise comparisons with physician plans, TheraAgent achieved an 86% overall win rate, particularly excelling in Targeting, Completeness, and Harm Control. The paper notes that real-world physician records often omit explicit thresholds and monitoring details due to workflow compression, which TheraAgent expands upon.
- TheraJudge shows significantly higher correlation with HealthBench than traditional text metrics, with Spearman 0.6669, Pearson 0.7052, and CCC 0.6467. BLEU/ROUGE/BERTScore show weak correlations.
- Cost is a significant trade-off. A single DeepSeek-R1 call averages 1,358 tokens and 30.6s. TheraAgent (3 rounds) requires 6 calls, 13,445 tokens, and 332.6s (9.9x cost); at 10 rounds, it reaches 20 calls, 87,005 tokens, and 753.5s (64.1x cost).

## Highlights & Insights
- The paper captures the essence of medical scenarios: treatment planning is not about "answering medical knowledge points" but generating safe decision drafts in an open space with multiple constraints and goals.
- The value of TheraJudge lies not just in final evaluation but in turning evaluation into an optimization signal for the next round. This behaves more like an internal controller for an agent than a standard LLM-as-judge.
- Score-aware memory is a practical detail. A common issue in self-reflection systems is treating all history as experience; in medical planning, allowing errors from low-score plans to re-enter the context can reinforce mistakes.
- The result where TheraAgent outperforms physician plans should be interpreted cautiously: it reflects that physician records are often concise working documents rather than a lack of clinical competence. The paper positions TheraAgent as a tool for structured drafts and safety reminders, not a physician replacement.

## Limitations & Future Work
- Inference costs are high, especially for the 10-round version. While acceptable for high-risk planning, it may not suit emergency medicine, real-time triage, or low-resource hospitals.
- Validation relies on strong models (DeepSeek-R1, GPT-4o); gains on smaller or locally deployed models require systematic evaluation.
- Inputs are currently text-based. It has not yet directly integrated laboratory time-series, imaging, vital sign monitoring, or structured EMR data, which are often required for real-world planning.
- TheraJudge may still produce erroneous evaluations. Even with high correlation to HealthBench, it is not a clinical gold standard; real-world deployment requires physician oversight and local guideline adaptation.

## Related Work & Insights
- **vs MedPlan**: MedPlan is more of a phased/RAG system based on clinical workflows, while TheraAgent focuses on multi-round self-improvement and internal judge feedback.
- **vs TxAgent**: TxAgent emphasizes treatment reasoning and tool ecosystems, whereas TheraAgent emphasizes iterative evaluation, memory, and rewriting of treatment texts.
- **vs General self-reflection agents**: Standard reflection agents use free-form self-evaluation. TheraAgent constrains reflection to clinical dimensions, guideline evidence, and expert examples, making it more suitable for safety-sensitive domains.
- **Inspirations for Future Research**: Medical agents should not just pursue longer chains-of-thought but should modularize evaluation dimensions, reflection memories, and risk controls for auditability, clearly defining when a case must be handed over to a human physician.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematizing self-improving agents for treatment planning; the TheraJudge + Memory combination is practical.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers HealthBench, physician blind reviews, judge agreement, cost, and ablation, though missing multi-modal clinical inputs.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation; framework diagrams and case analyses are persuasive.
- Value: ⭐⭐⭐⭐⭐ High reference value for the safe deployment of medical LLM agents, particularly in incorporating clinical evaluators into the reasoning loop.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](../../ICLR2026/llm_agent/agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](../../NeurIPS2025/llm_agent/zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [\[ACL 2026\] Why LLM Web Agents Fail: A Hierarchical Planning Perspective](why_do_llm-based_web_agents_fail_a_hierarchical_planning_perspective.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](../../ICLR2026/llm_agent/openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ICML 2026\] Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling](../../ICML2026/llm_agent/agent_jit_compilation_for_latency-optimizing_web_agent_planning_and_scheduling.md)

</div>

<!-- RELATED:END -->
