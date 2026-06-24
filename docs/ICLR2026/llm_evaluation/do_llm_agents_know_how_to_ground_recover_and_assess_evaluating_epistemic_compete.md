---
title: >-
  [Paper Note] Do LLM Agents Know How to Ground, Recover, and Assess? Evaluating Epistemic Competence in Information-Seeking Agents
description: >-
  [ICLR 2026][LLM Evaluation][Search Agents] The authors propose SeekBench—the first **process-level** evaluation framework for LLM search agents. It decomposes "ability to use evidence" into three epistemic competencies: groundedness, recovery, and calibration, with quantifiable metrics (RQI / ERF / CE). Utilizing 190 expert-annotated trajectories to calibrate a highly consistent annotation schema, the framework scales to 28,493 trajectories via LLM-as-judge…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Search Agents"
  - "Process-level Evaluation"
  - "Epistemic Competence"
  - "Evidence State"
  - "LLM-as-judge"
date: 2026-05-08
content_hash: 0781348bde38585e
---

# Do LLM Agents Know How to Ground, Recover, and Assess? Evaluating Epistemic Competence in Information-Seeking Agents

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=r0L9GwlnzP](https://openreview.net/forum?id=r0L9GwlnzP)  
**Code**: https://github.com/SHAO-Jiaqi757/SeekBench  
**Area**: LLM Evaluation / Search Agents / Benchmark  
**Keywords**: Search Agents, Process-level Evaluation, Epistemic Competence, Evidence State, LLM-as-judge  

## TL;DR
The authors propose SeekBench—the first **process-level** evaluation framework for LLM search agents. It decomposes "ability to use evidence" into three epistemic competencies: groundedness, recovery, and calibration, with quantifiable metrics (RQI / ERF / CE). Utilizing 190 expert-annotated trajectories to calibrate a highly consistent annotation schema, the framework scales to 28,493 trajectories via LLM-as-judge, revealing behavioral defects invisible to final answer accuracy.

## Background & Motivation

**Background**: Recent works utilize Reinforcement Learning (RL) to train LLM search agents for open-domain QA. These agents iteratively "identify information gaps → retrieve external evidence → reason over evidence → decide next action or provide an answer," implicitly learning decision strategies via RL. Currently, evaluation almost exclusively relies on **final answer** metrics such as exact match or F1.

**Limitations of Prior Work**: Correct answers do not guarantee a correct process. An agent might guess the correct answer while ignoring conflicting sources, failing to recognize ambiguity, or answering prematurely with insufficient evidence, thus achieving high benchmark scores without a trustworthy reasoning process. Figure 1 in the paper illustrates an example where two agents both output "455,000"—one by guessing and one through rigorous verification—yet answer-level metrics treat them identically.

**Key Challenge**: Information retrieval tasks possess an inherent difficulty distinct from code or mathematics—they lack an **objective verifier**. Code can be executed and proofs can be verified, but determining whether "a retrieved text segment supports a conclusion" cannot be automatically judged. Consequently, evaluation either degrades into answer-checking or requires expensive manual trajectory review; neither addresses at scale whether the agent is reasonably acquiring, evaluating, and applying knowledge.

**Goal**: To operationalize the abstract "epistemic competence" into properties **measurable on trajectories**, achieving both high accuracy (inter-annotator agreement) and scalability (automatic evaluation of thousands of steps).

**Key Insight**: Borrowing from Content Analysis in social sciences and construct validity in psychometrics, the authors first observe real trajectories to induce observable behavioral labels, infer latent competencies from these behaviors, and finally translate these constructs into quantitative metrics. This "observable features → latent competence → quantitative metrics" pipeline ensures the metrics are theoretically grounded.

**Core Idea**: Use an **evidence state** defined as $E=C+Q$ as a unified pivot. Groundedness, recovery, and calibration are defined as measurable functions over this evidence state. An LLM-as-judge pipeline is then used to replicate expert annotations, enabling process-level, large-scale, and interpretable agent evaluation.

## Method

### Overall Architecture

SeekBench formalizes a multi-turn trajectory for a search agent as $T=\langle \tau_1,\tau_2,\dots,\tau_T\rangle$, where non-terminal turns $\tau_t=\langle r_t, s_t, e_t\rangle$ consist of reasoning $r_t$, search $s_t$, and evidence $e_t$, while the terminal turn $\tau_T=\langle r_T, a_T\rangle$ provides the answer $a_T$. The framework is built in three stages: **Stage 1** uses content analysis to iteratively develop and calibrate an annotation schema, labeling each step with "functional type" and "quality attributes"; 190 expert-annotated trajectories (1,800+ steps) were used to refine 12 candidate fields into 8 high-consistency features (Cohen's $\kappa>0.8$). **Stage 2** induces three core epistemic competencies from observed behavioral variances (whether reasoning is grounded, strategy adjustment under poor results, and timing of answers). **Stage 3** translates each competence into a quantitative metric (RQI / ERF / CE), all based on a unified definition of "evidence state." Finally, an LLM-as-judge pipeline expands the evaluation to 28,493 trajectories across 283,950 steps.

### Key Designs

**1. Dual-dimension Annotation Schema: Labeling "What" and "How Well"**

Process-level evaluation faces the hurdle that reasoning steps serve different purposes: identifying gaps, synthesizing findings, or planning. Labeling them all as "reasoning" is insufficient. The authors set two orthogonal dimensions: **Functional Type** (Reasoning steps: InformationSynthesis, PlanFormation, StateAssessment; Search steps: Exploration, Repeat, Follow-up, Refinement) and **Quality Attributes** (e.g., whether reasoning is supported by evidence, whether evidence is clear and sufficient). This "what $\times$ how well" structure allows for decomposed competence measurement.

**2. Evidence State $E=C+Q$: The Unified Pivot**

The evidence state $E$ links the three competencies. For evidence retrieved at turn $t$ of trajectory $i$, two binary variables are annotated: Clarity $C_{i,t}\in\{0,1\}$ (unambiguity) and Sufficiency/Quality $Q_{i,t}\in\{0,1\}$ (whether it contains enough information to answer). The evidence state is defined as:

$$E_{i,t} := C_{i,t}+Q_{i,t}\in\{0,1,2\}$$

Where $E=0$ represents poor evidence, $E=1$ partial evidence, and $E=2$ good evidence. Groundedness, recovery, and calibration are all defined using $E_{i,t}$ as the coordinate axis.

**3. Three Competencies and Metrics: RQI, ERF, and CE**

*Groundedness — Reasoning Quality Index (RQI)*: Each reasoning step is given a binary grounded label $G_{i,t}\in\{0,1\}$ (factually supported by evidence). The model-level RQI is the average: $\text{RQI}_{\text{model}}:=\mathbb{E}_{i}[\mathbb{E}_{t}[G_{i,t}]]$. RQI can be decomposed by evidence state: $\text{RQI}_i=\sum_{k=0}^{2}P_t(E_{i,t}=k)\cdot\mathbb{E}_t[G_{i,t}\mid E_{i,t}=k]$, distinguishing between "poor grounding due to bad evidence" vs. "failure to use good evidence."

*Recovery — Evidence Recovery Function (ERF)*: A recovery event is defined as the first turn $T_{\text{recover},i}$ where an agent reaches $E=2$ or provides a correct answer. $\text{ERF}(t)$ is the proportion of trajectories recovered by turn $t$. Steeper curves indicate faster recovery from poor evidence. Kaplan–Meier survival analysis is used to handle right-censored data.

*Calibration — Calibration Error (CE)*: The ideal strategy $\pi^*(k)$ is to answer if and only if evidence is good ($E=2$). CE measures the deviation:

$$\text{CE}_i := \sum_{k=0}^{2} P(E_{i,t}=k)\,\bigl|P(answer_{i,t}=1\mid E_{i,t}=k)-\pi^*(k)\bigr|$$

This captures both overconfidence (answering at $k=0$) and over-cautiousness (not answering at $k=2$).

**4. LLM-as-judge Pipeline: Scaling to Tens of Thousands of Trajectories**

The authors used the calibrated schema as a prompt for LLMs to automate labeling. After cost-benefit analysis, GPT-4.1-mini was selected as the primary judge, achieving high consistency ($\kappa=0.731$) with human experts at a low cost (\$0.0087 per trajectory).

## Key Experimental Results

The evaluation covers agents based on Qwen-2.5-7B (Base, Few-shot, Search-R1, etc.) across 7 QA benchmarks (NQ, TriviaQA, HotpotQA, etc.) and GAIA.

### Main Results: Answer Ranking vs. Process Competence

| Dimension | Key Findings |
|------|---------|
| Answer F1 Ranking | ASEARCHER > Search-R1 > RESEARCH > Few-shot ≈ DEEPRESEARCHER > Base |
| Groundedness (RQI) | **Few-shot is highest (0.27)**, surpassing all RL agents. RL optimizes answer accuracy but fails to foster the ability to ground reasoning in evidence. |
| Grounding by Type | Information Synthesis (IS) is a relative strength; **Plan Formation (PF) is the greatest weakness for all agents (generally < 0.2)**. |
| Recovery (ERF) | ASEARCHER (highest F1) performs best in recovery, while DEEPRESEARCHER (lowest F1) is worst. Recovery correlates with final performance. |

### Calibration Analysis (CE, lower is better)

| Model | Overconfidence ↓ | Over-cautiousness ↓ | Calibration Error CE ↓ |
|------|-----------|-----------|--------------|
| Base | 0.631 | 0.030 | 0.329 |
| Few-shot | 0.511 | 0.024 | 0.317 |
| RL-trained | **0.353** | 0.085 | **0.309** |

RL training significantly reduces overconfident answering from 63.1% to 35.3%, leading to the lowest CE. RL teaches models "answer only when evidence is sufficient."

### Key Findings

- **Evidence quality drives accuracy**: RL agents exhibit 31.6% accuracy under $E=2$ but only 8.4% without evidence support.
- **Answer-level metrics mislead**: F1 scores fail to detect Search-R1's synthesis strengths or the fact that Base agents possess non-negligible reasoning capabilities.
- **Competence complementarity**: In "agent composition" experiments, Search-R1 is the best evidence collector, while the Base model gains the most from high-quality evidence (+2.42 F1), suggesting standard benchmarks overestimate RL gains in pure reasoning.

## Highlights & Insights

- **The Evidence State as a bridge**: $E=C+Q$ uses two binary labels to unify groundedness, recovery, and calibration into a single coordinate system, allowing for cross-metric comparison and root-cause failure analysis.
- **Social Science methodology**: The use of content analysis and construct validity provides a rigorous paradigm for subjective LLM evaluation problems that lack objective verifiers.
- **Counter-intuitive finding**: RL training **improves calibration** yet **damages groundedness**. This indicates that competence development is not monolithic, explaining why answer-level metrics can misguide R&D.
- **Kaplan–Meier Survival Analysis**: This statistical tool effectively handles variable-length trajectories and right-censored data in recovery estimation.

## Limitations & Future Work

- **Dependence on LLM-as-judge**: Biases of GPT-4.1-mini might propagate, and the subjective nature of "clarity" means inter-annotator agreement is not perfect.
- **Coarse granularity of $E$**: Binary $C$ and $Q$ terms do not capture "partially relevant" or "slight ambiguity" states.
- **Model Scope**: Evaluation is primarily focused on the Qwen-2.5-7B family; generalizability to different scales or architectures requires further verification.
- **Future Directions**: Refining evidence states into multi-level scales and utilizing these metrics as training signals for joint optimization of groundedness and calibration.

## Related Work & Insights

- **vs. Answer-level Evaluation**: Proves that focusing solely on $a_T$ underestimates base reasoning while overestimating RL gains. Process-level evaluation provides a finer diagnostic.
- **vs. Faithfulness/CoT Alignment**: Unlike previous works focusing on internal consistency, SeekBench formalizes the interaction between external evidence and agent behavior (grounding, recovery, and calibration) using a mathematical framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First process-level framework for search agents; elegant evidence state design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 28k+ trajectories, multiple benchmarks, and consistency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology and rigorous definitions.
- Value: ⭐⭐⭐⭐⭐ Provides a practical diagnostic tool for developing trustworthy retrieval agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents](researchrubrics_a_benchmark_of_prompts_and_rubrics_for_evaluating_deep_research_.md)
- [\[ICLR 2026\] From Reproduction to Replication: Evaluating Research Agents with Progressive Code Masking](from_reproduction_to_replication_evaluating_research_agents_with_progressive_cod.md)
- [\[ICLR 2026\] HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities](hackworld_evaluating_computer-use_agents_on_exploiting_web_application_vulnerabi.md)
- [\[ICLR 2026\] Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks](can_llms_refuse_questions_they_do_not_know_measuring_knowledge-aware_refusal_in_.md)
- [\[ICLR 2026\] CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale](cybergym_evaluating_ai_agents_real-world_cybersecurity_capabilities_at_scale.md)

</div>

<!-- RELATED:END -->
