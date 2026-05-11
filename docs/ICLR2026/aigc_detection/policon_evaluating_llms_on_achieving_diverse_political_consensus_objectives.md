---
title: >-
  [Paper Note] PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives
description: >-
  [ICLR 2026][AIGC Detection][Political Consensus] The PoliCon benchmark is constructed from 2,225 high-quality deliberation records from the European Parliament (2009–2022) to evaluate LLMs' ability to draft consensus res…
tags:
  - "ICLR 2026"
  - "AIGC Detection"
  - "Political Consensus"
  - "LLM Evaluation"
  - "European Parliament"
  - "Social Choice Theory"
  - "Vote Simulation"
date: 2026-05-08
content_hash: 3fa13b79d7ce3f16
---

# PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives

**Conference**: ICLR 2026
**arXiv**: [2505.19558](https://arxiv.org/abs/2505.19558)

**Code**: Available
**Area**: AIGC Detection / LLM Evaluation

**Keywords**: Political Consensus, LLM Evaluation, European Parliament, Social Choice Theory, Vote Simulation

## TL;DR

The PoliCon benchmark is constructed from 2,225 high-quality deliberation records spanning 13 years (2009–2022) of the European Parliament. By designing diverse voting mechanisms (simple majority / two-thirds majority / veto power), power structures, and political objectives (utilitarianism / Rawlsianism), the benchmark systematically evaluates the ability of LLMs to draft political consensus resolutions, revealing the shortcomings of frontier models on complex consensus tasks and their inherent partisan biases.

## Background & Motivation

- **Importance of political consensus**: In pluralistic societies, consensus-building—from infrastructure to welfare policy—is the foundation of the legitimacy and enforceability of collective decisions, yet the process is highly challenging due to value conflicts, power dynamics, and issue complexity.
- **Exploration of LLMs in governance**: Although LLMs have demonstrated potential in facilitating group deliberation, supporting democratic discussion, and resolving regional conflicts, their ability to reach consensus in genuinely complex political settings remains unsystematically studied.
- **Limitations of prior work**: Existing political science benchmarks focus on stance detection, ideology analysis, or text summarization, and none specifically evaluate LLMs' ability to draft political consensus resolutions under diverse consensus objectives.
- **Core problem**: Can LLMs bridge the divide among stakeholders with divergent positions and achieve different types of political consensus in real-world political environments?

## Method

### Data Collection and Preprocessing

Data were sourced from three channels: the official European Parliament website, HowTheyVote, and the VoteWatch Europe dataset, covering the 7th–9th parliamentary terms (2009–2022). After rigorous filtering (confirming completed final votes and information completeness) from 30,698 raw records, 2,225 high-quality entries were retained. Each entry consists of a six-tuple: $(issue, topic, background, stances, resolution, votes)$, where DeepSeek-R1 is used for background summarization and stance extraction, and rule-based synonym substitution diversifies the stance data. Voting data are processed by matching each MEP to their party and rounding support rates to integers in the range 0–9.

### Task Environment Design

PoliCon constructs the task environment via four adjustable factors:

| Factor | Description | Specific Settings |
|--------|-------------|-------------------|
| Political Issue | The political question under discussion and its thematic category | 5 coarse + 19 fine-grained topic types (security, economy, etc.) |
| Political Objective | The criterion for achieving consensus | Pass resolution / Rawlsianism / Utilitarianism |
| Participants | Stakeholders varying in number and position | 2, 4, or 6 parties |
| Power Structure | Influence differences arising from seat allocation | Randomly assigned seat proportions $\sum_{i=1}^{n} w_i = 1$ |

### Voting Mechanisms and Political Objectives

The voting mechanisms simulate real-world collective decision-making rules. The overall vote outcome is $u = \sum_{i=1}^{n} w_i u_i$, where $w_i$ is the seat proportion of party $p_i$ and $u_i$ is that party's vote score:

| Mechanism / Objective | Passing Condition | Real-World Analog |
|-----------------------|-------------------|-------------------|
| Simple Majority (SM) | $u \geq 5$ | Routine votes in most parliaments |
| Two-Thirds Majority (2/3M) | $u \geq 6.67$ | Major decisions such as constitutional amendments |
| Veto Power (VP) | $u \geq 5$ and $u_k \geq 6$ | UN Security Council permanent member veto |
| Rawlsianism (Rawls) | $u = \min_{i \in n}(u_i)$, maximizing the least advantaged party | Protecting minority group interests |
| Utilitarianism (Util) | $u = \sum_{i=1}^{n} u_i$, maximizing aggregate utility | Maximizing overall social welfare |

Combining 3 party-count settings × 5 configurations yields 15 task setups covering 28,620 specific political scenarios.

### Open-Ended Evaluation Framework

The evaluation framework is grounded in Social Choice Theory and comprises two modules:

1. **Vote Simulation Module**: Adopts an LLM-as-a-judge approach (GPT-4o-mini backbone) to produce a vote score in the range 0–9 for each party: $u_i = \text{JUDGE}(\cdot \mid \text{background}, s_i, \text{resolution})$, considering both consistency between the resolution and each party's stance and overall feasibility.
2. **Consensus Assessment Module**: Maps all votes to a quantitative score according to the specific task definition and determines whether the corresponding political consensus objective has been achieved.

## Experimental Setup

- **Evaluated models**: 6 representative LLMs—GPT-4o, Gemini-2.5-Flash (thinking), DeepSeek-V3.1 (thinking), Qwen2.5-72B, Qwen2.5-32B, and Llama-3.3-70B.
- **Inference parameters**: temperature=0.7, top-p=0.95.
- **Baselines**: Random (randomly selecting one party's stance as the resolution) and Greedy (selecting the stance of the party with the most seats as the resolution).
- **Evaluator validation**: Compared against ground-truth votes on approximately 41,800 test samples, achieving a Pearson correlation of 0.83; in human annotator agreement experiments, the mean error was only 1.61 and 72% of errors fell within $\pm 1.92$.

## Key Experimental Results

| Model | SM (2/4/6 parties) | 2/3M (2/4/6 parties) | VP (2/4/6 parties) | Rawls (2/4/6 parties) | Util (2/4/6 parties) |
|-------|--------------------|----------------------|--------------------|----------------------|----------------------|
| Random | 0.56/0.53/0.56 | 0.29/0.20/0.14 | 0.36/0.35/0.38 | 2.59/2.01/1.77 | 5.04/4.78/4.80 |
| Greedy | 0.80/0.74/0.73 | 0.45/0.37/0.28 | 0.46/0.44/0.44 | 2.61/2.02/1.74 | 5.07/4.79/4.79 |
| Qwen2.5-32B | 0.74/0.80/0.87 | 0.34/0.39/0.40 | 0.47/0.55/0.62 | 4.02/3.50/3.19 | 6.01/6.27/6.38 |
| Llama-3.3-70B | 0.72/0.78/0.86 | 0.37/0.45/0.48 | 0.46/0.55/0.63 | 3.98/3.42/3.11 | 6.08/6.40/6.56 |
| Qwen2.5-72B | 0.76/0.82/0.88 | 0.40/0.47/0.49 | 0.50/0.57/0.65 | 4.11/3.46/3.13 | 6.11/6.39/6.53 |
| GPT-4o | 0.83/0.87/0.92 | 0.51/0.57/0.63 | 0.54/0.62/0.69 | 4.50/3.80/3.42 | 6.40/6.62/6.80 |
| DeepSeek-V3.1 | 0.87/0.89/0.93 | 0.52/0.57/0.63 | 0.58/0.64/0.71 | 4.52/3.78/3.42 | 6.38/6.62/6.77 |
| Gemini-2.5 | **0.88/0.90**/0.90 | **0.53/0.57**/0.58 | **0.61/0.66**/0.70 | **4.60/3.91/3.51** | 6.39/6.56/6.68 |

**Key Findings**:
- Gemini-2.5 achieves the best overall performance, attaining top results on 60% of tasks, followed closely by DeepSeek-V3.1 and GPT-4o.
- Model pass rates reach 87–93% on SM tasks but drop sharply to 52–63% on 2/3M tasks.
- Thinking models (Gemini-2.5, DeepSeek-V3.1) generally outperform non-thinking models.
- As the number of parties increases, the pass rate for resolution tasks actually rises (because task construction preferentially selects parties with the greatest stance divergence, making reconciliation harder with fewer parties), whereas performance on the Rawls objective declines (more parties make it harder to satisfy all parties simultaneously).
- LLMs lack the ability to form coalitions among smaller parties to achieve collective welfare; successful proposals tend to rely on the support of the largest party.
- Topics involving security and civil rights are more challenging than those related to industrial development.

## Key Designs

1. **Multi-dimensional task environment construction**: By combining four factors—political issue, political objective, number of participants, and power structure—28,620 political scenarios covering diverse consensus objectives are generated from 2,225 real records, ensuring comprehensive and realistic evaluation.
2. **Social Choice Theory-based evaluation framework**: LLM-as-a-judge vote simulation is integrated with Social Choice Theory (simple majority / two-thirds majority / veto power / Rawlsianism / utilitarianism) to enable automated quantitative evaluation of open-ended text outputs.
3. **Partisan bias detection mechanism**: By randomly reassigning seat proportions and observing the resulting vote score distributions, the study finds that LLM scores still tend toward the real vote distribution (rather than the uniform distribution of the random baseline), thereby quantitatively revealing the inherent partisan biases of the models.

## Highlights & Insights

- ⭐⭐⭐ **Novelty**: The first benchmark to systematically evaluate LLM performance across multiple political consensus objectives; the integration of Social Choice Theory into the LLM evaluation framework represents a novel and practically significant problem formulation.
- ⭐⭐⭐ **Experimental Design**: The four-factor combination generates 28,620 scenarios with broad coverage; the high agreement between the evaluator and ground-truth votes (Pearson 0.83) provides a solid validation foundation.
- ⭐⭐ **Practicality**: While the work reveals LLM limitations and biases, it remains far from genuinely assisting political decision-making; the paper is primarily diagnostic and lacks concrete proposals for improving models' consensus-building capabilities.
- ⭐⭐ **Limitations**: The evaluator itself is based on GPT-4o-mini, introducing circularity concerns in model self-evaluation; random seat assignment increases diversity but may deviate from real power dynamics; generalizability beyond European parliamentary data to other political systems remains uncertain.
- **Potential extensions**: (1) Introducing multi-round negotiation mechanisms rather than single-shot resolution generation; (2) exploring prompting strategies or fine-tuning methods to improve performance on harder tasks such as two-thirds majority; (3) extending the framework to other collective decision-making settings such as corporate governance or community deliberation.

## Related Work & Insights

| Direction | Representative Work | Relation to This Paper |
|-----------|--------------------|-----------------------|
| LLM-assisted democratic deliberation | Tessler et al. 2024; Fish et al. 2023 | Prior work focuses on opinion aggregation and group statement generation; this paper further evaluates consensus achievement under formal political institutions (voting mechanisms + power structures). |
| Political science benchmarks | POLCA; Liang et al. 2025 | POLCA only judges whether statements appear in a final agreement; Liang focuses on UN position simulation; this paper is the first to construct a multi-objective political consensus evaluation benchmark. |
| LLM bias detection | Stammbach et al. 2024; Chalkidis & Brandl 2024 | Prior work detects whether models have inherent political leanings; this paper further reveals how such biases affect actual consensus-building performance. |
| Negotiation and game theory | Lewis et al. 2017; Bianchi et al. 2024 | Game-theoretic negotiation focuses on two-party transaction scenarios; this paper addresses multi-party, multi-objective, and varying power-structure settings in real political consensus problems. |

---
title: >-
  [Paper Notes] PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives
description: >-
  [ICLR 2026][Political Consensus] PoliCon is a benchmark constructed from 2,225 high-quality European Parliament deliberation records (2009–2022), evaluating LLMs' ability to draft consensus resolutions under diverse voting mechanisms, power structures, and political objectives. Results show that frontier models perform reasonably well on simple majority tasks but fall significantly short on two-thirds majority and security-related topics.
tags:
  - ICLR 2026
  - Political Consensus
  - LLM Evaluation
  - European Parliament
  - Social Choice Theory
  - Vote Simulation
---

# PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives

**Conference**: ICLR 2026
**arXiv**: [2505.19558](https://arxiv.org/abs/2505.19558)

**Code**: Available
**Area**: AIGC Detection / LLM Evaluation

**Keywords**: Political Consensus, LLM Evaluation, European Parliament, Social Choice Theory, Vote Simulation

## TL;DR

The PoliCon benchmark is constructed from 2,225 high-quality deliberation records from the European Parliament (2009–2022) to evaluate LLMs' ability to draft consensus resolutions under diverse voting mechanisms, power structures, and political objectives. Results show that frontier models perform reasonably well on simple majority tasks but fall significantly short on two-thirds majority tasks and security-related topics.

## Background & Motivation

Establishing political consensus is a prerequisite for effective governance in pluralistic societies. LLMs have demonstrated potential in facilitating group discussions and supporting democratic deliberation, yet their ability to achieve different consensus objectives in genuinely complex political settings has not been systematically evaluated. Existing political science benchmarks focus on stance classification or text analysis and do not assess LLMs' capacity to "find consensus."

PoliCon designs four adjustable factors: (1) political issue and thematic category; (2) political objective (simple majority / two-thirds majority / veto power / Rawlsianism / utilitarianism); (3) number of participating parties (2 / 4 / 6); (4) seat-based power structure. Their combination yields 28,620 scenarios.

## Method

### Overall Architecture

(1) Large-scale crawling and cleaning from the European Parliament website, VoteWatch, and HowTheyVote → (2) Cleaning and structuring with DeepSeek-R1 → (3) Construction of the evaluation framework (vote simulation + score mapping) → (4) Evaluation of 6 frontier LLMs.

### Key Designs

1. **Data collection and preprocessing**: Issues, debates, resolutions, and votes are matched; 2,225 complete records are filtered from 30,698 raw entries. Topics are organized into 5 coarse-grained and 19 fine-grained categories.

2. **Evaluation framework**: Two stages — (a) GPT-4o-mini simulates each party's voting proportion (0–9 score), achieving Pearson r=0.83 against real votes; (b) votes are mapped to quantitative scores according to the political objective.

3. **Five political objectives**: Simple Majority ($u \geq 5$), Two-thirds Majority ($u \geq 6.67$), Veto Power ($u \geq 5$ and veto party $\geq 6$), Rawlsianism ($\max \min_i u_i$), Utilitarianism ($\max \sum u_i$).

4. **Power structure**: Seat proportions are randomly assigned to expose LLMs' latent biases toward different parties.

### Loss & Training

No training is involved. This is a pure evaluation framework. LLM inference uses temperature=0.7 and top_p=0.95.

## Key Experimental Results

### Main Results (6-party setting)

| Model | SM | 2/3M | VP | Rawls | Util |
|-------|----|------|----|-------|------|
| Random | 0.56 | 0.14 | 0.38 | 1.77 | 4.80 |
| Greedy | 0.73 | 0.28 | 0.44 | 1.74 | 4.79 |
| GPT-4o | 0.87 | 0.51 | 0.66 | 2.36 | 5.38 |
| DeepSeek-V3.1 | **0.92** | **0.58** | **0.73** | **2.59** | **5.55** |

### Ablation Study

| Setting | Difficulty Change | Explanation |
|---------|------------------|-------------|
| 2 → 6 parties | SM stable; 2/3M drops sharply | More parties → harder to reach supermajority |
| Security topics | Significant drop across all models | Political sensitivity affects outputs |
| Dominant-party bias | Models tend to favor the dominant party | Rather than forming coalitions among smaller parties |

### Key Findings

- All models perform well on simple majority (>80%), but performance drops significantly on two-thirds majority (~40–58%).

- Security and defense topics are the most challenging category—likely because models' safety training constrains relevant outputs.

- LLMs tend to prioritize the stance of the party with the most seats rather than attempting to form coalitions among smaller parties, revealing an implicit "strong-party-first" bias.

- The Greedy baseline (always adopting the dominant party's stance) proves surprisingly effective in some settings, indicating that LLM strategies offer little improvement over simple heuristics.

- Agreement analysis with 20 human annotators confirms the reliability of the evaluation framework.

## Highlights & Insights

- The first benchmark to systematically evaluate LLMs' political consensus-building capabilities, with a well-designed setup (multiple objectives + multiple power structures).

- Reveals LLMs' implicit political bias: models tend to accommodate the dominant party rather than seeking genuine compromise.

- The Social Choice Theory-based evaluation framework provides an actionable assessment scheme for open-ended political tasks.

## Limitations & Future Work

- The benchmark is based solely on European Parliament data; performance under different political systems remains to be evaluated.

- The vote simulation evaluator (GPT-4o-mini) may itself carry biases.

- Real political negotiations involve dynamic processes such as compromise and issue-linkage, whereas the current setup is limited to single-round generation.

## Related Work & Insights

- Complements the group consensus work of Tessler et al. (2024) while focusing on more formal political settings.

- The framework is extensible to collective decision-making evaluation settings such as corporate governance and community deliberation.

## Rating

- Novelty: ⭐⭐⭐⭐ First political consensus evaluation benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 models + multiple settings + human validation
- Writing Quality: ⭐⭐⭐⭐ Clear structure
- Value: ⭐⭐⭐⭐ Significant implications for AI governance

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](../../ACL2026/aigc_detection/reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)
- [\[ACL 2026\] CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation](../../ACL2026/aigc_detection/citeguard_faithful_citation_attribution_for_llms_via_retrieval-augmented_validat.md)
- [\[NeurIPS 2025\] ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](../../NeurIPS2025/aigc_detection/asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](../../ACL2026/aigc_detection/who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[NeurIPS 2025\] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](../../NeurIPS2025/aigc_detection/can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)

</div>

<!-- RELATED:END -->
