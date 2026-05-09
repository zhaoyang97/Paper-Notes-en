---
title: >-
  [Paper Note] SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks
description: >-
  [NeurIPS 2025 (Datasets & Benchmarks Track, Spotlight)][Dialogue Systems][scientific literature evaluation] SciArena is a community-driven open evaluation platform for scientific literature tasks. It adopts a Chatbot Arena-style human preference voting paradigm to rank 47 foundation models, collecting over 20,000 votes, and releases SciArena-Eval as a meta-benchmark for assessing the ability of automated evaluation systems to judge answer quality on literature-grounded tasks.
tags:
  - "NeurIPS 2025 (Datasets & Benchmarks Track, Spotlight)"
  - Dialogue Systems
  - scientific literature evaluation
  - foundation models
  - human preference
  - Chatbot Arena
  - meta-benchmark
date: 2026-05-08
content_hash: 475bce88a53a7be8
---

# SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks

**Conference**: NeurIPS 2025 (Datasets & Benchmarks Track, Spotlight)
**arXiv**: [2507.01001](https://arxiv.org/abs/2507.01001)
**Code**: Available
**Area**: Dialogue Systems
**Keywords**: scientific literature evaluation, foundation models, human preference, Chatbot Arena, meta-benchmark

## TL;DR
SciArena is a community-driven open evaluation platform for scientific literature tasks. It adopts a Chatbot Arena-style human preference voting paradigm to rank 47 foundation models, collecting over 20,000 votes, and releases SciArena-Eval as a meta-benchmark for assessing the ability of automated evaluation systems to judge answer quality on literature-grounded tasks.

## Background & Motivation

**Background**: Scientific literature understanding and synthesis is a critical application scenario for foundation models, encompassing open-ended tasks such as paper-based question answering, literature review generation, and hypothesis proposal. Traditional benchmarks typically rely on closed-set tasks with automatically verifiable answers (e.g., multiple-choice or short-answer questions); however, answers to scientific literature tasks are often long-form and open-ended, making programmatic quality assessment infeasible.

**Limitations of Prior Work**: Existing scientific literature benchmarks are either restricted to simple extractive question answering or rely on fixed reference answers for scoring, failing to capture researchers' holistic preferences for model responses. Automated evaluation systems (e.g., LLM-as-judge) also offer limited guarantees of alignment with human judgments on such non-verifiable tasks.

**Key Challenge**: Evaluating model capabilities on scientific literature tasks requires domain expert involvement, yet large-scale and sustained human evaluation is prohibitively expensive. Achieving scalable, community-driven evaluation without sacrificing quality is a key challenge. Drawing on the success of Chatbot Arena, this paper constructs a community-driven evaluation platform tailored to scientific literature tasks, enabling researchers to compare model outputs through pairwise preference voting, while releasing a meta-evaluation benchmark to advance more reliable automated evaluation methods.

## Method

### Overall Architecture
SciArena adopts a two-tier framework combining human preference voting and an automated evaluation meta-benchmark. The lower tier is a web-based interactive platform where researchers submit scientific literature task queries; the system routes each query to two anonymous models and collects pairwise preference votes. The upper tier constructs the SciArena-Eval meta-benchmark from the collected preference data to evaluate automated evaluators.

### Key Designs

1. **Community-Driven Evaluation Platform**:

    - Function: Provides an open, collaborative web platform where researchers can submit scientific literature tasks (e.g., paper QA, literature synthesis) and cast pairwise comparison votes on model outputs.
    - Mechanism: Adopts the pairwise comparison paradigm from Chatbot Arena — upon query submission, the system randomly pairs two anonymous models, and users vote based on response quality (A is better / B is better / Tie).
    - Design Motivation: The quality of scientific literature task responses is difficult to assess programmatically, necessitating expert judgment from domain researchers. Chatbot Arena has already validated the effectiveness of community voting in general-purpose dialogue settings.

2. **Model Ranking and Elo Rating System**:

    - Function: Generates reliable model rankings from preference voting data.
    - Mechanism: Applies Elo rating or the Bradley-Terry model to convert pairwise votes into global rankings, supporting fair comparison across 47 foundation models.
    - Design Motivation: Pairwise comparisons are more stable and more consistent with human judgment than absolute scoring; the ranking system can be dynamically updated as new data are collected.

3. **SciArena-Eval Meta-Evaluation Benchmark**:

    - Function: Measures the accuracy of automated evaluation systems (e.g., LLM-as-judge) in judging the quality of responses to literature-grounded tasks.
    - Mechanism: Uses human votes as the gold standard and checks whether the pairwise judgments of automated evaluators align with human preferences.
    - Design Motivation: High-quality human evaluation is costly; reliable automated methods would substantially reduce evaluation overhead. However, the accuracy of existing methods on scientific literature tasks is unknown, motivating the need for a standardized meta-benchmark.

### Data Collection and Quality Assurance
The platform employs vote consistency analysis and statistical testing to verify data quality, ensuring that collected preference votes reflect genuine differences in model capability rather than random noise. All votes are provided by human researchers spanning multiple scientific disciplines, including computer science, physics, biology, and medicine. The platform incorporates bias-control measures such as model anonymization (to eliminate brand preference), random pairing (to avoid oversampling specific model pairs), and statistical anomaly detection on voting patterns.

## Key Experimental Results

### Main Results

| Dimension | Data |
|-----------|------|
| Supported models | 47 foundation models |
| Total votes | 20,000+ from human researchers |
| Coverage | Multiple scientific disciplines |
| Task types | Open-ended scientific literature tasks requiring long-form, literature-grounded responses |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| LLM-as-judge baseline | Agreement rate with human votes | Automated evaluation systems show substantial room for improvement in alignment with human judgments on scientific literature tasks |
| Different models as evaluators | Preference alignment | Significant variation in alignment with human preferences across different judge models |
| Data quality analysis | Vote consistency | Statistical analysis confirms high quality of the collected data |

### Key Findings
- Top-tier models (e.g., GPT-4, Claude series) rank highest on scientific literature understanding and synthesis tasks, with clear capability stratification across models.
- Automated evaluation methods exhibit relatively low alignment with human preferences, indicating that automatic evaluation of scientific literature tasks remains an open and challenging problem.
- Community voting data are confirmed to be of high quality through statistical analysis.
- Voting distributions and model performance vary across scientific disciplines, suggesting that domain specificity must be considered in scientific literature evaluation.
- Performance gaps between models are larger on complex tasks requiring deep literature understanding and synthesis, and smaller on simple factual question answering.

## Highlights & Insights
- The first community-driven evaluation platform for non-verifiable scientific literature tasks; awarded NeurIPS Spotlight, filling a critical gap in this evaluation landscape.
- Successfully transfers the Chatbot Arena paradigm to the scientific literature domain, validating the feasibility and effectiveness of community-driven evaluation in a specialized vertical domain.
- SciArena-Eval quantifies the shortcomings of automated evaluation methods, providing a clear improvement direction and baseline for subsequent research.
- Represents an infrastructure-level contribution whose value will continue to grow as the community expands and data accumulate.
- Covers diverse scientific task types (literature QA, review synthesis, hypothesis evaluation, etc.), with task designs closely aligned with real-world research needs.

## Limitations & Future Work
- Voting quality depends on researchers' willingness to participate and their level of expertise, which may result in uneven coverage across scientific disciplines.
- Pairwise comparison cannot directly yield absolute capability scores for specific sub-tasks; only relative rankings are produced.
- The evaluation requirements differ substantially across sub-tasks (e.g., QA vs. review generation), making a single unified ranking system difficult to generalize.
- Future work should explore combining citation verification and fact-checking methods to improve the reliability of automated evaluation.
- Coverage is currently limited primarily to English-language scientific literature; multilingual extension and participation from non-English academic communities remain a challenge.
- The disciplinary background of voters may affect the fairness of judgments on interdisciplinary task responses.
- As the number of participating models grows, the number of vote pairs available per model may be diluted, potentially reducing ranking precision.

## Related Work & Insights
- **Chatbot Arena**: The general-purpose dialogue evaluation platform and direct inspiration for this work, demonstrating the scalability and reliability of community voting-based evaluation.
- **LLM-as-judge** (e.g., MT-Bench): SciArena-Eval directly benchmarks the reliability of such methods in the scientific literature domain, revealing substantial room for improvement.
- **MMLU / SciQ and similar science benchmarks**: Traditional closed-set evaluation benchmarks that cannot assess model capabilities on open-ended scientific literature tasks.
- **Semantic Scholar / OpenReview**: Academic literature platforms upon which SciArena's model evaluation is built.
- Insight: The core difficulty in scientific literature evaluation lies in its *non-verifiability* — there is no single correct answer. This challenge is equally present in code review, medical diagnosis, legal reasoning, and other domains; SciArena's platform-based approach thus has broader transferability.

## Rating
- Novelty: ⭐⭐⭐⭐ — First community-driven evaluation platform for scientific literature tasks, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐ — Considerable scale (47 models / 20K+ votes), though in-depth analysis of detailed rankings and meta-evaluation is limited by available paper information.
- Writing Quality: ⭐⭐⭐⭐ — Problem definition is clear, platform design is well-motivated, and the rationale is thoroughly articulated.
- Value: ⭐⭐⭐⭐⭐ — An infrastructure-level contribution; NeurIPS Spotlight; with long-term impact on the scientific literature evaluation community.

<!-- end -->

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[AAAI 2026\] Canoe: Teaching LLMs to Maintain Contextual Faithfulness via Synthetic Tasks and RL](../../AAAI2026/dialogue/teaching_large_language_models_to_maintain_contextual_faithfulness_via_synthetic.md)
- [\[AAAI 2026\] Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation](../../AAAI2026/dialogue/auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)
- [\[ACL 2026\] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](../../ACL2026/dialogue/author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)
- [\[NeurIPS 2025\] Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)

<!-- RELATED:END -->
