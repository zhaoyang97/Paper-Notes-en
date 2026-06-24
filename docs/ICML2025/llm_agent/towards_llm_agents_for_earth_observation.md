---
title: >-
  [Paper Note] Towards LLM Agents for Earth Observation
description: >-
  [ICML 2025][LLM Agent][Earth Observation] This paper proposes UnivEARTH—a Earth Observation benchmark featuring 140 yes/no questions, covering 13 topics and 17 satellite sensors. Evaluation reveals that the best LLM Agent (generating code to use Google Earth Engine) achieves an accuracy of only 33%, primarily limited by the fact that 58% of the generated code fails to execute.
tags:
  - "ICML 2025"
  - "LLM Agent"
  - "Earth Observation"
  - "Google Earth Engine"
  - "benchmark"
  - "code generation"
date: 2026-05-08
content_hash: 550134c9459c78ac
---

# Towards LLM Agents for Earth Observation

**Conference**: ICML 2025  
**arXiv**: [2504.12110](https://arxiv.org/abs/2504.12110)  
**Code**: None  
**Area**: Agent  
**Keywords**: Earth Observation, LLM agent, Google Earth Engine, benchmark, code generation

## TL;DR

This paper proposes UnivEARTH—a Earth Observation benchmark featuring 140 yes/no questions, covering 13 topics and 17 satellite sensors. Evaluation reveals that the best LLM Agent (generating code to use Google Earth Engine) achieves an accuracy of only 33%, primarily limited by the fact that 58% of the generated code fails to execute.

## Background & Motivation

**Background**: Earth Observation (EO) provides critical data for fields such as environmental monitoring, disaster management, and climate science. Scientists routinely need to analyze planetary data like land use, surface reflectance, and chlorophyll content, which involves choosing suitable sensors, data products, locations, and timeframes. Although domain-specific automatic systems have been deployed for years (e.g., for forest fire detection), flexible general-purpose query capabilities remain scarce.

**Limitations of Prior Work**: Existing LLMs face severe challenges when generating Google Earth Engine (GEE) code: (1) GEE has over 400 image collections, making it difficult for models to select and name them correctly; (2) programming problems in earth sciences are underrepresented in pre-training data; (3) different sensors vary in data formats, spatial coverage, and temporal resolutions, requiring extensive domain knowledge.

**Key Challenge**: While LLMs excel at general programming tasks, in the specialized domain of Earth Observation, they must simultaneously possess domain knowledge (sensor selection, data product understanding) and programming capabilities (GEE API usage). Both aspects are severely underrepresented in current pre-training data.

**Goal**: (1) To construct a reliable EO QA benchmark—requiring knowledge of what questions to ask, what the answers are, and ensuring that the supporting data is accessible; (2) to systematically evaluate the actual capabilities of existing LLM Agents on EO tasks.

**Key Insight**: NASA Earth Observatory articles serve as authoritative data sources. Each article documents scientific conclusions derived from satellite imagery, naturally providing a "question-answer-evidence" triplet. Additionally, the answerability of each question is verified using the GEE JavaScript Editor.

**Core Idea**: Construct a high-quality QA benchmark through authoritative sources to test the EO capabilities of LLM Agents via "code generation for evidence retrieval," revealing that current models are far from reliable.

## Method

### Overall Architecture

The contributions of this work majorly focus on benchmark construction and evaluation. The pipeline for benchmark construction is: article collection $\rightarrow$ LLM-assisted QA-generation $\rightarrow$ GEE answerability verification $\rightarrow$ independent expert review. The evaluation pipeline is: provide LLMs with questions and GEE API access $\rightarrow$ LLMs generate analysis code $\rightarrow$ execute code locally $\rightarrow$ parse results and compare with ground truth.

### Key Designs

1. **Three-stage Pipeline for Benchmark Construction**:

    - Function: To ensure benchmark quality—scientifically correct, answerable, and unambiguous.
    - Mechanism: **Collection Phase**: Download NASA Earth Observatory articles (up to March 10, 2025), use Claude-3.5-Sonnet to analyze text and generate yes/no QA candidates, and manually add extra questions based on images in the articles. Filter out inapplicable articles regarding sensor specifications, non-satellite imagery, and transient observations (wind speed, tides). **Verification Phase**: For each question, write test implementations in the GEE JavaScript Editor to verify that the required datasets are available in GEE, replacing them with equivalent sources if necessary. **Review Phase**: 4 reviewers each assess half of the dataset across four dimensions (answer correctness, textual support, imagery support, geographic location validation), iteratively revising until Cohen's Kappa ($Q_1$) reaches perfect consensus.
    - Design Motivation: An EO QA benchmark did not previously exist—none of the questions, answers, or supporting evidence were readily available.

2. **LLM Agent Evaluation Framework**:

    - Function: Test the performance of LLMs on EO tasks when granted data access.
    - Mechanism: Three evaluation paradigms: (1) **Zero-shot**: Directly prompt the model to reason and generate code; (2) **3-shot**: Provide 3 out-of-benchmark QA code examples; (3) **Reflexion**: A 3-turn self-reflection loop, feeding the code, execution results, and error feedback back to the model for code re-generation. Code is executed locally, and results are parsed by GPT-4o-mini into yes/no/inconclusive answers. Metrics include accuracy, failure rate (code fails to run or data unavailable), and selective accuracy (accuracy excluding failures).
    - Design Motivation: To test whether models can "answer questions with evidence" rather than relying solely on parametric memory—thus requiring the generation of executable code.

3. **Data Utilization and Error Analysis**:

    - Function: Reveal the root causes of model failures.
    - Mechanism: Analyze the relationship between the number of unique image collections used by the model and its accuracy—revealing a correlation of $r = 0.87$, indicating that models using more diverse data sources achieve higher accuracy. Further analysis shows that the primary failure cause is the "Wrong Asset Name" error (incorrect image collection names), which strongly negatively correlates with accuracy. This indicates that the bottleneck lies in domain knowledge memorization rather than reasoning capability.
    - Design Motivation: Identifying the bottleneck is crucial for subsequent improvements—the direction points towards enhancing the model's memory of GEE library dataset names.

### Loss & Training

Since this work is a benchmark evaluation, it does not involve new training loss functions.

## Key Experimental Results

### Main Results

Performance of various LLM Agents using GEE (average of 8 trials):

| Model | Zero-shot Accuracy | 3-shot Accuracy | Reflexion Accuracy | Zero-shot Failure Rate |
|---|---|---|---|---|
| Claude-3.7-Sonnet | 32.4% | 30.6% | **33.0%** | 61.3% |
| DeepSeek-V3 | 28.4% | 32.8% | 24.3% | 64.3% |
| o3-mini | 25.7% | 33.0% | 25.1% | 70.0% |
| Claude-3.5-Sonnet | 27.0% | 23.9% | 27.8% | 67.5% |
| GPT-4o-mini | 8.3% | 13.1% | 5.8% | 89.1% |
| Llama-3.3-70B | 2.8% | 6.5% | 2.6% | 96.7% |

### Ablation Study

Relationship between data utilization diversity and performance:

| Metric | Correlation |
|---|---|
| Number of unique image collections vs. Accuracy | $r = 0.87$ (Strong positive correlation) |
| "Wrong Asset Name" error rate vs. Accuracy | Strong negative correlation |
| Best accuracy without Internet access | 49.0% (Using model parametric knowledge only) |

### Key Findings

- The best accuracy is only 33.0% (Claude-3.7-Sonnet, Reflexion), primarily because over 58% of the code fails to execute.
- Even when code executes successfully, there is still an approximately 20% chance of returning an incorrect answer.
- Accuracy with GEE data access (33%) is actually lower than without data access (49%), indicating that code generation ability is the primary bottleneck.
- Memorization of image collection names is highly correlated—models that use more diverse datasets perform significantly better ($r=0.87$).
- 3-shot learning generally outperforms zero-shot, but Reflexion does not always bring improvements and can even degrade performance for some models.

## Highlights & Insights

- Fills the gap in LLM evaluation benchmarks for the EO domain with a rigorous construction pipeline (authoritative NASA source + GEE answerability verification + expert review).
- Uncovers a counter-intuitive finding: equipping models with more tools (GEE API) can end up decreasing accuracy—tools become a burden when capabilities are insufficient.
- The $r=0.87$ correlation points to a clear direction for improvement: enhancing the model’s recall of GEE image collection names.
- The benchmark is highly relevant to real-world scientific problems, featuring active research topics such as lake dynamics in the Tibetan Plateau and global cropland expansion.

## Limitations & Future Work

- The benchmark size is relatively small (140 questions), limiting statistical robustness.
- It only considers a yes/no question format, failing to cover more complex open-ended EO queries.
- It does not contain "unanswerable" questions (cases where the ground truth is inconclusive).
- Testing is restricted to the GEE platform, omitting other remote sensing data platforms.
- Evaluation relies on GPT-4o-mini to parse answers, introducing additional uncertainty.
- No fine-tuned open-source model is provided for community replication and improvement.
- The source of questions is limited to the NASA Earth Observatory, lacking coverage of non-English literature and regional EO research.

## Related Work & Insights

- Shares a similar position as HumanEval (Chen et al., 2021) but for a vertical domain, serving as the first systematic evaluation of EO automation.
- GeoBench-VLM (Danish et al., 2024) focuses on vision models' understanding of geographic images, whereas this work evaluates code generation and scientific reasoning.
- Insight: Reliable applications of LLM Agents in specialized scientific domains still need to resolve two bottlenecks: "domain knowledge integration" and "tool-use capability."
- Fine-tuning Llama-3.1-8B on synthetic data achieves 25% accuracy (comparable to large commercial models), showing the potential of domain-specific fine-tuning.
- Qwen2.5-72B shows anomalous behavior—using fewer datasets but utilizing them more effectively, suggesting that deep mastery of a few tools may outperform broad but shallow coverage.

## Rating

⭐⭐⭐ (6.5/10)

The benchmark construction is high-quality, and the findings are valuable (especially the 33% accuracy and $r=0.87$). However, as a research contribution, it leans on the lighter side—mostly consisting of benchmark construction and evaluation of existing models without proposing a targeted solution. The benchmark dataset (140 questions) is small and limited to a yes/no format. Nonetheless, this work provides clear guidance for the EO community and the AI-for-Science direction.

Notably, this paper reveals a deeper issue: the bottleneck for LLM Agents in specialized scientific fields is not reasoning capacity, but the accurate recall of domain knowledge—"Wrong Asset Name" is the most common error pattern. This suggests that future improvements should focus on: (1) providing real-time retrieval of GEE data catalogs through RAG mechanisms; (2) increasing EO programming paradigms in fine-tuning data; (3) building structured tool descriptions of GEE APIs for agents to call. Although the 140 questions are few, they cover 13 topics and 17 sensors, prioritizing quality over quantity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation](../../ACL2026/llm_agent/opera_a_dataset_of_observation_persona_rationale_and_action_for_evaluating_llms_.md)
- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](../../ACL2025/llm_agent/legalagentbench_evaluating_llm_agents_in_legal_domain.md)
- [\[ACL 2025\] MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](../../ACL2025/llm_agent/multiagentbench_evaluating_the_collaboration_and_competition_of_llm_agents.md)
- [\[NeurIPS 2025\] AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents](../../NeurIPS2025/llm_agent/agentmisalignment_measuring_the_propensity_for_misaligned_behaviour_in_llm-based.md)
- [\[ICML 2025\] GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning](guardagent_safeguard_llm_agents_by_a_guard_agent_via_knowledge-enabled_reasoning.md)

</div>

<!-- RELATED:END -->
