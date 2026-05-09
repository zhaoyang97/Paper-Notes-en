---
title: >-
  [Paper Note] What AI Speaks for Your Community: Polling AI Agents for Public Opinion on Data Center Projects
description: >-
  [NeurIPS 2025][LLM Agent][AI agent polling] This paper proposes an LLM-based AI agent polling framework that synthesizes demographically representative virtual resident agents to conduct large-scale, low-cost public opinion surveys on data center projects. Cross-model and cross-region experiments demonstrate high thematic alignment between agent opinions and real-world polls.
tags:
  - NeurIPS 2025
  - LLM Agent
  - AI agent polling
  - public opinion
  - data center
  - LLM simulation
  - community engagement
date: 2026-05-08
content_hash: cabc6501f434c682
---

# What AI Speaks for Your Community: Polling AI Agents for Public Opinion on Data Center Projects

**Conference**: NeurIPS 2025
**arXiv**: [2511.22037](https://arxiv.org/abs/2511.22037)
**Code**: None
**Area**: LLM Agent
**Keywords**: AI agent polling, public opinion, data center, LLM simulation, community engagement

## TL;DR
This paper proposes an LLM-based AI agent polling framework that synthesizes demographically representative virtual resident agents to conduct large-scale, low-cost public opinion surveys on data center projects. Cross-model and cross-region experiments demonstrate high thematic alignment between agent opinions and real-world polls.

## Background & Motivation
**Background**: The surging demand for compute driven by AI foundation models has accelerated global data center construction. These facilities bring economic benefits such as local employment and tax revenue, while simultaneously imposing environmental burdens including water consumption, strain on power grids, and carbon emissions.

**Limitations of Prior Work**: Traditional public opinion surveys are costly in terms of labor and funding, and community feedback is typically collected only at public hearings after project commitments have already been made, precluding early integration of community voices. Limited community engagement and sampling bias further undermine the representativeness of collected feedback.

**Key Challenge**: Data center development requires early-stage, scalable, and diverse community input, yet no effective mechanism exists to achieve this.

**Goal**: How can representative community public opinion be obtained at low cost and at scale during the early planning stages of data center projects?

**Key Insight**: Leverage the reasoning capabilities and world knowledge of LLMs to create AI agents that simulate community residents, enabling scalable opinion simulation surveys.

**Core Idea**: Demographically synthesized virtual resident agents + multi-model LLM-driven structured questionnaires = a scalable early-screening tool for community opinion.

## Method

### Overall Architecture
The framework comprises six key stages: (1) establishing a data center proposal; (2) generating a representative AI agent sample from county-level demographic data using iterative proportional fitting (IPF); (3) constructing community profiles and project specifications as system prompts; (4) conducting structured polling with multiple LLMs (GPT-5, Gemini-2.5-Pro, Qwen-Max); (5) optional conformal prediction calibration; and (6) multi-level analysis (cross-model, cross-region, and comparison with human polls).

### Key Designs
1. **Regional Context Modeling**: Three layers of contextual information are incorporated — state-level data center electricity consumption data, county-level demographic profiles (retrieved via the Census Bureau ACS 5-year estimates API), and standardized data center project descriptions (energy usage, environmental impact, economic impact). Together, these constitute the "global context" provided to each agent.
2. **Virtual Agent Construction**: Twelve demographic dimensions are extracted from ACS data (educational attainment, marital status, household language, citizenship, employment status, household income, housing tenure, vehicle ownership, age group, sex, race, and ethnicity). IPF is used to synthesize joint probability distributions from marginal distributions, generating virtual agents with correlated demographic characteristics. The demographic distribution of the agent population is validated via chi-square goodness-of-fit tests.
3. **AI Agent Polling Execution**: The questionnaire contains 13 questions (12 closed-ended single/multiple choice + 1 open-ended text), covering five core domains: economic impact, environmental concerns, community engagement, anticipated personal impact, and overall project support. A system message/user prompt separation strategy is employed to optimize API calls (static regional context as a cacheable system message). Batch APIs from three LLMs are used to process thousands of agent responses in parallel.
4. **Conformal Prediction Calibration**: A methodology is proposed in which a small number of real survey responses are used to compute nonconformity scores; a threshold is then derived to construct confidence intervals for new agent survey results, providing statistical guarantees. (Not implemented in this work due to resource constraints; included for methodological completeness.)

### Loss & Training
- No training is required; the framework relies entirely on zero-shot LLM inference.
- Each experiment surveys 1,000 virtual agents per model per region.
- Total API cost is approximately \$36.2 per run (GPT-5: \$23.3, Gemini-2.5: \$11.2, Qwen: \$1.7).
- A single run typically exceeds 24 hours.

## Key Experimental Results

### Baseline Case Study (Taylor County, TX; GPT-5)

| Dimension | Key Result |
|-----------|-----------|
| Overall Attitude | 54.2% neutral, 43.6% positive, 2.2% negative |
| Perception of Economic Impact | 80% perceive mixed impact, 20% perceive positive |
| Level of Environmental Concern | 97% express concern |
| Trust in Government Regulation | 60% neutral, 40% distrust |
| Top 3 Open-Text Themes | Water conservation, utility costs, local employment |

### Cross-Model Comparison (Taylor County, $n=1000$)

| Dimension | GPT-5 | Gemini-2.5 | Qwen-Max |
|-----------|-------|------------|----------|
| Economic Attitude — Positive | ~20% | ~10% | 91% |
| Top Economic Priority | Diverse preferences | Diverse preferences | Near-unanimous: taxes and jobs |
| Government Distrust Rate | 40% | 32% | ~0% |
| Preferred Information Sources | Academic research (primary) | Academic research (primary) | Academic research + strong preference for local government |

### Cross-Region Comparison (GPT-5, $n=1000$)

| Dimension | Taylor County, TX | Loudoun County, VA |
|-----------|-------------------|--------------------|
| Net Support Rate | +41% (support 44%, oppose 2%) | −5% (support 10%, oppose 15%) |
| Top Condition to Increase Support | Lower utility rates (94%) | Stricter regulation (90%) |
| Economic Attitude — Positive | 20% | Lower |
| Distinctive Economic Concerns | Low income + high temperatures → utility bill concerns | Pressure on public services (extensive prior data center experience) |

### Comparison with Real Polls (Heatmap News National Poll)

| Metric | Heatmap National Poll | AI Agent Poll |
|--------|-----------------------|---------------|
| Net Support Rate | +2% (44% support vs. 42% oppose) | Taylor: +41%; Loudoun: −5% |
| Primary Perceived Benefit | Tax revenue + high-paying jobs | Tax revenue + infrastructure upgrades |
| Primary Concerns | Water resources + electricity use | Water consumption + rising utility bills |

### Key Findings
- **Water resources and utility costs** are the shared core concerns across regions and models; tax revenue is the commonly recognized primary benefit.
- **Model divergence reflects cultural and institutional differences embedded in training data**: Qwen exhibits greater optimism toward economic development and higher trust in government, likely reflecting a development-oriented economic paradigm prevalent in its training data.
- **Regional differences align with reality**: Taylor County (low income, limited employment) prioritizes economic gains; Loudoun County (data center–dense area) focuses on regulation and public service pressure.
- AI agent polls demonstrate **high thematic alignment** with real polls, though quantitative values are not directly comparable.

## Highlights & Insights
- This work represents the **first application of an AI agent polling framework to community opinion assessment for data centers**, addressing a gap in community engagement for AI infrastructure.
- **Cross-model comparison reveals intrinsic LLM biases**: The differing "cultural priors" of various LLMs lead to markedly different opinions on the same project from an identical population, which is itself a significant finding.
- **Significant cost-effectiveness**: Total API cost is only \$36.2 per run, far below the labor and resource costs of traditional surveys.
- **Methodological rigor in IPF-based agent synthesis**: 12 demographic dimensions combined with chi-square validation ensure representativeness.
- **Honesty in the Disclosure section**: The authors explicitly acknowledge that AI agent opinions cannot substitute for genuine community voices, positioning the framework as a screening tool.

## Limitations & Future Work
- **Conformal prediction calibration not implemented**: A core methodological component is absent due to insufficient resources, leaving the statistical guarantees unverified.
- **Only two counties tested**: Both already have existing data center infrastructure; applicability to communities with no data center experience remains unknown.
- **LLM bias issues**: Training data biases systematically influence agent outputs, and racial/cultural biases cannot be fully eliminated.
- **Lack of longitudinal validation**: It has not been verified whether agent polls can predict future real community responses.
- **Limitations of agent cognitive models**: The framework cannot simulate neurodivergent perspectives, "participatory silence," or other complex human behaviors.
- Potential improvements include incorporating additional LLMs for cross-validation, implementing calibration mechanisms, and extending the framework to other infrastructure domains.

## Related Work & Insights
- **vs. Traditional polling**: Conventional methods are costly and slow; the proposed framework is scalable and low-cost, but lacks the depth of genuine human interaction.
- **vs. LLM-based election simulation**: Political AI agent frameworks focus on electoral prediction; this work is the first to target physical infrastructure (data centers), with a more multidimensional problem structure spanning economic, environmental, and governance dimensions.
- **vs. public engagement tools such as PolicyPulse**: Such tools extract themes from existing public discourse, whereas this work actively generates opinions — a fundamentally different paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of AI agent polling to community opinion on data centers; novel entry point with practical relevance.
- Experimental Thoroughness: ⭐⭐⭐ Cross-model and cross-region analyses are convincing, but coverage is limited to two counties and calibration is not implemented.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, candid in the Disclosure section, methodologically rigorous yet accessible.
- Value: ⭐⭐⭐⭐ Provides a practical tool prototype for the responsible deployment of AI infrastructure, with meaningful policy reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generative AI Agents for Controllable and Protected Content Creation](generative_ai_agents_for_controllable_and_protected_content_creation.md)
- [\[NeurIPS 2025\] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers](lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)
- [\[NeurIPS 2025\] Enhancing Demand-Oriented Regionalization with Agentic AI and Local Heterogeneous Data for Adaptation Planning](enhancing_demand-oriented_regionalization_with_agentic_ai_and_local_heterogeneou.md)
- [\[NeurIPS 2025\] SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](suffixdecoding_extreme_speculative_decoding_for_emerging_ai_applications.md)
- [\[NeurIPS 2025\] PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer](panda_towards_generalist_video_anomaly_detection_via_agentic_ai_engineer.md)

</div>

<!-- RELATED:END -->
