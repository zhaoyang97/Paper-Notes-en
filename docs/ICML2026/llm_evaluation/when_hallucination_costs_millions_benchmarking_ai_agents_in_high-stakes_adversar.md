---
title: >-
  [Paper Note] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)
description: >-
  [ICML 2026][LLM Evaluation][Adversarial Evaluation] CAIA builds the first "adversarial high-stakes" agent benchmark using 17 frontier LLMs on 178 time-anchored real-world cryptocurrency tasks. Key findings: without tools…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Adversarial Evaluation"
  - "Cryptocurrency"
  - "tool selection"
  - "Pass@k trap"
  - "time-anchored benchmark"
date: 2026-05-08
content_hash: 4453bd7efe35057d
---

# When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)

**Conference**: ICML 2026  
**arXiv**: [2510.00332](https://arxiv.org/abs/2510.00332)  
**Code**: https://github.com/SurfAI/CAIA (Available, including Leaderboard and HuggingFace dataset)  
**Area**: Agent Evaluation / Security & Robustness / Financial AI  
**Keywords**: Adversarial Evaluation, Cryptocurrency, tool selection, Pass@k trap, time-anchored benchmark

## TL;DR
CAIA builds the first "adversarial high-stakes" agent benchmark using 17 frontier LLMs on 178 time-anchored real-world cryptocurrency tasks. Key findings: without tools, all models achieve only $12\text{--}28\%$ accuracy (near random guessing); with tools, even the strongest GPT-5 only reaches $67.4\%$ vs. $80\%$ for entry-level human analysts. More critically, $55.5\%$ of model tool calls lean toward "unreliable web searches" while bypassing authoritative on-chain data, causing Pass@k metrics to systematically mask dangerous "luck-based" trial-and-error behaviors.

## Background & Motivation

**Background**: In the past year, LLMs have repeatedly broken records on challenging closed-form benchmarks like ICPC and IMO, making "autonomous AI agent deployment" seem imminent. However, existing benchmarks (SWE-Bench, AppWorld, TheAgentCompany, etc.) mostly assume "tools are available, information is trustworthy, and other agents cooperate," measuring *competence* (capability upper bound) rather than *resilience* (survival in hostile environments).

**Limitations of Prior Work**: Fields such as finance, governance, and critical infrastructure—the actual targets for agent deployment—are rife with active deception, misinformation, and irreversible operations. An agent capable of winning an IMO gold medal might still trust phishing links or invest in compromised assets. Existing evaluations have never been specifically designed for "surviving in environments surrounded by attackers." Furthermore, mature metrics like Pass@k assume "more trials are better," whereas in high-stakes scenarios, a single first-time error can cause irreversible losses of millions of dollars.

**Key Challenge**: (1) Training data originates from the organized Web2, while deployment environments are Web3/real financial markets filled with malicious inducements; (2) Benchmarks are increasing in difficulty, but increased difficulty does not equate to increased robustness; (3) Metrics like Pass@5 represent "exploration victories" in controlled tasks but constitute "blind guessing" in irreversible decisions.

**Goal**: Construct a benchmark that directly quantifies agent performance under adversarial, high-stakes, and multi-source mixed data conditions; characterize specific failure modes of SOTA models (especially tool selection behavior), and elevate "adversarial robustness" to a measurable and mandatory deployment prerequisite.

**Key Insight**: The authors acutely select cryptocurrency as a "natural laboratory"—it simultaneously possesses (i) active attackers (honeypot contracts, flash loans, coordinated social engineering), (ii) high stakes ($\$30\text{B}$ in losses in 2024, irreversible on-chain transactions), and (iii) verifiable ground truth (transparent and immutable blockchain). This creates a "three-in-one" environment for adversarial evaluation that other financial scenarios cannot provide.

**Core Idea**: A four-in-one design featuring "adversarial-first + real financial loss + time-anchored + fine-grained failure diagnosis" to upgrade agent benchmarks from "can it be completed" to "can it be safely completed under active hostility."

## Method

### Overall Architecture
CAIA consists of 178 time-anchored real cryptocurrency analysis tasks covering 6 sub-categories. During evaluation, each model runs under two conditions: "no tools" and "with tools (23 professional tools + ReAct framework)." Each question is run 5 times independently using majority voting to report Pass@1/Pass@5. Token consumption and USD costs are recorded to provide cost-per-accuracy. All data was extracted via a 5-stage pipeline from over 10,000 real queries by 3,000+ practitioners. Contamination control (block height/timestamp anchoring) and liveness (continuous retirement of old tasks/addition of new ones) are explicitly defined in the specifications.

### Key Designs

1.  **5-Stage Adversarial-Priority + Time-Anchored Data Pipeline**:
    - **Function**: Filters 178 high-quality tasks that are authentic, verifiable, and resistant to training data contamination from 10,000 real queries.
    - **Mechanism**: (Stage 1) LLM-as-judge for initial screening of topic relevance, answer existence, and temperature anchoring, retaining the top $15\%$ (~1,000); (Stage 2) Review by 92 domain experts, with each question receiving at least 4 ratings; the top 200 are selected after removing outliers, leaving 186 prototypes after deduplication; (Stage 3) Formatting is standardized to force anchoring to specific block numbers or timestamps, ensuring answers are fully reproducible; (Stage 4) Construction of a "reproducible ground-truth toolchain" for each task—providing not just the answer but the toolcall sequence to reach it; (Stage 5) Categorization into On-Chain Analysis ($43.3\%$), Project Discovery ($27.5\%$), Tokenomics ($12.9\%$), Overlap ($7.9\%$), Trend Analysis ($4.5\%$), and General Knowledge ($3.9\%$) for fine-grained diagnosis.
    - **Design Motivation**: Traditional static benchmarks are susceptible to training data contamination and often feature "theoretically correct but non-executable" scenarios. Time-anchoring + reproducible toolchains solve both. The immutability of blockchain ensures ground truth is objective, avoiding the dilemma of choosing between "proprietary data" and "synthetic simulation."

2.  **Dual-Condition Evaluation + 23-Tool ReAct Framework**:
    - **Function**: Decouples "parametric knowledge" from "tool orchestration capability" to quantify respective weaknesses.
    - **Mechanism**: No-tool condition = closed-book exam, forcing models to use only parametric memory; Tool condition = open-book exam, providing 23 tools (Etherscan/CoinGecko/DefiLlama, market APIs, web search, Python, etc.). The authors ensure that "the correct answer is always obtainable through the appropriate tools," thereby isolating the challenge to *tool selection + synthesis* rather than information availability. All tool-using experiments run within a ReAct-style framework to eliminate implementation variance.
    - **Design Motivation**: Previous evaluations conflated tool capability, reasoning, and prompt engineering, obscuring true bottlenecks. CAIA isolates "knowledge vs. orchestration" dimensions.

3.  **6-Category Fine-Grained Failure Diagnosis + Cost-Aware Evaluation**:
    - **Function**: Decomposes a single accuracy metric into 6 analytical categories, tool distribution, cost efficiency, and Pass@k vs. majority voting comparisons to reveal hidden "lucky guessing" risks.
    - **Mechanism**: (a) Primary metric uses 5-round majority voting to mitigate sampling variance; (b) Simultaneous reporting of Pass@1 and Pass@5 highlights that Pass@k is a "dangerous metric" in high-stakes scenarios—where a model might have Pass@1=$26.4\%$ but Pass@5=$54.5\%$ (e.g., DeepSeek R1), indicating reliance on random attempts; (c) Mapping token and USD costs to calculate cost/score, revealing that cost and accuracy are not always positively correlated; (d) Failure mode analysis shows $55.5\%$ of tool calls bias towards unreliable web searches even when specialized APIs are available.
    - **Design Motivation**: Single accuracy metrics lack "depth" and hide the "cost of error." Combining behavior distribution (tool preference), stability (voting vs. single run), and economics (cost/score) provides deployment-grade diagnostics required for high-stakes scenarios.

### Loss & Training
CAIA is a benchmark rather than a training method and does not involve a loss function. Evaluation Protocol: Each question is run 5 times independently using majority voting. The human baseline was completed by 16 analysts from blockchain clubs and startups on a $10\%$ subset, achieving an average accuracy of $80\%$.

## Key Experimental Results

### Main Results
Evaluation of 17 models (GPT, Claude, Gemini, Grok, DeepSeek, etc.) under dual conditions:

| Model | No Tools (Majority) | With Tools (Majority) | With Tools (Pass@5) | With Tools Cost ($) |
|---|---|---|---|---|
| **GPT-5** | **0.275** | **0.674** | 81.5 (≈) | 0.021 |
| Claude Opus 4 | 0.135 | 0.573 | 71.9 | 1.114 |
| Claude Opus 4.1 | 0.135 | 0.563 | 69.0 | 0.936 |
| Claude Sonnet 4 | 0.118 | 0.567 | 66.9 | 0.229 |
| DeepSeek V3.1 | 0.157 | 0.492 | 71.2 | 0.022 |
| GPT-4.1 | 0.197 | 0.466 | 60.7 | 0.091 |
| Gemini 2.5 Pro | 0.225 | 0.449 | 61.2 | 0.041 |
| GPT-4o | 0.169 | 0.303 | 55.6 | 0.091 |
| **DeepSeek R1** | 0.208 | 0.174 | **54.5** | 0.012 |
| GPT-OSS 120B | 0.146 | (Pareto) | – | **0.0003** |
| **Human Analyst** | – | **0.80** | – | – |

The most significant contrast: DeepSeek R1's Pass@1 with tools is $26.4\%$ but Pass@5 surges to $54.5\%$, indicating it is effectively "guessing." GPT-OSS 120B approaches frontier performance at $\$0.0003/\text{query}$, representing the cost-accuracy Pareto frontier.

### Ablation Study

| Dimension | Key Observation | Description |
|---|---|---|
| Tool Availability | No Tools $12\text{--}28\%$ → Tools Max $67.4\%$ | Tools are useful but not a complete solution |
| Tool Selection Behavior | $55.5\%$ of calls are Web Search | Models prefer unreliable sources even when specialized tools exist |
| Pass@1 vs Pass@5 | Pass@5 $\gg$ Pass@1 across models | Reveals trial-and-error; equivalent to "gambling" in high-stakes settings |
| Category Distribution | On-Chain $43.3\%$ / Project Disc. $27.5\%$ | On-chain transaction analysis is the dominant and most difficult task |
| Human baseline | $80\%$ vs GPT-5 $67.4\%$ | Strongest models still lag by $12.6\text{pp}$ even with full tools |

### Key Findings
- **Tool selection catastrophe**: Models systematically prefer web search ($55.5\%$) even when professional on-chain tools provide ground truth. The root cause is an architecture-level inability to assess information reliability rather than a knowledge gap.
- **Pass@k is a misleading metric for high-stakes**: The gap between Pass@5 and Pass@1 reveals "illusionary capability." In finance or security, a first-time failure is "game over."
- **Closed-source $\neq$ inherently stronger**: GPT-OSS 120B achieves performance comparable to closed-source models at $\$0.0003/\text{query}$, nearly $1000\times$ cheaper than Claude Opus 4.
- **Fundamental limits of Web2 training**: Failures in Web3/Crypto scenarios stem from "out-of-distribution" training—models have not mastered on-chain data structures or SEO-based adversarial attacks.
- **Hallucinations have specific economic costs**: By anchoring questions to real block heights and amounts, wrong answers map directly to quantifiable financial losses.

## Highlights & Insights
- **Cryptocurrency as an adversarial testbed**: The argument that crypto is adversarial + irreversible + verifiable is exceptionally solid.
- **Expert Curation**: The 5-stage pipeline with 92 expert reviewers ensures high credibility, which is far harder to replicate than synthetic data generation.
- **Quantifiable Tool Selection**: Treating tool-call frequency distribution as an evaluation dimension rather than just accuracy is a profound methodological contribution to agent evaluation.
- **Pass@k Critique**: Advocating for majority voting + cost-aware evaluation in place of Pass@k aligns the benchmark with real-world deployment requirements.

## Limitations & Future Work
- Small sample size (178 tasks): While expert-vetted, the scale is small compared to SWE-Bench ($2294$).
- Domain specificity: Conclusions may not fully migrate to medical or political content manipulation domains.
- Passive vs. Active Adversaries: Focuses on the "hostile environment" but does not yet include active tool poisoning or prompt injection attacks.
- Scaffolding dependency: The ReAct framework may limit models that perform better under "plan-then-execute" architectures.
- Human baseline variance: The sample (16 humans) is relatively small.
- Dynamic Adversaries: Adversaries evolve; the benchmark requires continuous updates to remain effective.

## Related Work & Insights
- **vs SWE-Bench / AppWorld**: CAIA introduces the "adversarial + high-stakes + irreversible" evaluation dimension.
- **vs FinanceBench / FinQA**: CAIA utilizes blockchain transparency to bypass the "proprietary vs. synthetic" data trade-off.
- **vs $\tau$-Bench / WebArena**: Introduces tool selection preference as a new behavioral metric.
- **Insights**: (1) Focus evaluation on orchestration via "available but must be selected correctly" toolsets; (2) Replace Pass@k with cost-aware and behavior-distribution metrics; (3) Use real-world practitioner queries as seeds rather than synthetic data.

## Rating
- Novelty: ⭐⭐⭐⭐ (Strong combination of existing elements with a deep domain choice).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Excellent coverage of models, conditions, sampling, and cost).
- Writing Quality: ⭐⭐⭐⭐⭐ (Persuasive "Why crypto" arguments and clear terminology).
- Value: ⭐⭐⭐⭐⭐ (Critical guidance for model deployment in high-stakes fields).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](../../ICLR2026/llm_evaluation/astabench_benchmarking_ai_agents.md)
- [\[ICML 2026\] When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation](when_ai_benchmarks_plateau_a_systematic_study_of_benchmark_saturation.md)
- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[ICML 2026\] Automatic Layer Selection for Hallucination Detection](automatic_layer_selection_for_hallucination_detection.md)
- [\[AAAI 2026\] MindVote: When AI Meets the Wild West of Social Media Opinion](../../AAAI2026/llm_evaluation/mindvote_when_ai_meets_the_wild_west_of_social_media_opinion.md)

</div>

<!-- RELATED:END -->
