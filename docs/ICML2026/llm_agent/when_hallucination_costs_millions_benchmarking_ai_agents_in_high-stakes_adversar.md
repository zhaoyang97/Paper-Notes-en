---
title: >-
  [Paper Note] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)
description: >-
  [ICML 2026][LLM Agent][Adversarial Evaluation] CAIA establishes the first "adversarial high-stakes" agent benchmark using 17 cutting-edge large models on 178 time-anchored real-world cryptocurrency tasks. Key findings: w…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Adversarial Evaluation"
  - "Cryptocurrency"
  - "Tool Selection"
  - "Pass@k Pitfall"
  - "Time-Anchored Benchmark"
date: 2026-05-08
content_hash: 77227a809b6042ee
---

# When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)

**Conference**: ICML 2026  
**arXiv**: [2510.00332](https://arxiv.org/abs/2510.00332)  
**Code**: https://github.com/SurfAI/CAIA (available, includes Leaderboard and HuggingFace dataset)  
**Area**: Agent Evaluation / Safety & Robustness / Financial AI  
**Keywords**: Adversarial Evaluation, Cryptocurrency, Tool Selection, Pass@k Pitfall, Time-Anchored Benchmark

## TL;DR
CAIA establishes the first "adversarial high-stakes" agent benchmark using 17 cutting-edge large models on 178 time-anchored real-world cryptocurrency tasks. Key findings: without tools, all models achieve only 12–28% accuracy (near random guessing); with tools, even the strongest GPT-5 reaches only 67.4% vs. human junior analysts at 80%. More critically, 55.5% of model tool calls prefer "unreliable web search" over authoritative on-chain data, causing Pass@k metrics to systematically mask the dangerous "trial-and-error luck" behavior.

## Background & Motivation

**Background**: Over the past year, large models have repeatedly set new records on challenging closed benchmarks such as ICPC and IMO, making "autonomous AI agent deployment" appear within reach. However, existing benchmarks (SWE-Bench, AppWorld, TheAgentCompany, etc.) almost universally assume "tools are available, information is trustworthy, and other agents cooperate," measuring *competence* (upper bound of ability) rather than *resilience* (survivability in adversarial environments).

**Limitations of Prior Work**: Domains where agents are truly needed—finance, governance, critical infrastructure—are rife with active deception, misinformation, and irreversible actions. An agent capable of winning IMO gold may still fall for phishing links or purchase compromised assets. Existing evaluations have never been designed specifically for "surviving in adversarial environments"; meanwhile, mature metrics like Pass@k assume "just try more times," but in high-risk scenarios, a single mistake can cause millions in irreversible losses.

**Key Challenge**: (1) Training data comes from orderly Web2, but deployment environments are maliciously manipulated Web3/real financial markets; (2) Benchmarks are getting harder, but increased difficulty does not equate to increased robustness; (3) Metrics like Pass@5, which reward exploration in controlled tasks, become "blind trial-and-error" in irreversible decision contexts.

**Goal**: Construct a benchmark that directly quantifies agent performance under adversarial, high-risk, and multi-source data conditions; characterize the specific failure modes of current SOTA models (especially tool selection behavior), and elevate "adversarial robustness" to a measurable, mandatory deployment prerequisite.

**Key Insight**: The authors astutely select cryptocurrency as a "natural laboratory"—it uniquely combines (i) active attackers (honeypot contracts, flash loans, coordinated social engineering), (ii) high risk (2024 saw \$30B in losses, on-chain transactions are irreversible), and (iii) verifiable ground truth (blockchain is fully transparent and immutable)—a "three-in-one" adversarial evaluation setting unmatched by other financial domains.

**Core Idea**: Integrate "adversarial priority + real financial loss + time anchoring + fine-grained failure diagnosis" to upgrade agent benchmarks from "can it complete the task" to "can it complete the task safely under active adversarial conditions."

## Method

### Overall Architecture
CAIA comprises 178 time-anchored real-world cryptocurrency analysis tasks across 6 subcategories. Each model is evaluated under "no tools" and "with tools (23 professional tools + ReAct framework)" conditions, running each task independently 5 times with majority voting and reporting Pass@1/Pass@5. Token consumption and dollar cost are recorded to provide cost-per-accuracy. All data are extracted via a 5-stage pipeline from 10,000+ real queries by 3,000+ practitioners. Benchmark design explicitly incorporates contamination control (block height/timestamp anchoring) and liveness (continuous retirement/addition of tasks) into the protocol.

### Key Designs

1. **5-Stage Adversarial-First + Time-Anchored Data Pipeline**:

    - **Function**: Selects 178 high-quality tasks from 10,000 real queries that are authentic, verifiable, and resistant to training data contamination.
    - **Mechanism**: (Stage 1) LLM-as-judge screens for topic relevance, answer existence, and temperature anchoring, retaining the top 15% (~1,000 tasks); (Stage 2) 92 domain experts review, each task evaluated by at least 4 reviewers, averaging after removing highest/lowest, yielding top 200, deduplicated to 186 prototypes; (Stage 3) Standardizes format, forcibly anchoring each task to a specific block number or timestamp for full reproducibility; (Stage 4) Constructs a "reproducible ground-truth toolchain" for each task—not just the standard answer, but the tool invocation chain to reach it; tasks failing reproducibility are removed, leaving 178; (Stage 5) Categorizes into On-Chain Analysis (43.3%), Project Discovery (27.5%), Tokenomics (12.9%), Overlap (7.9%), Trend Analysis (4.5%), General Knowledge (3.9%) for fine-grained diagnosis.
    - **Design Motivation**: Traditional static benchmarks are prone to training data contamination and may appear correct but fail in execution; time anchoring + reproducible toolchains address both issues. Blockchain immutability ensures objective ground truth, avoiding the dilemma of "proprietary data vs. synthetic simulation" in traditional finance benchmarks.

2. **Dual-Condition Evaluation + 23-Tool ReAct Framework**:

    - **Function**: Decouples "model knowledge" from "tool orchestration ability," quantifying weaknesses on both sides.
    - **Mechanism**: No-tools condition = closed-book, forcing the model to rely solely on parametric memory, measuring basic understanding; with-tools condition = open-book, providing 23 tools (Etherscan/CoinGecko/DefiLlama, market data APIs, web search, Python interpreter, etc.), with the guarantee that "the correct answer is always obtainable via appropriate tools," focusing the challenge on *tool selection + synthesis* rather than "information unavailability." All with-tools experiments use a unified ReAct-style framework (standard dispatch, result parsing, iterative reasoning) to eliminate implementation variance.
    - **Design Motivation**: Previous agent evaluations conflated tool ability, model reasoning, and prompt engineering, obscuring true bottlenecks; CAIA's engineering constraint that "answers are always tool-accessible" isolates "knowledge vs. orchestration" as independent dimensions, enabling clear attribution of failures to tool selection.

3. **6-Class Fine-Grained Failure Diagnosis + Cost-Aware Evaluation**:

    - **Function**: Decomposes single accuracy into 6 analytical categories + tool invocation distribution + cost efficiency + Pass@k vs. majority vote comparison, exposing the "trial-and-error luck" risk hidden by traditional metrics.
    - **Mechanism**: (a) Main metric uses 5-run majority vote to mitigate large model sampling variance; (b) Reports both Pass@1 and Pass@5, explicitly noting that Pass@k is a "dangerous metric" in high-risk scenarios—some models achieve Pass@1=26.4% but Pass@5=54.5% (DeepSeek R1 with tools), indicating reliance on random attempts; (c) Records per-task token consumption and dollar cost, calculating cost/score, revealing that cost and accuracy are not necessarily correlated (GPT-OSS 120B outperforms some closed models at 1/100th the cost); (d) Failure mode analysis shows 55.5% of tool calls prefer unreliable web search—even when specialized on-chain APIs can provide ground truth, models are misled by SEO-optimized misinformation and social platform manipulation.
    - **Design Motivation**: Single accuracy is a "width-zero" report, masking "why it failed" and "the cost of failure"; combining behavior distribution (tool selection preference), stability (majority vote vs. single run), and economics (cost/score) enables deployment-level judgment—precisely the diagnostic depth required for high-stakes scenarios.

### Loss & Training
CAIA is a benchmark, not a training method, and does not involve loss functions. Evaluation protocol: each task is run independently 5 times with majority vote; human baseline is established by 16 university blockchain club members and early-stage startup junior analysts on a stratified 10% subset, averaging 80% accuracy.

## Key Experimental Results

### Main Results
Seventeen models (GPT-4.1/4o/5/o3/OSS-120B, Claude Sonnet/Opus 4/4.1, Gemini 2.5 Flash/Pro, Grok 4/Fast, DeepSeek R1/V3.1, Kimi K2, Llama 4 Maverick, Qwen 3 235B) evaluated under both conditions:

| Model | No Tools Majority Vote | With Tools Majority Vote | With Tools Pass@5 | With Tools Cost ($) |
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
| **Human Junior Analyst** | – | **0.80** | – | – |

Most striking: DeepSeek R1 with tools achieves Pass@1=26.4% but Pass@5 surges to 54.5%, indicating reliance on "blind trial-and-error"; GPT-OSS 120B achieves near state-of-the-art performance at \$0.0003/query, representing the cost-accuracy Pareto frontier.

### Ablation Study

| Dimension | Key Observation | Note |
|---|---|---|
| Tool Availability | No tools 12–28% → With tools up to 67.4% | Tools help, but are not the ceiling |
| Tool Selection Behavior | 55.5% calls are web search | Even when specialized on-chain tools provide answers, models prefer unreliable sources |
| Pass@1 vs Pass@5 | Many models Pass@5 ≫ Pass@1 | Reveals trial-and-error, equivalent to "gambling" in high-risk scenarios |
| Category Distribution | On-Chain 43.3% / Project Disc. 27.5% / Tokenomics 12.9% | On-chain analysis dominates, most tests tool invocation |
| Human Baseline | 80% vs GPT-5 67.4% | Even strongest model + full tools lags by 12.6pp |

### Key Findings
- **Tool selection catastrophe**: Models systematically prefer web search (55.5%), even when specialized on-chain tools provide ground truth; the root cause is not "lack of information" but "agents' inability to assess information source reliability," an architectural flaw rather than a knowledge gap.
- **Pass@k is misleading in high-risk scenarios**: The gulf between Pass@5 and Pass@1 exposes the pseudo-capability of "just try more"—in finance, healthcare, and security, a single mistake is game over; traditional metrics are completely distorted.
- **Closed-source ≠ inherently stronger**: GPT-OSS 120B achieves results comparable to or better than several closed models at \$0.0003/query, with cost/score nearly 1,000 times lower than Claude Opus 4, profoundly impacting deployment economics.
- **Fundamental limits of Web2 training**: Model failures in crypto/Web3 are due to "out-of-distribution"—they have not seen on-chain data structures or experienced SEO attack scenarios, suggesting similar collapses in cybersecurity, content moderation, and other adversarial domains.
- **Frequency of "hallucination" has concrete economic cost**: Tasks are anchored to real block heights and amounts; wrong answers directly map to quantifiable financial losses, turning hallucination from "looks wrong" to "how much money is lost."

## Highlights & Insights
- **Cryptocurrency as an adversarial testbed is rigorously justified**: The authors clearly decompose "why crypto" into adversarial + irreversible + verifiable, providing a model motivation template for benchmark papers.
- **5-stage pipeline + 92 expert reviewers + 3,000+ real queries as seeds**: The data curation workload and expertise are key to the benchmark's credibility, far harder to replicate than synthetic tasks.
- **Explicit quantification of tool selection as observable behavior**: Using "frequency distribution of 23 tool calls" as an evaluation dimension, not just accuracy, is a profound methodological contribution to agent evaluation.
- **Pass@k critique**: Clearly points out Pass@k's misleading nature in high-risk/irreversible scenarios, advocating for majority vote + cost-aware evaluation, correcting methodology for the agent benchmark community.

## Limitations & Future Work
- 178 tasks is relatively small: Although each is expert-reviewed, the scale is still smaller than SWE-Bench (2294), AppWorld (750), etc., so statistical noise is non-negligible; the authors promise continuous updates to mitigate this.
- Limited to cryptocurrency: While the authors argue "crypto is the adversarial extreme," attack patterns in other domains (medical misdiagnosis, political content manipulation) differ structurally from on-chain deception, so transferability needs validation.
- Current adversarial evaluation mainly reflects "hostile information environments," lacking more active attacks such as prompt-injection, jailbreak, or tool poisoning; future extensions are possible.
- The ReAct framework, while unifying implementation, may limit some models' performance (some may excel under plan-then-execute); the impact of different agentic scaffolding is not isolated.
- Human baseline is only 16 people × 10% subset = 18 tasks/person, with high variance; the 80% figure should be seen as a ballpark rather than a precise threshold.
- Lacks "adversary evolution over time" dynamic evaluation—adversaries iterate their methods, so the benchmark needs continuous updates to stay sharp; the authors promise this but do not specify frequency or mechanism.

## Related Work & Insights
- **vs SWE-Bench / AppWorld / TheAgentCompany**: These benchmarks measure task completion in controlled environments; CAIA is the first to make "adversarial + high-risk + irreversible" core evaluation dimensions.
- **vs FinanceBench / FinQA**: Traditional finance benchmarks mostly use proprietary data or synthetic simulations; CAIA leverages blockchain's transparency and immutability to bypass the "proprietary vs. synthetic" dilemma.
- **vs τ-Bench / WebArena**: Those benchmarks focus on tool-use engineering metrics; CAIA introduces tool selection preference distribution as a new behavioral measure.
- **Insights**: (1) The design "answers are accessible but require correct tool selection" can be extended to other tool-rich domains, forcing evaluation to focus on orchestration ability; (2) Pass@k should be replaced by cost-aware + first-attempt accuracy + behavioral distribution, especially for deployment-oriented evaluation; (3) Using real practitioner queries as benchmark seeds better reflects actual capability than synthetic data.

## Rating
- Novelty: ⭐⭐⭐⭐ The benchmark is new, though "adversarial + real + time-anchored" elements have been seen individually; the key is the combination and the deep choice of crypto as testbed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 17 models × dual conditions × 5 samples × 6 categories × cost + human baseline, covering a wide range of dimensions.
- Writing Quality: ⭐⭐⭐⭐⭐ "Why crypto" justification, "Tool selection catastrophe" naming, and "Pass@k critique" are highly communicative, almost policy white-paper level.
- Value: ⭐⭐⭐⭐⭐ Directly warns of LLM agent deployment risks in finance and other high-stakes domains, providing guidance for model developers, regulators, and users alike.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Adversarial Environments Mislead Agentic AI](../../ACL2026/llm_agent/how_adversarial_environments_mislead_agentic_ai.md)
- [\[ICML 2026\] Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History](persona2web_benchmarking_personalized_web_agents_for_contextual_reasoning_with_u.md)
- [\[ICML 2026\] Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](scaling_benchmarking_and_reasoning_of_vision-language_agents_for_mobile_gui_navi.md)
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)
- [\[ICML 2026\] Adaptive Querying with AI Persona Priors](adaptive_querying_with_ai_persona_priors.md)

</div>

<!-- RELATED:END -->
