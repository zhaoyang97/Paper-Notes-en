---
title: >-
  [Paper Note] Are Large Language Models Sensitive to the Motives Behind Communication?
description: >-
  [NeurIPS 2025][LLM Agent][motivational vigilance] Three progressive experiments systematically evaluate whether LLMs possess "motivational vigilance"—the ability to recognize the intentions and incentives of information…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "motivational vigilance"
  - "strategic communication"
  - "information credibility"
  - "rational model"
date: 2026-05-08
content_hash: fd3bbe09a9097d16
---

# Are Large Language Models Sensitive to the Motives Behind Communication?

**Conference**: NeurIPS 2025
**arXiv**: [2510.19687](https://arxiv.org/abs/2510.19687)  
**Code**: None  
**Area**: LLM Agent / Social Cognition
**Keywords**: motivational vigilance, strategic communication, information credibility, rational model

## TL;DR

Three progressive experiments systematically evaluate whether LLMs possess "motivational vigilance"—the ability to recognize the intentions and incentives of information sources and adjust trust accordingly. In controlled experiments, frontier non-reasoning LLMs perform close to the rational model (Pearson's $r > 0.9$) and resemble humans more than the rational model does; however, vigilance drops sharply in real-world YouTube sponsored content ($r < 0.2$), and simple prompt steering partially restores it (raising $r$ to 0.31).

## Background & Motivation

**Background**: LLMs are increasingly deployed as agents acting in the real world on behalf of users—handling emails, browsing the web, and making purchasing decisions. Information encountered in these settings inherently carries the sender's motives—a salesperson's recommendation, a sponsor's advertisement, or a competitor's review. Humans naturally identify and discount information from biased sources through *epistemic vigilance*.

**Limitations of Prior Work**: LLMs are known to exhibit multiple vigilance failures: (1) jailbreaking—malicious instructions bypass safety mechanisms; (2) sycophancy—blindly conforming to users' erroneous beliefs; (3) pop-ups and distracting content hijacking agent behavior. The common root cause is a lack of deep understanding of the motives behind information sources, yet a systematic evaluation framework has been absent.

**Key Challenge**: LLM training paradigms prioritize instruction-following and user satisfaction over monitoring the incentives and veracity of information sources—yet the latter is precisely what enables reliable real-world agent behavior.

**Goal**: To what extent do LLMs possess motivational vigilance? How does performance differ between controlled and naturalistic settings? Can simple interventions improve vigilance?

**Key Insight**: The rational model from cognitive science (Oktar et al., 2024/2025) is used as a normative benchmark, and LLM vigilance is evaluated across three experimental paradigms of increasing difficulty.

**Core Idea**: LLMs exhibit near-rational motivational vigilance in simple controlled settings, but in naturalistic contexts with rich contextual information, the additional content diverts attention from motivation-relevant cues.

## Method

### Overall Architecture

Three progressive experiments: (1) *Basic capacity*—can LLMs distinguish intentional communication from incidental information? (2) *Fine-grained calibration*—can LLMs precisely adjust trust based on speaker benevolence and incentives? (3) *Real-world generalization*—can LLMs maintain vigilance in YouTube sponsored content? Each experiment uses the rational Bayesian model of Oktar et al. as a normative benchmark.

### Key Designs

1. **Experiment 1: Distinguishing Intentional Communication from Incidental Observation**
    - **Function**: Tests whether LLMs adjust their judgments based on the source of information (deliberate "advice" from a motivated party vs. incidentally "overheard" ground truth).
    - **Mechanism**: Adapted from Watson & Morgan (2024)'s two-player judgment task—Player 2 receives either Player 1's deliberate advice or a peeked true answer. Cooperative/competitive payoff structures modulate information credibility. The LLM plays Player 2 and decides whether and how much to update its answer.
    - **Design Motivation**: This is the most fundamental prerequisite for motivational vigilance—the ability to distinguish whether information carries strategic intent.

2. **Experiment 2: Fine-Grained Vigilance Calibration Against a Rational Model**
    - **Function**: Tests whether LLMs precisely adjust trust in recommendations based on two key factors: benevolence $\lambda$ and incentive $R_S$.
    - **Mechanism**: Uses the rational Bayesian model of Oktar et al. as a normative benchmark. The probability that a speaker chooses an utterance is $P_S(u) \propto \exp\{\beta_S \cdot \sum R_{Joint} \pi_L(a|u)\}$, where $R_{Joint} = \lambda R_L + (1-\lambda)R_S$. A vigilant listener infers product quality as $P_L(R_L|u) \propto P_S(u|...) P(R_S) P(R_L) P(\lambda)$. Sixteen speaker–incentive combinations are tested across three domains: finance, real estate, and healthcare.
    - **Design Motivation**: Provides a quantitative benchmark—assessing not only whether LLMs are vigilant, but whether the degree of vigilance is rational.

3. **Experiment 3: Ecological Validity in YouTube Sponsored Content**
    - **Function**: Tests whether LLM vigilance generalizes across 300 real YouTube sponsored segments.
    - **Mechanism**: Sponsorship timestamps are obtained from SponsorBlock; video metadata and transcripts are collected; brand names are masked to prevent prior knowledge interference. LLMs separately estimate product quality ($P(R_L|u)$), sponsorship revenue ($R_S$), and channel credibility ($\lambda$). LLM inferences are correlated with rational model predictions. The effect of "vigilance prompt steering"—prompting the model to attend to speaker incentives and intentions—is also tested.
    - **Design Motivation**: Controlled experiments cannot represent real deployment conditions; naturalistic sponsored content, replete with distracting information, constitutes a genuine test of vigilance.

## Key Experimental Results

### Main Results

| Model | Controlled: Bayesian–LLM $r$ | Controlled: LLM–Human $r$ | Real-World $r$ | +Steering $r$ |
|---|---|---|---|---|
| GPT-4o | 0.911 | **0.943** | 0.024–0.121 | **0.137–0.312** |
| Claude 3.5 Sonnet | 0.845 | **0.941** | 0.033–0.190 | **0.200–0.283** |
| Llama 3.3-70B | 0.876 | 0.922 | −0.011–0.098 | **0.029–0.152** |
| o1 | 0.705 | 0.861 | — | — |
| DeepSeek-R1 | 0.326 | 0.643 | — | — |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Non-reasoning vs. reasoning models | Bayesian correlation | Non-reasoning (0.8–0.9) >> reasoning (0.3–0.7) |
| Direct output vs. CoT | Influence multiplier | CoT makes LLMs more susceptible to Player 1's influence (>60%) |
| Agent role vs. assistant role | DeepSeek $r$ | Agent: 0.793 → Assistant: −0.141 (vigilance completely lost) |
| Small models (3B/8B) | Bayesian $r$ | 0.29–0.61—scale determines vigilance capacity |
| Default prompt vs. vigilance steering | Real-world $r$ | Steering improves $r$ by 0.1–0.2 on average; most gains significant ($p < .05$) |

### Key Findings
- Frontier non-reasoning LLMs resemble humans more than the rational model does in controlled settings ($r_{LLM\text{-}Human} > r_{Bayesian\text{-}Human}$), suggesting that LLMs capture supra-rational heuristic biases present in human vigilance.
- Reasoning models (o1/o3-mini/DeepSeek-R1) perform worse on vigilance tasks—reasoning steps may interfere with intuitive motivational judgment.
- CoT increases LLMs' susceptibility to social information (in the opposite direction from human bias)—CoT amplifies trust rather than skepticism.
- The primary cause of vigilance collapse in naturalistic settings is that additional contextual information diverts attention from motivation-relevant cues.
- Simple prompt steering (reminding the model to attend to motives) can partially restore vigilance, suggesting that the capability exists but is not activated by default.

## Highlights & Insights
- The first systematic evaluation of LLM motivational vigilance using a rational model from cognitive science.
- The large gap between controlled and real-world settings (0.9 → 0.02) exposes a fundamental limitation of laboratory-based evaluation.
- The counterintuitive finding that reasoning models exhibit weaker vigilance carries important warnings for the deployment of reasoning LLMs as agents.
- The effectiveness of prompt steering points to a low-cost improvement pathway.

## Limitations & Future Work
- YouTube sponsored content represents a single scenario; other forms of motivated communication (misinformation, phishing emails, etc.) remain untested.
- Whether the rational model is a sufficient normative benchmark is debatable—humans themselves are not fully rational.
- Steering prompts require manual design and are difficult to generalize to all motivated communication settings.
- Evaluation is limited to English; cross-lingual and cross-cultural differences in motivational understanding may exist.

## Related Work & Insights
- **vs. Jailbreaking research**: Jailbreaking tests LLM resistance to malicious instructions; this paper tests rational discounting of motivated information—the latter is more representative of agent deployment conditions.
- **vs. Theory of Mind research**: ToM is a prerequisite for vigilance (requiring understanding of others' beliefs and intentions); this paper focuses on how vigilance translates ToM information into calibrated belief updating.
- **vs. Sycophancy research**: Sycophancy is a symptom of vigilance failure—failing to question users' erroneous beliefs; this paper provides a framework for quantifying the relevant capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First introduction of the cognitive-science motivational vigilance framework into LLM evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three progressive experiments, multiple models, human comparisons, and mitigation strategies.
- Writing Quality: ⭐⭐⭐⭐ Cross-disciplinary writing is clear; the rational model is introduced with sufficient detail.
- Value: ⭐⭐⭐⭐⭐ Important implications for LLM agent deployment safety; reveals that controlled experiments cannot substitute for real-world evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [\[ACL 2026\] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](../../ACL2026/llm_agent/implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)
- [\[NeurIPS 2025\] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](agenttts_large_language_model_agent_for_testtime_computeopti.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](../../ACL2026/llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)

</div>

<!-- RELATED:END -->
