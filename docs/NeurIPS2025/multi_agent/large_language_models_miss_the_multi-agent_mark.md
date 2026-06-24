---
title: >-
  [Paper Note] Large Language Models Miss the Multi-Agent Mark
description: >-
  [NeurIPS 2025][Multi-Agent][Multi-agent systems] This position paper systematically surveys 1,400+ papers to argue that current LLM-based multi-agent systems (MAS LLMs) deviate from foundational MAS theory along four dimensions: LLMs lack native social behavior, environment design is LLM-centric, asynchronous coordination and standard communication protocols are absent, and emergent behaviors lack quantification. The paper warns that the field risks reinventing the wheel whil…
tags:
  - "NeurIPS 2025"
  - "Multi-Agent"
  - "Multi-agent systems"
  - "Position paper"
  - "Social intelligence"
  - "Asynchronous communication"
  - "Emergent behavior"
date: 2026-05-08
content_hash: e457bbe5bb753d63
---

# Large Language Models Miss the Multi-Agent Mark

**Conference**: NeurIPS 2025
**arXiv**: [2505.21298](https://arxiv.org/abs/2505.21298)  
**Code**: None  
**Area**: LLM/NLP
**Keywords**: Multi-agent systems, Position paper, Social intelligence, Asynchronous communication, Emergent behavior

## TL;DR

This position paper systematically surveys 1,400+ papers to argue that current LLM-based multi-agent systems (MAS LLMs) deviate from foundational MAS theory along four dimensions: LLMs lack native social behavior, environment design is LLM-centric, asynchronous coordination and standard communication protocols are absent, and emergent behaviors lack quantification. The paper warns that the field risks reinventing the wheel while ignoring 40 years of MAS research.

## Background & Motivation

**Background**: MAS LLMs have grown explosively in recent years, with applications spanning software engineering, multi-robot planning, data analysis, scientific reasoning, and social simulation. Frameworks such as AutoGen, MetaAgent, and CAMEL have proliferated rapidly.

**Limitations of Prior Work**: The field has extensively borrowed MAS terminology—"agent," "collaboration," "emergence"—without genuinely engaging with foundational MAS theory. This has led to a series of fundamental problems: LLMs are pretrained in isolation to respond to user requests and have never been trained to interact with other agents; environment design is built entirely around LLMs, neglecting their inherent limitations (hallucination, non-determinism, absence of long-term memory); communication relies on natural language, an expensive and ambiguous medium; and claims of emergent behavior are largely based on observational descriptions rather than quantitative metrics.

**Key Challenge**: MAS has accumulated over 40 years of theoretical and practical development since the 1980s–90s, yet builders of current MAS LLMs rarely reference this body of work in their framework designs. For instance, agent frameworks released by Google, Anthropic, Microsoft, and OpenAI cite only ML research while ignoring decades of MAS literature.

**Goal**: To systematically analyze the gap between MAS LLMs and classical MAS, identify overlooked core issues, and propose constructive research directions for each.

**Key Insight**: Classical MAS theory serves as the benchmark, and current practice is examined along four dimensions: social intelligence, environment design, coordination and communication, and emergent behavior. The authors are CS and MAS researchers from Oxford, KCL, and Sussex with deep backgrounds in traditional MAS.

**Core Idea**: Most LLM systems that claim to be "multi-agent" lack genuine multi-agent properties; the community should return to MAS foundations to avoid redundant effort.

## Method

### Overall Architecture

The paper adopts a critical survey methodology, systematically reviewing approximately 112 MAS LLMs benchmark/evaluation papers, 1,400+ MAS LLMs papers, and 60+ papers on emergent behavior. Analysis proceeds along four dimensions, each concluding with proposed research directions.

### Key Designs

1. **Argument 1: LLMs Lack Native Social Behavior**

    - Core argument: Intelligent agents in MAS require three capabilities—reactivity, proactiveness, and social ability. LLMs possess the first two, but social ability is entirely injected via prompts or imposed by an orchestrator, not acquired through training.
    - Key evidence: Cemri et al. find that 37% of MAS LLMs failures stem from inter-agent alignment and coordination issues; LLMs perform poorly on Theory of Mind benchmarks; most MAS LLMs effectively degrade into ensembles (majority voting) rather than genuine collaboration.
    - Research directions: Incorporate multi-agent cooperation and competition scenarios during pretraining; leverage text-feedback-based training methods such as TextGrad to enable agents to learn social behavior through interaction.

2. **Argument 2: Environment Design Is LLM-Centric**

    - Core argument: Traditional MAS environment design makes no assumptions about agent architecture, whereas current MAS LLMs assume all agents are LLMs communicating via natural language. This overlooks three inherent LLM deficiencies: non-determinism (temperature set to 0 is not fully deterministic), hallucination (deviation from assigned identities/roles), and absence of long-term memory.
    - Key data: A survey of 112 MAS LLMs papers finds that most operate in partially observable, determinism-assumed, discrete-time, text-based environments.
    - Illustrative cases: In CAMEL, two LLMs inadvertently swap roles and fall into infinite message loops; in MetaAgent, LLMs hallucinate capabilities and deviate from assigned identities.
    - Research directions: Design multimodal environments to reduce natural-language mediation; replace free-form text with structured formats; integrate formal planners or neuro-symbolic methods.

3. **Argument 3: Absence of Asynchronous Coordination and Standard Communication**

    - Core argument: Asynchrony is a core characteristic of genuine MAS, yet a survey of 1,400+ MAS LLMs papers identifies only 22 that explicitly address asynchronous interaction. Natural language communication is costly and ambiguous, and established structured agent communication standards such as KQML and FIPA ACL are entirely ignored.
    - AutoGen case: Although the framework supports asynchronous APIs, developers must manually define asynchronous calls for every action and event; doing asynchronous programming in a synchronous language is highly error-prone.
    - Research directions: Frameworks should treat asynchrony as the default and synchrony as the exception; borrow concurrent modeling formalisms such as Petri nets to analyze reachability and boundedness in MAS LLMs; establish standard agent communication protocols analogous to Google A2A.

### Argument 4: Emergent Behavior Lacks Quantification

A survey of 60+ papers claiming to study emergent behavior finds that the vast majority offer only qualitative observation without quantitative metrics. For example, the emergent concepts in Generative Agents (the Stanford town simulation) are never formally defined, and results largely amount to "run the system for a while and observe interesting behavior." Project Sid (a Minecraft civilization simulation) claims LLMs can achieve AI civilization milestones, but evidence is purely descriptive. Research directions: establish falsifiable definitions of emergent behavior, distinguish weak emergence (derivable from lower-level dynamics) from strong emergence (requiring new assumptions), and draw on mature definitions from economics and systems theory.

## Key Experimental Results

### Survey Statistics

| Survey Dimension | Corpus Size | Key Finding |
|---|---|---|
| Environment feature analysis | 112 MAS LLMs papers | Most assume partial observability, determinism, and text-based environments |
| Asynchrony survey | 1,400+ papers | Only 22 explicitly support asynchronous interaction |
| Emergent behavior papers | 60+ papers | Very few define measurable quantitative metrics |
| Failure analysis | Cemri et al. (cited) | 37% of failures stem from inter-agent coordination issues |

### Key Findings

- **MAS LLMs Are "Multi-Agent" in Name Only**: Most systems are essentially LLM pipelines or ensembles controlled by an orchestrator.
- **Non-Scalability of Natural Language Communication**: The ambiguity and high computational cost of natural language make communication in large-scale agent networks unsustainable.
- **Conflation of Emergence and Hallucination**: The field currently lacks a methodology for distinguishing genuinely emergent behavior from LLM hallucinations or coincidental outputs.

## Highlights & Insights

- **Data-Driven Critical Perspective**: Rather than a purely opinion-based piece, this paper draws on a systematic survey of 1,400+ papers, lending strong empirical credibility to its arguments.
- **Bridging Two Communities**: The paper connects 40 years of accumulated MAS theory (KQML, BDI architectures, concurrent systems theory) with the LLM Agent community, identifying a wealth of reusable theoretical tools.
- **Every Critique Is Paired with a Constructive Direction**: The paper not only diagnoses problems but also provides specific, actionable research paths—for instance, "modeling MAS LLMs state transitions with Petri nets" is a concrete and well-targeted entry point.

## Limitations & Future Work

- **Bias Toward Classical MAS**: The paper may underestimate the practical value of LLMs in certain scenarios where they operate effectively without strict adherence to MAS theory.
- **Lack of Empirical Comparison**: No controlled experiment is conducted comparing "an LLM system designed according to MAS principles" against "existing systems" to empirically validate the paper's claims.
- **Limited Impact on Industrial Practice**: Industry prioritizes practical utility over theoretical rigor.
- **The Definition Problem for Emergent Behavior**: Although the paper criticizes the lack of quantification in existing work, it does not itself provide an operational definition of emergence metrics.

## Related Work & Insights

- **vs. CAMEL / AutoGen / MetaAgent**: The paper critically analyzes the MAS shortcomings of these frameworks while acknowledging their engineering value in practical applications.
- **vs. Classical MAS Textbooks (Wooldridge)**: The paper extensively cites Wooldridge's canonical definitions as theoretical reference points.
- **Noteworthy Emerging Directions**: Google A2A and MCP (Model Context Protocol) are actively attempting to standardize inter-agent communication, resonating with the research directions advocated in this paper.

## Rating

- Novelty: ⭐⭐⭐⭐ — Unique perspective; systematically examines the LLM Agent ecosystem through the lens of classical MAS theory.
- Experimental Thoroughness: ⭐⭐⭐ — Large survey corpus, but empirical experiments validating the claims are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear argumentative structure; each claim is supported by data and accompanied by responses to alternative views.
- Value: ⭐⭐⭐⭐ — An important cautionary note and theoretical supplement for LLM Agent researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)
- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](../../ACL2026/multi_agent/agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[ICLR 2026\] Cache-to-Cache: Direct Semantic Communication Between Large Language Models](../../ICLR2026/multi_agent/cache-to-cache_direct_semantic_communication_between_large_language_models.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](../../AAAI2026/multi_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[ICLR 2026\] Emergent Coordination in Multi-Agent Language Models](../../ICLR2026/multi_agent/emergent_coordination_in_multi-agent_language_models.md)

</div>

<!-- RELATED:END -->
