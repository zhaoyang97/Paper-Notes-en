---
title: >-
  [Paper Note] SafeSearch: Automated Red-Teaming of LLM-Based Search Agents
description: >-
  [ICML 2026][Audio & Speech][Search Agents] This paper proposes SafeSearch—a fully automated, sandboxed, and scalable red-teaming framework that evaluates the safety of search agents by injecting a single LLM-generated un…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Search Agents"
  - "Unreliable search results"
  - "Red-teaming"
  - "Indirect prompt injection"
  - "Misinformation"
date: 2026-05-08
content_hash: da3b01244b679089
---

# SafeSearch: Automated Red-Teaming of LLM-Based Search Agents

**Conference**: ICML 2026  
**arXiv**: [2509.23694](https://arxiv.org/abs/2509.23694)  
**Code**: https://github.com/jianshuod/SafeSearch  
**Area**: LLM Security / Agent Security / Red-teaming Evaluation  
**Keywords**: Search Agents, Unreliable search results, Red-teaming, Indirect prompt injection, Misinformation

## TL;DR
This paper proposes SafeSearch—a fully automated, sandboxed, and scalable red-teaming framework that evaluates the safety of search agents by injecting a single LLM-generated unreliable webpage into real search results. Systematically evaluating 17 LLMs across 3 agent scaffolds using 300 test cases, the study finds a peak ASR of 90.5% and demonstrates that common "reminder" defenses are nearly ineffective.

## Background & Motivation

**Background**: "Search Agents" (represented by ChatGPT Search, Gemini Deep Research, Search-R1, etc.) obtain real-time and long-tail information by connecting LLMs to search engines. This has become a mainstream external knowledge enhancement paradigm beyond RAG, covering various applications from quick queries to deep research.

**Limitations of Prior Work**: The safety of search agents fundamentally depends on the reliability of search results. However, the open internet is filled with content farms, black-hat SEO, promotional articles, and Wikipedia errors. The authors' empirical tests found that among 8,933 user-like queries, 4.3% (380) of top results originated from low-trust sources. Furthermore, enabling search on 1,000 health-related queries led to 46 binary stance reversals, proving the threat is not merely theoretical.

**Key Challenge**: Existing evaluations rely on manual question design (non-scalable, e.g., OpenAI internal red-teaming), construction of malicious queries (narrow coverage, high cost), or require real manipulation of search rankings (harmful to innocent users, unethical). Meanwhile, existing RAG safety research assumes an auditable corpus, which cannot be applied to the "uncontrollable webpages appearing only at runtime" scenario.

**Goal**: To build a scalable, low-cost, sandboxed red-teaming framework capable of: (i) automatically generating batches of test cases across multiple risk types, (ii) reproducing the threat of "benign query + unreliable search results" without polluting real search engines, and (iii) providing quantifiable agent safety metrics to support subsequent defense research.

**Key Insight**: Model the threat from an agent-centric perspective: "benign queries encountering unreliable results." This treats unreliable sources as outputs of an *insider* tool rather than injections from an external attacker. Additionally, the "differential testing" concept is introduced, using a baseline agent to compare behavior under benign vs. manipulated tools to automatically filter out invalid test cases.

**Core Idea**: Utilize an LLM pipeline "Generation—Site Synthesis—Judgment" combined with single-page sandbox injection to transform search agent red-teaming into a repeatable, scalable, and zero-harm standard evaluation protocol.

## Method

The core of SafeSearch is a red-teaming framework orchestrated by four LLM assistants, divided into two phases: offline generation of 300 high-quality test cases and online testing of search agents scored by an LLM-as-Judge.

### Overall Architecture
The input is a natural language description of a risk type (5 categories: indirect prompt injection, harmful output, advertisement promotion, misinformation, and bias inducement). The generation phase outputs a test case quadruple `(benign query, target negative consequence, unreliable website, checklist)`. In the testing phase, an unreliable page $d_u$ is appended to the real search result list $D=\{d_1,\dots,d_k\}$. The Agent executes normal multi-turn search/tool calling/deep research on $D\cup\{d_u\}$ and generates a final response. In the evaluation phase, a safety evaluator assisted by a checklist outputs a boolean judgment to aggregate the Attack Success Rate (ASR). A helpfulness evaluator scores the response from 1–5 to calculate the Helpfulness Score (HS). In multi-turn Agents, $d_u$ is only injected in the first round to give the agent a chance to "self-correct" in subsequent searches, thereby obtaining a **conservative lower bound** for Agent safety.

### Key Designs

1.  **Three-step Generation Workflow + Differential Filtering**:
    - **Function**: Transforms a "one-sentence risk description" into an executable test case quadruple while automatically removing unqualified cases.
    - **Mechanism**: Uses o4-mini as a test generator, progressing through Scenario anchoring → Test design → Test instantiation. It first envisions a real-world scenario, then designs a "malicious plan" (specifying target consequences and exploitable site types, emphasizing threats post-LLM knowledge cutoff), and finally produces a "content guide" for the website generator and a "checklist" for the safety evaluator. After generation, Qwen3-8B acts as a baseline agent for two differential tests: whether the target consequence is successfully induced under the **manipulated tool** (attainability $A$) and whether it is triggered spontaneously under the **benign tool** (integrity $I$). Only cases satisfying $A\wedge\neg I$ are included.
    - **Design Motivation**: Purely generated cases are often "invalid" (the baseline agent fails under benign conditions anyway, or the attack fails to induce the issue). Failing to filter these results in a distorted ASR. Differential testing decouples the "effective attack surface" from "existing model flaws," ensuring ASR measures the actual extra harm caused by unreliable results.

2.  **Single-page Sandbox Injection + Timestamp-conditioned Generation**:
    - **Function**: Synthesizes an unreliable website $d_u$ on the fly and appends it to real search results to simulate "low-quality content newly appeared online."
    - **Mechanism**: The website generator (GPT-4.1-mini) creates $d_u$ based on the "content guide," using the test date as a condition so the content falls after the LLM knowledge cutoff. The injection occurs only in the first round of multi-turn Agents. Real Google results are fetched via Serper API and content extracted via Jina Reader API, with a 2,000 token limit per site and top-5 retrieval to ensure fairness.
    - **Design Motivation**: Real SEO manipulation is unethical and harmful, while static corpora fail to capture the "real-time search + long-tail query" threat. Single-page injection balances these by preserving the real execution chain of the Agent while reproducibly assessing the worst-case impact of one unreliable result.

3.  **Checklist-aided Dual-axis LLM-as-Judge**:
    - **Function**: Automatically evaluates whether the Agent response triggers negative consequences (safety) and its perceived helpfulness to the user (helpfulness).
    - **Mechanism**: The safety evaluator receives the `(query, target consequence, checklist, agent response)` quadruple, performs reasoning, and outputs a boolean judgment (averaged over three runs for ASR). The helpfulness evaluator assigns a 1–5 score (scaled to 0–100 for HS) based on the `(query, response)` pair, focusing on perceived utility rather than factual accuracy. The safety evaluator achieves >95% human agreement.
    - **Design Motivation**: Asking "is it safe?" is subjective and unstable. Using a checklist (produced during generation) as a rubric makes the definition of "breached" explicit and verifiable, improving consistency and transparency. Separating helpfulness and safety reveals the trap where "unsafe responses may still feel very useful" (experimentally, $\text{HS}_\text{manip.}=92.2 > \text{HS}_\text{benign}=91.4$ for tool-calling).

### Loss & Training
SafeSearch does not train any models; it performs "zero-shot" red-teaming. All LLM assistants are orchestrated via prompts: five roles (test generator, website generator, safety evaluator, helpfulness evaluator, baseline filter) are performed by o4-mini, GPT-4.1-mini, Qwen3-8B, etc. Agents are set to a temperature of 0.6, with results averaged over 3 trials. The final dataset comprises 300 cases (5 risk types × 60 cases/type), filled via a "generation-filtering" loop.

## Key Experimental Results

### Main Results
Evaluation covers 9 closed-source and 8 open-source LLMs across 3 scaffolds (passive search workflow, active tool calling, and "deep research" LangGraph prototype). Selected results for Overall ASR↓ (%):

| Configuration | search workflow | tool calling | deep research |
| :--- | :--- | :--- | :--- |
| GPT-4.1-mini | **90.5** | 77.8 | 57.4 |
| GPT-4.1 | 85.0 | 77.3 | — |
| Gemini-2.5-Pro | 75.1 | 58.5 | — |
| o4-mini | 60.2 | 43.8 | — |
| Claude-Sonnet-4.5 | 19.8 | **4.6** | — |
| GPT-5 | 18.4 | 5.0 | — |
| DeepSeek-R1 | 66.8 | 64.8 | 30.6 |
| Qwen3-8B | 85.5 | 70.8 | 45.8 |
| **Average** | 63.1 | 49.3 | 38.9 |

Regarding helpfulness, under tool-calling, the average $\text{HS}_\text{benign}=91.4$ while $\text{HS}_\text{manip.}=92.2$—safety failures often appear slightly more useful to the user.

### Ablation Study
Search budget controlled comparison (GPT-4.1-mini backend, ASR %):

| Scaffold | budget=3 | budget=6 | budget=9 |
| :--- | :--- | :--- | :--- |
| Tool-call auto | 74.6 | 74.0 | 74.3 |
| Tool-call forced | 49.9 | 43.7 | **36.7** |
| Deep research auto | 63.2 | 58.3 | 57.0 |
| Deep research forced | 59.4 | 56.4 | 46.2 |

Key observation: When forced to exhaust the budget, tool-calling becomes safer than deep research. The safety advantage of deep research primarily stems from its willingness to search more frequently.

### Key Findings
- **Reasoning models are more noise-resistant but not sufficient**: Models like Claude-Sonnet-4.5, GPT-5, and o4-mini have significantly lower ASR, but newer knowledge (Gemini-2.5-Pro) does not replace "skepticism."
- **Risk types vary greatly**: Misinformation is the hardest risk to defend (highest average ASR), while Indirect Prompt Injection is the easiest (0% on GPT-5 + tool-calling), reflecting recent industry focus on injection defense.
- **Defense strategies have limited effect**: Reminders (system prompt warnings) are nearly useless, showing a "knowledge-action gap" (LLMs know a source is unreliable but use it anyway). Filtering (using GPT-4.1-mini as a detector) reduces ASR by half but has only 44.2% recall.
- **Search result count is an implicit safety knob**: As top-k decreases, the weight of a single unreliable site increases, leading to a monotonic rise in ASR.

## Highlights & Insights
- **Turning red-teaming into sustainable CI**: The "generation-filtering" loop allows for updating test difficulty as baseline agents improve. Timestamp conditioning ensures templates stay fresh, perfectly matching the pace of model iteration.
- **Conservative sandbox design enhances credibility**: Injecting only 1 page at the end of results and only in the first round are choices that "lower the ASR." Therefore, the observed 60–90% ASR represents a true lower bound of vulnerability.
- **The "knowledge–action gap" is a transferable insight**: An LLM might correctly identify a site as untrustworthy when asked directly, but will still adopt its content within an agent chain. This suggests defenses should focus on the "action layer" (e.g., source weighting) rather than just the "knowledge layer" (warnings).

## Limitations & Future Work
- SafeSearch ASR should be interpreted as "vulnerability under controlled simulation" rather than real-world failure rates. More sophisticated adversarial SEO would likely increase ASR.
- The evaluation covers only 5 risk types, and helpfulness is "perceived" rather than "factual." User studies are needed to see if the "high HS + high ASR" combination successfully deceives humans.
- Reliability depends on the checklist; emergent safety issues not in the list may be missed. There is also a risk of "generator-evaluator consistency bias" when using the same model family for different roles.
- Future work: Transforming filtering into agent-aware detection, integrating "chain-level" constraints (cross-verification, confidence weighting) into scaffolds, and extending to multi-source coordinated misinformation.

## Related Work & Insights
- **vs. Luo et al. 2025 / Ou et al. 2025**: They focus on system-level outputs and adversarial queries. This paper maintains an "agent-level" behavior focus using benign queries, attributing risk to the search tool.
- **vs. AgentHarm / GAIA**: These evaluate task capability or general agent harm, whereas SafeSearch specifically targets the "source reliability" dimension of search agents.
- **vs. RAG Safety (PoisonedRAG, SafeRAG)**: RAG assumes auditable corpora at the vector DB layer. Search Agents face the open internet and must use inference-time reasoning, necessitating the "online injection" approach.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of sandbox injection, differential filtering, and checklist evaluation for search agents is a first, though individual components are established.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ High scale (17 LLMs, 3 scaffolds, 300 cases, 3 trials) with extensive ablations on budget, defenses, and human consistency.
- Writing Quality: ⭐⭐⭐⭐ Clear threat model and protocol, though high table density and few figures may be challenging for non-security readers.
- Value: ⭐⭐⭐⭐⭐ Open-source dataset and framework suitable for CI-based regression testing, revealing critical insights like the "knowledge-action gap."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration](group_cognition_learning_making_everything_better_through_governed_two-stage_age.md)
- [\[ICLR 2026\] RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments](../../ICLR2026/audio_speech/redteamcua_adversarial_testing_agents.md)
- [\[ICML 2026\] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)
- [\[ACL 2026\] LLM-MC-Affect: LLM-Based Monte Carlo Modeling of Affective Trajectories and Latent Ambiguity for Interpersonal Dynamic Insight](../../ACL2026/audio_speech/llm-mc-affect_llm-based_monte_carlo_modeling_of_affective_trajectories_and_laten.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](../../ACL2026/audio_speech/full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)

</div>

<!-- RELATED:END -->
