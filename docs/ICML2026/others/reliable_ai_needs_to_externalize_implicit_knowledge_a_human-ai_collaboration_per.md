---
title: >-
  [Paper Note] Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective
description: >-
  [ICML 2026 (Position Paper)][Implicit Knowledge] This ICML position paper argues that current AI reliability methods (RAG, self-consistency, RLHF, Agent Memory) can only verify explicit knowledge…
tags:
  - "ICML 2026 (Position Paper)"
  - "Implicit Knowledge"
  - "Knowledge Objects"
  - "Human-in-the-loop"
  - "Verification Economics"
  - "RLHF Alternative"
date: 2026-05-08
content_hash: 5cc31c27dcb17ef5
---

# Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2605.02010](https://arxiv.org/abs/2605.02010)  
**Code**: None (Position paper, no open-source implementation)  
**Area**: AI Reliability / Human-AI Collaboration / Knowledge Management  
**Keywords**: Implicit Knowledge, Knowledge Objects, Human-in-the-loop, Verification Economics, RLHF Alternative

## TL;DR
This ICML position paper argues that current AI reliability methods (RAG, self-consistency, RLHF, Agent Memory) can only verify explicit knowledge, while the true power of AI stems from "implicit knowledge"—representing 80-95% of training data—which remains undocumented by humans. The authors propose Knowledge Objects (KOs) as infrastructure to externalize implicit AI reasoning into structured artifacts that are human-checkable, verifiable, and endorsable, allowing the cost of a single human verification to yield long-term compound interest across a community.

## Background & Motivation

**Background**: LLM capabilities in knowledge-intensive tasks are advancing rapidly—75% of ChatGPT conversations involve knowledge work (Chatterji 2025), Copilot generates millions of code suggestions daily, and RCTs show that AI collaboration can bring 20-40% productivity gains. However, the same systems continue to fail at scale—professional legal AI hallucinates on 17-34% of queries (Magesh 2025), 28.6% of citations in GPT-4 medical reviews are fabricated, and general LLMs exhibit error rates of 58-88% on verifiable legal questions.

**Limitations of Prior Work**: The authors identify a fatal flaw shared by four categories of mainstream reliability methods: **(1) RAG** only verifies "what the document says," not "how the AI reasons"; **(2) Internal verification (Self-Consistency, Uncertainty, LLM-as-Judge)** uses AI to evaluate AI, leading to the consistent reproduction of systemic errors (true hit rate of 65% for 99% confidence intervals); **(3) Training-based methods (SFT/RLHF/DPO)** shove knowledge into a parameter black box, making it impossible to trace or correct incrementally, with sycophancy persisting at 78.5% post-alignment; **(4) Agent Memory (MemGPT/Reflexion/MemoryBank)** stores data without verification states, causing erroneous information to accumulate.

**Key Challenge**: AI-learned knowledge exists in two layers: **Explicit Knowledge** (papers, documents, databases, 5-20%) which can be cited; and **Implicit Knowledge** (reasoning patterns, debugging heuristics, domain intuition, 80-95%) embedded in conversation logs, commit histories, and experiment logs. Implicit knowledge has never been formally extracted because "recording cost > perceived benefit." LLMs learn both indiscriminately, acquiring expert judgment alongside systemic biases—but only explicit knowledge is currently verifiable.

**Goal**: To establish an infrastructure that allows AI to "externalize" learned implicit knowledge into artifacts that humans can inspect, correct, and cumulatively verify, transforming the hidden cost of "re-evaluating AI output every time" into a compound interest model of "verify once, reuse forever."

**Key Insight**: Drawing from Nonaka’s organizational knowledge theory (1994) and Polanyi’s tacit knowledge theory (1966), the authors posit that implicit knowledge is not "unrecordable" but rather "marginal recording cost > perceived marginal utility." By using AI to automatically extract implicit patterns into structured candidates, humans only need to perform "lightweight verification," flipping verification economics from "do it every time" to "do it once, benefit continuously."

**Core Idea**: Position Knowledge Objects (KOs) as the "hub" for human-AI collaboration—AI externalizes implicit knowledge into structured products (claim + evidence + scope + validation metadata), while humans verify, correct, and endorse them. Verification status is treated as a first-class citizen for persistence and retrieval.

## Method

### Overall Architecture
The paper is not a methodological study but proposes a **conceptual framework + five core attributes + call to action**. The architecture is the "KO-Hub" collaboration paradigm: Environment $\rightarrow$ Task $\rightarrow$ (AI System + Human) Collaboration $\rightarrow$ Generation of Interaction Data $\rightarrow$ AI externalizes candidate KOs from interaction data $\rightarrow$ Human verification/correction/rejection $\rightarrow$ Verified KOs enter a Collective Human Knowledge pool $\rightarrow$ Subsequent tasks can retrieve these verified KOs. This loop converts human verification from a disposable judgment into a cumulative, searchable asset.

### Key Designs

1.  **Formal Definition and Five Attributes of Knowledge Objects**:
    - **Function**: Solidifies implicit knowledge into objects humans can "see-verify-endorse," rather than inaccessible representations in parameters.
    - **Mechanism**: Definition 4.1 specifies a KO must contain four elements: (i) knowledge claim or procedure, (ii) supporting evidence or reasoning, (iii) explicit scope and limitations, and (iv) validation metadata (logging who, when, and under what conditions it was verified). Five mandatory attributes are proposed: **Understandable** (readable for domain experts, not embeddings), **Verifiable** (records verification state, not a one-off judgment), **Traceable** (traceable to source and endorsement), **Controllable** (human-editable), and **Reusable** (one-time verification ammortized for future users). The first three address invisibility/unverifiability/untraceability, while the last two ensure cost-efficiency.
    - **Design Motivation**: Contrast RAG (explicit citation only) and Agent Memory (persistence without state). KOs make "human verification state" a first-class attribute.

2.  **Verification Economics Inversion**:
    - **Function**: Replaces the hidden total cost of "every user independently evaluating AI output" with a model where "one expert verifies, $N$ users benefit."
    - **Mechanism**: Polanyi's "we know more than we can tell" stems from extraction costs being higher than immediate benefits; KOs utilize AI to shift "extraction" costs from humans to machines. Humans perform "low-cost verification" like confirming scope tags. Verification evolves from private judgment to public asset. Using Wikipedia as an analogy, high-risk knowledge requires expert review (Featured Articles), while common patterns are labeled as "unvalidated."
    - **Design Motivation**: Addressing concerns that "verification will be a bottleneck," the authors argue that *not* verifying is the actual hidden cost. The current status quo is what's unsustainable.

3.  **Complementary Positioning of KOs**:
    - **Function**: Distinguishes KOs from Knowledge Graphs, wikis, and Agent Memory.
    - **Mechanism**: Table 1 compares impacts on implicit knowledge—RAG=Untouched (reasoning stays internal), Self-Verification=Unexposed (only yields confidence), Training=Absorbed (black box), Agent Memory=Unstructured. KOs are the **only** design transforming implicit knowledge into externally inspectable products. Procedural KOs are likened to Voyager's skill library or Agent Workflow Memory's reusable workflows.
    - **Design Motivation**: KOs do not replace existing KM systems but fill the "AI-generated yet human-verifiable" gap; traditional wikis manage "what humans have written," while KOs manage "what AI has learned but humans haven't verified."

### Loss & Training
As a position paper, there is no training objective. Instead, Section 6 provides an "Action List": ML researchers develop extraction algorithms and evaluation frameworks; system builders implement five-attribute infrastructure and UIs; organizations handle governance and incentives; the research community shares benchmarks and open datasets.

## Key Experimental Results
This is an ICML position paper without empirical experiments; the following tables organize the core evidentiary arguments.

### Quantification of Failure Modes (Cited Literature)

| Failure Mode | Data | Source |
|---|---|---|
| Legal AI hallucinations post-RAG | 17-34% of queries | Magesh 2025 |
| General LLM legal error rate | 58-88% | Dahl 2024 |
| GPT-4 medical review fake citations | 28.6% | Chelli 2024 |
| Sycophancy residual post-alignment | 78.5% | Sharma 2024 |
| Accuracy change via prompt formatting | Up to 76 percentage points | Sclar 2024 |
| Real hit rate in 99% confidence interval| 65% | Geng 2024 |

### Coverage of Implicit Knowledge by Existing Methods

| Method | Explicit Knowledge | Implicit Knowledge | KO Potential |
|---|---|---|---|
| RAG | ✅ Cited documents | ❌ Reasoning invisible | ✅ Externalize as KO |
| Self-Verification | △ Consistency check | ❌ Only looks at confidence | ✅ KO provides ref |
| Training (SFT/RLHF) | △ Embedded in params | ❌ Black box, no traceability | ✅ KO is explicit artifact |
| Agent Memory | ✅ Fact storage | △ No verification state | ✅ KO includes metadata |

### Key Findings
- **Implicit knowledge accounts for 80-95% of organizational knowledge** (Dalkir 2017) and is the core source of LLM capability, yet it is the least verifiable part—stronger models are ironically better at learning implicit negative patterns (Lin 2022, McKenzie 2023).
- **Refutation of five counter-arguments** (e.g., "KG already solved this," "AI can self-verify," "human bottleneck"): This is the most informative section, clarifying the distinction between KOs and alternatives.
- **Core Reframe**: The reliability issue is not an AI algorithm problem, but an infrastructure problem. Without a vehicle for "cumulative human verification," algorithmic improvements remain local optimizations.

## Highlights & Insights
- **"Verification Economics" perspective**: Reconstructs AI reliability from a "training-time algorithm problem" to an "inference-time infrastructure problem," treating LLMs as "organizational members" within a KM system.
- **Implicit vs. Explicit knowledge reframe**: Uses Polanyi’s 1966 theory to explain why RAG treats the symptoms not the cause, then reverses the economic logic of "why it wasn't recorded" to show "how AI can help record it."
- **Bridge to Agentic AI**: Linking Agent "skills" (like Voyager) to "procedural KOs" suggests this framework is highly compatible with the future of agentic AI where validated skills become organizational building blocks.
- **High-level pre-rebuttal**: The five points in Section 5 show a strong grasp of domain consensus, making it more persuasive than a pure manifesto.

## Limitations & Future Work
- The paper lacks specific technical specifications (schemas), extraction algorithms, or UI designs, leaving the implementation gap for others to fill.
- Does not detail how "scaled human verification" handles conflicting expert views, knowledge decay, or malicious "verification poisoning."
- While claiming KOs are the "hub," it lacks quantitative economic modeling of the losses incurred by *not* implementing KOs.
- Boundary definitions between KOs and recent alignment routes like Process Reward Models (PRMs) could be clearer.
- Potential for "verification inflation" (low-quality bulk verification) requires reputation mechanisms (like PageRank for validators) which are not discussed.

## Related Work & Insights
- **vs RAG (Lewis 2020)**: RAG links to "external explicit documents," but is powerless against implicit reasoning in generation. KOs freeze reasoning as a verifiable artifact.
- **vs Constitutional AI / RLHF**: These bake preferences into parameters; KOs keep them external, traceable, and editable.
- **vs Agent Memory (MemGPT/Reflexion/A-MEM)**: Memory optimizes AI retrieval; KOs optimize human verification. Adding validation to Memory is a "patch," not a "first-class citizen" design.
- **vs Wikipedia / Stack Overflow**: These are explicit platforms; KO is the conceptual blueprint for an "AI implicit knowledge community platform."
- **vs Process Reward Model (PRM)**: PRMs use models to evaluate steps; the two could be combined where PRM provides initial confidence and KO provides human ground truth.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces organizational KM theory into AI reliability; clean "Knowledge Object" coinage.
- Experimental Thoroughness: ⭐⭐ Position paper with no experiments; all evidence is literature-based.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear logical chain (Status $\rightarrow$ Pain $\rightarrow$ Existing failures $\rightarrow$ Why $\rightarrow$ KO $\rightarrow$ Rebuttal $\rightarrow$ Action).
- Value: ⭐⭐⭐⭐ Provides new terminology and framework with high potential to catalyze new infrastructure and benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[AAAI 2026\] Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration](../../AAAI2026/others/align_when_they_want_complement_when_they_need_human-centere.md)
- [\[ICML 2026\] Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems](mapping_human_anti-collusion_mechanisms_to_multi-agent_ai_systems.md)
- [\[NeurIPS 2025\] Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications](../../NeurIPS2025/others/military_ai_needs_technically-informed_regulation_to_safeguard_ai_research_and_i.md)
- [\[AAAI 2026\] Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis](../../AAAI2026/others/intrinsic_barriers_and_practical_pathways_for_human-ai_alignment_an_agreement-ba.md)

</div>

<!-- RELATED:END -->
