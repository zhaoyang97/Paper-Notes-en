---
title: >-
  [Paper Note] Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective
description: >-
  [ICML 2026 (Position Paper)][Implicit Knowledge] This ICML position paper argues that all current AI reliability methods (RAG / Self-Consistency / RLHF / Agent Memory) can only verify explicit knowledge…
tags:
  - "ICML 2026 (Position Paper)"
  - "Implicit Knowledge"
  - "Knowledge Objects"
  - "Human-in-the-Loop"
  - "Verification Economics"
  - "RLHF Alternative"
date: 2026-05-08
content_hash: 474864b6cbd66e30
---

# Position: Reliable AI Needs to Externalize Implicit Knowledge: A Human-AI Collaboration Perspective

**Conference**: ICML 2026 (Position Paper)  
**arXiv**: [2605.02010](https://arxiv.org/abs/2605.02010)  
**Code**: None (position paper, no open-source implementation)  
**Area**: AI Reliability / Human-AI Collaboration / Knowledge Management  
**Keywords**: Implicit Knowledge, Knowledge Objects, Human-in-the-Loop, Verification Economics, RLHF Alternative

## TL;DR
This ICML position paper argues that all current AI reliability methods (RAG / Self-Consistency / RLHF / Agent Memory) can only verify explicit knowledge, while the true power of AI comes from the 80-95% of "implicit knowledge" in training data that has never been formally recorded by humans. The author proposes Knowledge Objects (KOs) as infrastructure—externalizing AI's implicit reasoning into structured artifacts that humans can inspect, verify, and endorse, enabling the cost of a single human verification to compound across the community over time.

## Background & Motivation

**Background**: LLMs have made rapid progress on knowledge-intensive tasks—75% of ChatGPT conversations are knowledge work (Chatterji 2025), Copilot generates millions of code suggestions daily, and RCTs show AI collaboration can boost productivity by 20-40%. However, these same systems still make large-scale errors—professional legal AI hallucinates on 17-34% of queries (Magesh 2025), GPT-4 fabricates 28.6% of citations in medical reviews, and general LLMs have error rates of 58-88% on verifiable legal questions.

**Limitations of Prior Work**: The author identifies a fatal flaw shared by four mainstream reliability approaches: **(1) RAG** can only verify "what documents say," not "how AI reasons"; **(2) Internal verification (Self-Consistency, Uncertainty, LLM-as-Judge)** uses AI to judge AI, so systematic errors are consistently reproduced (99% confidence interval actual hit rate is only 65%); **(3) Training-based methods (SFT/RLHF/DPO)** embed knowledge in parameter black boxes, making it untraceable and unincrementally correctable, with sycophancy persisting at 78.5% post-alignment; **(4) Agent Memory (MemGPT/Reflexion/MemoryBank)** stores data without verification status, allowing errors to accumulate and pollute memory.

**Key Challenge**: AI learns knowledge at two levels—**explicit knowledge** (papers, documents, databases, 5-20%) can be cited and traced; **implicit knowledge** (reasoning patterns, debugging routines, domain intuition, 80-95%) is embedded in conversation logs, commit histories, and experiment logs, and has never been formally extracted due to "recording cost > perceived benefit." LLMs learn both indiscriminately, acquiring both expert judgment and systematic bias—yet only explicit knowledge can be verified.

**Goal**: To establish infrastructure that enables AI to "externalize" its learned implicit knowledge into artifacts that humans can inspect, correct, and cumulatively verify, transforming the hidden cost of "re-evaluating every AI output" into a compounding model of "verify once, reuse forever."

**Key Insight**: Drawing on Nonaka's organizational knowledge theory (1994) and Polanyi's tacit knowledge theory (1966), the author argues that implicit knowledge is not "unrecordable," but rather "the marginal cost of recording exceeds the perceived marginal utility at the time." If AI can automatically extract implicit patterns into structured candidates, humans only need to perform "lightweight verification," flipping verification economics from "verify every time" to "verify once, benefit continuously."

**Core Idea**: Treat Knowledge Objects (KOs) as the "hub" of human-AI collaboration—AI externalizes implicit knowledge into structured artifacts (claim + evidence + scope + validation metadata), humans verify, correct, and endorse, and the verification status is persisted and retrievable as a first-class citizen.

## Method

### Overall Architecture
This is not a methods paper, but rather proposes a **conceptual framework + five attributes + call to action**. The core architecture is the "KO-Hub" collaboration paradigm: Environment → Task → (AI System + Human) collaboration → generate Interaction Data → AI externalizes candidate KOs from interaction data → Human verifies/corrects/rejects → verified KOs enter the Collective Human Knowledge pool → subsequent tasks can retrieve these verified KOs. This closed loop transforms "human verification" from a one-off, ephemeral judgment into an accumulable, queryable asset.

### Key Designs

1. **Formal Definition and Five Attributes of Knowledge Objects**:

    - Function: Solidify implicit knowledge into objects that humans can "see, verify, and endorse," rather than inaccessible representations embedded in parameters.
    - Mechanism: Definition 4.1 stipulates that a KO must include four elements—(i) knowledge claim or procedure, (ii) supporting evidence or reasoning, (iii) explicit scope and limitations, (iv) validation metadata (who, when, under what conditions verified). On this basis, five essential attributes are proposed: **Understandable** (readable, assessable by domain experts, not just embeddings), **Verifiable** (recordable verification status, not one-off judgments), **Traceable** (provenance—who endorsed, source, how it was modified), **Controllable** (humans can modify, annotate, reject), **Reusable** (verification can be reused by subsequent users). The first three address the core issues of "invisible / unverifiable / untraceable" identified in the paper, while the last two allow verification costs to be amortized.
    - Design Motivation: In contrast to RAG, which only solves "explicit citation," and Agent Memory, which only solves "persistent storage without state," KO is the first to treat "human verification status" as a first-class property of objects.

2. **Verification Economics Inversion**:

    - Function: Transform the hidden total cost of "every user independently evaluating AI output" into a compounding model where "one expert verification benefits N subsequent users."
    - Mechanism: Polanyi argued that implicit knowledge is "inexpressible" because human extraction cost > immediate benefit; KO leverages AI to automatically externalize as structured candidates, shifting the "extraction" cost from humans to AI, so humans only need to perform "low-cost verification"—e.g., confirming/scoring/adding scope tags. Verification shifts from "ephemeral private judgment" to "persistent public asset." The author draws an analogy to Wikipedia's layered model: 99.9% of articles use community consensus, 0.1% undergo Featured Article scrutiny; KO systems are similarly layered—high-risk knowledge requires expert verification, ordinary patterns are simply marked "unvalidated."
    - Design Motivation: Addressing concerns that "verification will become a bottleneck," the author argues that not verifying is the largest hidden cost—currently, the total time spent by each user independently evaluating LLM outputs far exceeds the "verify once, reuse many times" model; the real scalability issue is the status quo.

3. **KO's Complementary Positioning with Existing Methods**:

    - Function: Clearly distinguish KO from knowledge graphs, wikis, and Agent Memory to avoid being mistaken for reinventing the wheel.
    - Mechanism: Table 1 systematically compares four existing methods' handling of "implicit knowledge"—RAG=Untouched (reasoning remains unverified inside the model), Self-Verification=Unexposed (only produces confidence, no external reference), Training=Absorbed (becomes parameter black box, invisible and untraceable), Agent Memory=Unstructured (persistent storage but no verification status). KO is the **only** design that externalizes implicit knowledge into externally checkable artifacts. The author also discusses KO's specific forms in agent scenarios—Voyager's executable skill library and Agent Workflow Memory's reusable workflows are early forms of "procedural KOs."
    - Design Motivation: The author emphasizes that KO is not a replacement for existing KM systems, but fills the missing layer of "AI-generated yet human-verifiable" knowledge; traditional wikis manage "what humans have written," KO manages "what AI has learned but humans have not yet verified."

### Loss & Training
As a position paper, there is no training objective, but Section 6 provides an "action list" for four stakeholder groups: ML researchers are responsible for developing KO candidate extraction algorithms and KO quality evaluation frameworks; system builders for implementing the five-attribute infrastructure + verification UI + provenance API; organizations for governance frameworks (who can verify which claims) + piloting in high-risk domains + verification incentives; the research community for sharing benchmarks + open datasets + interoperability standards.

## Key Experimental Results
As an ICML position paper, there are no empirical experiments; the following tables organize the core argumentative evidence from the paper.

### Quantification of Failure Modes (Cited by Author)

| Failure Mode | Quantitative Data | Source |
|---|---|---|
| Legal AI still hallucinates after RAG | 17-34% queries | Magesh 2025 |
| General LLM legal error rate | 58-88% | Dahl 2024 |
| GPT-4 fabricated citations in medical reviews | 28.6% | Chelli 2024 |
| Residual sycophancy after alignment | 78.5% | Sharma 2024 |
| Prompt format changes affect accuracy | Up to 76 percentage points | Sclar 2024 |
| 99% confidence interval actual hit rate | 65% | Geng 2024 |

### Coverage of Implicit Knowledge by Existing Methods

| Method | Explicit Knowledge | Implicit Knowledge | Can KO Supplement? |
|---|---|---|---|
| RAG | ✅ Cites documents | ❌ Reasoning process invisible | ✅ Externalizes reasoning as KO |
| Self-Verification | △ Consistency check | ❌ Only confidence | ✅ KO provides external reference |
| Training (SFT/RLHF/DPO) | △ Embedded in parameters | ❌ Black box, untraceable | ✅ KO is explicit artifact |
| Agent Memory | ✅ Stores facts | △ No verification status | ✅ KO includes validation metadata |

### Key Findings
- **Implicit knowledge constitutes 80-95% of organizational knowledge** (Dalkir 2017) and is the core source of LLM capability, but is also the least verifiable—stronger models are even better at learning implicit bad patterns (Lin 2022, McKenzie 2023).
- **The author's rebuttal of five opposing viewpoints** (KG already solves this / existing systems will naturally add validation / humans will be the bottleneck / AI can self-verify / structuring reduces usability) is the most informative part of the paper, demonstrating KO's true difference from each alternative.
- **Core reframe**: The reliability problem is not an AI algorithm problem, but an infrastructure problem; without a carrier for "cumulative human verification," any algorithmic improvement is merely local optimization.

## Highlights & Insights
- **"Verification economics" is a novel perspective**: Reframes AI reliability from a "training-time algorithm problem" to an "inference-time infrastructure problem," borrowing from Nonaka's organizational knowledge management theory and treating LLMs as "organizational members" in KM system design—a rare interdisciplinary bridge.
- **The implicit vs explicit knowledge reframe is incisive**: The author uses Polanyi's 1966 tacit knowledge theory to explain why RAG is only a stopgap, then reverses the "why not recorded" economic logic to "how AI can help record"—a graceful shift in stance.
- **Agent skill = procedural KO bridge**: The author incorporates Voyager's code skill library and Agent Workflow Memory's workflow induction as procedural forms of KO, indicating this framework is especially suited for the agentic AI era, where validated skills can become "organizational building blocks."
- **High-level preemptive rebuttal of opposing views**: Section 5's five rebuttals (especially "AI self-verification is sufficient" and "structuring reduces adaptability") show the author's command of field consensus; this structure is far more persuasive in a position paper than a pure manifesto.

## Limitations & Future Work
- The position paper does not provide concrete KO specifications (schema), automatic extraction algorithms, or UI design; the gap from "proposal" to "implementable system" remains large—the author leaves these to ML researchers and system builders.
- How to handle "expert disagreement," "knowledge decay," and "malicious verification poisoning" in large-scale human verification is not discussed; Wikipedia has mature governance, but migrating to enterprise KO repositories will be more complex.
- The author claims KO is the "hub of human-AI collaboration," but does not quantify "how much is lost without KO"; readers must rely on intuition and case studies, lacking economic model support.
- The boundaries with recent alignment approaches like LLM-as-Judge / Constitutional AI / Process Reward Model (which also "make reasoning explicit") could be more clearly delineated.
- Even with KO infrastructure, preventing "verification inflation" (proliferation of low-quality verification) will require a PageRank-like verifier reputation mechanism—this is barely discussed.

## Related Work & Insights
- **vs RAG (Lewis 2020)**: RAG can only attach "external explicit documents," powerless over implicit reasoning in model generation; KO solidifies reasoning itself as a verifiable artifact—an orthogonal supplement to RAG.
- **vs Constitutional AI / RLHF**: These approaches embed human preferences in parameters, while KO keeps human preferences/verification external, traceable, and revisable; the former is efficient but a black box, the latter is transparent but requires new infrastructure.
- **vs Agent Memory (MemGPT/Reflexion/A-MEM)**: Memory systems optimize AI retrieval performance, KO optimizes human verification performance—the author clarifies the differing design philosophies, noting that adding validation to Memory is a "patch," not a "first-class citizen."
- **vs Wikipedia / Stack Overflow**: These are explicit knowledge community platforms; KO is a conceptual blueprint for an "AI implicit knowledge community platform." Borrowing Wikipedia's layered verification (Featured Article) and Stack Overflow's voting endorsement mechanisms are obvious engineering directions.
- **vs Process Reward Model (PRM, OpenAI's let's verify)**: PRM uses the model to evaluate steps as another way to address unverifiable reasoning; the two can be combined—PRM provides initial confidence, KO provides human final ground truth.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces organizational knowledge management theory to AI reliability; the Knowledge Object concept is clean, though KG/wiki have early conceptual precedents
- Experimental Thoroughness: ⭐⭐ Position paper, no experiments; all quantitative evidence is from literature review
- Writing Quality: ⭐⭐⭐⭐⭐ Argument logic chain (status quo→pain points→existing methods fail→why→KO→rebuttal→action) is very clear; the five-part rebuttal is well-argued
- Value: ⭐⭐⭐⭐ Provides the community with a new term (KO) and organizational language, with potential to inspire new infrastructure and evaluation; but practical implementation will require substantial follow-up work

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Align When They Want, Complement When They Need! Human-Centered Ensembles for Adaptive Human-AI Collaboration](../../AAAI2026/others/align_when_they_want_complement_when_they_need_human-centere.md)
- [\[NeurIPS 2025\] Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications](../../NeurIPS2025/others/military_ai_needs_technically-informed_regulation_to_safeguard_ai_research_and_i.md)
- [\[NeurIPS 2025\] A Sustainable AI Economy Needs Data Deals That Work for Generators](../../NeurIPS2025/others/a_sustainable_ai_economy_needs_data_deals_that_work_for_gene.md)
- [\[AAAI 2026\] Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis](../../AAAI2026/others/intrinsic_barriers_and_practical_pathways_for_human-ai_alignment_an_agreement-ba.md)
- [\[ICML 2026\] AI Cap-and-Trade: Efficiency Incentives for Accessibility and Sustainability](ai_cap-and-trade_efficiency_incentives_for_accessibility_and_sustainability.md)

</div>

<!-- RELATED:END -->
