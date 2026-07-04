---
title: >-
  [Paper Note] Position: 'AI Alignment' Encompasses Competing Technical Priorities
description: >-
  [ICML 2026][AI Safety][AI Alignment] This ICML position paper argues that "AI alignment" is a **polysemous** term: the ML literature contains at least three high-level alignment ideals that are competing rather than merely different (Task Reliability / Social Judiciousness / Takeover Avoidance). In practice, advancing one type of alignment often **actively undermines** another. The authors explain these tensions via two cross-cutting distinctions—"threat model differences" an…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "AI Alignment"
  - "Position Paper"
  - "Threat Models"
  - "Conceptual Analysis"
  - "Alignment Tension"
date: 2026-05-08
content_hash: d511719da569d5a4
---

# Position: 'AI Alignment' Encompasses Competing Technical Priorities

**Conference**: ICML 2026  
**arXiv**: [2606.14315](https://arxiv.org/abs/2606.14315)  
**Code**: None (Position Paper)  
**Area**: AI Safety / AI Alignment / Position Paper  
**Keywords**: AI Alignment, Position Paper, Threat Models, Conceptual Analysis, Alignment Tension

## TL;DR
This ICML position paper argues that "AI alignment" is a **polysemous** term: the ML literature contains at least three high-level alignment ideals that are competing rather than merely different (Task Reliability / Social Judiciousness / Takeover Avoidance). In practice, advancing one type of alignment often **actively undermines** another. The authors explain these tensions via two cross-cutting distinctions—"threat model differences" and "positive/negative alignment differences"—and offer five specific recommendations for researchers.

## Background & Motivation
**Background**: "Alignment" is naturally a **binary relation**—to say $x$ is aligned with $y$ is to say $x$ conforms to $y$ in some sense. Thus, discussing "AI alignment" requires answering two questions: **Q1** What is the target property $y$? **Q2** What is the object $x$ that must satisfy $y$? Definitions of "AI alignment" in the literature are diverse: ranging from "making AI follow human values" and "conforming to the designer's intended goals/interests/values" to "developers tuning models according to the social norms of user communities," as well as various sub-concepts like Thick, Collective, Socioaffective, and Decolonial alignment.

**Limitations of Prior Work**: Numerous papers provide only a cursory definition of "AI alignment," conflating these **different answers to Q1 and Q2**. The authors argue that this **polysemy** obscures the fact that many seemingly "technical" alignment disagreements are actually **normative disagreements**. When different researchers speak of "making AI more aligned," they may be pursuing goals that are **fundamentally impossible to achieve simultaneously**.

**Key Challenge**: The issue is not that "everyone has the same goal but uses different methods," but that "everyone disagrees on **what to align and what to align to**." The paper emphasizes that high-level alignment ideals in practice often **disagree on $x$ (the object to be aligned)**, not just on $y$ (the target property)—some target "locally measurable AI behavior," some target "socio-technical systems deployed in real-world contexts," and others target "the optimization goals of future AGI/ASI."

**Goal**: To decouple "AI alignment" into three high-level ideals and characterize the practical tensions between them, demonstrating that "AI alignment" involves **competing rather than merely different technical priorities**, and to provide five recommendations for clearer research and communication.

## Method

### Overall Architecture
The paper is not an experimental work but a **conceptual analysis argument chain**: it first (Section 2) uses the Q1/Q2 scale to segment "AI alignment" into **three high-level ideals**, showing they provide different answers to "what to align and what to align to" (see Table 1). Then (Section 3), it introduces **two cross-cutting distinctions**—threat models (harms from misdirected competence vs. harms from incompetence) and positive/negative alignment—to demonstrate how these three ideals **conflict with each other** in real-world interventions (see Table 2). Finally (Section 4), the analysis concludes with **five recommendations for the research community**. The core thesis is: because different alignment concepts are driven by different threat models or emphasize different "benefits" and "harms" of AI, "making AI more aligned" contains **competing priorities** that cannot be pursued as a single technical objective.

Importantly, the paper distinguishes between "competing" and "different": "different" implies multiple goals can be pursued in parallel without interference; "competing" implies that at the level of practical intervention, advancing one goal comes at the expense of another. Sections 2 and 3 constitute the main argument: Section 2 proves "AI alignment" is **polysemous**, while Section 3 proves this polysemy **obscures normative disagreements**, making seemingly technical discussions implicit battles over value positions. The authors specifically warn: these three ideals disagree on $x$ (the object) itself, not just $y$ (the target)—this is the root of their irreconcilability.

### Key Designs

**1. Three Alignment Ideals: Segmenting Polysemous "AI Alignment" via Q1/Q2**

The authors group scattered alignment usages into three mutually exclusive high-level ideals. **Task Reliability (Def 2.1)**: AI is aligned if it does what we ask; $x$ is "locally measurable AI behavior" and $y$ is "developer intent." InstructGPT-style "Alignment as Fine-Tuning" (making model behavior conform to user/developer expectations) is a sub-category. **Social Judiciousness (Def 2.2)**: AI is misaligned if its output in a deployment context "creates, perpetuates, or exacerbates undesirable social trends"—it views AI as a **socio-technical system** rather than a pure technical artifact; $y$ is "some external normative standard." The authors further distinguish two failure sources: **Training Data Conservatism (Def 2.3)** (harmful behavior from biased/unrepresentative data) and **Malicious Use (Def 2.4)** (powerful/malicious actors using AI for their ends). **Takeover Avoidance (Def 2.5)**: AI is misaligned if it "optimizes for undesirable outcomes" in the real world, stemming from concerns that future AGI/ASI "optimization goals are unfriendly and hidden" (i.e., deceptive alignment); $x$ is the "optimization goal of AGI/ASI." Their answers to Q1/Q2 are summarized below:

| Alignment Ideal | What is Aligned ($x$) | Aligned to What ($y$) |
|------|------|------|
| Takeover Avoidance | Optimization goals of AGI/ASI | Non-takeover goals |
| Social Judiciousness | AI deployed in real-world contexts | External normative standards |
| Task Reliability | Locally measurable AI behavior | Developer intent |

**2. Threat Model Cross-Cut: Harms from Misdirected Competence vs. Incompetence**

The first cross-cutting distinction categorizes threat models by the "source of negative outcomes." **Harms from Misdirected Competence (Def 3.2)**: Danger comes from AI being **too capable** at certain tasks—Takeover Avoidance falls here (future strong AI is dangerous because it is too capable). **Harms from Incompetence (Def 3.3)**: Danger comes from AI **not being capable enough**—e.g., social harms caused by models learning superficial/biased correlations in predictive policing, healthcare, or facial recognition. Social Judiciousness spans both. This distinction creates conflict: researchers focusing on Social Judiciousness (viewed as incompetence harms) support studies like "reducing LLM hallucinations" to **increase capability**; however, those focused on Takeover Avoidance **oppose** this—lower hallucinations and stronger situational awareness might make misaligned AI better at "scheming" or "sandbagging" during evaluations. The same intervention is progress for one ideal and a setback for the other.

**3. Positive vs. Negative Alignment: Evaluation Scope Differences**

The second distinction concerns the **scope when evaluating AI behavior**. **Positive Alignment** specifies properties we **want** the AI to have; **Negative Alignment** specifies properties we **don't want** the AI to have. The authors acknowledge these are formally equivalent in first-order logic (if $A$ is the behavior domain, $Wa$ is "want $a$," and $Da$ is "does $a$," then positive $\forall a:(Da\to Wa)$ and negative $\forall a:(\neg Wa\to\neg Da)$ are equivalent); however, **evaluation domains differ in practice**—checking "did it do the one thing I wanted" is usually easier than "did it avoid everything it shouldn't do." This characterizes the tension between Task Reliability and other ideals: training a model to score higher on math benchmarks (positive alignment progress) may push hallucination rates higher (negative alignment regression; e.g., o3/o4-mini vs. GPT-4.5 on PersonQA); a recommendation system maximizing CTR (Task Reliability success) may trigger addiction/polarization (Social Judiciousness failure); training an LLM to produce "non-offensive" Chain-of-Thought may teach the model to **hide its intent** (Takeover Avoidance deterioration).

| Alignment Ideal | Threat Model | Positive/Negative Alignment |
|------|------|------|
| Takeover Avoidance | Competence | Negative |
| Social Judiciousness | Either | Negative |
| Task Reliability | N/A | Positive |

**4. Five Recommendations: Operational Prescriptions for the Research Community**

The argument concludes with five recommendations: ① Do not conflate **scientific/technical scope distinctions** with **policy distinctions**; ② **Methodological disagreements** (underlying threat model/normative conflicts) should be explicitly acknowledged rather than hidden under neutral technical language; ③ Distinguish "AI alignment" as a **high-level ideal** from the **alignment proxies** used in empirical research, introducing proxy concepts as needed (or even separate review tracks/conferences by ideal); ④ Use **finer-grained concepts** to identify the **source and nature** of AI harms/benefits; ⑤ In empirical work and communication with non-technical audiences (especially policymakers), **explicitly acknowledge the diversity** of the "alignment" concept.

The shared spirit is "define before debating": many alignment debates stall because parties use the same word for different concepts. The authors particularly value Recommendation ③—empirical papers often equate "proxy improvement" with "better alignment," but a proxy is only a **local representative** of a high-level ideal; they are not equivalent.

## Key Experimental Results
This is a position/conceptual analysis paper with **no experiments, no datasets, and no quantitative metrics**. Its "evidence" consists of literature reviews and real-world tension cases. The core tension cases are summarized below:

| Intervention/Case | "Progress" for | "Regression" for |
|------|------|------|
| Reducing LLM hallucinations | Social Judiciousness (less misinformation) | Takeover Avoidance (better "scheming/sandbagging") |
| Math benchmark scaling (o3 vs. GPT-4.5) | Positive Alignment / Task Reliability | Negative Alignment (higher PersonQA hallucinations) |
| Recommender CTR maximization | Task Reliability (meets developer goal) | Social Judiciousness (addiction/polarization) |
| Training "non-offensive" CoT | Task Reliability (desirable local output) | Takeover Avoidance (model learns to hide intent) |

These cases share a common structure: **the same technical intervention is simultaneously an improvement for one type of alignment and a regression for another**.

### Key Findings
- **"AI Alignment" is polysemous, and polysemy hides normative conflict**: Many technical debates are actually disputes over values or threat models.
- **The three ideals can conflict pairwise**: Specifically, "capability improvement" is a good thing for Social Judiciousness but a disaster for Takeover Avoidance, forming the sharpest tension.
- **Scope differences in Positive/Negative alignment** cause "achieving the desired" and "avoiding the undesired" to diverge in practice.

## Highlights & Insights
- **A useful analytical tool (Q1/Q2 + two cross-cuts)**: Organizing chaotic alignment discourse into a 2×3 structure allows one to immediately locate a paper's alignment "coordinates."
- **The "capability as a double-edged sword" insight is impactful**: It challenges the default assumption that all alignment research is moving in the same direction, reminding the community that "hallucination reduction" is not a stance-neutral goal.
- **Actionable recommendations**: Recommendation ③ regarding "high-level ideal vs. alignment proxy" is particularly useful for empirical writing to avoid misinterpreting proxy gains as global alignment progress.
- **Redefining the "unit of collaboration" in alignment research**: It suggests the community should treat different ideals as sometimes **competing** research programs rather than a unified whole.

## Limitations & Future Work
- **Non-exhaustive**: Only three ideals are focused on; concepts like Collective or Bidirectional alignment are difficult to fit into the current taxonomy.
- **Lack of empirical testing**: Tensions are supported by logic and literature but lack quantitative evidence on how prevalent these conflicts are in practice.
- **Overlapping boundaries**: Social Judiciousness spans two threat models and opposes Task Reliability on positive/negative alignment; real-world papers often fall into multiple categories.
- **Vague implementation path for "should" suggestions**: How to implement "separate review pools" or "explicitly acknowledge conflict" within the current peer-review system is not detailed.
- **Future Directions**: Turning the 2×3 framework into an annotation system for alignment papers, performing empirical coding statistics on the frequency of "concept conflation," or formalizing "Positive/Negative evaluation domain differences" to guide benchmark design.

## Related Work & Insights
- **vs. Standard Definitions (Russell / Yudkowsky / InstructGPT)**: These works provide universal definitions; this paper contributes by showing they are **competing answers** to Q1/Q2.
- **vs. Thick / Collective / Socioaffective / Decolonial Alignment**: These are categorized under "Social Judiciousness," though Collective alignment remains an outlier.
- **vs. Deceptive Alignment / Takeover Risk Literature (Carlsmith, Hubinger, etc.)**: This paper does not add new takeover arguments but treats "Takeover Avoidance" as one of three ideals to explain structural conflicts with the "capability for safety" route.
- **vs. Algorithmic Fairness / Socio-technical AI Ethics (Bender, etc.)**: These focus on "Social Judiciousness." This paper's value-add is positioning them alongside Task Reliability and Takeover Avoidance to reveal where "reducing social harm" paths conflict with other alignment goals.

## Rating
- Novelty: ⭐⭐⭐⭐ Structures polysemy into Q1/Q2 + cross-cuts; rare and clear perspective.
- Experimental Thoroughness: ⭐⭐⭐ Position paper with strong logic-based arguments but no empirical metrics.
- Writing Quality: ⭐⭐⭐⭐ Rigorous progression from concept to definition to proposition.
- Value: ⭐⭐⭐⭐ Provides a common language for the alignment community; highly relevant for research communication and policy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Embodied AI Requires a Privacy-Utility Trade-off](position_embodied_ai_requires_a_privacy-utility_trade-off.md)
- [\[ICML 2026\] Position: AI Researchers Must Help Lead Arms Control to Mitigate Military AI Risks](ai_researchers_must_help_lead_arms_control_to_mitigate_military_ai_risks.md)
- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)
- [\[ICML 2026\] Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation](position_retire_the_positive_backdoor_label_--_secret_alignment_requires_strict_.md)
- [\[ICML 2025\] Distributed and Decentralised Training: Technical Governance Challenges in a Shifting AI Landscape](../../ICML2025/ai_safety/distributed_and_decentralised_training_technical_governance_challenges_in_a_shif.md)

</div>

<!-- RELATED:END -->
