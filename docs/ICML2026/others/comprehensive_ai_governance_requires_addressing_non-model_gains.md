---
title: >-
  [Paper Note] Comprehensive AI Governance Requires Addressing Non-Model Gains
description: >-
  [ICML2026][AI Governance] This position paper argues that the current model-centric AI governance paradigm is becoming less effective as "non-model gains" (inference, systems…
tags:
  - "ICML2026"
  - "AI Governance"
  - "Non-Model Gains"
  - "Frontier AI Safety"
  - "Inference Scaling"
  - "Multi-layer Governance"
date: 2026-05-08
content_hash: 9d3df6f1ca1c1d02
---

# Comprehensive AI Governance Requires Addressing Non-Model Gains

**Conference**: ICML2026  
**arXiv**: [2606.00047](https://arxiv.org/abs/2606.00047)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: AI Governance, Non-Model Gains, Frontier AI Safety, Inference Scaling, Multi-layer Governance  

## TL;DR
This position paper argues that the current model-centric AI governance paradigm is becoming less effective as "non-model gains" (inference, systems, and asset gains) increase in importance. It advocates for a multi-layered complementary governance portfolio involving systems, entities, agents, and cloud layers to fill regulatory gaps.

## Background & Motivation

**Background**: Current frontier AI governance relies primarily on "model-level governance," which manages risks by evaluating and mitigating the dangerous capabilities of models. This paradigm assumes that model capabilities are primarily determined by compute and data during the training phase, thereby positioning frontier model developers as the core nodes for safety efforts. International regulations (such as the EU AI Act) and corporate self-regulation frameworks (such as Anthropic's RSP and OpenAI's Preparedness Framework) focus on pre-deployment evaluation and mitigation.

**Limitations of Prior Work**: With the rise of inference models (e.g., OpenAI o1 series), complex scaffolding systems (e.g., Google DeepMind’s Big Sleep zero-day discovery), and collaborations between national security agencies and AI companies, an increasing share of capability improvements comes from post-deployment "non-model gains" rather than pre-training scaling. These gains are low-cost, diffuse rapidly, and are difficult for model developers to fully anticipate before deployment.

**Key Challenge**: The effectiveness of model-level governance depends on the developer's ability to perform **exhaustive elicitation** of a model's downstream capabilities before deployment. However, non-model gains cause the capability space to expand continuously post-deployment, leading to three governance failures: elicitation failure (inability to foresee all enhancement methods), mitigation failure (difficulty in controlling known dangerous capabilities), and lag costs (increased expected harm from un-elicited capabilities).

**Goal**: (1) Formalize the concept of "non-model gains" and establish a taxonomy; (2) Analyze how each type of gain undermines model-level governance; (3) Propose complementary governance solutions that go beyond the model level.

**Key Insight**: Starting from governance practice, the authors observe that the contribution of pre-training scaling to frontier model performance is declining (while the importance of inference scaling and post-training rises), and low-resource actors can approach frontier capabilities through systems gains. This implies that the "leverage points" of governance are diffusing from model developers toward downstream actors.

**Core Idea**: Non-model gains are systematically undermining the foundational assumptions of model-level governance. A multi-layer governance portfolio encompassing systems, entities, agents, and the cloud must be established to address these shifts.

## Method

### Overall Architecture
The paper constructs an analytical framework of "Non-Model Gains $\rightarrow$ Governance Failure Mechanisms $\rightarrow$ Complementary Governance Solutions." The input is the current AI governance ecosystem and capability trends, and the output is a set of recommendations for multi-layered governance. The core analysis is divided into three stages: defining a taxonomy of non-model gains, analyzing the impact of these gains on model-level governance, and proposing corresponding governance supplements.

### Key Designs

1.  **Taxonomy of Non-Model Gains**:
    - **Function**: Categorizes post-deployment capability enhancement systems into three current and three forward-looking gains.
    - **Mechanism**: Current gains include: **Inference gain**, achieving performance through scaling inference compute (e.g., chain-of-thought in reasoning models), allowing small models to mimic large models (e.g., Qwen3-4B reaching o3-mini levels via recursive self-aggregation); **Systems gain**, enhancing capabilities through post-training enhancements like scaffolding, tool-use, and multi-agent orchestration, which can diffuse freely once the "recipe" is discovered; **Asset gain**, capability boosts from accessing restricted assets (e.g., government classified data, specialized hardware), where specific datasets can provide performance gains equivalent to $1000\times$ pre-training compute. Forward-looking gains include embodiment, continual learning, and diffusion effects.
    - **Design Motivation**: This taxonomy operationalizes the vague concept of "non-model capability enhancement," facilitating a systematic analysis of governance impacts and the design of targeted countermeasures.

2.  **Governance Failure Modes**:
    - **Function**: Reveals how each type of non-model gain specifically weakens the three pillars of model-level governance.
    - **Mechanism**: Inference gains close the gap between frontier and sub-frontier models, allowing malicious actors to bypass regulation using open-weight models. Systems gains, characterized by low cost and rapid diffusion, prevent developers from anticipating all downstream modifications. Asset gains, due to the confidentiality of restricted assets, prevent evaluators from testing relevant scenarios. Together, these gains transform pre-deployment evaluation from "difficult but feasible" to "structurally insufficient."
    - **Design Motivation**: By clarifying failure mechanisms, the paper demonstrates that simply improving model-level governance is inadequate and requires a paradigm shift.

3.  **Multi-Layer Governance Portfolio**:
    - **Function**: Proposes a four-layer complementary governance framework to cover risk nodes unreachable by model-level governance.
    - **Mechanism**: **System governance** requires providers of systems that significantly enhance base model capabilities to assume risk management responsibilities; **Entity governance** focuses on organizational structures, incentives, and decision processes rather than individual models; **Agent governance** manages the delegation parameters and autonomous interactions of AI agents, including access boundaries, behavioral constraints, and unique agent IDs; **Cloud governance** implements safety oversight at the inference level through KYC, content monitoring, and compute pattern monitoring. Societal resilience serves as a backstop for all governance layers.
    - **Design Motivation**: Different types of non-model gains require different governance nodes; a single layer cannot cover all risk vectors.

## Key Experimental Results

### Mapping Non-Model Gains to Governance Solutions

| Non-Model Gain Type | Governance Failure Mechanism | Recommended Governance Layer |
| :--- | :--- | :--- |
| Inference Gain | Closes frontier/sub-frontier gap; open models bypass regulation | Model-level + Entity + Cloud |
| Systems Gain | Low cost, rapid diffusion; developers cannot foresee all scaffolds | Model-level + Entity/Agent/System |
| Asset Gain | Classified assets cannot be evaluated; few high-capability actors | Post-deployment monitoring + NatSec cooperation |
| Embodiment Gain | Information risk translates to physical security risk | Supply chain alignment + System |
| Continual Learning | Safety training may be forgotten; model behavior drifts | Post-deployment monitoring |
| Diffusion Effects | Monoculture risks, cascading failures | Entity/Agent + Societal Resilience |

### Empirical Evidence for Inference Gains

| Case | Mechanism of Gain | Effect |
| :--- | :--- | :--- |
| Qwen3-4B + Recursive Self-aggregation | Inference-time compute scaling | 4B parameter model reaches o3-mini (high) level |
| DeepSeek-V3.2 vs Gemini 3 | 1.5-2.5x token consumption | Sub-frontier model exceeds frontier on multiple benchmarks |
| Specialized Dataset Fine-tuning | Asset gain | Equivalent to $1000\times$ pre-training compute |
| Big Sleep (Google DeepMind) | Systems gain (scaffold + tools) | First LLM agent to discover zero-day vulnerabilities |

### Key Findings
- Inference gain is currently the most quantifiable type of non-model gain, supported by emerging inference scaling laws that enable prediction.
- Systems gains are the hardest to defend against—low-resource actors have demonstrated the ability to build complex adversarial scaffolds (e.g., the Claude Code scaffold constructed by a state-sponsored group in 2025).
- While asset gains affect the fewest actors, they present the greatest potential harm and are nearly impossible to evaluate in advance due to their confidential nature.
- Cloud governance faces significant commercial, technical, and legal hurdles (privacy regulations, cross-provider coordination, maturity of confidential computing).

## Highlights & Insights
- **Formal Taxonomy of Non-Model Gains**: The paper clearly deconstructs the previously vague concept of "post-deployment capability enhancement" into inference, systems, and asset categories, establishing causal links to governance failure and providing an actionable framework for policymakers.
- **Core Insight on the "Shift of Governance Leverage Points"**: As the importance of non-model gains rises, safety responsibility should not rest solely on model developers but should be distributed along the value chain to system integrators, deployment platforms, and end-users. This perspective offers a significant correction to current regulatory approaches that focus excessively on model developers.
- **The Double-Edged Sword of Inference Gains as "Democratization Tools"**: Inference scaling allows small models to approach frontier performance. While this lowers the barrier to accessing AI capabilities, it also lowers the threshold for malicious use, which has direct implications for safety policies regarding open-weight models.

## Limitations & Future Work
- As a position paper, it lacks empirical validation; the feasibility and effectiveness of the proposed governance solutions have not been systematically evaluated.
- Insufficient analysis of interaction effects between non-model gains (e.g., the combinatorial amplification of inference gain $\times$ systems gain), whereas actual risk scenarios often involve overlapping gains.
- The cloud governance section acknowledges major challenges in privacy law, technical implementation, and commercial viability, making short-term implementation difficult.
- Potential conflicts between governance schemes (e.g., how entry barriers in entity governance might stifle competition and innovation) are not fully discussed.
- Future work could involve quantifying the magnitude of capability improvements from various non-model gains to build predictive models for guiding the allocation of governance resources.

## Related Work & Insights
- Complements model-level safety frameworks like Anthropic RSP and OpenAI Preparedness Framework by identifying their blind spots.
- Directly relates to METR’s research on capability elicitation, as this paper argues for the structural limitations of such work.
- Echoes empirical research by Epoch AI on the shifting relative importance of the three scaling paradigms (pre-training, post-training, and inference).
- **Insight**: When evaluating the safety of AI systems, "system-level capabilities" should be considered separately from "model-level capabilities," as the former may significantly exceed the latter.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Model Readiness: Institutional Readiness for AI Deployment in Public Systems](beyond_model_readiness_institutional_readiness_for_ai_deployment_in_public_syste.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](../../NeurIPS2025/others/fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)
- [\[AAAI 2026\] Bridging the Skills Gap: A Course Model for Modern Generative AI Education](../../AAAI2026/others/bridging_the_skills_gap_a_course_model_for_modern_generative_ai_education.md)
- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[ICLR 2026\] The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?](../../ICLR2026/others/the_hot_mess_of_ai_how_does_misalignment_scale_with_model_intelligence_and_task_.md)

</div>

<!-- RELATED:END -->
