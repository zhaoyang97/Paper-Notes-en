---
title: >-
  [Paper Note] Comprehensive AI Governance Requires Addressing Non-Model Gains
description: >-
  [ICML2026][AI Governance] This position paper argues that the current model-centric AI governance paradigm is experiencing diminishing effectiveness as "non-model gains" (inference gains, systems gains, and asset gains) become increasingly significant. It calls for a multi-layered complementary governance framework—including system, entity, agent, and cloud governance—to fill existing regulatory gaps.
tags:
  - "ICML2026"
  - "AI Governance"
  - "non-model gains"
  - "frontier AI safety"
  - "inference scaling"
  - "multi-layered governance"
date: 2026-05-08
content_hash: 09a493e7baff1c72
---

# Comprehensive AI Governance Requires Addressing Non-Model Gains

**Conference**: ICML2026  
**arXiv**: [2606.00047](https://arxiv.org/abs/2606.00047)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: AI Governance, non-model gains, frontier AI safety, inference scaling, multi-layered governance  

## TL;DR
This position paper argues that the current model-centric AI governance paradigm is experiencing diminishing effectiveness as "non-model gains" (inference gains, systems gains, and asset gains) become increasingly significant. It calls for a multi-layered complementary governance framework—including system, entity, agent, and cloud governance—to fill existing regulatory gaps.

## Background & Motivation

**Background**: Current frontier AI governance relies primarily on "model-level governance," which manages risks by evaluating and mitigating dangerous capabilities of models. This paradigm assumes that model capabilities are primarily determined by compute and data during the training phase, making frontier model developers the central nodes for safety efforts. Regulatory frameworks (e.g., EU AI Act) and corporate self-regulation (e.g., Anthropic RSP, OpenAI Preparedness Framework) focus on pre-deployment assessment and mitigation.

**Limitations of Prior Work**: With the rise of reasoning models (e.g., OpenAI o1), complex scaffolding systems (e.g., Google DeepMind's Big Sleep for zero-day discovery), and cooperation between national security agencies and AI firms, an increasing share of capability improvements comes from post-deployment "non-model gains" rather than pre-training scaling. These gains are low-cost, diffuse rapidly, and are difficult for model developers to fully foresee before deployment.

**Key Challenge**: The effectiveness of model-level governance depends on the developer's ability to **exhaustively elicit** a model's downstream capabilities before deployment. However, non-model gains cause the capability space to expand continuously post-deployment, leading to three governance failures: elicitation failure (inability to foresee all enhancement methods), mitigation failure (difficulty in controlling known dangerous capabilities), and overhang costs (increased potential harm from unelicited capabilities).

**Goal**: (1) Formalize the concept of "non-model gains" and establish a taxonomy; (2) Analyze how each gain category undermines model-level governance; (3) Propose complementary governance solutions that move beyond the model level.

**Key Insight**: The authors observe from governance practice that the proportional contribution of pre-training scaling in frontier models is declining while the importance of inference scaling and post-training is rising. Furthermore, low-resource actors can approach frontier capabilities through systems gains, meaning "governance leverage points" are diffusing from developers to downstream actors.

**Core Idea**: Non-model gains are systematically undermining the foundational assumptions of model-level governance. A multi-layered governance portfolio encompassing systems, entities, agents, and the cloud must be established to address these challenges.

## Method

### Overall Architecture
As a position paper, the "method" consists of a logical chain of arguments: first, naming and categorizing post-deployment capability increases as "non-model gains"; second, deconstructing how each category breaches specific assumptions of model-level governance; and finally, prescribing a multi-layered governance framework that no longer focuses solely on model developers. The three steps follow a causal progression—without clear classification, failure points cannot be located, and without locating failure points, effective governance layers cannot be prescribed.

### Key Designs

**1. Taxonomy of Non-Model Gains: Breaking down "post-deployment capability expansion" into analytical categories**

Model-level governance defaults to the idea that capabilities are determined by training compute and data, treating developers as the primary safeguards. However, what truly drives capability expansion post-deployment are "non-model gains," which are often ambiguous and hard to quantify. The authors categorize current gains into three types: **Inference gains** achieve performance through increased inference-time compute (e.g., Chain-of-Thought scaling in reasoning models, where a Qwen3-4B model can approach o3-mini levels via recursive self-aggregation); **Systems gains** are built through post-training scaffolding, tool calls, and multi-agent orchestration (once a "recipe" is discovered, it diffuses freely at zero cost); and **Asset gains** arise from access to restricted assets (classified government data, specialized hardware), where a single high-quality dataset can provide an improvement equivalent to $1000\times$ pre-training compute. The authors also foresightedly list embodiment, continual learning, and diffusion effects as future gains. This nomenclature transforms a vague phenomenon into specific objects of analysis for governance impacts.

**2. Analysis of Governance Failure Mechanisms: Explaining how each gain punctures governance pillars**

The authors argue that even improved model-level governance cannot resolve these issues by analyzing the three pillars of model-level governance (elicitation, mitigation, and overhang costs). Inference gains close the gap between frontier and sub-frontier models, allowing malicious actors to bypass frontier regulations using open-weight models. Systems gains are low-cost and diffuse so quickly that developers cannot predict what scaffolds will be built downstream. Asset gains are the most elusive—as restricted assets are secret, evaluators cannot even test the relevant scenarios. The combination of these factors moves pre-deployment assessment from "difficult but feasible" to "structurally insufficient."

**3. Multi-Layered Governance Framework: Assigning governance layers to specific failure points**

Since different gains leak through different stages, a single gatekeeper is insufficient. The authors propose a four-layer complementary governance portfolio. **System governance** requires providers of systems that significantly boost base model capabilities to share risk management responsibilities, addressing systems gains. **Entity governance** shifts focus from individual models to organizational structures, incentives, and decision processes. **Agent governance** manages the delegation parameters and autonomous interactions of AI agents, including access boundaries and unique agent IDs. **Cloud governance** utilizes KYC, content monitoring, and compute pattern monitoring at the inference layer to monitor inference gains. Finally, **Societal resilience** serves as a safety net for when all other layers fail. The logic is not to stack regulation indiscriminately, but to ensure every risk vector has a corresponding node of intervention.

## Key Experimental Results

### Mapping Non-Model Gains to Governance Solutions

| Non-Model Gain Type | Governance Failure Mechanism | Recommended Governance Layer |
| :--- | :--- | :--- |
| Inference Gain | Closes frontier/sub-frontier gap; bypasses regulation via open models | Model-level improvement + Entity + Cloud |
| Systems Gain | Low cost, rapid diffusion; unpredictable scaffolds | Model-level improvement + Entity/Agent/System |
| Asset Gain | Restricted assets cannot be evaluated; high-capability actors | Post-deployment monitoring + NatSec cooperation |
| Embodiment Gain | Information risk translates to physical security risk | Supply chain alignment + System governance |
| Continual Learning | Safety training may be forgotten; model drift | Post-deployment monitoring |
| Diffusion Effects | Monoculture risks, cascading failures | Entity/Agent governance + Societal resilience |

### Empirical Evidence of Inference Gains

| Case | Mechanism of Gain | Effect |
| :--- | :--- | :--- |
| Qwen3-4B + Recursive Self-Aggregation | Inference-time compute scaling | 4B model reaches o3-mini (high) performance |
| DeepSeek-V3.2 vs Gemini 3 | $1.5\text{--}2.5\times$ token consumption | Sub-frontier model exceeds frontier on multiple benchmarks |
| Specialized Dataset Fine-tuning | Asset gain | Equivalent to $1000\times$ pre-training compute |
| Big Sleep (Google DeepMind) | Systems gain (scaffold + tools) | First LLM agent to discover zero-day vulnerabilities |

### Key Findings
- Inference gain is currently the most quantifiable type of non-model gain, supported by emerging inference scaling laws.
- Systems gains are the hardest to prevent—low-resource actors have demonstrated the ability to build complex adversarial scaffolds (e.g., the 2025 Claude Code scaffold by state-sponsored groups).
- Asset gains affect the fewest actors but carry the highest potential harm and are nearly impossible to assess in advance due to their secretive nature.
- Cloud governance faces major commercial, technical, and legal hurdles (privacy laws, cross-provider coordination, and maturity of confidential computing).

## Highlights & Insights
- **Formalized Taxonomy of Non-Model Gains**: Clearly deconstructing the vague concept of "post-deployment capability increases" into inference/system/asset categories and establishing causal chains to governance failure provides an actionable framework for policy.
- **Core Insight on "Shift in Governance Leverage Points"**: As non-model gains become more important, safety responsibility should not rest solely on model developers but should be distributed along the value chain to system integrators, deployment platforms, and end-users.
- **Dual-Effect of Inference Gains**: Inference scaling as a tool for "democratization" is a double-edged sword; it lowers the barrier to frontier-level AI capabilities while simultaneously lowering the barrier for malicious use, which has direct implications for open-weight model safety policies.

## Limitations & Future Work
- As a position paper, it lacks empirical validation, and the feasibility/effectiveness of the proposed governance solutions have not been systematically evaluated.
- Insufficient analysis of interaction effects between non-model gains (e.g., the amplification when inference gains $\times$ systems gains combine); real-world risks often involve multiple overlapping gains.
- Acknowledges significant challenges for cloud governance regarding privacy laws and technical/commercial viability in the short term.
- Does not fully discuss potential conflicts between governance solutions (e.g., how entity governance entry barriers might stifle competition and innovation).
- Future work could involve quantifying the magnitude of capability increases from various non-model gains to build predictive models for resource allocation in governance.

## Related Work & Insights
- Complements model-level safety frameworks like Anthropic RSP and OpenAI Preparedness Framework by identifying their blind spots.
- Directly relates to METR's research on elicitation; this paper demonstrates the structural limitations of exhaustive elicitation.
- Echoes empirical research by Epoch AI regarding the shifting relative importance of the three scaling paradigms (pre-training/post-training/inference).
- Insight: When evaluating AI system safety, "system-level capabilities" should be assessed separately from "model-level capabilities," as the former may significantly exceed the latter.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PAI-Bench: A Comprehensive Benchmark For Physical AI](../../CVPR2026/others/pai-bench_a_comprehensive_benchmark_for_physical_ai.md)
- [\[ICML 2026\] Position: Evaluation of ML Resource Utilization Requires Model Life Cycle Assessment](evaluation_of_ml_resource_utilization_requires_model_life_cycle_assessment.md)
- [\[ICML 2026\] Beyond Model Readiness: Institutional Readiness for AI Deployment in Public Systems](beyond_model_readiness_institutional_readiness_for_ai_deployment_in_public_syste.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](../../NeurIPS2025/others/fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)
- [\[AAAI 2026\] Bridging the Skills Gap: A Course Model for Modern Generative AI Education](../../AAAI2026/others/bridging_the_skills_gap_a_course_model_for_modern_generative_ai_education.md)

</div>

<!-- RELATED:END -->
