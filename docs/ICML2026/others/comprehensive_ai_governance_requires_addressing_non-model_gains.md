---
title: >-
  [Paper Note] Comprehensive AI Governance Requires Addressing Non-Model Gains
description: >-
  [ICML 2026][Others][Paper Note] This paper is a position paper arguing that the current model-centric AI governance paradigm is seeing diminishing effectiveness in the context of the growing importance of "non-model gains" (inference gains, system gains, and asset gains). It calls for a multi-layered complementary approach—comprising system governanc
tags:
  - ICML 2026
  - Others
date: 2026-05-08
content_hash: fd4844e6cc81c8ad
---
# Comprehensive AI Governance Requires Addressing Non-Model Gains

**Conference**: ICML2026  
**arXiv**: [2606.00047](https://arxiv.org/abs/2606.00047)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: AI Governance, Non-Model Gains, Frontier AI Safety, Inference Scaling, Multi-layered Governance  

## TL;DR
This paper is a position paper arguing that the current model-centric AI governance paradigm is seeing diminishing effectiveness in the context of the growing importance of "non-model gains" (inference gains, system gains, and asset gains). It calls for a multi-layered complementary approach—comprising system governance, entity governance, agent governance, and cloud governance—to fill regulatory gaps.

## Background & Motivation

**Background**: Current frontier AI governance relies primarily on "model-level governance," which manages risks by evaluating and mitigating the dangerous capabilities of models. This paradigm assumes that a model's capabilities are primarily determined by the compute and data used during the training phase, thereby placing frontier model developers as the central nodes for safety efforts. International regulations (such as the EU AI Act) and corporate self-regulatory frameworks (like Anthropic’s RSP and OpenAI’s Preparedness Framework) focus on pre-deployment evaluation and mitigation.

**Limitations of Prior Work**: With the rise of reasoning models (e.g., OpenAI o1 series), complex scaffolding systems (e.g., Google DeepMind’s Big Sleep for zero-day vulnerability discovery), and collaborations between national security agencies and AI companies, an increasing number of capability improvements do not stem from pre-training scaling but from post-deployment "non-model gains." These gains are characterized by low cost, rapid diffusion, and the difficulty for model developers to fully foresee them before deployment.

**Key Challenge**: The effectiveness of model-level governance depends on the developer's ability to **exhaustively elicit** a model's downstream capabilities prior to deployment. However, non-model gains cause the capability space to expand continuously post-deployment, leading to three governance failures: elicitation failure (inability to foresee all enhancement methods), mitigation failure (difficulty in controlling known dangerous capabilities), and lagging costs (increased expected harm from unelicited capabilities).

**Goal**: (1) Formalize the concept of "non-model gains" and establish a taxonomic system; (2) Analyze how each category of gain undermines model-level governance; (3) Propose complementary governance solutions that go beyond the model level.

**Key Insight**: Starting from governance practice, the authors observe that the proportional contribution of pre-training scaling in frontier models is declining while the importance of inference scaling and post-training is rising. Furthermore, low-resource actors can approach frontier capabilities through system gains, meaning the "leverage points" of governance are diffusing from model developers toward downstream actors.

**Core Idea**: Non-model gains are systematically undermining the underlying assumptions of model-level governance. A multi-layered governance portfolio including system, entity, agent, and cloud levels must be established to address this.

## Method

### Overall Architecture
As a position paper, the "method" consists of a linked chain of arguments: first, naming and categorizing post-deployment capability increases as "non-model gains"; second, deconstructing how each category pierces the assumptions of model-level governance; and finally, prescribing a multi-layered governance framework that no longer focuses solely on model developers. The progression is causal—without clear classification, failure points cannot be located; without locating failure points, effective governance layers cannot be prescribed.

### Key Designs

**1. Non-Model Gain Taxonomic System: Deconstructing "Post-deployment Capability Expansion" into Three Analyzable Categories**

Model-level governance defaults to the view that capabilities are determined by training-period compute and data, positioning developers as the primary safety gatekeepers. However, what truly drives capability expansion post-deployment are "non-model gains," which are often vague and difficult to quantify. The authors categorize current gains into three types: **Inference Gain** relies on increasing inference-time compute for performance (e.g., Chain-of-Thought scaling in reasoning models, allowing small models to approach large ones—for instance, Qwen3-4B matching o3-mini levels through recursive self-aggregation); **Systems Gain** is built through post-training scaffolding, tool calls, and multi-agent orchestration (once a "recipe" is discovered, it can diffuse freely at zero cost); **Asset Gain** comes from accessing restricted assets (classified government data, specialized hardware), where a single high-quality dataset might provide an improvement equivalent to a $1000\times$ increase in pre-training compute. Beyond these, the authors also prospectively list embodiment, continual learning, and diffusion effects as future gains. The value of this naming convention lies in transforming a generalized phenomenon into specific objects for analyzing how they undermine governance.

**2. Governance Failure Mechanism Analysis: Explaining How Each Category Undermines Core Pillars**

The authors argue that even improving model-level governance cannot resolve these issues. They analyze how each type of gain strikes at the three pillars of model-level governance (elicitation, mitigation, and lagging costs). Inference gains narrow the gap between frontier and sub-frontier models, allowing malicious actors to use open-weight models to bypass frontier-specific regulations. Systems gains are low-cost and diffuse rapidly, making it impossible for developers to foresee what scaffolds will be constructed downstream. Asset gains are particularly challenging—since the assets themselves are confidential, evaluators cannot even test relevant scenarios. The combination of these factors moves pre-deployment evaluation from "difficult but feasible" to "structurally insufficient."

**3. Multi-layered Governance Framework: Assigning Governance Layers to Failure Points**

Since different gains leak through different stages, a single gatekeeper is insufficient. The authors propose a four-layer complementary governance portfolio. **System Governance** requires providers of systems that significantly elevate base model capabilities to assume risk management responsibilities, directly addressing systems gains. **Entity Governance** shifts focus from individual models to organizational structures, incentive mechanisms, and decision-making processes. **Agent Governance** manages delegation parameters and autonomous interactions of AI agents, including access boundaries, behavioral constraints, and unique agent IDs. **Cloud Governance** utilizes KYC, content monitoring, and compute pattern monitoring at the inference layer to oversee inference gains. Finally, **Societal Resilience** serves as a backstop when all other layers fail. The logic is not to stack regulation indiscriminately, but to ensure every risk vector has a corresponding node.

## Key Experimental Results

### Mapping Non-Model Gains to Governance Solutions

| Non-Model Gain Type | Governance Failure Mechanism | Recommended Governance Layer |
|:--- |:--- |:--- |
| Inference Gain | Narrows frontier/sub-frontier gap; open models bypass regs | Model improvements + Entity + Cloud |
| Systems Gain | Low cost, rapid diffusion; unpredictable scaffolds | Model improvements + Entity/Agent/System |
| Asset Gain | Un-evaluable assets; few high-capability actors | Post-deployment monitoring + NatSec coop |
| Body Gain (Embodiment) | Information risk transforms into physical safety risk | Supply chain alignment + System governance |
| Continual Learning | Safety training may be forgotten; behavioral drift | Post-deployment monitoring |
| Diffusion Effects | Monoculture risks; cascading failures | Entity/Agent governance + Societal resilience |

### Empirical Evidence of Inference Gains

| Case | Mechanism | Effect |
|:--- |:--- |:--- |
| Qwen3-4B + Recursive Self-Aggregation | Inference-time compute scaling | 4B model reaches o3-mini (high) level |
| DeepSeek-V3.2 vs Gemini 3 | 1.5-2.5x token consumption | Sub-frontier outperforms frontier on benchmarks |
| Specialized Dataset Fine-tuning | Asset gain | Equivalent to $1000\times$ pre-training compute |
| Big Sleep (Google DeepMind) | Systems gain (scaffold + tools) | First LLM agent to find zero-day vulnerabilities |

### Key Findings
- Inference gain is currently the most quantifiable type of non-model gain, supported by emerging inference scaling laws.
- Systems gain is the most difficult to prevent—low-resource actors have demonstrated the ability to build complex adversarial scaffolds.
- Asset gain affects the fewest actors but carries the greatest potential harm and is nearly impossible to evaluate in advance due to its confidential nature.
- Cloud governance faces significant commercial, technical, and legal hurdles (privacy laws, cross-provider coordination, maturity of confidential computing).

## Highlights & Insights
- **Formal Classification of Non-Model Gains**: Clearly deconstructing the vague concept of "post-deployment capability increase" into inference/system/asset categories establishes a causal chain for governance failure, providing an actionable framework for policy.
- **Insight on "Shift in Governance Leverage Points"**: As the importance of non-model gains rises, safety responsibility should not rest solely on model developers but should be distributed along the value chain to system integrators, deployment platforms, and end-users.
- **Double-Edged Sword of Inference Gains**: Inference scaling democratizes high-performance AI but also lowers the barrier to entry for malicious use, which has direct implications for safety policies regarding open-weight models.

## Limitations & Future Work
- As a position paper, it lacks empirical validation; the feasibility and effectiveness of the proposed governance layers have not been systematically evaluated.
- Insufficient analysis of interaction effects between non-model gains (e.g., the multiplicative effect of inference gain $\times$ systems gain).
- Parts of the cloud governance proposal face major privacy, technical, and commercial challenges that make short-term implementation difficult.
- Potential conflicts between governance solutions (e.g., entity governance entry barriers stifling innovation) are not fully discussed.
- Future work could quantify the magnitude of capability increases from various non-model gains to build predictive models for resource allocation.

## Related Work & Insights
- Complementary to model-level safety frameworks like Anthropic RSP and OpenAI Preparedness, identifying their blind spots.
- Directly related to METR's research on capability elicitation, arguing for its structural limitations.
- Resonates with empirical studies from Epoch AI regarding the shifting importance of the three scaling paradigms (pre-training, post-training, and inference).
- **Insight**: When evaluating AI system safety, "system-level capability" and "model-level capability" must be considered separately, as the former may far exceed the latter.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PAI-Bench: A Comprehensive Benchmark For Physical AI](../../CVPR2026/others/pai-bench_a_comprehensive_benchmark_for_physical_ai.md)
- [\[ICML 2026\] Beyond Model Readiness: Institutional Readiness for AI Deployment in Public Systems](beyond_model_readiness_institutional_readiness_for_ai_deployment_in_public_syste.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](../../NeurIPS2025/others/fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)
- [\[AAAI 2026\] Bridging the Skills Gap: A Course Model for Modern Generative AI Education](../../AAAI2026/others/bridging_the_skills_gap_a_course_model_for_modern_generative_ai_education.md)
- [\[CVPR 2026\] Drainage: A Unifying Framework for Addressing Class Uncertainty](../../CVPR2026/others/drainage_a_unifying_framework_for_addressing_class_uncertainty.md)

</div>

<!-- RELATED:END -->
