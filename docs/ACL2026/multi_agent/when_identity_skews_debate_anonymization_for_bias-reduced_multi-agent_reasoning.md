---
title: >-
  [Paper Note] When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning
description: >-
  [ACL 2026][Multi-Agent][Paper Note] This paper highlights that LLMs in Multi-Agent Debate (MAD) change their positions based on "who said it" rather than "what was said." It quantifies and mitigates this identity-driven bias through response anonymization and the Identity Bias Coefficient (IBC).
tags:
  - ACL 2026
  - Multi-Agent
date: 2026-05-08
content_hash: 0586cfc643316336
---
This paper note provides an English translation of the key aspects of the study "When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning."

## TL;DR
This paper highlights that LLMs in Multi-Agent Debate (MAD) change their positions based on "who said it" rather than "what was said." It quantifies and mitigates this identity-driven bias through response anonymization and the Identity Bias Coefficient (IBC).

## Background & Motivation
**Background**: Multi-Agent Debate (MAD) assumes that allowing multiple LLMs to answer independently, read peer responses, and revise their own stances can amplify correct reasoning and reduce errors. Traditional research focuses on communication protocols and agent diversity, assuming agents update beliefs based on argument quality.

**Limitations of Prior Work**: In practice, agents receive content with source labels (e.g., "my answer" vs. "peer answer"). Models often defer excessively to peers or stubbornly stick to their own answers regardless of evidence quality, potentially leading the system away from the correct answer.

**Key Challenge**: MAD protocols leak identity information, which distorts belief updates into a competition between "self" and "other" weights. Systems need agents to reference each other without irrational conformity or self-persistence.

**Goal**: The paper aims to integrate conformity and self-bias into one framework, measure these biases during disagreements, and reduce them through protocol-level changes without extra training.

**Key Insight**: Since identity labels trigger different weights for the same information, the debate process can be modeled as a Bayesian belief update. Observing which side an agent follows during a disagreement allows for the estimation of identity-driven behavioral influence.

**Core Idea**: Response anonymization removes identity markers from the debate, forcing agents to evaluate arguments solely on content. The IBC then measures the magnitude of the bias removed.

## Method

### Overall Architecture
The study uses a MAD setting where multiple agents answer a question, see their own and peers' previous answers, and then revise. Analysis focuses on disagreement samples ($y_{i,t-1} \neq y_{j,t-1}$), as these reveal whether an agent follows a peer or persists in its own answer.

The authors define Conformity (probability of following a peer) and Obstinacy (probability of sticking to one's own answer) within a Dirichlet-Compound-Multinomial belief update model. Response Anonymization is then implemented by removing source labels from prompts, effectively forcing equal weights for all responses ($w_i = w_j$).

### Key Designs
1. **Conformity and Obstinacy Metrics**:
    - **Function**: Converts abstract "conformity" and "stubbornness" into computable indicators.
    - **Mechanism**: Calculates probabilities only when $y_{i,t-1} \neq y_{j,t-1}$ to isolate identity influence.
    - **Design Motivation**: These metrics separate debate dynamics into foundational tendencies (e.g., being corrected vs. being misled) rather than just looking at final accuracy.

2. **Identity-Weighted Bayesian Model**:
    - **Function**: Decomposes the difference between Conformity and Obstinacy into content belief and identity weight.
    - **Mechanism**: Models internal belief as Dirichlet parameters, where self and peer evidence carry specific weights.
    - **Design Motivation**: Provides a testable low-dimensional explanation for why identity labels influence model adoption of answers.

3. **Response Anonymization and IBC**:
    - **Function**: Removes the identity channel and estimates the bias magnitude.
    - **Mechanism**: $\text{IBC} = \Delta_{vanilla} - \Delta_{anonymized}$. Positive values indicate conformity; negative values indicate self-bias.
    - **Design Motivation**: A zero-cost, model-agnostic protocol change that treats the issue as an information control problem.

## Key Experimental Results

### Main Results
Testing Qwen2.5, Llama3.1, Mistral, and GPT-OSS on benchmarks like GPQA and MMLU revealed that identity bias is pervasive, typically manifesting as positive IBC (conformity).

| Model / Dataset | Vanilla $\Delta$ | Anonymized $\Delta$ | IBC | Finding |
| :--- | :--- | :--- | :--- | :--- |
| Qwen-32B / MMLU | 0.608 | 0.024 | 0.584 | Strong conformity; virtually eliminated by anonymization |
| Qwen-7B / HellaSwag | 0.507 | -0.032 | 0.539 | High peer weight; becomes slightly self-biased when anonymous |
| Llama-8B / MMLU | 0.151 | -0.157 | 0.307 | Anonymization reveals underlying content belief differences |

### Ablation Study
- **Vanilla vs. Anonymous**: 18 of 20 cases showed positive IBC in vanilla settings.
- **Trust Analysis**: Anonymization reduces Subversion (correct to incorrect) by 64.3% in Qwen-32B, while Correction only drops by 14.9%.

### Key Findings
- MAD failures often stem from protocol issues (identity labels) rather than just reasoning failures.
- Large models like Qwen-32B are not immune to identity-driven conformity.
- Anonymization selectively reduces "incorrect" stance changes triggered by identity.

## Highlights & Insights
- Integrates sycophancy and self-bias into a single "identity bias" framework.
- Response Anonymization is a simple, effective protocol change applicable to any multi-agent system.
- IBC serves as a useful diagnostic tool for identifying dangerous identity signals in protocols.

## Limitations & Future Work
- Complexity: Other factors like response length or formatting might also influence adoption weights.
- Expert Systems: In some cases, identity is a valid signal (e.g., expert vs. non-expert); future work should distinguish between harmful and helpful labels.
- Task Type: Currently focused on short-answer reasoning; effects on long-form generation are unknown.

## Related Work & Insights
- Compared to standard MAD research, this paper highlights that the source is as important as the content.
- Extends sycophancy research from user-model interaction to model-model interaction.
- Suggests that "persona" based debates may inadvertently introduce weight biases.

## Rating
- Novelty: ⭐⭐⭐⭐☆
- Experimental Thoroughness: ⭐⭐⭐⭐☆
- Writing Quality: ⭐⭐⭐⭐☆
- Value: ⭐⭐⭐⭐⭐

## Related Papers

- [\[ACL 2026\] Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate](latent_agents_a_post-training_procedure_for_internalized_multi-agent_debate.md)
- [\[ACL 2026\] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)
- [\[ICML 2026\] When Cloud Agents Meet Device Agents: Lessons from Hybrid Multi-Agent Systems](../../ICML2026/multi_agent/when_cloud_agents_meet_device_agents_lessons_from_hybrid_multi-agent_systems.md)
- [\[ACL 2026\] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)
- [\[ACL 2025\] CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate](../../ACL2025/multi_agent/cortexdebate_debating_sparsely_and_equally_for_multi-agent_debate.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ACL 2026\] Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate](latent_agents_a_post-training_procedure_for_internalized_multi-agent_debate.md)
- [\[ACL 2026\] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)
- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)

</div>

<!-- RELATED:END -->
