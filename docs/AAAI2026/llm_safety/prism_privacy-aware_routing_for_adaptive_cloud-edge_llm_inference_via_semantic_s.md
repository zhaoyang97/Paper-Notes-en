---
title: >-
  [Paper Note] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration
description: >-
  [AAAI 2026][LLM Safety][Privacy Protection] This paper proposes PRISM, a framework that dynamically routes user prompts to one of three inference modes—cloud-only, edge-only, or collaborative—via a context-aware soft gating mechanism. In the collaborative mode, an adaptive two-layer local differential privacy (LDP) scheme and semantic sketch collaboration are employed to achieve a three-way balance among privacy, utility, and efficiency.
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Privacy Protection"
  - "Cloud-Edge Collaborative Inference"
  - "LLM Privacy"
  - "Differential Privacy"
  - "Semantic Sketch"
date: 2026-05-08
content_hash: 440129634c858d06
---

# PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration

**Conference**: AAAI 2026
**arXiv**: [2511.22788](https://arxiv.org/abs/2511.22788)  
**Code**: [https://github.com/Junfei-Z/PRISM](https://github.com/Junfei-Z/PRISM)  
**Area**: AI Security
**Keywords**: Privacy Protection, Cloud-Edge Collaborative Inference, LLM Privacy, Differential Privacy, Semantic Sketch

## TL;DR

This paper proposes PRISM, a framework that dynamically routes user prompts to one of three inference modes—cloud-only, edge-only, or collaborative—via a context-aware soft gating mechanism. In the collaborative mode, an adaptive two-layer local differential privacy (LDP) scheme and semantic sketch collaboration are employed to achieve a three-way balance among privacy, utility, and efficiency.

## Background & Motivation

LLMs are typically deployed in the cloud to meet large-scale inference demands, which introduces two core issues:

**Privacy Risk**: User prompts frequently contain sensitive personal information (e.g., medical records, financial data), and transmitting them in full to the cloud poses a risk of privacy leakage.

**Communication Overhead**: Transmitting large volumes of prompts incurs significant latency and energy consumption.

Limitations of existing cloud-edge collaborative approaches:

- **Binary Routing Decisions**: Simple threshold-based routing classifies prompts as either "process locally" or "process in the cloud," which is prone to misclassification—either overburdening edge devices or leaking privacy.
- **Uniform Noise Injection**: Prompt-level differential privacy methods (e.g., Split-and-Denoise, DP-Forward) apply uniform noise to all tokens/dimensions regardless of sensitivity, causing two problems:
    - Unnecessary noise is introduced for non-sensitive queries (e.g., "What is the capital of France?").
    - Uniform perturbation severely distorts semantics, causing cloud LLMs to produce vague or evasive responses (e.g., "I cannot provide information about [MASKED_ENTITY]").

**Core Motivation**: There is a need for an **adaptive privacy protection mechanism that is aware of prompt semantics**, dynamically selecting a protection strategy based on the specific sensitivity of each prompt.

## Method

### Overall Architecture

PRISM consists of four stages:

1. **Sensitivity Profiling**: The edge device performs entity-level sensitivity assessment on the prompt.
2. **Soft Gating**: A soft classifier selects the execution mode (cloud / edge / collaborative) based on the sensitivity features.
3. **Adaptive Two-Layer LDP**: For sensitive entities on the collaborative path, differential privacy budgets are allocated according to category-level risk.
4. **Semantic Sketch Collaboration**: The cloud LLM generates a semantic sketch from the perturbed prompt, and the edge SLM refines the final response using the original context.

### Key Designs

#### 1. **Sensitivity Profiling Module**

A lightweight edge module with two outputs:

- **Risk Score $R(P)$**: Reflects the overall privacy sensitivity of the prompt.

$$R(P) = \sum_{i=1}^m w_{c_i} \cdot \mathbb{I}(e_i)$$

where $w_{c_i}$ is the predefined sensitivity weight for each entity category (e.g., $w_{\text{PERSON}} > w_{\text{NATIONALITY}}$).

- **Context Indicator $\Delta$**: Detects the presence of privacy-related linguistic cues (first-person pronouns, named person entities).

$$\Delta = \max_{x_j \in P} \mathbb{I}(x_j \in \mathcal{F})$$

**Key Design Intent**: The same entity can carry different sensitivity levels in different contexts. "Tokyo" in "I plan to travel to Tokyo" involves personal information (due to "I"), whereas "Tokyo" in "Which country is Tokyo in?" constitutes public knowledge. When $\Delta > 0$, all entities are conservatively flagged for protection.

#### 2. **Soft Gating Routing Mechanism**

Unlike hard-threshold routing, PRISM employs an entropy-regularized soft classifier:

$$\boldsymbol{\pi} = \text{softmax}(f_\theta(\mathbf{z})) \in \mathbb{R}^3$$

This produces a probability distribution over three modes: $\boldsymbol{\pi} = (\pi_{\text{cloud}}, \pi_{\text{collab}}, \pi_{\text{local}})$.

**Training Loss** comprises a task loss and an entropy regularization term:

$$\mathcal{L}_{\text{gating}} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{H}(\boldsymbol{\pi})$$

- Low entropy: encourages confident routing decisions.
- High entropy: accommodates ambiguous or uncertain cases.
- At inference time, $\arg\max$ is taken to ensure deterministic routing.

#### 3. **Adaptive Two-Layer Local Differential Privacy**

This constitutes the core technical contribution of the paper. To address both the vulnerability of naive anonymization (e.g., replacing entities with `<NAME>`) to linkage attacks and the over-perturbation caused by uniform LDP, a hierarchical adaptive LDP scheme is proposed:

**Budget Allocation**: Privacy budgets are adaptively allocated based on entity category sensitivity weights $w_{c_i}$:

$$\varepsilon_1 = \varepsilon_{\text{total}} \cdot \frac{w_{c_i}}{w_{c_i} + (1 - w_{c_i}) \cdot \alpha}, \quad \varepsilon_2 = \varepsilon_{\text{total}} - \varepsilon_1$$

- **High-sensitivity entities** (e.g., NAME): $w_{c_i} \to 1$; more budget is allocated to $\varepsilon_1$ (category layer) to conceal the entity type.
- **Low-sensitivity entities** (e.g., LOCATION): $w_{c_i} \to 0$; more budget is allocated to $\varepsilon_2$ (value layer) to preserve semantic consistency.

Both layers use the **Randomized Response** mechanism to achieve $\varepsilon$-LDP.

**Privacy Guarantee** (Theorem 1): The composite mechanism $M = M_2 \circ M_1$ satisfies $(\varepsilon_1 + \varepsilon_2)$-LDP, guaranteed by the sequential composition theorem.

#### 4. **Semantic Sketch Collaboration**

The perturbed prompt $P^*$ is sent in plaintext to the cloud (avoiding embedding transmission and tokenizer synchronization issues). Collaboration proceeds in two steps:

**Cloud Sketch Generation**: Using few-shot prompting, the cloud LLM generates a structured semantic sketch $S$ from $P^*$:

$$S = \mathcal{G}_{\text{cloud}}(\mathcal{C}_{\text{cloud}})$$

The sketch adopts a concise format, omitting obfuscated sensitive entities while maintaining semantic alignment.

**Edge Refinement**: The edge SLM combines the sketch $S$ with the original prompt $P$ (retained locally) to reconstruct the final response:

$$\hat{R} = \mathcal{G}_{\text{edge}}(\mathcal{C}_{\text{edge}})$$

### Loss & Training

- Gating module: cross-entropy loss with entropy regularization.
- Privacy mechanism: $\varepsilon$-LDP guarantee established via mathematical proof.
- Inference relies on in-context learning; no additional training is required.

## Key Experimental Results

### Main Results

| Method | Completion Time (s) | Energy (J) | Inference Quality (IQ) | Notes |
|---|---|---|---|---|
| Cloud-Only | 5.13 | 296 | 8.14 | No privacy protection |
| **PRISM** | **7.92** | **687** | **6.88** | Best privacy-preserving method |
| Uniform LDP | 20.56 | 1708 | 5.72 | Uniform noise injection |
| Selective LDP | 21.22 | 1771 | 5.94 | Selective noise injection |
| Edge-Only | 17.84 | 1574 | 5.09 | Pure edge inference |

**PRISM incurs only 1.54× the latency and 2.32× the energy consumption of Cloud-Only**, while providing strong privacy protection, whereas Uniform/Selective LDP require approximately 4× the latency and 6× the energy consumption.

### Ablation Study

| Cloud LLM + Edge SLM | Completion Time (s) | Energy (J) | Inference Quality |
|---|---|---|---|
| GPT-4o + Phi-3.5-mini-3.5B | 8.29 | 684 | 7.00 |
| GPT-4o + Qwen1.5-1.8B | 7.08 | 632 | 6.91 |
| GPT-4o + StableLM-2-1.6B | 7.34 | 658 | **7.16** |
| Qwen3-235B + Phi-3.5-mini-3.5B | 8.59 | 739 | **7.22** |
| Qwen3-235B + TinyLLaMA-1.1B | 8.11 | 698 | 7.19 |

All 8 cloud-edge model combinations maintain IQ ≥ 6.9 (except TinyLLaMA), demonstrating the framework's adaptability to heterogeneous deployments.

### Key Findings

1. **Adaptive routing is critical for efficiency**: PRISM routes non-sensitive prompts directly to the cloud, low-risk prompts through the collaborative path, and high-risk prompts to local processing, thereby avoiding the computational waste of a one-size-fits-all strategy.
2. **PRISM performance is stable across varying privacy budgets**: Under different values of $\varepsilon$, PRISM exhibits minimal fluctuation in inference quality, energy consumption, and latency, whereas Uniform/Selective LDP degrade significantly as the budget tightens.
3. **Semantic sketch design is effective**: Rather than generating a complete response directly, the cloud produces a structured sketch that the edge refines, preserving both privacy and response quality.

## Highlights & Insights

- **Entity-level fine-grained privacy protection**: Prompts are no longer treated as monolithic units; instead, each entity's sensitivity is assessed individually.
- **Theoretical rigor of two-layer LDP**: A complete privacy guarantee proof and a monotonicity proof for budget allocation are provided.
- **Real hardware evaluation**: Energy consumption and latency are measured on an actual RTX 3070 edge device rather than relying solely on simulation.
- **A semi-synthetic dataset spanning 4 domains is constructed**: covering travel, medical, banking, and general knowledge.

## Limitations & Future Work

1. **NER Dependency**: Sensitivity profiling relies on the accuracy of named entity recognition; missed detections by NER may lead to privacy leakage.
2. **Limited Dataset Scale**: Only 40 prompts per domain are included, lacking large-scale real-world validation.
3. **Single Edge Device**: Multi-edge-device collaborative inference scenarios are not considered.
4. **Sketch Quality Sensitive to Perturbation Level**: At very low privacy budgets, sketch quality may degrade sharply.
5. **Inference Quality Evaluation Relies on GPT-4o**: Using an LLM as the judge introduces potential evaluation bias.

## Related Work & Insights

- Cloud-edge collaborative LLM inference is a critical direction for privacy protection; the progression from simple binary routing to PRISM's three-mode soft routing represents a meaningful advance.
- The two-layer LDP concept is generalizable to other privacy-sensitive scenarios (e.g., gradient protection in federated learning).
- The "cloud generates sketch, edge refines" paradigm of semantic sketch collaboration is applicable to other privacy-preserving generation tasks.
- Future work could integrate federated learning to train personalized sensitivity profiling modules.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (three-mode routing + adaptive two-layer LDP + semantic sketch; rich and layered contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐ (real hardware evaluation, 8 model combinations, but dataset scale is limited)
- Writing Quality: ⭐⭐⭐⭐⭐ (rigorous theoretical proofs, clear system architecture description)
- Value: ⭐⭐⭐⭐⭐ (addresses practical LLM privacy protection problems with deployment potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](../../ACL2026/llm_safety/privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)
- [\[AAAI 2026\] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [\[NeurIPS 2025\] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](../../NeurIPS2025/llm_safety/cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)
- [\[AAAI 2026\] RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence](radarllm_empowering_large_language_models_to_understand_human_motion_from_millim.md)
- [\[AAAI 2026\] SafeNlidb: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](safenlidb_a_privacy-preserving_safety_alignment_framework_for_llm-based_natural_.md)

</div>

<!-- RELATED:END -->
