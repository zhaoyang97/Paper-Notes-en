---
title: >-
  [Paper Note] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems
description: >-
  [ICLR 2026][Social Computing][Mandela effect] This paper presents the first systematic study of the Mandela effect (collective false memory) in LLM-based multi-agent systems. It introduces the ManBench benchmark (4,838 questions, 5 interaction protocols), demonstrates that all 13 evaluated LLMs are susceptible to this effect, and proposes prompt-level and model-level mitigation strategies that reduce false memory by 74.40% on average.
tags:
  - ICLR 2026
  - Social Computing
  - Mandela effect
  - multi-agent systems
  - collective false memory
  - cognitive bias
  - misinformation
date: 2026-05-08
content_hash: f72da46349b9a08f
---

# When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems

**Conference**: ICLR 2026
**arXiv**: [2602.00428](https://arxiv.org/abs/2602.00428)
**Code**: [github.com/bluedream02/Mandela-Effect](https://github.com/bluedream02/Mandela-Effect)
**Area**: Social Computing
**Keywords**: Mandela effect, multi-agent systems, collective false memory, cognitive bias, misinformation

## TL;DR

This paper presents the first systematic study of the Mandela effect (collective false memory) in LLM-based multi-agent systems. It introduces the ManBench benchmark (4,838 questions, 5 interaction protocols), demonstrates that all 13 evaluated LLMs are susceptible to this effect, and proposes prompt-level and model-level mitigation strategies that reduce false memory by 74.40% on average.

## Background & Motivation

**Background**: LLM-driven multi-agent systems are widely applied to complex tasks (e.g., public policy analysis, social governance, contract review), with a core advantage in simulating social dynamics such as discussion and consensus building.

**Limitations of Prior Work**: Prior research has focused on individual agent errors (hallucinations) or simple conformity behavior, overlooking the distinctive characteristics of **collective cognitive bias** in multi-agent systems. The Mandela effect—shared false memories among a group—involves persuasive false evidence propagating through interactions and being internalized as persistent memory, which is fundamentally different from one-time hallucinations or short-term compliance.

**Key Challenge**: Existing work treats hallucinations as stateless, one-shot failures, ignoring the process by which social interaction can **consolidate false beliefs into long-term memory**. A standardized benchmark for evaluating this phenomenon is lacking.

**Goal**: Construct the ManBench benchmark with 4 categories of Mandela-effect-susceptible tasks (4,838 questions total), design 5 interaction protocols (varying group composition and memory time scale) to inject and measure collective false memory, and propose prompt-level (cognitive anchoring, source scrutiny) and model-level (SFT alignment) mitigation strategies.

## Method

### Overall Architecture

ManBench consists of three components: (1) 20 tasks and 4,838 multiple-choice questions curated from BIG-Bench Hard, classified into 4 knowledge domains; (2) 5 interaction protocols (1 baseline + 4 false memory injection); and (3) an evaluation metric system for quantifying the Mandela effect.

### Key Design 1: Interaction Protocol Design

The 5 protocols vary along two dimensions:

**Group Composition Dimension**:
- **Generic Group**: Undifferentiated agents take turns providing false evidence to form a simple social consensus.
- **Role-based Group**: Five specialized roles collaboratively construct a multi-layered false narrative:
    - Error Conclusion Initiator: proposes the incorrect answer
    - Detail Support Provider: adds fabricated but plausible details
    - Group Consensus Reinforcer: creates the illusion of majority agreement
    - Authority Endorser: lends authority via academic terminology as a domain expert
    - Questioning Compromiser: initially questions but is subsequently persuaded

**Memory Time Scale Dimension**:
- **Short-term**: Evaluated immediately within the same conversational context.
- **Long-term**: Evaluated after memory consolidation (distilling the conversation into a belief summary) and memory retrieval (answering in a new conversation based on the belief summary).

These dimensions combine to yield 4 protocols: GS (Generic Short-term), GL (Generic Long-term), RS (Role-based Short-term), and RL (Role-based Long-term).

### Key Design 2: Evaluation Metric System

- **Error rate** $\text{Err}^P = |\mathcal{Q}_{\times}^P| / |\mathcal{Q}|$
- **Reality shift rate** $\sigma^P = |\mathcal{Q}_{\times}^P \cap \mathcal{Q}_{\checkmark}^B| / |\mathcal{Q}_{\checkmark}^B|$: proportion of questions answered correctly at baseline but incorrectly after interaction
- **Maximum reality shift rate** $\sigma_{max}$: total proportion of correct memories overturned under any protocol

$$\sigma_{max} = |(\mathcal{Q}_{\times}^{GS} \cup \mathcal{Q}_{\times}^{GL} \cup \mathcal{Q}_{\times}^{RS} \cup \mathcal{Q}_{\times}^{RL}) \cap \mathcal{Q}_{\checkmark}^B| / |\mathcal{Q}_{\checkmark}^B|$$

### Key Design 3: Mitigation Strategies

**Prompt-level**:

- **Cognitive Anchoring**: An "inside-out" approach requiring agents to first establish an internal knowledge anchor, maintain skepticism toward external claims, and demand evidence before updating beliefs.
- **Source Scrutiny**: An "outside-in" approach recasting the agent's role from passive recipient to discourse analyst, identifying rhetorical patterns and unnatural consensus.

**Model-level**: SFT on a balanced dataset comprising a resilience set (training resistance to false narratives) and a cooperation set (training acceptance of correct guidance), preventing the model from over-filtering all social input.

## Key Experimental Results

### Main Results

Error rates and reality shift rates (%) for 13 LLMs:

| Model | Baseline Err | RS Err | σ^GS | σ^RS | σ^RL |
|-------|-------------|--------|------|------|------|
| GPT-5 | 17.63 | 41.59 | 27.42 | 31.03 | 1.67 |
| Claude 4 Sonnet | 20.48 | 45.87 | 15.45 | 35.21 | 26.56 |
| GPT-4o | 25.96 | 64.16 | 46.04 | 55.95 | 33.61 |
| Qwen3-235B | 25.48 | 74.75 | 66.98 | 68.69 | 56.85 |
| Llama3.1-8B | 44.58 | 99.67 | 61.69 | 99.47 | 32.10 |
| Claude 3.5 Haiku | 32.00 | 70.38 | 53.26 | 63.67 | 55.63 |

### Ablation Study

**Mitigation strategy effectiveness (σ values for GPT-4o, %)**:

| Method | σ^GS | σ^GL | σ^RS | σ^RL |
|--------|------|------|------|------|
| No defense | 46.04 | 36.53 | 55.95 | 33.61 |
| Cognitive Anchoring | 17.8 | 14.7 | 17.0 | 15.2 |
| Source Scrutiny | 26.5 | 16.0 | 25.2 | 14.5 |

**Model-level defense (Llama3.1-8B)**:

| Training Set | σ^RS | σ^C (correct guidance shift) |
|-------------|------|------------------------------|
| No training | 99.47 | — |
| Resilience set only | 18.2 | 38.5 (over-rejection) |
| Resilience + cooperation set | 21.5 | 1.1 (cooperative ability preserved) |

### Key Findings

1. **All LLMs are susceptible**: Even the strongest model, GPT-5, sees its error rate double under the role-based short-term protocol (17.6% → 41.6%), while Qwen3-235B surges to 74.8%.
2. **Role-based > Generic groups**: Strategic narrative construction is more effective at injecting false memory than simple consensus; Claude 4 Sonnet's σ rises from 15.45% (GS) to 35.21% (RS).
3. **False memories can consolidate into long-term beliefs**: Claude 3.5 Haiku's σ decreases only marginally from 63.67% (RS) to 55.63% (RL), whereas GPT-5 demonstrates strong self-correction capacity (31.0% → 1.67%).
4. **Inverted-U effect of group size**: Role-based groups exert maximum influence at 6 agents; larger groups paradoxically trigger "skeptical vigilance," prompting self-correction.
5. **Model scaling does not reliably help**: Within the Qwen3 series, $\sigma_{max}$ increases rather than decreases as parameters scale from 8B to 235B (89.3% → 92.2%).

## Highlights & Insights

- **From hallucination to social false memory**: Unlike traditional hallucination research, this work reveals a memory manipulation mechanism driven by social interaction, representing a risk unique to multi-agent systems.
- **"Suspicion-triggered vigilance"**: The counterintuitive finding that larger coordinated groups reduce susceptibility suggests LLMs possess a latent capacity to detect implausible social dynamics.
- **Necessity of balanced training**: Training solely for resistance to misinformation causes models to over-reject all external input; the resilience-plus-cooperation training scheme preserves discriminative ability.
- **Knowledge domain analysis**: Even in general knowledge domains with a baseline error rate of only 9.4%, σ reaches 48%; in specialized domains it rises to 67.5%, indicating that a strong knowledge base does not confer immunity.

## Limitations & Future Work

- ManBench adopts a multiple-choice format, which simplifies the complexity of real-world unstructured dialogue.
- Open-ended discussion and dynamic role-switching scenarios remain unexplored.
- The generalizability of mitigation strategies requires further validation (currently verified preliminarily on the medical domain benchmark MedMCQA).
- Introducing a "critic" agent for cross-validation and reflection is a promising direction.
- The Mandela effect across different cultural and linguistic contexts has not been examined.

## Related Work & Insights

- **LLM social influence**: Weng et al. (2025) study conformity; Xu et al. (2024) study persuadability. This paper extends the focus from short-term compliance to long-term memory consolidation.
- **Multi-agent systems**: MetaGPT (Hong et al., 2024) and AutoGen (Wu et al., 2024) demonstrate multi-agent collaboration capabilities but overlook the risk of collective cognitive bias.
- **Hallucination and factual robustness**: Huang et al. (2025) study malicious agents injecting false information; this paper focuses on memory manipulation induced by social persuasion.

## Rating

⭐⭐⭐⭐

The paper addresses a novel research question, presenting the first systematic study of the Mandela effect in multi-agent systems. ManBench is thoughtfully designed (4 task categories × 5 protocols × 13 models), and the discovered inverted-U group size effect and scaling paradox offer important insights. The mitigation strategies—particularly the balanced training scheme—are practically valuable. The experimental scope is large and the analytical dimensions are rich.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/social_computing/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [\[ICLR 2026\] Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)
- [\[ICLR 2026\] Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI](functional_embeddings_enable_aggregation_of_multi-area_seeg_recordings_over_subj.md)
- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](../../ACL2026/social_computing/on_the_step_length_confounding_in_llm_reasoning_data_selection.md)

</div>

<!-- RELATED:END -->
