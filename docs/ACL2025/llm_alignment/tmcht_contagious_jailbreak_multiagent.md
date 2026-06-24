---
title: >-
  [Paper Note] A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns
description: >-
  [ACL 2025][LLM Alignment][Multi-agent attack] Proposes TMCHT (a large-scale multi-agent multi-topology text attack evaluation framework) and ARCJ (Adversarial Replicative Contagious Jailbreak) method—enhancing the retrieval probability of toxic samples by optimizing the retrieval suffix + enabling the self-replicating contagious capability of toxic information by optimizing the replicative suffix, solving the "toxicity dissipation" problem faced by single-agent attack methods…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Multi-agent attack"
  - "contagious jailbreak"
  - "memory attack"
  - "toxicity dissipation"
  - "ARCJ"
date: 2026-05-08
content_hash: 4b465db08cf8ba01
---

# A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns

**Conference**: ACL 2025  
**arXiv**: [2410.16155](https://arxiv.org/abs/2410.16155)  
**Code**: None  
**Area**: LLM Alignment  
**Keywords**: Multi-agent attack, contagious jailbreak, memory attack, toxicity dissipation, ARCJ

## TL;DR

Proposes TMCHT (a large-scale multi-agent multi-topology text attack evaluation framework) and ARCJ (Adversarial Replicative Contagious Jailbreak) method—enhancing the retrieval probability of toxic samples by optimizing the retrieval suffix + enabling the self-replicating contagious capability of toxic information by optimizing the replicative suffix, solving the "toxicity dissipation" problem faced by single-agent attack methods in multi-agent systems.

## Background & Motivation

**Background**: LLMs are widely used as agents, where memory is a key component but susceptible to jailbreak attacks.  
**Limitations of Prior Work**: Existing attacks only focus on single-agent memory or shared memory, whereas in real-world scenarios, multi-agent systems typically employ independent memories.  
**Key Challenge**: The effectiveness of single-agent attack methods drops sharply in non-complete graph topologies (line/star) and large-scale (100+ agents) systems.  
**Goal**: Achieve effective cross-agent jailbreak propagation in multi-agent systems with independent memories.  
**Key Insight**: Discovery of the "toxicity dissipation" phenomenon—toxic suffixes gradually disappear when propagating between agents, leading to retrieval failure.  
**Core Idea**: Empower toxic information with "self-replicating" capabilities to maintain toxicity during propagation.

## Method

### Overall Architecture

TMCHT task setting: One attacker agent $\rightarrow$ in a given social topology (graph/line/star) $\rightarrow$ through a limited number of dialogue rounds $\rightarrow$ misleads the agents across the entire society. The ARCJ method consists of two stages: first optimizing the retrieval suffix, and then optimizing the replicative suffix.

### Key Designs

1. **Retrieval Suffix Optimization**:
    - **Function**: Makes toxic samples more likely to be selected during memory retrieval.
    - **Mechanism**: Minimize the distance between the toxic sample embedding and the query embedding: $\min_{\delta_r} \text{dist}(\text{Embed}(x + \delta_r), \text{Embed}(q))$, where $\delta_r$ is the retrieval suffix.
    - **Design Motivation**: Toxic samples must first be retrieved to take effect—if benign content has higher similarity, toxic samples will never be utilized.

2. **Replicative Suffix Optimization (Contagious Capability)**:
    - **Function**: Forces the attacked LLM to automatically replicate toxic information in its responses.
    - **Mechanism**: $\min_{\delta_c} -\log P(x + \delta_r + \delta_c | \text{context})$, maximizing the probability of reproducing toxic text in LLM responses.
    - **Design Motivation**: Addresses the core of "toxicity dissipation"—standard attacks have suffixes rewritten/lost during propagation. The replicative suffix ensures that responses from downstream agents also contain the complete toxic content.

3. **Multi-Topology Evaluation Framework (TMCHT)**:
    - **Function**: Defines multi-agent attack evaluation tasks under three social topologies: graph, line, and star.
    - **Mechanism**: The attacker only communicates with neighboring agents and must influence the entire network within limited rounds; evaluation metrics are ASR (Attack Success Rate) and influence range.
    - **Design Motivation**: The communication topology of real-world multi-agent systems is not a complete graph—line and star are the most challenging structures.

### Loss & Training

The retrieval suffix is optimized using gradients (GCG-style discrete token search), and the replicative suffix is optimized using a maximum likelihood objective. The two are optimized serially.

## Key Experimental Results

### Main Results

Attack success rate under different topologies (%):

| Method | Graph | Line | Star | 100 Agents |
|------|:---:|:---:|:---:|:---:|
| Single-agent baseline | ~40 | 20.69 | 19.19 | 32.25 |
| ARCJ (Ours) | ~65 | **44.20** | **38.94** | **85.18** |
| Gain | — | +23.51 | +18.95 | **+52.93** |

### Ablation Study

Contributions of each component:

| Configuration | Line ASR | Star ASR |
|------|:---:|:---:|
| No suffix | ~15 | ~12 |
| Retrieval suffix only | 20.69 | 19.19 |
| Retrieval + replicative suffix | **44.20** | **38.94** |

### Key Findings

1. **Toxicity dissipation is the core barrier to multi-agent attacks**: Toxic suffixes gradually disappear during propagation.
2. **Replicative suffix contributes the most**: From 20.69% $\rightarrow$ 44.20% (line), showing a larger contribution compared to the retrieval suffix.
3. **100-agent system is particularly vulnerable**: The baseline is only 32.25%, but ARCJ reaches 85.18%—scaling up unexpectedly makes the system easier to attack.
4. **Reveals contagion risks in multi-agent architectures**: Independent memories do not guarantee security.

## Highlights & Insights

- **Discovery and naming of the "toxicity dissipation" phenomenon**—precisely describing the root cause of multi-agent attack failures.
- **Unique "self-replicating" idea**—allowing toxic information to propagate like a virus among agents.
- **100-agent experimental scale**—the first to validate multi-agent attacks on such a large scale.
- **Important revelation that independent memory $\neq$ safety**.

## Limitations & Future Work

- Only validated on text tasks; multimodal multi-agent systems are not covered.
- Attackers require white-box access to optimize suffixes.
- Defense methods are not fully explored.
- Social topology structures are simple, while real network topologies are more complex.

## Related Work & Insights

- **vs. Single-Agent Memory Attack (Chen et al. 2024)**: Attack only on single agents—Ours extends this to multi-agent propagation.
- **vs. Shared Memory Attack (Ju et al. 2024)**: Shared memory is naturally propagative—Ours solves the propagation challenge in independent memories.
- **vs. GCG (Zou et al. 2023)**: Original adversarial suffix method—Ours incorporates replication capability.
- **Insights**: The security of multi-agent systems needs to be considered from the perspective of propagation dynamics.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The discovery of toxicity dissipation + the design of the self-replicating suffix are highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple topologies + multiple scales + ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition and vivid "Town" metaphor.
- **Value**: ⭐⭐⭐⭐ Reveals the contagion risks of multi-agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning](world_modeling_makes_a_better_planner_dual_preference_optimization_for_embodied_.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](../../NeurIPS2025/llm_alignment/what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [\[ACL 2025\] LLMs Caught in the Crossfire: Malware Requests and Jailbreak Challenges](llms_caught_in_the_crossfire_malware_requests_and_jailbreak_challenges.md)
- [\[ACL 2025\] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [\[ACL 2025\] Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction](rewrite_to_jailbreak_discover_learnable_and_transferable_implicit_harmfulness_in.md)

</div>

<!-- RELATED:END -->
