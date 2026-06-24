---
title: >-
  [Paper Note] Distributed and Decentralised Training: Technical Governance Challenges in a Shifting AI Landscape
description: >-
  [ICML 2025][AI Safety][Distributed training] This paper systematically distinguishes between two emerging paradigms: distributed training (multi-data centre) and decentralised training (community-driven). It analyzes how low-communication training algorithms (such as DiLoCo) enable these two paradigms and delves into the challenges and opportunities they present for technical AI governance (compute structuring, capability proliferation, and shut-down capability).
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Distributed training"
  - "decentralised training"
  - "compute governance"
  - "low-communication algorithms"
  - "AI policy"
date: 2026-05-08
content_hash: 6f1c0b76b1f1c7ab
---

# Distributed and Decentralised Training: Technical Governance Challenges in a Shifting AI Landscape

**Conference**: ICML 2025  
**arXiv**: [2507.07765](https://arxiv.org/abs/2507.07765)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Distributed training, decentralised training, compute governance, low-communication algorithms, AI policy

## TL;DR

This paper systematically distinguishes between two emerging paradigms: distributed training (multi-data centre) and decentralised training (community-driven). It analyzes how low-communication training algorithms (such as DiLoCo) enable these two paradigms and delves into the challenges and opportunities they present for technical AI governance (compute structuring, capability proliferation, and shut-down capability).

## Background & Motivation

Current frontier LLM training highly relies on centralized large-scale data centres. The power required for a single training run has exceeded 100 megawatts and is projected to grow to over 5 gigawatts by 2030. This centralized setup was once viewed as a natural chokepoint for AI governance—easy to detect and concentrated in the hands of a few entities.

However, recent algorithmic advances are disrupting this landscape:

**Energy bottlenecks driving distribution**: Building a single hyper-scale cluster faces permit and energy supply bottlenecks, prompting hyper-scalers to adopt multi-data centre training (e.g., GPT-4.5 and Gemini-1.5 are already trained across data centres).

**The rise of decentralised AI**: Since early 2024, startups such as Prime Intellect and Nous Research have raised approximately $145 million, aiming to train open-source models that compete with frontier models using community-contributed compute.

**Lagging policy discourse**: In policy discourse, "distributed" and "decentralised" are frequently conflated and lack precise definitions, severely hindering effective governance discussions.

The core motivation of this paper is to provide a clear conceptual framework for the AI policy community to understand the technical foundations, differences, and governance implications of these two training paradigms.

## Method

### Overall Architecture

The paper proposes a two-dimensional analytical framework (see Figure 1 in the paper), classifying training paradigms into four quadrants based on two dimensions: **number of parties** and **number of compute locations**:

| Quadrant | Parties | Compute Locations | Representative Cases |
|------|--------|---------|---------|
| Centralized | Single | Single | Grok 3 (xAI) |
| Distributed | Single | Multiple | GPT-4.5 (OpenAI) |
| Decentralised | Multiple | Multiple | INTELLECT-1 |
| Hypothetical | Multiple | Single | No instances yet |

**Definitions of Core Terms**:

- **Distributed Training**: Training across multiple physically dispersed compute pools, but coordinated by a centralized entity. A more precise term would be "multi-data centre training" or "geographically distributed training".
- **Decentralised Training**: Utilizing community-contributed compute resources, without a centralized coordinating entity.

### Key Designs

#### 1. Low-Communication Data-Parallel Training Algorithms

Traditional data parallelism requires gradient synchronization after each training batch, which incurs massive communication overhead. Taking an internet connection as an example: if training a single batch takes 4 seconds and synchronization takes 60 seconds, the GPU utility is only 6.25%.

The core breakthrough of **DiLoCo (Distributed Low Communication) and subsequent algorithms**:

- **Dual-optimizer architecture**: A local optimizer (such as an SGD variant) handles local updates, and a global optimizer adjusts global parameters after periodic synchronization.
- **500x reduction in synchronization frequency**: Reducing synchronization from every step to once every 500 steps.
- Taking internet training as an example again, synchronizing once every 500 steps (~33 minutes) increases GPU utility from 6.25% to **over 97%**.

Summary of key algorithmic developments:

| Algorithm | Core Mechanism | Communication Reduction | Key Features |
|------|---------|---------|---------|
| DiLoCo | Dual optimizers + sparse synchronization | ~500× | Robust to asynchronous and heterogeneous data |
| Async Local-SGD | Asynchronous local SGD | Significant | Adaptable to unstable connections |
| DeMo | Decoupled momentum optimization | Significant | Reduces transmitted data volume |
| Streaming DiLoCo | Overlapping communication and computation | ~500× | Close to a "free lunch" |
| Eager Updates | Eager update strategy | ~500× | Further optimizes DiLoCo |

Important additional features of these algorithms:

- **Robustness to asynchronous communication**: Robust to scenarios where some replicas fail to participate in synchronization, making them suitable for unstable network connections.
- **Robustness to heterogeneous data distribution**: Performance does not degrade when different workers train on data from different distributions—a key to unlocking private data.
- **Effect scaling with model size**: Charles et al. (2025) found that under certain settings, DiLoCo even outperforms traditional data-parallel training, and this advantage grows with model size.

#### 2. Peer-to-Peer (P2P) Communication Architectures

Decentralised training further eliminates centralized bottlenecks:

- **No centralized synchronization node**: Model synchronization is achieved through direct node-to-node communication (peer-to-peer), rather than through a central parameter server.
- **Trustless participation**: New nodes can join the training using cryptographic proofs without requiring centralized authorization.
- **Fault tolerance and dynamic reorganization**: Single GPU failures do not interrupt training; the network automatically reorganizes and accepts replacement GPUs.
- **Architecture closer to blockchain**: Such as Ethereum's decentralized structure, rather than traditional data parallelism.

Prime Intellect and Nous Research have successfully pre-trained a 10-billion parameter model (INTELLECT-1) on GPUs across different continents at standard internet speeds.

#### 3. Natural Fit Between Reasoning Models and Decentralisation

The paper provides an in-depth analysis of why reasoning models are particularly suited for decentralized environments:

- **Traditional SSL training**: 1:1 ratio of forward to backward propagation; every forward pass requires weight updates and synchronization.
- **RL post-training**: The ratio of forward to backward propagation can reach **1000:1**; the model explores a large number of "thinking trajectories" before performing a single weight update.
- **Lowered hardware barrier**: The computation and memory requirements for generating thinking trajectories are lower than those for full backpropagation; consumer-grade hardware (such as Apple M3 Ultra, 512GB RAM) can run the inference of DeepSeek-R1 (671B parameters).
- **Further reduced communication demand**: Only thinking trajectories and reward values need to be transmitted, avoiding the transfer of full gradients.

### Loss & Training

As this is a governance analysis paper, it does not propose a new loss function. However, the analysis of training strategies shows that:

- DiLoCo employs an **outer optimizer** to periodically correct global parameters, systemically guaranteeing the stability of model convergence under sparse synchronization conditions.
- In decentralized scenarios, each node independently runs a local optimizer, ensuring the credibility of compute contributions through cryptographic verification (such as the TopLoc locality-sensitive hashing scheme).
- Post-training RL strategies such as GRPO are naturally suited for low-bandwidth environments because the generation phase is an "inference-only" operation.

## Key Experimental Results

### Main Results

This is a position paper and does not contain traditional experiments. The core empirical evidence is sourced from prior work:

| Training Paradigm | Model | Scale | Communication Conditions | Conclusion |
|---------|------|------|---------|------|
| Decentralised Pre-training | INTELLECT-1 | 10B | Cross-continent / Internet | Feasibility proven, but performance has not yet caught up with models of the same scale |
| Distributed Training | GPT-4.5 | Undisclosed | Multi-data centre | Already adopted by frontier models |
| Distributed Training | Gemini-1.5 | Undisclosed | Multi-data centre | Already adopted by frontier models |
| Precedent: Distributed Computing | Folding@Home | 280,000 GPUs | Internet | Peak compute exceeded $10^{18}$ FLOPS |

### Ablation Study

Analysis of the impact of communication frequency on GPU utility:

| Synchronization Frequency | Connection Type | GPU Utility | Note |
|---------|---------|---------|------|
| Every 1 step (Traditional) | High-speed interconnect / Single cluster | ~80% | Traditional data parallelism, communication overhead ~20% |
| Every 1 step | Internet | **6.25%** | Cross-geographic training completely infeasible |
| Every 500 steps (DiLoCo) | Internet | **>97%** | Low-communication algorithms make cross-geographic training feasible |

### Key Findings

1. **Scaling effects of low-communication algorithms**: The performance of DiLoCo and subsequent methods improves as model size increases, suggesting they can keep pace with the scaling paradigm.
2. **Explosive growth of the decentralised startup ecosystem**: Since early 2024, at least 6 startups have raised approximately $145 million, aiming to train o3-grade models.
3. **INTELLECT-1 synchronization interval**: Synchronization occurs every 38 minutes and can be deliberately extended to obfuscate communication patterns.
4. **Post-training is better suited for decentralisation**: The high forward-to-backward propagation ratio of RL post-training naturally reduces communication demands.

## Highlights & Insights

1. **Highly valuable conceptual clarification**: Clearly distinguishing between distributed vs. decentralised establishes a foundation for policy discussions. Previously, these two terms were frequently conflated in policy literature, leading to governance measures lacking precise targeting.
2. **Balanced double-edged sword perspective**: The paper analyzes governance challenges (structured compute evading regulation, capability proliferation, and lack of shut-down capabilities) while acknowledging the positive values of decentralisation (privacy-preserving training unlocking more data, and mitigating the concentration of power).
3. **Unique analysis of reasoning models**: It provides a profound insight into the natural alignment between the RL post-training of reasoning models and decentralized training, an analysis that is rare in existing literature.
4. **"Marginal risk" framework**: Adopting the marginal risk analysis framework from Kapoor et al., it argues that if decentralized AI lags significantly behind the frontier and defensive technologies are sufficient, government intervention may not be necessary.

## Limitations & Future Work

1. **Lack of quantitative analysis**: The paper lacks a concrete estimation of the total compute capacity that decentralized training pools can aggregate, relying only on a rough analogy to Folding@Home.
2. **Superficial safety discussion**: There is a lack of discussion on how safety alignment is executed in decentralized training (e.g., decentralized collection of human feedback in RLHF).
3. **Limited technical depth**: The analysis of algorithms like DiLoCo remains at an overview level, without addressing technical details such as theoretical guarantees of convergence or specific communication compression schemes.
4. **Lack of a game-theoretic perspective**: The strategic interaction between hyper-scalers and decentralized communities is not analyzed—would the former take actions to suppress the latter?
5. **Insufficient international governance dimension**: Issues such as jurisdictional conflicts in cross-border decentralized training and international coordination mechanisms receive little attention.
6. **Limited empirical cases**: INTELLECT-1 and INTELLECT-2 are currently the only case studies of decentralized large models, and their performance remains questionable, leaving the generalizability of the conclusions to be validated.

## Related Work & Insights

- **DiLoCo series** (Douillard et al., 2023; 2025): The algorithmic foundation of low-communication training, enabling cross-geographic training.
- **Sastry et al., 2024**: A theoretical framework for compute governance, upon which this paper discusses the impact of new paradigms on governance assumptions.
- **Kapoor et al., 2024**: The marginal risk framework, providing a methodological tool to evaluate the risks and benefits of decentralized AI.
- **Seferis & Fist, 2025**: A concrete technical proposal for structured compute detection.
- **Hivemind** (Ryabinin et al., 2020): A decentralized deep learning framework in PyTorch, which is part of the technical foundation of INTELLECT-1.
- **Federated Learning and Privacy Protection** (Sani et al., 2024): The data privacy direction of decentralized pre-training.

**Inspiration for subsequent research**: The governance challenges of decentralized training may catalyze a new paradigm of "on-chain AI governance" (combining blockchain and AI safety), an interdisciplinary direction that warrants attention.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ★★★★☆ | First to systematically clarify the differences between the two training paradigms for the AI policy community |
| Technical Depth | ★★★☆☆ | Overview nature, lacking algorithmic details, but policy analysis is thorough |
| Practicality | ★★★★☆ | Directly useful for policymakers and governance researchers |
| Clarity | ★★★★★ | Excellent writing, clear conceptual definitions, and rigorous argumentative structure |
| Overall Rating | ★★★★☆ | High-quality governance analysis paper, filling an important gap in AI policy |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Challenges to the Interpretation of Disaggregated Evaluations of AI](../../NeurIPS2025/ai_safety/understanding_challenges_to_the_interpretation_of_disaggregated_evaluations_of_a.md)
- [\[ICML 2026\] Position: 'AI Alignment' Encompasses Competing Technical Priorities](../../ICML2026/ai_safety/ai_alignment_encompasses_competing_technical_priorities.md)
- [\[NeurIPS 2025\] Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification](../../NeurIPS2025/ai_safety/keep_it_real_challenges_in_attacking_compression-based_adversarial_purification.md)
- [\[ECCV 2024\] Resilience of Entropy Model in Distributed Neural Networks](../../ECCV2024/ai_safety/resilience_of_entropy_model_in_distributed_neural_networks.md)
- [\[ICML 2025\] Identifying and Understanding Cross-Class Features in Adversarial Training](identifying_and_understanding_cross-class_features_in_adversarial_training.md)

</div>

<!-- RELATED:END -->
