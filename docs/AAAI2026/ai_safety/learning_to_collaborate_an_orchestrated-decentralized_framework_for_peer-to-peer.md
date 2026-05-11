---
title: >-
  [Paper Note] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning
description: >-
  [AI Safety] This paper proposes KNEXA-FL, a framework that models P2P collaboration as a contextual bandit problem via a Central Pairing Manager (CPM) that never accesses model parameters. Using LinUCB to learn optimal p…
tags:
  - "AI Safety"
date: 2026-05-08
content_hash: d112395d2a5504eb
---

# Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning

- **Conference**: AAAI 2026
- **arXiv**: [2601.17133](https://arxiv.org/abs/2601.17133)
- **Code**: [FujitsuResearch/knexa-fl](https://github.com/FujitsuResearch/knexa-fl)
- **Area**: ai_safety (Federated Learning / Decentralized Collaboration / Privacy Preservation)
- **Keywords**: Decentralized Federated Learning, P2P Collaboration, Knowledge Distillation, Contextual Bandit, LLM Fine-tuning, LoRA, Privacy Preservation

## TL;DR

This paper proposes KNEXA-FL, a framework that models P2P collaboration as a contextual bandit problem via a Central Pairing Manager (CPM) that never accesses model parameters. Using LinUCB to learn optimal pairing strategies, KNEXA-FL achieves approximately 50% higher Pass@1 than random P2P in heterogeneous LLM federated learning, while avoiding the catastrophic collapse observed in centralized distillation.

## Background & Motivation

### Root Cause

Domain-specific fine-tuning of LLMs requires diverse data across organizations, yet data sovereignty and privacy requirements prohibit raw data sharing. Federated learning (FL) offers a solution but introduces a dilemma:

1. **Fragility of centralized FL**: Traditional FL relies on a central aggregator, creating a single point of failure and vulnerability to model inversion attacks that can reconstruct sensitive training data.
2. **Inefficiency of decentralized FL**: Decentralized FL removes the central server but typically degenerates into random or static P2P pairing, ignoring agent heterogeneity and potentially causing negative transfer.

### Paper Goals

Existing work either accepts the security risks of central aggregation or the efficiency losses of random P2P pairing. This paper argues that such a binary framing is overly restrictive, and proposes **orchestrated decentralization**: employing a **non-aggregating Central Pairing Manager** to intelligently orchestrate P2P interactions, thereby reconciling security and efficiency.

## Method

### Overall Architecture: KNEXA-FL

The system comprises three logical components:

- **LLM Agents ($\mathcal{A}$)**: Autonomous entities, each holding a frozen base model $W_0$ and a trainable PEFT module $\phi_i$ (e.g., LoRA), fine-tuned on private non-IID data $D_i$. A local gateway with a Guardrail Filter prevents sensitive data leakage.
- **CPM ($\mathcal{P}$)**: The Central Pairing/Profiling Manager, which receives only abstract profile vectors $\mathbf{p}_i$ and never accesses raw data or model parameters.
- **Secure P2P Channels**: Encrypted channels established between paired agents for ephemeral knowledge exchange.

### Key Design 1: Adaptive Knowledge Distillation (AKD)

AKD is the core knowledge exchange mechanism, employing **text-level distillation** rather than logit-level distillation, which naturally supports heterogeneous models with different architectures and tokenizers:

1. The teacher agent $a_j$ generates text predictions $y_j(x)$ on a shared transfer set $\mathcal{X}_u$.
2. The student agent $a_i$ re-encodes the teacher's text using its own tokenizer to obtain "soft" target sequences $\tilde{y}_j(x)$.
3. The student optimizes a combined loss:

$$\mathcal{L}^{\text{kd}}_{\text{total},i} = (1-\alpha_{\text{kd}})\mathcal{L}_i(D_i) + \alpha_{\text{kd}} \mathbb{E}_{x \in \mathcal{X}_u}\left[\mathcal{L}_{\text{CE}}(\tilde{y}_j(x), p_i(\cdot|x))\right]$$

where the first term is the local training loss, the second term is the cross-entropy loss aligning with teacher outputs, and $\alpha_{\text{kd}}$ controls the distillation weight. Text-level distillation entirely circumvents tokenizer mismatch, making distillation well-defined for any pair of heterogeneous models.

### Key Design 2: LinUCB-Based Intelligent Pairing (CPM)

The CPM models P2P pairing as a **contextual combinatorial bandit problem**. The core steps are:

**Profile Construction**: Each agent transmits a privacy-preserving profile vector $\mathbf{p}_i \in \mathbb{R}^{d_p}$ containing:
- Static features: LLM family, PEFT configuration
- Dynamic features: task performance, perplexity, privacy-preserving embeddings of data distribution
- Historical/trust features: historical interaction success rate, trust score

**Context Vector**: For a candidate pair $(a_i, a_j)$, a context vector is constructed as $\mathbf{x}_{ij}^{(t)} = \varphi(\mathbf{p}_i^{(t)}, \mathbf{p}_j^{(t)}, S_{net}^{(t)})$.

**LinUCB Selection**: The estimated pairing reward $\hat{r}_{ij} = \hat{\boldsymbol{\theta}}^\top \mathbf{x}_{ij}$ is computed, and $K_p$ disjoint pairs are greedily selected based on UCB scores:

$$\text{UCB}_{ij} = \hat{\boldsymbol{\theta}}^\top \mathbf{x}_{ij} + \beta \sqrt{\mathbf{x}_{ij}^\top \mathbf{A}^{-1} \mathbf{x}_{ij}}$$

**Reward Signal**: The receiving agent feeds back a scalar reward based on interaction effectiveness:

$$r_{ij}^{(t)} = \gamma(\mathcal{L}_i^{\text{pre}} - \mathcal{L}_i^{\text{post}}) - \delta \cdot \text{KB}_{ij}^{(t)}$$

The first term captures the reduction in local loss; the second penalizes communication overhead. The CPM updates its bandit parameters $(\mathbf{A}, \mathbf{b})$ accordingly, progressively learning the optimal pairing strategy.

### Protocol Flow (Per Round)

1. **Asynchronous Profiling Phase**: Each agent performs local PEFT updates in parallel and sends its profile to the CPM.
2. **Centralized Pairing Phase**: The CPM computes LinUCB scores and selects the optimal disjoint pairing set.
3. **P2P Exchange Phase**: Paired agents directly exchange knowledge via AKD.
4. **Strategy Update Phase**: Receiving agents compute rewards and feed them back to the CPM, which updates its bandit model.

### Security Design

- **Data Minimization**: Only teacher predictions (text/logits) are exchanged; raw data is never shared.
- **Encrypted Communication**: mTLS with end-to-end payload encryption; the CPM cannot decrypt knowledge packages.
- **Non-Aggregating CPM**: Eliminates the single-point-of-failure risk of centralized aggregation.
- **Learning-Based Governance**: The bandit naturally learns to down-weight malicious or low-quality peers; trust emerges as a property of observed utility.

## Key Experimental Results

### Experimental Setup

- **Task**: Code generation (HumanEval + MBPP combined, 464 problems, 348/116 train/test split)
- **Heterogeneity Simulation**: Dirichlet distribution with $\alpha=0.1$ for data allocation (strongly non-IID)
- **6-Client Federation**: Qwen1.5-0.5B, Cerebras-GPT-590M, bloom-560m, pythia-410m, and other diverse models
- **All models use LoRA**, with trainable parameters accounting for 2.2%–3.0%
- **Evaluation Metrics**: Pass@k (k=1,5,10), CodeBLEU

### Main Results (Table 1)

| Method | Pass@1 (%) | Pass@5 (%) | Pass@10 (%) | CodeBLEU |
|--------|-----------|-----------|------------|---------|
| LocalOnly | 2.22 | 5.42 | 5.55 | 0.260 |
| FedID-CentralKD | 1.11 | 5.56 | 5.56 | 0.181 |
| Central-KD | 2.00 (peak 18.33)† | 7.80 | 10.00 | 0.268 |
| Heuristic-P2P | 6.67 | 16.67 | 27.78 | 0.392 |
| Random-P2P | 8.89 | 22.40 | 27.80 | 0.239 |
| **KNEXA-FL** | **13.33** | **31.25** | **44.44** | **0.344** |

†Central-KD is highly unstable: it peaks at 18.33% before collapsing to 2.00%.

### Ablation Study (Table 2): Pairing Quality (Peak Pass@1 on Transfer Set)

| Pairing Strategy | Peak Student Pass@1 |
|-----------------|-------------------|
| Random-P2P | 33.33% |
| KNEXA-FL (CPM-Guided) | **86.70%** |

CPM-guided pairing enables students to reach 86.70% on the transfer set—2.6× that of random pairing—demonstrating that the CPM successfully identifies highly synergistic knowledge transfer relationships.

## Key Findings

1. **Centralized distillation catastrophically collapses under high heterogeneity**: Central-KD briefly reaches 18.33% Pass@1 before collapsing to 2.00%; forcing heterogeneous models to distill from a single averaged "ensemble teacher" overwrites specialized knowledge.
2. **Naïve heuristic pairing is counterproductive**: Heuristic-P2P (maximizing JS divergence) underperforms Random-P2P (6.67% vs. 8.89%), indicating that pursuing data diversity alone does not constitute effective collaboration.
3. **The CPM learns a non-trivial diversity–compatibility trade-off**: Rather than maximizing heterogeneity, it selects synergistically compatible pairs while maintaining high JS divergence (≈0.64).
4. **Gains benefit the entire federation**: Even the smallest model, pythia-410m, substantially surpasses its isolated training performance, indicating that collaborative gains are not confined to stronger participants.

## Highlights & Insights

- **Elegant problem formulation**: Formalizing P2P pairing as a contextual bandit is a novel contribution; this is the first application of online learning to coordinate federated LLM collaboration.
- **Clever architectural design**: The CPM acts as a "matchmaker" rather than an "aggregator," removing the security risks of central aggregation while retaining the efficiency of intelligent orchestration.
- **High generality of text-level distillation**: Circumventing tokenizer mismatch enables effective knowledge exchange between arbitrary pairs of heterogeneous models.
- **Emergent trust mechanism**: The bandit naturally learns to down-weight low-quality or malicious peers without predefined rules.
- **In synthetic experiments, CPM gain reaches 48.5%** (32-client, high-heterogeneity setting), with performance robustly approaching the oracle upper bound as federation size increases.

## Limitations & Future Work

1. **Limited federation scale**: The main experiments involve only 6 clients; performance and WAN latency effects at large scale (e.g., 100+ clients) remain unvalidated.
2. **Single data partition scheme**: Only the Dirichlet distribution is used to simulate non-IID settings; more realistic semantic-level (e.g., user-profile-level) data partitioning is absent.
3. **Low absolute performance**: The best Pass@1 is only 13.33%, and all base models are small (~500M parameters); effectiveness on mainstream large models (7B+) is unverified.
4. **Incomplete baseline coverage**: Comparisons with advanced centralized FL optimizers such as FedProx and SCAFFOLD are missing.
5. **Security claims are largely theoretical**: Advanced mechanisms such as differential privacy and zero-knowledge proofs are deferred to future work.

## Related Work & Insights

- **Federated/Decentralized Learning**: FedAvg → FedProx/SCAFFOLD (addressing statistical drift but retaining central aggregator) → Gossip Learning (decentralized but random pairing) → IPLS (static one-time grouping) → KNEXA-FL (online learning for continuous orchestration).
- **Parameter-Efficient LLM Federation**: FATE-LLM, FedLoRA (centralized PEFT aggregation) → FedSKD, KD-PDFL (P2P distillation without intelligent pairing) → KNEXA-FL (intelligent pairing + heterogeneous PEFT).
- **Security and Governance**: Byzantine-robust aggregation, reputation/blockchain-based trust → KNEXA-FL (learning-based governance, trust as an emergent property of observed utility).
- **Orchestration and Multi-Agent Systems**: International Data Spaces (IDS), coalition formation → KNEXA-FL (knowledge data spaces + bandit-based coordination).

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: Formulating P2P pairing as a contextual bandit is original; the non-aggregating CPM design is pioneering in the security–efficiency trade-off space.
- **Experimental Thoroughness** ⭐⭐⭐: The heterogeneous setup is well-designed and ablations are thorough, but the federation scale is small, absolute performance is low, and comparisons with mainstream FL methods are lacking.
- **Writing Quality** ⭐⭐⭐⭐: The structure is clear, with a coherent problem–solution–experiment narrative; security analysis and theoretical insights are internally consistent.
- **Value** ⭐⭐⭐⭐: Code is open-sourced, the framework is broadly applicable, and it addresses practical cross-organizational LLM collaboration scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](../../ICLR2026/ai_safety/skirting_additive_error_barriers_for_private_turnstile_streams.md)

</div>

<!-- RELATED:END -->
