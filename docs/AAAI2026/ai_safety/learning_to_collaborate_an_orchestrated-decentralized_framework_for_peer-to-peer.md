---
title: >-
  [Paper Note] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning
description: >-
  [AI Safety] This paper proposes the KNEXA-FL framework, which models P2P collaboration as a contextual bandit problem via a model-agnostic Central Pairwise Matcher (CPM). By using LinUCB to learn the optimal pairing strategy, it achieves an approximately 50% increase in Pass@1 compared to random P2P in heterogeneous LLM federated learning, while avoiding the catastrophic collapse associated with centralized distillation.
tags:
  - "AI Safety"
date: 2026-05-08
content_hash: f96b26b3f5d8ddcb
---

# Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning

- **Conference**: AAAI 2026
- **arXiv**: [2601.17133](https://arxiv.org/abs/2601.17133)
- **Code**: [FujitsuResearch/knexa-fl](https://github.com/FujitsuResearch/knexa-fl)
- **Area**: ai_safety (Federated Learning / Decentralized Collaboration / Privacy Protection)
- **Keywords**: Decentralized Federated Learning, P2P Collaboration, Knowledge Distillation, Contextual Bandit, LLM Fine-tuning, LoRA, Privacy Protection

## TL;DR

This paper proposes the KNEXA-FL framework, which models P2P collaboration as a contextual bandit problem via a model-agnostic Central Pairwise Matcher (CPM). By using LinUCB to learn the optimal pairing strategy, it achieves an approximately 50% increase in Pass@1 compared to random P2P in heterogeneous LLM federated learning, while avoiding the catastrophic collapse associated with centralized distillation.

## Background & Motivation

### Key Challenge

Domain fine-tuning of LLMs requires diverse data across organizations, but data sovereignty and privacy requirements prohibit raw data sharing. Federated Learning (FL) provides a formulation to solve this, but faces a dilemma:

1. **Vulnerability of Centralized FL**: Traditional FL relies on a central aggregator, introducing single-point-of-failure challenges and vulnerability to model inversion attacks (which can reconstruct sensitive training data).
2. **Inefficiency of Decentralized FL**: Decentralized FL eliminates the central server but typically degenerates into random or static P2P pairing, ignoring agent heterogeneity and leading to potential negative transfer.

### Paper Positioning

Existing works either accept the security risks of centralized aggregation or accept the efficiency losses of random P2P. This paper argues that this binary opposition is too restrictive and proposes **orchestrated decentralization**: utilizing a **non-aggregating central pairwise matcher** to intelligently orchestrate P2P interactions, balancing both security and efficiency.

## Method

### Overall Architecture: KNEXA-FL

The system consists of three logical components:

- **LLM Agents ($\mathcal{A}$)**: Autonomous entities, each holding a frozen base model $W_0$ and a trainable PEFT module $\phi_i$ (e.g., LoRA), fine-tuned on private non-IID data $D_i$. A local gateway includes a Guardrail Filter to prevent sensitive data leakage.
- **CPM ($\mathcal{P}$)**: Central Pairwise Matcher / Profiler, which only receives abstract profiles $\mathbf{p}_i$ and never accesses raw data or model parameters.
- **Secure P2P Channel**: An encrypted channel established between paired agents for transient knowledge exchange.

### Key Designs

#### Design 1: Adaptive Knowledge Distillation (AKD)

AKD is the core knowledge exchange mechanism. It uses **text-level distillation** instead of logit-level distillation, natively supporting heterogeneous models (with different architectures and tokenizers):

1. The teacher agent $a_j$ generates a text prediction $y_j(x)$ on the shared transfer set $\mathcal{X}_u$.
2. The student agent $a_i$ re-encodes the teacher's text using its own tokenizer to obtain a "soft" target sequence $\tilde{y}_j(x)$.
3. The student optimizes the joint fusion loss:

$$\mathcal{L}^{\text{kd}}_{\text{total},i} = (1-\alpha_{\text{kd}})\mathcal{L}_i(D_i) + \alpha_{\text{kd}} \mathbb{E}_{x \in \mathcal{X}_u}\left[\mathcal{L}_{\text{CE}}(\tilde{y}_j(x), p_i(\cdot|x))\right]$$

where the first term is the local training loss, the second term is the cross-entropy loss for aligning with the teacher's output, and $\alpha_{\text{kd}}$ controls the distillation weight. Text-level distillation completely bypasses the tokenizer mismatch issue, making distillation between any model pair well-defined.

#### Design 2: LinUCB-based Intelligent Pairing (CPM)

The CPM models P2P pairing as a **contextual combinatorial bandit problem**. The core steps are:

**Profile Construction**: Each agent sends a privacy-preserving profile vector $\mathbf{p}_i \in \mathbb{R}^{d_p}$, which includes:
- Static features: LLM family, PEFT configuration.
- Dynamic features: Task performance, perplexity, privacy-preserving embeddings of data distribution.
- Historical/Trust features: Historical interaction success rate, trust scores.

**Context Vector**: For a potential pairing $(a_i, a_j)$, the context is constructed as $\mathbf{x}_{ij}^{(t)} = \varphi(\mathbf{p}_i^{(t)}, \mathbf{p}_j^{(t)}, S_{net}^{(t)})$.

**LinUCB Selection**: Estimate the pairing reward $\hat{r}_{ij} = \hat{\boldsymbol{\theta}}^\top \mathbf{x}_{ij}$, and greedily select $K_p$ disjoint pairings based on UCB scores:

$$\text{UCB}_{ij} = \hat{\boldsymbol{\theta}}^\top \mathbf{x}_{ij} + \beta \sqrt{\mathbf{x}_{ij}^\top \mathbf{A}^{-1} \mathbf{x}_{ij}}$$

**Reward Signal**: The receiver feedback a scalar reward based on the interaction utility:

$$r_{ij}^{(t)} = \gamma(\mathcal{L}_i^{\text{pre}} - \mathcal{L}_i^{\text{post}}) - \delta \cdot \text{KB}_{ij}^{(t)}$$

The first term is the reduction in local loss, and the second term penalizes communication overhead. The CPM uses this to update the bandit parameters $(\mathbf{A}, \mathbf{b})$, progressively learning the optimal pairing strategy.

### Protocol Flow (Per Round)

1. **Asynchronous Profiling Phase**: Each agent performs local PEFT updates in parallel, generates its profile, and sends it to the CPM.
2. **Centralized Pairing Phase**: The CPM computes LinUCB scores and selects the optimal disjoint pairing set.
3. **P2P Exchange Phase**: Paired agents directly exchange knowledge via AKD.
4. **Policy Update Phase**: Receivers calculate rewards and feed them back to the CPM, which updates the bandit model.

### Security Design

- **Data Minimization**: Only teacher predictions (text/logits) are exchanged; raw data is never shared.
- **Encrypted Communication**: mTLS + end-to-end payload encryption prevents the CPM from decrypting knowledge packets.
- **Non-aggregating CPM**: Eliminates the single point of failure risk inherent in centralized aggregation.
- **Learning-based Governance**: The bandit naturally learns to downweight malicious or low-quality peers, making trust an emergent property of observed utility.

## Key Experimental Results

### Experimental Setup

- **Task**: Code generation (combined HumanEval + MBPP, 464 questions, 348/116 train/test split)
- **Heterogeneity Simulation**: Dirichlet distribution with $\alpha=0.1$ for training data allocation (highly non-IID)
- **6-Client Federation**: Different models including Qwen1.5-0.5B, Cerebras-GPT-590M, bloom-560m, pythia-410m, etc.
- **LoRA is used for all**, with trainable parameters accounting for 2.2%–3.0%.
- **Evaluation Metrics**: Pass@k (k=1,5,10), CodeBLEU

### Main Results

**Table 1: Main Experimental Results**

| Method | Pass@1 (%) | Pass@5 (%) | Pass@10 (%) | CodeBLEU |
|------|-----------|-----------|------------|---------|
| LocalOnly | 2.22 | 5.42 | 5.55 | 0.260 |
| FedID-CentralKD | 1.11 | 5.56 | 5.56 | 0.181 |
| Central-KD | 2.00 (Peak 18.33)† | 7.80 | 10.00 | 0.268 |
| Heuristic-P2P | 6.67 | 16.67 | 27.78 | 0.392 |
| Random-P2P | 8.89 | 22.40 | 27.80 | 0.239 |
| **KNEXA-FL** | **13.33** | **31.25** | **44.44** | **0.344** |

†Central-KD is highly unstable, collapsing to 2.00% after peaking at 18.33%.

### Ablation Study

**Table 2: Pairing Quality Ablation (Peak Pass@1 on the Transfer Set)**

| Pairing Strategy | Peak Student Pass@1 |
|---------|---------------|
| Random-P2P | 33.33% |
| KNEXA-FL (CPM-guided) | **86.70%** |

CPM-guided pairing enables the student to achieve 86.70% on the transfer set, which is 2.6 times higher than random pairing, indicating that CPM can discover highly synergistic knowledge transfer relationships.

## Key Findings

1. **Centralized Distillation Collapses Catastrophically Under High Heterogeneity**: Central-KD briefly reached 18.33% Pass@1 before collapsing to 2.00%—forcing heterogeneous models to distill from a single, averaged "ensemble teacher" overwrites specialized knowledge.
2. **Naïve Heuristic Pairing Can Be Harmful**: Heuristic-P2P (maximizing Jensen-Shannon divergence) underperforms compared to Random-P2P (6.67% vs 8.89%), demonstrating that purely pursuing data diversity does not equate to effective collaboration.
3. **CPM Learns a Non-Trivial Diversity-Compatibility Trade-off**: Rather than simply maximizing heterogeneity, CPM selects synergistic and compatible pairings while maintaining a high JS divergence ($\approx0.64$).
4. **Gains Benefit the Entire Federation**: Even the smallest model, Pythia-410m, significantly outperformed its isolated training performance, showing that the collaboration gains are not limited to the strongest models.

## Highlights & Insights

- **Elegant Problem Modeling**: Formulates the P2P pairing problem as a contextual bandit, addressing the coordination of federated LLMs using online learning for the first time.
- **Ingenious Architecture Design**: The CPM acts solely as a "matchmaker" rather than a "regulator," removing the security risks of centralized aggregation while preserving the efficiency of intelligent orchestration.
- **High Versatility of Text-Level Distillation**: Bypasses tokenizer mismatches, allowing arbitrary heterogeneous model pairs to effectively exchange knowledge.
- **Emergent Trust Mechanism**: The bandit naturally learns to downweight low-quality or malicious peers without requiring pre-defined rules.
- **CPM Gains Reach up to 48.5% in Synthetic Experiments** (32-client highly heterogeneous scenario), with performance robustly approaching the oracle upper bound as the federation scales up.

## Limitations & Future Work

1. **Limited Scale of Federation**: The main experiments only involve 6 clients, without yet validating practical performance and WAN latency impact under large-scale (e.g., 100+) federations.
2. **Single Data Partitioning Approach**: Only the Dirichlet distribution is used to simulate non-IID distributions, lacking more realistic semantic-level (e.g., user-profile-level) data partitioning.
3. **Low Absolute Performance**: The best Pass@1 is only 13.33%. The base models are all small models (~500M params), and effectiveness on mainstream large models (7B+) is not yet validated.
4. **Insufficient Baseline Coverage**: No comparison is made with advanced centralized FL optimizers such as FedProx and SCAFFOLD.
5. **Theoretical Security Claims**: Advanced security mechanisms such as differential privacy and zero-knowledge proofs are left for future work.

## Related Work & Insights

- **Federated/Decentralized Learning**: FedAvg $\rightarrow$ FedProx/SCAFFOLD (resolving statistical drift but retaining a central aggregator) $\rightarrow$ Gossip Learning (decentralized but with random pairing) $\rightarrow$ IPLS (static one-time grouping) $\rightarrow$ KNEXA-FL (continuous orchestration with online learning)
- **Parameter-Efficient LLM Federated Learning**: FATE-LLM, FedLoRA (centralized PEFT aggregation) $\rightarrow$ FedSKD, KD-PDFL (P2P distillation without intelligent pairing) $\rightarrow$ KNEXA-FL (intelligent pairing + heterogeneous PEFT)
- **Security & Governance**: Byzantine-robust aggregation, reputation/blockchain-based trust $\rightarrow$ KNEXA-FL (learning-based governance, trust as an emergent property of observed utility)
- **Orchestration & Multi-Agent Systems**: International Data Spaces (IDS), coalition formation $\rightarrow$ KNEXA-FL (knowledge data spaces + bandit coordination)

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty** ⭐⭐⭐⭐: The idea of modeling P2P pairing as a contextual bandit is highly novel, and the non-aggregating CPM design makes pioneering trade-offs between security and efficiency.
- **Experimental Thoroughness** ⭐⭐⭐: The heterogeneous setup is reasonably designed and thoroughly ablated, but the federation scale is small, absolute performance is low, and comparisons with mainstream FL methods are lacking.
- **Writing Quality** ⭐⭐⭐⭐: The structure is clear, with a complete problem-solution-experiment logic, and self-consistent security analysis and theoretical insights.
- **Value** ⭐⭐⭐⭐: The code is open-source and the framework is highly generic, making it applicable to practical scenarios of cross-organization LLM collaboration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[ICML 2025\] Retraining with Predicted Hard Labels Provably Increases Model Accuracy](../../ICML2025/ai_safety/retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)

</div>

<!-- RELATED:END -->
