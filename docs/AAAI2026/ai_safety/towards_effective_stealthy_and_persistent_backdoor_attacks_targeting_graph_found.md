---
title: >-
  [Paper Note] Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models
description: >-
  [AAAI2026][AI Safety][backdoor attack] Proposes GFM-BA, the first systematic backdoor attack method targeting the pre-training phase of Graph Foundation Models (GFMs). By employing three modules—label-free trigger association, node-adaptive trigger generation, and persistent backdoor anchoring—it simultaneously addresses the three major challenges of effectiveness, stealthiness, and persistence.
tags:
  - "AAAI2026"
  - "AI Safety"
  - "backdoor attack"
  - "graph foundation model"
  - "GNN security"
  - "trigger generation"
  - "adversarial ML"
date: 2026-05-08
content_hash: 0a510e3a362790e8
---

# Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models

**Conference**: AAAI2026  
**arXiv**: [2511.17982](https://arxiv.org/abs/2511.17982)  
**Code**: [RingBDStack/GFM-BA](https://github.com/RingBDStack/GFM-BA)  
**Area**: AI Security  
**Keywords**: backdoor attack, graph foundation model, GNN security, trigger generation, adversarial ML

## TL;DR
Proposes GFM-BA, the first systematic backdoor attack method targeting the pre-training phase of Graph Foundation Models (GFMs). By employing three modules—label-free trigger association, node-adaptive trigger generation, and persistent backdoor anchoring—it simultaneously addresses the three major challenges of effectiveness, stealthiness, and persistence.

## Background & Motivation
GFMs are pre-trained on multi-domain graph data and then adapted to downstream tasks. Users routinely utilize open-source pre-trained models directly. This introduces a realistic threat surface for backdoor attacks: attackers control the pre-training phase, inject backdoors, and then release the backdoored models.

Fundamental differences between traditional GNN backdoor attacks and the GFM scenario:

| Condition | Traditional GNN | GFM |
|---|---|---|
| Downstream labels available | ✓ | ✗ |
| In-domain training/inference | ✓ | ✗ (Cross-domain) |
| Fixed model parameters | ✓ | ✗ (Downstream fine-tuning) |

This raises three key challenges:

**Effectiveness**: Downstream labels are unknown during pre-training. How can one ensure that the trigger causes the target classification error?

**Stealthiness**: Node feature distributions differ significantly across different domains. Thus, fixed triggers are easily detected by anomaly detection.

**Persistence**: Downstream fine-tuning may erase the backdoor behavior (backdoor forgetting).

Limitations of prior work: GCBA requires downstream labels; CrossBA cannot control target labels and degrades to adversarial evasion attacks.

## Method

### Module 1: Label-Free Trigger Association
- Uses a pre-trained GNN to extract node embeddings of the pre-training graph, and selects $k$ prototype embeddings via **Farthest Point Sampling (FPS)**.
- The greedy strategy of FPS ensures that the prototypes dispersedly cover the embedding space. Theoretical proof (Proposition 1) demonstrates that when inter-class separation is sufficiently large, FPS is more likely to cover multiple downstream classes.
- During downstream injection, the attacker performs a small number of probing queries to map the target label to the corresponding prototype.

### Module 2: Node-Adaptive Trigger Generator
- An MLP is used to dynamically generate trigger features based on the target node features $\mathbf{x}_i$ and the target embedding $\mathbf{e}_j$: $\mathbf{x}_{ij}^{tri} = \text{MLP}([\mathbf{x}_i \| \mathbf{e}_j])$.
- The trigger is designed as a 3-node fully connected subgraph inserted into the neighborhood of the target node.
- Bi-objective optimization: $\mathcal{L}_{eff}$ ensures that the triggered node embedding aligns with the target prototype; $\mathcal{L}_{ste}$ ensures the trigger features are similar to the target node features (preserving graph homophily).
- Key Insight: It does not modify the pre-trained model parameters, leveraging the latent backdoor logic already present in the encoder.

### Module 3: Persistent Backdoor Anchoring
- Experimental observation: Most pre-trained parameters change minimally during downstream fine-tuning.
- Uses graph mixup to synthesize cross-domain graphs to simulate potential downstream distributions.
- Identifies fine-tuning-sensitive parameters based on model-pruning importance estimation.
- Applies random perturbations $\theta_k \leftarrow \theta_k + \epsilon|\theta_k|$ to the sensitive parameters, training the trigger generator to remain effective under these perturbations.
- Persistence loss: $\mathcal{L}_{per} = \text{Var}(\{\mathcal{L}_{eff}^j\}) + \text{Mean}(\{\mathcal{L}_{eff}^j\})$

## Key Experimental Results

### Attack Effectiveness (ASR %, Target-Controlled Scenario)

| Method | Cora | CiteSeer | PubMed | Photo | Computers |
|---|---|---|---|---|---|
| GCBA_M (GCOPE) | 4.77 | 5.98 | 21.65 | 3.48 | 4.62 |
| CrossBA (GCOPE) | 14.29 | 16.67 | 33.33 | 9.25 | 7.98 |
| **GFM-BA (GCOPE)** | **90.40** | **89.06** | **100.00** | **84.53** | **78.54** |
| CrossBA (SAMGPT) | 13.61 | 16.67 | 33.33 | 12.10 | 9.20 |
| **GFM-BA (SAMGPT)** | **100.00** | **100.00** | **100.00** | **99.80** | **100.00** |

Target-Controlled ASR obtains a **66-91% gain** over the strongest baseline, CrossBA.

### Stealthiness (ASR after Edge Purification)
GFM-BA maintains a high ASR (100% on GCOPE) even after edge purification defense, outperforming baselines on average by **36.81%** (GCOPE), **19.98%** (MDGPT), and **36.73%** (SAMGPT). Clean accuracy does not decrease.

### Persistence (ASR Drop after Fine-Tuning)

| Method | Cora Drop | Photo Drop | Computers Drop |
|---|---|---|---|
| CrossBA (SAMGPT) | ↓4.74 | ↓9.40 | ↓0.60 |
| **GFM-BA (SAMGPT)** | **↓1.34** | **↓4.00** | **↓1.40** |
| CrossBA (MDGPT) | ↓1.36 | ↓4.60 | ↓2.40 |
| **GFM-BA (MDGPT)** | **↓0.68** | **↓0.60** | **↓0.80** |

ASR drops minimally after fine-tuning (mostly <2%), demonstrating significantly better persistence than the baselines.

## Highlights & Insights
- **Label-free attack paradigm**: Bypasses dependency on downstream labels by selecting prototype embeddings through FPS, which is a key breakthrough in GFM backdoor attacks.
- **Adaptive trigger generation**: The node-adaptive design preserves graph homophily, significantly improving stealthiness.
- **No modification to model parameters**: Leverages the latent logic of the pre-trained encoder, leaving clean utility unaffected.
- **Theoretical support**: Propositions 1 and 2 provide theoretical foundations for FPS coverage and parameter-insensitivity anchoring, respectively.

## Limitations & Future Work
- Only node classification tasks are validated; graph classification and link prediction scenarios are not explored.
- The coverage of FPS prototypes might fail under highly imbalanced class distributions.
- The attack assumption requires a small number of downstream probing queries to map prototypes to labels, which might be impractical in some scenarios.
- Defense evaluation is limited to simple edge purification, lacking tests against stronger defenses like spectral filtering and model pruning.
- The optimality of the 3-node fixed trigger structure is not investigated.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The first to systematically address the three major challenges of GFM backdoor attacks, with a groundbreaking label-free design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 5 datasets x 3 victim GFMs x 3 baselines, including ablation studies and hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐ — The motivation and challenges are clearly analyzed, and the method description is rigorous.
- Value: ⭐⭐⭐⭐ — Reveals major security risks in GFMs, promoting trustworthy AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Stealthy and Effective Backdoor Attacks on Lane Detection: A Naturalistic Data Poisoning Approach](../../CVPR2026/ai_safety/towards_stealthy_and_effective_backdoor_attacks_on_lane_detection_a_naturalistic.md)
- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)
- [\[CVPR 2026\] Unleashing Stealthy Backdoor Pandemic by Infecting a Single Diffusion Model](../../CVPR2026/ai_safety/unleashing_stealthy_backdoor_pandemic_by_infecting_a_single_diffusion_model.md)
- [\[CVPR 2026\] DASH: A Meta-Attack Framework for Synthesizing Effective and Stealthy Adversarial Examples](../../CVPR2026/ai_safety/dash_a_meta-attack_framework_for_synthesizing_effective_and_stealthy_adversarial.md)
- [\[ICLR 2026\] TrojanTO: Action-Level Backdoor Attacks Against Trajectory Optimization Models](../../ICLR2026/ai_safety/trojanto_action-level_backdoor_attacks_against_trajectory_optimization_models.md)

</div>

<!-- RELATED:END -->
