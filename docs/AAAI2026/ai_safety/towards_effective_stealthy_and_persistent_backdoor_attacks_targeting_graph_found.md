---
title: >-
  [Paper Note] Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models
description: >-
  [AAAI2026][AI Safety][backdoor attack] This paper proposes GFM-BA, the first systematic backdoor attack method targeting the pre-training phase of Graph Foundation Models (GFMs). It addresses three core challenges — effectiveness, stealthiness, and persistence — through three modules: label-free trigger association, node-adaptive trigger generation, and persistent backdoor anchoring.
tags:
  - AAAI2026
  - AI Safety
  - backdoor attack
  - graph foundation model
  - GNN security
  - trigger generation
  - adversarial ML
date: 2026-05-08
content_hash: 00a2f1c86bfc52e4
---

# Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models

**Conference**: AAAI2026
**arXiv**: [2511.17982](https://arxiv.org/abs/2511.17982)
**Code**: [RingBDStack/GFM-BA](https://github.com/RingBDStack/GFM-BA)
**Area**: AI Security
**Keywords**: backdoor attack, graph foundation model, GNN security, trigger generation, adversarial ML

## TL;DR
This paper proposes GFM-BA, the first systematic backdoor attack method targeting the pre-training phase of Graph Foundation Models (GFMs). It addresses three core challenges — effectiveness, stealthiness, and persistence — through three modules: label-free trigger association, node-adaptive trigger generation, and persistent backdoor anchoring.

## Background & Motivation
GFMs are pre-trained on multi-domain graph data and subsequently adapted to downstream tasks; users commonly deploy open-source pre-trained models directly. This creates a realistic attack surface: an adversary who controls the pre-training stage can inject a backdoor and release the poisoned model.

Fundamental differences between traditional GNN backdoor attacks and the GFM setting:

| Condition | Traditional GNN | GFM |
|---|---|---|
| Downstream labels available | ✓ | ✗ |
| Same-domain train/inference | ✓ | ✗ (cross-domain) |
| Fixed model parameters | ✓ | ✗ (downstream fine-tuning) |

These differences give rise to three core challenges:

**Effectiveness**: Downstream labels are unavailable at pre-training time — how can triggers be designed to induce targeted misclassification?

**Stealthiness**: Node feature distributions vary substantially across domains, making fixed triggers easily detectable by anomaly detection.

**Persistence**: Downstream fine-tuning may erase the injected backdoor behavior (backdoor forgetting).

Limitations of prior work: GCBA requires downstream labels; CrossBA cannot control the target label and degrades to an adversarial evasion attack.

## Method

### Module 1: Label-Free Trigger Association
- Node embeddings of the pre-training graph are extracted using the pre-trained GNN, and $k$ prototype embeddings are selected via **Farthest Point Sampling (FPS)**.
- The greedy strategy of FPS ensures prototypes are spread across the embedding space. Proposition 1 theoretically proves that when inter-class separation is sufficiently large, FPS is more likely to cover multiple downstream classes.
- At downstream injection time, the attacker identifies the prototype corresponding to the desired target label through a small number of probe queries.

### Module 2: Node-Adaptive Trigger Generator
- An MLP dynamically generates trigger features from the target node feature $\mathbf{x}_i$ and target embedding $\mathbf{e}_j$: $\mathbf{x}_{ij}^{tri} = \text{MLP}([\mathbf{x}_i \| \mathbf{e}_j])$
- The trigger is designed as a 3-node fully connected subgraph inserted into the target node's neighborhood.
- Dual-objective optimization: $\mathcal{L}_{eff}$ aligns the triggered node's embedding with the target prototype; $\mathcal{L}_{ste}$ enforces similarity between trigger features and target node features to preserve graph homophily.
- Notably, pre-trained model parameters are not modified; the method exploits latent backdoor logic already present in the encoder.

### Module 3: Persistent Backdoor Anchoring
- Empirical observation: the majority of pre-trained parameters change minimally during downstream fine-tuning.
- Graph mixup is used to synthesize cross-domain graphs that simulate potential downstream distributions.
- Fine-tuning-sensitive parameters are identified via model-pruning-based importance estimation.
- Random perturbations $\theta_k \leftarrow \theta_k + \epsilon|\theta_k|$ are applied to sensitive parameters, and the trigger generator is trained to remain effective under such perturbations.
- Persistence loss: $\mathcal{L}_{per} = \text{Var}(\{\mathcal{L}_{eff}^j\}) + \text{Mean}(\{\mathcal{L}_{eff}^j\})$

## Key Experimental Results

### Attack Effectiveness (ASR %, Target-Controlled Setting)

| Method | Cora | CiteSeer | PubMed | Photo | Computers |
|---|---|---|---|---|---|
| GCBA_M (GCOPE) | 4.77 | 5.98 | 21.65 | 3.48 | 4.62 |
| CrossBA (GCOPE) | 14.29 | 16.67 | 33.33 | 9.25 | 7.98 |
| **GFM-BA (GCOPE)** | **90.40** | **89.06** | **100.00** | **84.53** | **78.54** |
| CrossBA (SAMGPT) | 13.61 | 16.67 | 33.33 | 12.10 | 9.20 |
| **GFM-BA (SAMGPT)** | **100.00** | **100.00** | **100.00** | **99.80** | **100.00** |

Target-Controlled ASR exceeds the strongest baseline (CrossBA) by **66–91%**.

### Stealthiness (ASR After Edge Purification Defense)
GFM-BA maintains high ASR after edge purification (100% on GCOPE), outperforming baselines by an average of **36.81%** (GCOPE), **19.98%** (MDGPT), and **36.73%** (SAMGPT), with no degradation in clean accuracy.

### Persistence (ASR Drop After Fine-Tuning)

| Method | Cora Drop | Photo Drop | Computers Drop |
|---|---|---|---|
| CrossBA (SAMGPT) | ↓4.74 | ↓9.40 | ↓0.60 |
| **GFM-BA (SAMGPT)** | **↓1.34** | **↓4.00** | **↓1.40** |
| CrossBA (MDGPT) | ↓1.36 | ↓4.60 | ↓2.40 |
| **GFM-BA (MDGPT)** | **↓0.68** | **↓0.60** | **↓0.80** |

ASR degradation after fine-tuning is minimal (mostly <2%), demonstrating substantially superior persistence over baselines.

## Highlights & Insights
- **Label-free attack paradigm**: Selecting prototype embeddings via FPS bypasses the dependency on downstream labels, representing a key breakthrough for GFM backdoor attacks.
- **Adaptive trigger generation**: The node-adaptive design preserves graph homophily, significantly enhancing stealthiness.
- **No modification to model parameters**: The method exploits latent logic in the pre-trained encoder without affecting clean accuracy.
- **Theoretical grounding**: Propositions 1 and 2 provide theoretical foundations for FPS coverage and parameter-insensitive anchoring, respectively.

## Limitations & Future Work
- Evaluation is limited to node classification; graph classification and link prediction scenarios are not addressed.
- FPS prototype coverage may degrade under highly imbalanced class distributions.
- The attack assumes the adversary can perform a small number of downstream probe queries to match prototypes to labels, which may be infeasible in certain settings.
- Defense evaluation relies solely on simple edge purification; stronger defenses such as spectral filtering and model pruning are not assessed.
- The optimality of the fixed 3-node trigger structure is not investigated.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic treatment of the three core GFM backdoor attack challenges; the label-free design is a genuine breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 5 datasets × 3 victim GFMs × 3 baselines, with ablation studies and hyperparameter analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation and challenge analysis are clearly articulated; methodological descriptions are rigorous.
- **Value**: ⭐⭐⭐⭐ — Exposes security vulnerabilities in GFMs and advances research in trustworthy AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](../../NeurIPS2025/ai_safety/stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)
- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)
- [\[AAAI 2026\] TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models](toporeformer_mitigating_adversarial_attacks_using_topological_purification_in_oc.md)
- [\[ICLR 2026\] Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](../../ICLR2026/ai_safety/beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)
- [\[NeurIPS 2025\] A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers](../../NeurIPS2025/ai_safety/a_set_of_generalized_components_to_achieve_effective_poison-only_clean-label_bac.md)

</div>

<!-- RELATED:END -->
