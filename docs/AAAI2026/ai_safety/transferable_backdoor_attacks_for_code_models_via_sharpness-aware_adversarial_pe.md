---
title: >-
  [Paper Note] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation
description: >-
  [AAAI 2026][AI Safety][backdoor attack] This paper proposes STAB (Sharpness-aware Transferable Adversarial Backdoor), which trains a surrogate model via SAM to converge to flat regions of the loss landscape and employs G…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "backdoor attack"
  - "code models"
  - "transferability"
  - "Sharpness-Aware Minimization"
  - "adversarial perturbation"
date: 2026-05-08
content_hash: dcb32036474d284a
---

# Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation

**Conference**: AAAI 2026
**arXiv**: [2602.11213](https://arxiv.org/abs/2602.11213)  
**Code**: [github.com/ChangShuyu/STAB](https://github.com/ChangShuyu/STAB)  
**Area**: AI Security
**Keywords**: backdoor attack, code models, transferability, Sharpness-Aware Minimization, adversarial perturbation

## TL;DR

This paper proposes STAB (Sharpness-aware Transferable Adversarial Backdoor), which trains a surrogate model via SAM to converge to flat regions of the loss landscape and employs Gumbel-Softmax optimization to generate context-aware adversarial triggers. STAB is the first approach to simultaneously achieve cross-dataset transferability and stealthiness in backdoor attacks against code models.

## Background & Motivation

### Problem Definition

Pre-trained code models (e.g., PLBART, CodeT5) have become critical components of the modern software supply chain. Because their training data originates from public code repositories, these models are susceptible to backdoor attacks via data poisoning—producing malicious outputs when a trigger is present while behaving normally on clean inputs.

### Root Cause: Transferability vs. Stealthiness Trade-off

Existing code backdoor attacks face a fundamental trade-off:

| Attack Type | Transferability | Stealthiness | Core Issue |
|-------------|----------------|--------------|------------|
| Static triggers (Fixed/Grammar) | ✓ Good | ✗ Poor | Insert fixed dead code, 100% detectable by defenses such as KillBadCode |
| Dynamic triggers (AFRAIDOOR) | ✗ Poor | ✓ Good | Greedy search converges to sharp minima, causing failure across datasets |

### Limitations of Prior Work

**AFRAIDOOR assumes that the poisoned data and the victim's training data share the same distribution**—yet in realistic scenarios, adversaries poison public repositories while victims collect data from diverse sources, making distribution mismatch inevitable.

**AFRAIDOOR employs greedy search**, optimizing each identifier independently, which leads to convergence at suboptimal local minima. The perturbations discovered by greedy search reside in **sharp loss regions containing dataset-specific patterns**, causing large performance fluctuations under small parameter changes.

### Key Observation

**Models that converge to flat loss regions learn more generalizable features.** Flat regions capture code patterns that are universally present across different datasets, rather than dataset-specific artifacts confined to narrow parameter spaces. Consequently, adversarial perturbations discovered in flat regions transfer more effectively across datasets.

## Method

### Overall Architecture

STAB is a three-stage pipeline:

1. **Sharpness-Aware Surrogate Model Training**: Train the surrogate model with SAM to converge to a flat loss region.
2. **Adversarial Trigger Optimization**: Use Gumbel-Softmax differentiable optimization to generate a trigger distribution.
3. **Trigger Generation and Deployment**: Sample discrete triggers from the optimized distribution and deploy them.

### Threat Model

The adversary constructs a poisoned dataset $\mathcal{D}_p = \{(x_i \oplus t_i, y^*)\}_{i=1}^m$, injecting triggers paired with target output $y^*$. The victim only incorporates a subset $\mathcal{D}_p' \subseteq \mathcal{D}_p$, and the adversary can poison only a small fraction $\epsilon = |\mathcal{D}_p'|/|\mathcal{D}|$ of the victim's training data. The key constraint is $\mathcal{D}_s \neq \mathcal{D}_v$ (cross-dataset scenario).

### Key Designs

#### 1. Sharpness-Aware Surrogate Model Training

**Core Idea**: SAM optimization is applied to drive the surrogate model toward flat minima, thereby discovering backdoor patterns that generalize across datasets.

The training objective adopts a min-max formulation:

$$\min_{\theta_s} \mathcal{L}_{\text{SAM}}(\theta_s, \mathcal{D}_s) = \min_{\theta_s} \max_{\|\delta\|_2 \leq \rho} \mathcal{L}(\theta_s + \delta, \mathcal{D}_s)$$

The SAM optimization alternates between two steps:
- Compute the worst-case perturbation within the allowed radius: $\delta^* = \rho \cdot \frac{\nabla_{\theta_s}\mathcal{L}(\theta_s, \mathcal{D}_s)}{\|\nabla_{\theta_s}\mathcal{L}(\theta_s, \mathcal{D}_s)\|_2}$
- Compute the gradient at the perturbed parameters and update: $\theta_s \leftarrow \theta_s - \eta \cdot \nabla_{\theta_s}\mathcal{L}(\theta_s + \delta^*, \mathcal{D}_s)$

**Design Motivation**: Flat minima encode more generalizable code features (semantic and syntactic patterns), enabling the adversarial triggers generated in subsequent stages to remain effective under distribution shift.

#### 2. Gumbel-Softmax Trigger Optimization

**Core Idea**: The discrete identifier selection problem is reformulated as a continuous differentiable optimization, enabling end-to-end joint optimization over all trigger tokens.

**Steps**:
- Parse the code AST and identify all modifiable identifiers $\{v_j\}_{j=1}^k$.
- Initialize a learnable proxy distribution matrix $\mathbf{\Pi} \in \mathbb{R}^{L \times |\mathcal{V}_t|}$.
- Generate soft token representations via the Gumbel-Softmax function:
  $$\tilde{\mathbf{z}}_i = \text{softmax}\left(\frac{\log(\boldsymbol{\pi}_i) + \mathbf{g}_i}{\tau}\right)$$
- Feed the weighted embedding $\mathbf{e} = \tilde{\mathbf{z}}^T \mathbf{E}$ as a differentiable input to the surrogate model.

The **optimization objective** is a composite loss: $\mathcal{L}_{\text{trigger}} = \mathcal{L}_a + \lambda \cdot (\mathcal{L}_c + \mathcal{L}_d)$

| Loss Term | Formula | Role |
|-----------|---------|------|
| Attack loss $\mathcal{L}_a$ | $\mathbb{E}_{x \sim \mathcal{D}_s'}[-\log P(y^* \| \mathcal{M}_s^*(\mathbf{e}))]$ | Ensures backdoor activation to produce the malicious target output |
| Consistency loss $\mathcal{L}_c$ | $\sum_{j=1}^k \sum_{l,l' \in P_j} \text{MMD}(\boldsymbol{\pi}_l, \boldsymbol{\pi}_{l'})$ | Enforces the same trigger token for all occurrences of the same identifier |
| Diversity loss $\mathcal{L}_d$ | $-\sum_{1 \leq i < j \leq k} \text{MMD}(\bar{\boldsymbol{\pi}}_i, \bar{\boldsymbol{\pi}}_j)$ | Encourages distinct trigger tokens for different identifiers to avoid repetitive patterns |

**Design Motivation**: MMD (Maximum Mean Discrepancy) constraints, rather than hard constraints, ensure syntactic correctness while preserving differentiability. The consistency loss guarantees global naming coherence for code identifiers; the diversity loss additionally improves stealthiness.

#### 3. Trigger Generation and Deployment

- Sample discrete tokens from the optimized distribution matrix $\mathbf{\Pi}^*$ using Gumbel-Softmax at very low temperature.
- If distinct identifiers sample the same token, resample to preserve code validity.
- Replace the original identifiers to produce the final poisoned code.
- Inject the poisoned code into public repositories and exploit open-source participation mechanisms (stars, forks) to increase repository visibility.

### Loss & Training

- **Surrogate model training**: SAM optimization with $\rho = 0.02$; standard cross-entropy loss.
- **Trigger optimization**: Gumbel-Softmax temperature $\tau = 1.0$; $N = 100$ iterations; weight $\lambda = 0.1$.
- **Victim model**: Default poisoning rate $\epsilon = 5\%$; fine-tuned for 15 epochs with early stopping.

## Key Experimental Results

### Main Results

**Datasets**: Py150 (150K Python files), CodeSearchNet/CSN (400K+ Python functions), PyTorch/PyT (218K Python package libraries), forming 9 surrogate–victim dataset combinations.

**Tasks**: Method Name Prediction (MNP) and Code Summarization (CS).

| Surrogate Model | Victim Dataset | AFRAIDOOR Avg ASR | STAB Avg ASR | Gain |
|----------------|---------------|-------------------|-------------|------|
| PLBART | Py150 | 68.48% | 76.89% | +8.4% |
| PLBART | CSN | 90.72% | 94.55% | +3.8% |
| PLBART | PyT | 76.07% | 79.43% | +3.4% |
| CodeT5 | Py150 | 69.20% | 76.93% | +7.7% |
| CodeT5 | CSN | 90.13% | 95.13% | +5.0% |
| CodeT5 | PyT | 78.43% | 80.51% | +2.1% |

STAB achieves an average ASR of 80.1% on cross-dataset transfer, surpassing AFRAIDOOR by 12.4%, while maintaining comparable BLEU performance on clean data.

**Post-defense results** (ASR-D, using KillBadCode defense):

| Attack | Py150 | CSN | PyT | Overall |
|--------|-------|-----|-----|---------|
| Fixed | 0% | 0% | 0% | Completely fails under defense |
| Grammar | 0% | 0% | 0% | Completely fails under defense |
| AFRAIDOOR | ~60–70% | ~80% | ~65–70% | High variance |
| STAB | ~70–73% | ~85–91% | ~72–77% | Consistently high |

### Ablation Study

| Configuration | Py150 ASR | Py150 ASR-D | CSN ASR | PyT ASR | Note |
|--------------|----------|------------|---------|---------|------|
| STAB (full) | 76.89% | 70.17% | 94.55% | 79.43% | Best performance |
| w/o SAM | 72.21% | 63.92% | 91.12% | 77.84% | Lower ASR + larger std. dev. |
| w/o Gumbel-Softmax | 74.58% | 67.31% | 93.85% | 76.71% | Comparable initial ASR but notably lower ASR-D |

**Stealthiness comparison** (KillBadCode detection rate; lower is stealthier):

| Attack | Recall | F1 |
|--------|--------|----|
| Fixed | 99.99% | ~40% |
| Grammar | 99.99% | ~39% |
| AFRAIDOOR | ~27% | ~14% |
| STAB | ~23% | ~10% |

### Key Findings

1. **SAM is the core driver of transferability**: Removing SAM not only reduces ASR but also substantially increases standard deviation (from ~0.3% to ~0.9%), confirming that a flat loss landscape is critical for stability.
2. **Gumbel-Softmax is the core driver of stealthiness**: Replacing it with greedy search yields comparable initial ASR but significantly lower ASR-D and degraded stealthiness.
3. **Optimal sharpness parameter $\rho = 0.02$**: Too small provides insufficient flatness guidance; too large causes the victim model to encounter excessively varied trigger patterns, making it difficult to learn consistent associations.
4. **Poisoning rate analysis**: Higher poisoning rates generally improve ASR, but marginal gains diminish beyond a threshold; STAB maintains high ASR-D even under constrained poisoning budgets.

## Highlights & Insights

1. **A novel perspective on loss landscape geometry**: This work is the first to apply the theoretical insight that "flat minima favor generalization" to the transferability of backdoor attacks, an elegant conceptual contribution.
2. **Differentiable solution to discrete optimization**: Gumbel-Softmax reformulates discrete code token selection as continuous optimization, with MMD constraints preserving syntactic correctness.
3. **A more realistic threat model**: The cross-dataset scenario abandons the unrealistic assumption that poisoned and victim data share the same distribution.
4. **Fragility of static attacks under defense**: KillBadCode detects static triggers with 100% recall, whereas STAB substantially lowers the detection rate.

## Limitations & Future Work

1. **Only two code models evaluated** (PLBART, CodeT5); generalization to larger models such as CodeLlama remains unexplored.
2. **Only Python code evaluated**; cross-language transferability is unknown.
3. **Triggers restricted to identifier renaming**; other code transformations (e.g., expression-level mutations) are not explored.
4. **Ethical concerns**: Although the stated purpose is to advance defensive research, the paper effectively introduces a more potent attack vector.
5. **Limited defense evaluation**: Only three defenses are assessed (SS, ONION, KillBadCode); more advanced defensive strategies are not considered.

## Related Work & Insights

- **SAM (Sharpness-Aware Minimization)**: Originally proposed to improve model generalization; this paper innovatively repurposes it to enhance backdoor attack transferability.
- **Gumbel-Softmax**: A discrete sampling technique from the VAE literature, applied here to code token selection.
- **AFRAIDOOR**: The current state-of-the-art dynamic attack and the primary baseline for comparison.
- **KillBadCode**: The state-of-the-art code backdoor defense, which detects anomalies in code naturalness via n-gram language models.
- **Insight**: Loss landscape geometry (flat vs. sharp) is an underappreciated but critical dimension in both offensive and defensive research.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Applying the SAM theoretical insight to backdoor attack transferability is highly innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (3 datasets × 2 models × 9 combinations + comprehensive ablation)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, rigorous methodology, high-quality figures)
- Value: ⭐⭐⭐⭐ (Reveals a new threat to code model security and advances defensive research)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models](towards_effective_stealthy_and_persistent_backdoor_attacks_targeting_graph_found.md)
- [\[AAAI 2026\] TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models](toporeformer_mitigating_adversarial_attacks_using_topological_purification_in_oc.md)
- [\[ICLR 2026\] Membership Privacy Risks of Sharpness Aware Minimization](../../ICLR2026/ai_safety/sam_membership_privacy_risks.md)
- [\[CVPR 2026\] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](../../CVPR2026/ai_safety/when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)
- [\[AAAI 2026\] Authority Backdoor: A Certifiable Backdoor Mechanism for Authoring DNNs](authority_backdoor_a_certifiable_backdoor_mechanism_for_authoring_dnns.md)

</div>

<!-- RELATED:END -->
