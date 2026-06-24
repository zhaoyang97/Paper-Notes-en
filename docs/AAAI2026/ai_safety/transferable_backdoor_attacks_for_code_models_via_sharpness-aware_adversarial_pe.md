---
title: >-
  [Paper Note] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation
description: >-
  [AAAI 2026][AI Safety][backdoor attack] Proposes STAB (Sharpness-aware Transferable Adversarial Backdoor), which trains a surrogate model using SAM to converge to flat regions of the loss landscape and optimizes context-aware adversarial triggers using Gumbel-Softmax. This is the first work to simultaneously achieve cross-dataset transferability and stealthiness in backdoor attacks against code models.
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "backdoor attack"
  - "code models"
  - "transferability"
  - "Sharpness-Aware Minimization"
  - "adversarial perturbation"
date: 2026-05-08
content_hash: 6eaaa6b5c55a7f60
---

# Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation

**Conference**: AAAI 2026  
**arXiv**: [2602.11213](https://arxiv.org/abs/2602.11213)  
**Code**: [github.com/ChangShuyu/STAB](https://github.com/ChangShuyu/STAB)  
**Area**: AI Safety  
**Keywords**: backdoor attack, code models, transferability, Sharpness-Aware Minimization, adversarial perturbation

## TL;DR

Proposes STAB (Sharpness-aware Transferable Adversarial Backdoor), which trains a surrogate model using SAM to converge to flat regions of the loss landscape and optimizes context-aware adversarial triggers using Gumbel-Softmax. This is the first work to simultaneously achieve cross-dataset transferability and stealthiness in backdoor attacks against code models.

## Background & Motivation

### Problem Definition

Pre-trained code models (e.g., PLBART, CodeT5) have become crucial components of the modern software supply chain. However, because training data is collected from public repositories, they are vulnerable to backdoor attacks via data poisoning—where the model produces malicious outputs when the trigger is present, while behaving normally on clean inputs.

### Key Challenge: Trade-off between Transferability and Stealthiness

Existing code backdoor attacks face a fundamental trade-off:

| Attack Type | Transferability | Stealthiness | Core Problem |
|---------|--------|--------|---------|
| Static Triggers (Fixed/Grammar) | ✓ Good | ✗ Poor | Inserts fixed dead code, which is 100% detected by defenses like KillBadCode |
| Dynamic Triggers (AFRAIDOOR) | ✗ Poor | ✓ Good | Greedy search leads to convergence in sharp local minima, failing across datasets |

### Limitations of Prior Work

1. **AFRAIDOOR assumes that the poisoned data and the victim's training data share the same distribution**—however, in real-world scenarios, attackers poison public repositories while victims collect data from diverse sources, leading to inevitable distribution shifts.
2. **AFRAIDOOR uses greedy search** to independently optimize each identifier, which leads to convergence in sub-optimal local minima.
3. Greedy perturbations discover **dataset-specific patterns in sharp regions of the loss landscape**, meaning minor parameter updates cause drastic performance fluctuations.

### Key Observation

**Models that converge to flat regions learn more generalizable features**. Flat regions capture universal code patterns across different datasets rather than dataset-specific artifacts in narrow parameter spaces. Therefore, adversarial perturbations discovered in flat regions transfer more effectively across datasets.

## Method

### Overall Architecture

STAB is a three-stage pipeline:

1. **Sharpness-Aware Surrogate Model Training**: Train the surrogate model to reach a flat loss region using SAM.
2. **Adversarial Trigger Optimization**: Generate a trigger distribution using differentiable optimization via Gumbel-Softmax.
3. **Trigger Generation and Deployment**: Sample discrete triggers from the optimized distribution and deploy them.

### Threat Model

The attacker constructs a poisoned dataset $\mathcal{D}_p = \{(x_i \oplus t_i, y^*)\}_{i=1}^m$ by injecting triggers and pairing them with target outputs $y^*$. The victim's training set only contains a subset $\mathcal{D}_p' \subseteq \mathcal{D}_p$, where the attacker can only poison a small fraction of the victim's training data $\epsilon = |\mathcal{D}_p'|/|\mathcal{D}|$. The key constraint is that $\mathcal{D}_s \neq \mathcal{D}_v$ (cross-dataset scenario).

### Key Designs

#### 1. **Sharpness-Aware Surrogate Model Training**: Training surrogate models on flat loss landscapes

Mechanism: Train the surrogate model using SAM optimization to make it converge to flat minima, enabling the discovery of cross-dataset generalizable backdoor patterns.

The training objective uses a min-max formulation:

$$\min_{\theta_s} \mathcal{L}_{\text{SAM}}(\theta_s, \mathcal{D}_s) = \min_{\theta_s} \max_{\|\delta\|_2 \leq \rho} \mathcal{L}(\theta_s + \delta, \mathcal{D}_s)$$

The SAM optimization process alternates between:
- Finding the worst-case perturbation that maximizes loss within a permitted radius: $\delta^* = \rho \cdot \frac{\nabla_{\theta_s}\mathcal{L}(\theta_s, \mathcal{D}_s)}{\|\nabla_{\theta_s}\mathcal{L}(\theta_s, \mathcal{D}_s)\|_2}$
- Computing the gradient at the perturbed parameters and updating: $\theta_s \leftarrow \theta_s - \eta \cdot \nabla_{\theta_s}\mathcal{L}(\theta_s + \delta^*, \mathcal{D}_s)$

**Design Motivation**: Flat minima encode more general code features (semantic and syntactic patterns), making the subsequently generated adversarial triggers remain effective under distribution shifts.

#### 2. **Gumbel-Softmax Trigger Optimization**: Differentiable discrete trigger optimization

Mechanism: Formulate the discrete identifier selection problem as a continuous differentiable optimization problem to achieve end-to-end joint optimization of all trigger tokens.

**Steps**:
- Parse the code AST to identify all modifiable identifiers $\{v_j\}_{j=1}^k$.
- Initialize a learnable surrogate distribution matrix $\mathbf{\Pi} \in \mathbb{R}^{L \times |\mathcal{V}_t|}$.
- Use the Gumbel-Softmax function to generate soft token representations:
  $$\tilde{\mathbf{z}}_i = \text{softmax}\left(\frac{\log(\boldsymbol{\pi}_i) + \mathbf{g}_i}{\tau}\right)$$
- Feed the weighted embeddings $\mathbf{e} = \tilde{\mathbf{z}}^T \mathbf{E}$ as a differentiable input into the surrogate model.

The **optimization objective** is a composite loss: $\mathcal{L}_{\text{trigger}} = \mathcal{L}_a + \lambda \cdot (\mathcal{L}_c + \mathcal{L}_d)$

| Loss Term | Formula | Function |
|--------|------|------|
| Attack Loss $\mathcal{L}_a$ | $\mathbb{E}_{x \sim \mathcal{D}_s'}[-\log P(y^* \| \mathcal{M}_s^*(\mathbf{e}))]$ | Ensures backdoor activation and generates the target malicious output |
| Consistency Loss $\mathcal{L}_c$ | $\sum_{j=1}^k \sum_{l,l' \in P_j} \text{MMD}(\boldsymbol{\pi}_l, \boldsymbol{\pi}_{l'})$ | Ensures the same identifier uses the same trigger token across all its occurrences in the code |
| Diversity Loss $\mathcal{L}_d$ | $-\sum_{1 \leq i < j \leq k} \text{MMD}(\bar{\boldsymbol{\pi}}_i, \bar{\boldsymbol{\pi}}_j)$ | Ensures different identifiers use different trigger tokens, avoiding repetitive patterns |

**Design Motivation**: Using Maximum Mean Discrepancy (MMD) constraints rather than hard constraints ensures syntactic correctness while maintaining differentiability. Consistency loss ensures global naming consistency of code identifiers, while diversity loss enhances stealthiness.

#### 3. **Trigger Generation and Deployment**: Poisoned sample generation and deployment

- Sample discrete tokens from the optimized distribution matrix $\mathbf{\Pi}^*$ using Gumbel-Softmax with an extremely low temperature.
- Resample if different identifiers sample the same token to ensure code validity.
- Replace original identifiers to generate the final poisoned code.
- Inject the poisoned code into public code repositories and leverage open-source engagement mechanisms (stars, forks) to increase repository visibility.

### Loss & Training

- **Surrogate model training stage**: SAM optimization, $\rho = 0.02$, standard cross-entropy loss.
- **Trigger optimization stage**: Gumbel-Softmax temperature $\tau = 1.0$, $N = 100$ iterations, weight $\lambda = 0.1$.
- **Victim model**: Default poisoning rate $\epsilon = 5\%$, fine-tuned for 15 epochs with an early stopping strategy.

## Key Experimental Results

### Main Results

**Datasets**: Py150 (150K Python files), CodeSearchNet/CSN (400K+ Python functions), PyTorch/PyT (218K Python libraries), forming 9 surrogate-victim dataset combinations.

**Tasks**: Method Name Prediction (MNP) and Code Summarization (CS).

| Surrogate Model | Victim Dataset | AFRAIDOOR Avg ASR | STAB Avg ASR | Gain |
|---------|-----------|-----------------|-------------|------|
| PLBART | Py150 | 68.48% | 76.89% | +8.4% |
| PLBART | CSN | 90.72% | 94.55% | +3.8% |
| PLBART | PyT | 76.07% | 79.43% | +3.4% |
| CodeT5 | Py150 | 69.20% | 76.93% | +7.7% |
| CodeT5 | CSN | 90.13% | 95.13% | +5.0% |
| CodeT5 | PyT | 78.43% | 80.51% | +2.1% |

STAB achieves an average ASR of 80.1% in cross-dataset transferability, outperforming AFRAIDOOR by 12.4%. Meanwhile, it maintains comparable BLEU performance on clean data.

**Defended Performance** (ASR-D, using KillBadCode defense):

| Attack Method | Py150 | CSN | PyT | Overall Performance |
|---------|-------|------|------|---------|
| Fixed | 0% | 0% | 0% | Fails completely under defense |
| Grammar | 0% | 0% | 0% | Fails completely under defense |
| AFRAIDOOR | ~60-70% | ~80% | ~65-70% | High fluctuation |
| STAB | ~70-73% | ~85-91% | ~72-77% | Highly consistent |

### Ablation Study

| Configuration | Py150 ASR | Py150 ASR-D | CSN ASR | PyT ASR | Description |
|------|----------|------------|---------|---------|------|
| STAB (Full) | 76.89% | 70.17% | 94.55% | 79.43% | Best performance |
| w/o SAM | 72.21% | 63.92% | 91.12% | 77.84% | ASR drops + standard deviation increases |
| w/o Gumbel-Softmax | 74.58% | 67.31% | 93.85% | 76.71% | Initial ASR is similar, but post-defense ASR-D drops significantly |

**Stealthiness Comparison** (KillBadCode detection rate, lower is stealthier):

| Attack Method | Recall | F1 |
|---------|--------|-----|
| Fixed | 99.99% | ~40% |
| Grammar | 99.99% | ~39% |
| AFRAIDOOR | ~27% | ~14% |
| STAB | ~23% | ~10% |

### Key Findings

1. **SAM is crucial for transferability**: Removing SAM not only drops ASR but also significantly increases the standard deviation (from ~0.3% to ~0.9%), demonstrating that flat loss landscapes are vital for stability.
2. **Gumbel-Softmax is crucial for stealthiness**: Replacing it with greedy search yields similar initial ASR, but the post-defense ASR-D drops significantly, indicating degraded stealthiness.
3. **Optimal sharpness parameter is $\rho = 0.02$**: If too small, flatness guidance is insufficient; if too large, the victim model encounters excessively diverse trigger patterns, making it hard to learn consistent associations.
4. **Poisoning rate analysis**: Higher poisoning rates generally increase ASR, but marginal gains diminish after a certain threshold. STAB maintains high ASR-D even under limited poisoning budgets.

**Loss & Training**:

## Highlights & Insights

1. **A novel perspective on loss landscape geometry**: First to apply the theoretical insight "flat minima benefit generalization" to the transferability of backdoor attacks. The approach is highly elegant.
2. **Differentiable solution for discrete optimization**: Employs Gumbel-Softmax to convert discrete code token selection into continuous optimization, while leveraging MMD constraints to preserve syntactic correctness.
3. **More realistic threat model**: Abandons the assumption that poisoned data and victim data share the identical distribution, focusing instead on cross-dataset scenarios.
4. **Vulnerability of static attacks under defense**: KillBadCode detects static triggers with 100% success rate, whereas STAB drastically reduces detection rates.

## Limitations & Future Work

1. **Evaluated on only two code models** (PLBART, CodeT5); has not been extended to larger models like CodeLlama.
2. **Evaluated only on Python code**; cross-language transferability remains unknown.
3. **Triggers are limited to identifier renaming**; other code transformations (e.g., expression transformations) have not been explored.
4. **Ethical concerns**: Although the claimed purpose is to advance defense research, the study practically introduces a more perilous attack method.
5. **Limited defense evaluations**: Evaluates only three defenses (SS, ONION, KillBadCode); more advanced defensive strategies are not considered.

## Related Work & Insights

- **SAM (Sharpness-Aware Minimization)**: Originally designed to improve generalization; this paper innovatively applies it to enhance the transferability of backdoor attacks.
- **Gumbel-Softmax**: A discrete sampling trick from the domain of VAEs, applied here to code token selection.
- **AFRAIDOOR**: The current SOTA dynamic attack, acting as the primary baseline for comparison.
- **KillBadCode**: A SOTA code backdoor defense that utilizes n-gram language models to detect anomalies in code naturalness.
- **Insight**: Loss landscape geometry (flat vs. sharp) is an under-explored, yet critical dimension in adversarial security.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Highly innovative application of SAM theoretical insights to backdoor transferability)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive ablation plus evaluation on 3 datasets x 2 models x 9 combinations)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, rigorous methodology, and high-quality illustrations)
- Value: ⭐⭐⭐⭐ (Exposes a new threat vector in code model security, stimulating future defense research)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Membership Privacy Risks of Sharpness Aware Minimization](../../ICLR2026/ai_safety/sam_membership_privacy_risks.md)
- [\[AAAI 2026\] Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models](towards_effective_stealthy_and_persistent_backdoor_attacks_targeting_graph_found.md)
- [\[AAAI 2026\] TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models](toporeformer_mitigating_adversarial_attacks_using_topological_purification_in_oc.md)
- [\[ICLR 2026\] TrojanTO: Action-Level Backdoor Attacks Against Trajectory Optimization Models](../../ICLR2026/ai_safety/trojanto_action-level_backdoor_attacks_against_trajectory_optimization_models.md)
- [\[CVPR 2026\] Eliminate Distance Differences Induced by Backdoor Attacks: Layer-Selective Training and Clipping to Mask Backdoor Models](../../CVPR2026/ai_safety/eliminate_distance_differences_induced_by_backdoor_attacks_layer-selective_train.md)

</div>

<!-- RELATED:END -->
