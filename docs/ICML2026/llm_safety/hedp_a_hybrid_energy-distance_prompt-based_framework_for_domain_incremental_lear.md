---
title: >-
  [Paper Note] HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning
description: >-
  [ICML 2026][LLM Safety][Domain Incremental Learning] Drawing physical intuition from Helmholtz free energy, the prompt parameters for each domain are trained to follow an energy curve that is "compressed to boundary $\Th…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Domain Incremental Learning"
  - "Prompt Learning"
  - "Energy-based Models"
  - "Helmholtz Free Energy"
  - "CLIP"
date: 2026-05-08
content_hash: 4fc012079e028d5f
---

# HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning

**Conference**: ICML 2026  
**arXiv**: [2605.05776](https://arxiv.org/abs/2605.05776)  
**Code**: Available (Public repository attached in the paper)  
**Area**: Continual Learning / Domain Incremental Learning / Prompt Learning  
**Keywords**: Domain Incremental Learning, Prompt Learning, Energy-based Models, Helmholtz Free Energy, CLIP

## TL;DR
Drawing physical intuition from Helmholtz free energy, the prompt parameters for each domain are trained to follow an energy curve that is "compressed to boundary $\Theta$ and aligned to midline $\Delta$." During inference, a combination of energy and distance factors is used to jointly weight the prompts from each domain, achieving performance gains of 1.76, 3.12, and 2.57 percentage points on unknown domains across three DIL benchmarks: CDDB, DomainNet, and CORe50.

## Background & Motivation

**Background**: Domain Incremental Learning (DIL) requires models to be trained sequentially on multiple domains (e.g., autonomous driving detection under different weather conditions) without replaying data from old domains. During inference, the model must maintain accuracy on known domains while generalizing to unknown domains. The mainstream approach involves freezing a pre-trained large model (e.g., CLIP) and learning a set of prompt parameters for each domain; representative methods include CP-Prompt, S-Prompts, MoP-CLIP, and ESN.

**Limitations of Prior Work**: (1) There is a persistent trade-off between known and unknown domains—CP-Prompt performs well on known domains but poorly on unknown ones, while MoP-CLIP shows the opposite; (2) "How to select which domain's prompt" at inference time is the core problem. Existing methods use either distance (prone to misjudgment in overlapping regions) or clustering (blurry boundaries at coarse granularities); (3) Single-domain prompts easily overfit to their own distributions, which can actually increase overlap with other prompts in the shared space. t-SNE visualizations show that Domain B samples in the CLIP space can actually be closer to the cluster points of Domain A.

**Key Challenge**: Domains are both similar and distinct within the shared feature space. A single signal (distance or energy) only captures one aspect—distance reflects the global semantic structure (learned by CLIP), while energy reflects the local distribution sensitivity tuned by the prompt. While they have different error modes, neither is stable enough individually.

**Goal**: (1) Align the energy distributions of each domain's prompt during training so that energy truly reflects whether a sample belongs to that domain; (2) Design a hybrid signal of energy and distance during inference to combine their strengths and cancel out their respective weaknesses.

**Key Insight**: The paper analogies the statistical distribution of data in the feature space to a physical energy field, utilizing Helmholtz free energy $E(x) = -kT \ln[\sum_y e^{H(x)[y]/kT}]$ to calculate a scalar energy for each sample relative to each prompt. Ideally, a domain-specific prompt should assign low energy to its own samples and high energy to samples from other domains.

**Core Idea**: An energy regularization term, combining "boundary loss + midline loss," is used to constrain the energy distribution of each domain prompt to a unified scale. During inference, energy and distance factors are summed and passed through a softmax to obtain hybrid weights for a weighted summation of predictions from each domain prompt.

## Method

### Overall Architecture
The CLIP ViT-B/16 backbone is frozen. **Training**: For each domain, a set of visual prompts $P_v^S$ and text prompts $P_t^S$ are trained independently. The loss consists of classification cross-entropy $\mathcal{L}_{ce}$ plus energy regularization $\lambda \mathcal{L}_{reg}$ with weight $\lambda = 0.05$. **Inference**: For each test sample, the model simultaneously calculates (a) the distance $D^i(x)$ to each domain's cluster center in the frozen CLIP space; and (b) the energy $E^i(x)$ under each prompt model. These are normalized into relative factors $EF^i$ and $DF^i$, summed as $F^i(x) = EF^i/\alpha + DF^i/\beta$, and softmax-weighted to get $W^i$. The final prediction is a mixture: $P_{mix}(x) = \sum_i W^i P^i(x)$.

### Key Designs

1.  **Energy Regularization Loss (Boundary + Midline)**:
    - **Function**: Ensures the energy distribution produced by each domain prompt is neither too dispersed nor too concentrated, allowing for direct cross-domain energy comparison.
    - **Mechanism**: Energy is defined as $E(x) = -kT \ln[\sum_{y=1}^U e^{H(x)[y]/kT}]$ (Helmholtz free energy). The regularization consists of two parts: the boundary loss $\mathcal{L}_{border} = \frac{1}{|\mathcal{D}_t|}\sum \max(0, E(x) - \Theta)$ punishes energy exceeding $\Theta = -32$, pushing domain samples toward the low-energy side; the midline loss $\mathcal{L}_{midline} = |\Delta - \frac{1}{|\mathcal{D}_t|}\sum E(x)|$ pulls the mean toward $\Delta = -40$.
    - **Design Motivation**: The authors plotted energy distribution comparisons under four regularization combinations (none / boundary only / midline only / complete). Boundary-only yields values below $\Theta$, but relative positions remain chaotic. Midline-only leaves the distribution shape unconstrained, leading to inversions like $E^B(x^A) < E^A(x^A)$. Combining both stability satisfies $E^s(x^s) < E^i(x^s) (\forall i \neq s)$.

2.  **Hybridization of Energy and Distance Factors**:
    - **Function**: Calculates a comprehensive "similarity factor" for each domain during inference to determine the weight of that domain's prompt in the prediction.
    - **Mechanism**: The energy factor $EF^i(x) = E_{\min} - E^i(x)$ uses a negative offset within the range $(-\infty, 0]$, where larger values indicate lower energy and higher confidence under the $i$-th prompt. The distance factor $DF^i(x) = D_{\min} - D^i(x)$ uses $K$-means to calculate $K$ cluster centers for each domain in the frozen CLIP space, then computes the cosine distance to the nearest center. The hybrid factor is $F^i(x) = EF^i(x)/\alpha + DF^i(x)/\beta$, with $W^i = \text{softmax}(F^i)$.
    - **Design Motivation**: Using a first-order Taylor expansion in the appendix, the authors demonstrate that $\nabla_x EF$ aligns with prompt parameter directions (capturing domain statistical differences), while $\nabla_x DF$ aligns with frozen CLIP semantic directions (capturing global semantics). These gradients are approximately orthogonal, meaning their error modes are uncorrelated. Perturbations that cause one signal to fail are unlikely to affect the other, thus cancelling out errors. At $\alpha = \beta = 0.6$, the balance between distance-bias for known domains and energy-bias for unknown domains aligns with theoretical expectations.

3.  **Energy Regularization Implicitly Smooths the Energy Landscape**:
    - **Function**: Stabilizes energy function shifts for samples from unknown domains.
    - **Mechanism**: Proposition 2 in the appendix proves that constraining energy output within $(-\infty, \Theta]$ with a mean at $\Delta$ is equivalent to implicitly compressing the local Lipschitz constant $K$ of the energy function on the data manifold. For an OOD sample $x_{out} = x_{in} + \Delta_x$, the energy shift satisfies $|E(x_{out}) - E(x_{in})| \leq K\|\Delta_x\|$. A smaller $K$ prevents OOD samples from suddenly falling into the low-energy regions of known domains, thereby resisting catastrophic forgetting.
    - **Design Motivation**: Traditional energy training often creates "energy cliffs" where OOD samples are incorrectly classified as low-energy if they approach the boundary. Compressing the distribution into a compact range "softens the terrain," providing an energy buffer for OOD samples.

### Loss & Training
The total loss is $\mathcal{L}_{total} = \mathcal{L}_{ce} + \lambda \mathcal{L}_{reg}$, where $\mathcal{L}_{reg} = \mathcal{L}_{border} + \mathcal{L}_{midline}$. Hyperparameters: $\Theta = -32, \Delta = -40, K = 5, \alpha = \beta = 0.6, \lambda = 0.05$. Optimization uses SGD with cosine annealing and an initial learning rate of 0.01.

## Key Experimental Results

### Main Results

| Dataset | Scenario | Prev. SOTA | HEDP | Gain |
|---------|----------|------------|------|------|
| CDDB-Hard | Known AA / AF | CP-Prompt 93.65 / -0.25 | **93.72 / -0.08** | +0.07 / +0.17 |
| CDDB-Hard | Unknown AA | MoP-CLIP 81.98 | **83.74** | +1.76 |
| DomainNet | Known AA (All) | CP-Prompt 73.15 | **74.19** | +1.04 |
| DomainNet | Unknown AA | MoP-CLIP 63.97 | **67.09** | +3.12 |
| CORe50 | Unknown AA | ESN 91.80 | **94.37** | +2.57 |

HEDP achieves the best results on both known and unknown domains simultaneously, eliminating the previous trade-off.

### Ablation Study

| Scheme | Energy Boundary | Energy Midline | Energy Factor | Distance Factor | CDDB Unknown | DomainNet Unknown | CORe50 Unknown |
|--------|-----------------|----------------|---------------|-----------------|--------------|-------------------|----------------|
| 1 (Dist Only) | ✗ | ✗ | ✗ | ✓ | 75.80 | 64.97 | 93.17 |
| 2 (Unreg Energy) | ✗ | ✗ | ✓ | ✗ | 77.31 | 63.55 | 92.06 |
| 3 (+Boundary) | ✓ | ✗ | ✓ | ✗ | 79.22 | 65.05 | 92.98 |
| 4 (+Midline) | ✗ | ✓ | ✓ | ✗ | 79.12 | 65.01 | 93.77 |
| 5 (Full Energy) | ✓ | ✓ | ✓ | ✗ | 81.52 | 65.59 | 94.07 |
| 6 (Full HEDP) | ✓ | ✓ | ✓ | ✓ | **83.74** | **67.09** | **94.66** |

### Key Findings
- **Boundary and Midline must be used together**: Using either alone is 2-3 points worse than full regularization, proving they capture different distribution characteristics (maximum value constraint vs. central tendency alignment).
- **Energy and Distance are complementary, not redundant**: Adding the distance factor to Scheme 5 (to get Scheme 6) increases CDDB Unknown by 2.22 points; adding the energy factor to Scheme 1 increases CDDB Unknown by 7.94 points. The complementary effect is especially pronounced in unknown domains.
- **Hyperparameter impact on unknown domains follows a diagonal distribution**: Grid heatmaps for $\alpha$ and $\beta$ show known domains favor "distance dominance" while unknown domains favor an "energy + distance balance," proving different underlying mechanisms.
- Changing the number of clusters $K$ has minimal impact, suggesting the distance factor serves as a "global topological stabilizer" rather than a fine-grained discriminator.

## Highlights & Insights
- **From Physical Intuition to ML Design**: The application of Helmholtz free energy from statistical physics is natural. The "energy boundary + midline" correspond to potential well depth and zero point in physics, providing strong interpretability.
- **Gradient Orthogonality Argument**: The use of first-order Taylor expansions to argue that error gradients for energy and distance are orthogonal provides theoretical support for the "complementary signals" beyond pure empirical observation.
- **"Terrain Smoothing" Side Effect of Energy Regularization**: The incidental discovery that constraining the energy distribution within a compact interval implicitly reduces the Lipschitz constant, thereby enhancing OOD robustness, is a trick that could be independently transferred to any OOD detection task.
- For prompt-based continual learning, "how to select the prompt" is often more important than "how to train the prompt," a perspective this paper explores to its fullest.

## Limitations & Future Work
- Inference latency increases linearly with the number of domains—each test sample must pass through all domain prompt models to calculate energy. The authors acknowledge this as a scalability bottleneck and suggest dynamic prompt selection for future work.
- $\Theta$ and $\Delta$ are manually tuned hyperparameters, and while results saturate as $\Delta$ is pushed further, a self-adaptive mechanism is lacking.
- Experiments are limited to vision classification. Whether the energy physical intuition holds for NLP or VLM reasoning tasks remains to be verified.
- The argument that "energy is an SFT-invariant signal" is somewhat thin and relies heavily on visualization.

## Related Work & Insights
- **vs CP-Prompt** (ACMMM 2024): CP-Prompt is strong on known domains but weak on unknown ones; HEDP uses the energy factor to compensate for the lack of unknown generalization.
- **vs MoP-CLIP** (WACV 2024): MoP-CLIP uses coarse clustering for prompt mixing; HEDP combines clustering (distance factor) with internal prompt energy for finer discrimination.
- **vs ESN** (AAAI 2023): ESN introduces a temperature-adjustable energy metric but only uses energy. HEDP adds "energy regularization" to explicitly constrain the distribution shape, leading to significantly better results.
- **vs ELI** (CVPR 2022): ELI also uses energy for incremental learning, but task-wise energy manifolds are less suitable for prompt-based DIL scenarios. HEDP attaches energy directly to the prompt output, making it more lightweight.

## Rating
- Novelty: ⭐⭐⭐⭐ The "energy regularization + distance/energy hybrid" combination is new, and the physical analogy is natural.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of three datasets + full ablation + hyperparameter grids + energy distribution visualization across both known and unknown domains.
- Writing Quality: ⭐⭐⭐⭐ The narrative is clear, and the inclusion of gradient orthogonality and Lipschitz arguments in the appendix adds theoretical depth.
- Value: ⭐⭐⭐⭐ Simultaneously solves the known/unknown trade-off in prompt-based DIL, making it practically useful for engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning](../../ICLR2026/llm_safety/sabre-fl_selective_and_accurate_backdoor_rejection_for_federated_prompt_learning.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](../../ACL2026/llm_safety/adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[ICLR 2026\] VeriTrail: Closed-Domain Hallucination Detection with Traceability](../../ICLR2026/llm_safety/veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes.md)

</div>

<!-- RELATED:END -->
