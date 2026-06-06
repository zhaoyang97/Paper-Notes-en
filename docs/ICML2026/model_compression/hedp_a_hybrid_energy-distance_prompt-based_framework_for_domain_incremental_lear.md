---
title: >-
  [Paper Note] HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning
description: >-
  [ICML 2026][Model Compression][Domain Incremental Learning] Inspired by the physical intuition of Helmholtz free energy, each domain's prompt parameters are trained to form an "energy curve compressed to boundary $\Theta…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Domain Incremental Learning"
  - "Prompt Learning"
  - "Energy Model"
  - "Helmholtz Free Energy"
  - "CLIP"
date: 2026-05-08
content_hash: 807336d57dbbac0d
---

# HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning

**Conference**: ICML 2026  
**arXiv**: [2605.05776](https://arxiv.org/abs/2605.05776)  
**Code**: Available (public repository linked at the end of the paper)  
**Area**: Continual Learning / Domain Incremental / Prompt Learning  
**Keywords**: Domain Incremental Learning, Prompt Learning, Energy Model, Helmholtz Free Energy, CLIP

## TL;DR
Inspired by the physical intuition of Helmholtz free energy, each domain's prompt parameters are trained to form an "energy curve compressed to boundary $\Theta$ and aligned to midline $\Delta$." During inference, an energy factor and a distance factor are jointly used to weight each domain prompt. This approach improves performance on unknown domains by 1.76 / 3.12 / 2.57 percentage points on the CDDB / DomainNet / CORe50 DIL benchmarks, respectively.

## Background & Motivation

**Background**: Domain Incremental Learning (DIL) requires models to be trained sequentially on multiple domains (e.g., autonomous driving detection under different weather conditions) without replaying old domain data. At inference, the model must maintain accuracy on seen (known) domains and generalize to unseen (unknown) domains. The mainstream approach is to freeze a pre-trained large model (e.g., CLIP) and learn a set of prompt parameters for each domain; representative methods include CP-Prompt, S-Prompts, MoP-CLIP, and ESN.

**Limitations of Prior Work**: (1) There is always a trade-off between known and unknown domains—CP-Prompt excels on known domains but underperforms on unknown ones, while MoP-CLIP is the opposite; (2) The core problem during inference is "which domain's prompt to use," with existing methods relying either on distance (prone to misclassification in overlapping regions) or clustering (coarse granularity leads to fuzzy boundaries); (3) Single-domain prompts tend to overfit their own distribution, increasing overlap with other domain prompts in the shared space. t-SNE visualization shows that Domain B samples may be closer to Domain A's cluster center in CLIP space.

**Key Challenge**: Domains are both similar and different in the shared feature space; a single signal (distance or energy) only captures one aspect—distance reflects global semantic structure (learned by CLIP), while energy reflects the local distribution sensitivity induced by the prompt. Their error patterns differ, but neither is stable enough alone.

**Goal**: (1) Align the energy distributions of each domain prompt during training so that energy reliably indicates "whether a sample belongs to this domain"; (2) Design a hybrid energy + distance signal for inference, leveraging their strengths and offsetting each other's weaknesses.

**Key Insight**: The statistical distribution of data in feature space is analogous to an energy field in physics. Using Helmholtz free energy $E(x) = -kT \ln[\sum_y e^{H(x)[y]/kT}]$, a scalar energy is computed for each sample with respect to each prompt. Ideally, a domain's prompt assigns low energy to in-domain samples and high energy to out-of-domain samples.

**Core Idea**: An energy regularization term combining "boundary loss + midline loss" constrains each domain prompt's energy distribution to a unified scale. During inference, the energy and distance factors are summed and softmaxed to yield hybrid weights, which are used to aggregate predictions from all domain prompts.

## Method

### Overall Architecture
The CLIP ViT-B/16 backbone is frozen. **Training**: Each domain independently trains a set of visual prompts $P_v^S$ and text prompts $P_t^S$, with the loss being classification cross-entropy $\mathcal{L}_{ce}$ plus energy regularization $\lambda \mathcal{L}_{reg}$, where $\lambda = 0.05$. **Inference**: For each test sample, (a) compute the distance $D^i(x)$ to each domain's cluster center in the frozen CLIP space; (b) compute the energy $E^i(x)$ under each prompt model. Both are normalized to relative factors $EF^i, DF^i$, summed as $F^i(x) = EF^i/\alpha + DF^i/\beta$, softmaxed to obtain weights $W^i$, and the final prediction is a weighted sum $P_{mix}(x) = \sum_i W^i P^i(x)$.

### Key Designs

1. **Energy Regularization Loss (Boundary + Midline)**:

    - **Function**: Ensures each domain prompt's energy distribution is neither too dispersed nor too concentrated, making cross-domain energies directly comparable.
    - **Mechanism**: Energy is defined as $E(x) = -kT \ln[\sum_{y=1}^U e^{H(x)[y]/kT}]$ (Helmholtz free energy). The regularization consists of two parts: boundary loss $\mathcal{L}_{border} = \frac{1}{|\mathcal{D}_t|}\sum \max(0, E(x) - \Theta)$ penalizes energies exceeding $\Theta = -32$, pushing in-domain samples to the low-energy side; midline loss $\mathcal{L}_{midline} = |\Delta - \frac{1}{|\mathcal{D}_t|}\sum E(x)|$ aligns the mean to $\Delta = -40$.
    - **Design Motivation**: The authors compare energy distributions under four regularization settings (none / boundary only / midline only / both): boundary alone may push all energies below $\Theta$ but with disordered relative positions; midline alone leaves the distribution shape unconstrained, possibly resulting in $E^B(x^A) < E^A(x^A)$. Only the combination reliably ensures $E^s(x^s) < E^i(x^s) (\forall i \neq s)$.

2. **Hybrid of Energy Factor and Distance Factor**:

    - **Function**: Computes a composite "similarity factor" for each domain during inference, determining the weight of each domain prompt in the final prediction.
    - **Mechanism**: The energy factor $EF^i(x) = E_{\min} - E^i(x)$ is a negative offset, range $(-\infty, 0]$, with higher values indicating lower energy and higher confidence for prompt $i$; the distance factor $DF^i(x) = D_{\min} - D^i(x)$ uses $K$-means to compute $K$ cluster centers per domain in the frozen CLIP space, then calculates the cosine distance from the sample to the nearest center. The hybrid factor is $F^i(x) = EF^i(x)/\alpha + DF^i(x)/\beta$, and $W^i = \text{softmax}(F^i)$.
    - **Design Motivation**: The appendix uses first-order Taylor expansion to show that $\nabla_x EF$ aligns with the prompt parameter direction (capturing domain statistical differences), while $\nabla_x DF$ aligns with the frozen CLIP semantic direction (capturing global semantics); their gradients are approximately orthogonal, and error patterns are uncorrelated. Thus, perturbations that cause one signal to fail are unlikely to affect the other, and their combination cancels out errors. With $\alpha = \beta = 0.6$, the bias between known and unknown domains is balanced, matching theoretical expectations.

3. **Energy Regularization Implicitly Smooths the Energy Landscape**:

    - **Function**: Stabilizes the energy function's response to unknown domain samples.
    - **Mechanism**: Proposition 2 in the appendix proves that constraining energy outputs to $(-\infty, \Theta]$ with mean at $\Delta$ implicitly compresses the local Lipschitz constant $K$ of the energy function on the data manifold; for OOD samples $x_{out} = x_{in} + \Delta_x$, the energy shift satisfies $|E(x_{out}) - E(x_{in})| \leq K\|\Delta_x\|$, and a smaller $K$ means OOD samples are less likely to fall into low-energy regions of known domains, thus resisting catastrophic forgetting.
    - **Design Motivation**: Traditional energy training can create "energy cliffs," where OOD samples near the boundary are incorrectly assigned low energy. Compressing the energy distribution into a compact interval "softens the landscape," providing an energy buffer for OOD samples.

### Loss & Training
The total loss is $\mathcal{L}_{total} = \mathcal{L}_{ce} + \lambda \mathcal{L}_{reg}$, with $\mathcal{L}_{reg} = \mathcal{L}_{border} + \mathcal{L}_{midline}$. Hyperparameters: $\Theta = -32, \Delta = -40, K = 5, \alpha = \beta = 0.6, \lambda = 0.05$. SGD with cosine annealing, initial learning rate 0.01.

## Key Experimental Results

### Main Results

| Dataset | Scenario | Prev. SOTA | HEDP | Gain |
|---------|----------|------------|------|------|
| CDDB-Hard | Known AA / AF | CP-Prompt 93.65 / -0.25 | **93.72 / -0.08** | +0.07 / +0.17 |
| CDDB-Hard | Unknown AA | MoP-CLIP 81.98 | **83.74** | +1.76 |
| DomainNet | Known AA (all) | CP-Prompt 73.15 | **74.19** | +1.04 |
| DomainNet | Unknown AA | MoP-CLIP 63.97 | **67.09** | +3.12 |
| CORe50 | Unknown AA | ESN 91.80 | **94.37** | +2.57 |

HEDP achieves the best results on both known and unknown domains, eliminating the trade-off.

### Ablation Study

| Scheme | Energy Boundary | Energy Midline | Energy Factor | Distance Factor | CDDB Unknown | DomainNet Unknown | CORe50 Unknown |
|--------|----------------|---------------|--------------|----------------|--------------|-------------------|----------------|
| 1 (Distance only) | ✗ | ✗ | ✗ | ✓ | 75.80 | 64.97 | 93.17 |
| 2 (No regularized energy) | ✗ | ✗ | ✓ | ✗ | 77.31 | 63.55 | 92.06 |
| 3 (+Boundary) | ✓ | ✗ | ✓ | ✗ | 79.22 | 65.05 | 92.98 |
| 4 (+Midline) | ✗ | ✓ | ✓ | ✗ | 79.12 | 65.01 | 93.77 |
| 5 (Full energy) | ✓ | ✓ | ✓ | ✗ | 81.52 | 65.59 | 94.07 |
| 6 (Full HEDP) | ✓ | ✓ | ✓ | ✓ | **83.74** | **67.09** | **94.66** |

### Key Findings
- **Boundary + Midline Must Be Used Together**: Using either alone is 2-3 points worse than full regularization, indicating they capture different distribution characteristics (max constraint vs. mean alignment).
- **Energy and Distance Are Complementary, Not Redundant**: Adding the distance factor to scheme 5 increases CDDB unknown by 2.22 points; adding the energy factor to scheme 1 increases CDDB unknown by 7.94 points. The complementary effect is especially pronounced on unknown domains.
- **Hyperparameter Effects on Unknown Domains Are Diagonal**: Grid heatmaps of $\alpha, \beta$ show known domains favor "distance-dominated," while unknown domains favor "energy+distance balanced," confirming their different mechanisms.
- The number of clusters $K$ has little impact, indicating the distance factor mainly acts as a "global topological stabilizer" rather than for fine discrimination.

## Highlights & Insights
- **From Physical Intuition to ML Design**: The use of Helmholtz free energy from statistical physics is natural; "energy boundary + midline" correspond to potential well depth and zero point in physics, offering strong interpretability.
- **Gradient Orthogonality Argument**: First-order Taylor expansion demonstrates that the error gradients of energy and distance are orthogonal, providing theoretical support for the "complementary signals" claim, making it more credible than empirical weighting schemes.
- **"Landscape Smoothing" Side Effect of Energy Regularization**: Constraining the energy distribution to a compact interval implicitly reduces the Lipschitz constant, enhancing OOD robustness—this trick can be independently transferred to any OOD detection task.
- For prompt-based continual learning, "how to select prompts" is more important than "how to train prompts," and this work fully exploits this perspective.

## Limitations & Future Work
- Inference latency grows linearly with the number of domains—each test sample must pass through all domain prompt models to compute energy. The authors acknowledge this as a scalability bottleneck and suggest dynamic prompt selection for future work.
- Both $\Theta$ and $\Delta$ are manually tuned hyperparameters, and pushing $\Delta$ further improves performance but eventually saturates; there is a lack of adaptive mechanisms.
- Experiments are limited to visual classification tasks and have not been extended to NLP or VLM inference tasks; whether the physical intuition of energy holds for text generation remains to be verified.
- The argument that "energy is an SFT-invariant signal" is relatively weak, relying mainly on visualizations.

## Related Work & Insights
- **vs CP-Prompt** (ACMMM 2024): CP-Prompt is already strong on known domains but weak on unknown ones; HEDP uses the energy factor to address the generalization gap on unknown domains.
- **vs MoP-CLIP** (WACV 2024): MoP-CLIP uses coarse clustering for prompt mixing, while HEDP combines clustering (distance factor) with internal prompt energy, achieving finer discrimination.
- **vs ESN** (AAAI 2023): ESN introduces a temperature-adjustable energy metric but uses only energy; HEDP adds "energy regularization" to explicitly constrain distribution shape, yielding significantly better results.
- **vs ELI** (CVPR 2022): ELI also uses energy for incremental learning, but task-wise energy manifolds are less suitable for prompt-based DIL; HEDP directly applies energy to prompt outputs, making it more lightweight.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "energy regularization + distance/energy hybrid" is new, and the physical analogy is natural.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + complete ablation + hyperparameter grid + energy distribution visualization, covering both known and unknown domains.
- Writing Quality: ⭐⭐⭐⭐ The narrative is clear, and the appendix provides gradient orthogonality and Lipschitz arguments, enhancing theoretical depth.
- Value: ⭐⭐⭐⭐ Simultaneously addresses the known/unknown trade-off in prompt-based DIL, making it practically valuable for engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AREA: Attribute Extraction and Aggregation for CLIP-Based Class-Incremental Learning](area_attribute_extraction_and_aggregation_for_clip-based_class-incremental_learn.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](../../CVPR2026/model_compression/towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)
- [\[ICML 2026\] Multi-Adapter Representation Interventions via Energy Calibration](multi-adapter_representation_interventions_via_energy_calibration.md)

</div>

<!-- RELATED:END -->
