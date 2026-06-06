---
title: >-
  [Paper Note] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts
description: >-
  [ICCV 2025][Image Generation][Machine Unlearning] This paper proposes a Meta-Unlearning framework for diffusion models that augments standard unlearning objectives with a meta-objective…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Machine Unlearning"
  - "Diffusion Models"
  - "Malicious Fine-tuning Defense"
  - "Meta-Learning"
  - "Concept Erasure"
date: 2026-05-08
content_hash: c32970598c9fe92e
---

# Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts

**Conference**: ICCV 2025
**arXiv**: [2410.12777](https://arxiv.org/abs/2410.12777)  
**Code**: None  
**Area**: Image Generation
**Keywords**: Machine Unlearning, Diffusion Models, Malicious Fine-tuning Defense, Meta-Learning, Concept Erasure

## TL;DR

This paper proposes a Meta-Unlearning framework for diffusion models that augments standard unlearning objectives with a meta-objective, causing benign knowledge associated with unlearned concepts to self-destruct upon malicious fine-tuning, thereby preventing relearning of erased concepts. The framework is compatible with most existing unlearning methods and requires only the addition of a simple meta-objective.

## Background & Motivation

**Background**: With the proliferation of large-scale diffusion models such as Stable Diffusion, machine unlearning has become a critical mechanism for preventing the generation of harmful or copyright-infringing content. Methods such as ESD, SDD, UCE, and RECE can effectively "forget" specific concepts from pretrained models.

**Limitations of Prior Work**: Studies show that even after a model is correctly unlearned and publicly released, malicious users can restore erased concepts through minimal fine-tuning. Fine-tuning on entirely unrelated benign data may also partially recover forgotten capabilities.

**Key Challenge**: Unlearning methods only modify the model's output behavior for targeted concepts, while certain benign concepts retained in the model (e.g., "skin") remain semantically correlated with erased concepts (e.g., "nudity"), providing a "bridge" that facilitates malicious fine-tuning.

**Goal**: How can an unlearned model resist concept relearning when subjected to malicious fine-tuning?

**Key Insight**: Inspired by meta-learning (MAML) — simulating the attacker's fine-tuning process during training to preemptively arrange the parameter space such that optimization along the fine-tuning direction causes benign knowledge to self-destruct.

**Core Idea**: Simulate malicious fine-tuning within the unlearning process and optimize a meta-objective such that the gradient angle between the forget set and retain set exceeds 90°, so that fine-tuning that reduces the forget-set loss automatically increases the retain-set loss.

## Method

### Overall Architecture

The Meta-Unlearning framework consists of two components: (1) a standard unlearning objective $\mathcal{L}_{unlearn}$ (which can be any of ESD/SDD/UCE/RECE) to ensure the model forgets specified concepts; and (2) a meta-objective $\mathcal{L}_{meta}$ to ensure the model cannot relearn erased concepts after malicious fine-tuning. The entire process is formulated as a bilevel optimization, described in Algorithm 1.

### Key Designs

1. **Meta Objective**:

    - **Function**: Prevent the model from relearning erased concepts after malicious fine-tuning, while triggering the self-destruction of associated benign knowledge.
    - **Mechanism**:
    $\mathcal{L}_{meta}(\theta^{FT}) = -\mathcal{L}_{DM}(\theta^{FT}; \mathcal{D}_{FT}) - \zeta[\mathcal{L}_{DM}(\theta^{FT}; \mathcal{D}_{retain}) - \mathcal{L}_{DM}(\theta; \mathcal{D}_{retain})]$
      where $\theta^{FT} = \theta - \tau \nabla_\theta \mathcal{L}_{FT}(\theta; \mathcal{D}_{FT})$ denotes the parameters after simulated malicious fine-tuning.
    - After first-order approximation, two critical terms emerge: (1) $\|\nabla_\theta \mathcal{L}_{DM}(\theta; \mathcal{D}_{FT})\|_2^2$ — minimizing this term reduces gradient norm, slowing relearning; (2) $\nabla_\theta \mathcal{L}_{DM}(\theta; \mathcal{D}_{FT})^\top \nabla_\theta \mathcal{L}_{DM}(\theta; \mathcal{D}_{retain})$ — minimizing this term drives the gradient angle beyond 90°, realizing the self-destruction mechanism whereby reducing the forget-set loss automatically increases the retain-set loss.
    - **Design Motivation**: Conventional unlearning only considers safety at release time, neglecting post-release attack risks. The meta-objective preemptively "plants traps" so that parameter updates along the fine-tuning trajectory are detrimental to the model.

2. **Compatibility Design (Integration with Existing Methods)**:

    - **Function**: Enable seamless integration of the meta-unlearning framework into both optimization-based (ESD/SDD) and closed-form (UCE/RECE) unlearning methods.
    - **Mechanism**: For optimization-based methods (ESD/SDD), the gradient $g$ incorporates contributions from both $\mathcal{L}_{unlearn}$ and $\mathcal{L}_{meta}$; for closed-form methods (UCE/RECE), the closed-form solution $\theta^{UN}$ is first obtained, and then the meta-objective is optimized starting from $\theta^{UN}$.
    - **Implementation**: Meta-gradients can be computed via automatic differentiation without complex implementation.

3. **Bilevel Optimization Structure**:

    - Outer loop ($N$ steps, lr=$\omega$): Updates model parameters to jointly satisfy the unlearning and meta-objectives.
    - Inner loop ($M$ steps, lr=$\tau$): Simulates the malicious fine-tuning process by updating $M$ steps from the current parameters.
    - Weight factors $\gamma_1$ and $\gamma_2$ control the contributions of the unlearning objective and meta-objective, respectively.

### Loss & Training

The malicious fine-tuning dataset $\mathcal{D}_{FT} \subset \mathcal{D}_{forget}$ is sampled from the forget set at each step. Three types of fine-tuning datasets are generated using FLUX.1 for evaluation: HRM-s (single harmful prompt), HRM-m (multiple harmful prompts), and CLEAN (benign prompts). Experiments are conducted on SD-v1-4 and SDXL.

## Key Experimental Results

### Main Results

**SD-v1-4 Nudity Score (selected from Tab.1 and Tab.2)**:

| Method | Type | Before FT | FT HRM-m 50 steps | FT HRM-m 100 steps | FT CLEAN 100 steps |
|------|------|------|------|------|------|
| SD-1.4 Original | - | 97.18 | - | - | - |
| ESD-u-1 | Unlearn | 6.34 | 19.01 | 21.83 | 13.38 |
| ESD-u-1 | **Meta-Unlearn** | **0.00** | **8.45** | **13.38** | **2.11** |
| ESD-u-3 | Unlearn | 3.52 | 26.76 | 38.73 | 4.93 |
| ESD-u-3 | **Meta-Unlearn** | **0.00** | **3.52** | **19.01** | **2.82** |
| SDD | Unlearn | 1.41 | 33.10 | 57.04 | 16.20 |
| SDD | **Meta-Unlearn** | **0.00** | **20.42** | **45.07** | **5.63** |
| UCE | Unlearn | 16.90 | 36.62 | 44.37 | 25.35 |
| UCE | **Meta-Unlearn** | **1.41** | **24.65** | **28.17** | **5.63** |
| RECE | Unlearn | 4.93 | 16.20 | 19.72 | 9.86 |
| RECE | **Meta-Unlearn** | **4.23** | **7.04** | **10.56** | **5.63** |

**Generation Quality Evaluation (Tab.2)**:

| Method | FID | CLIP Score |
|------|-----|-----------|
| Original SD | 16.71 | 31.09 |
| ESD-u-1 Unlearn | 16.01 | 30.32 |
| ESD-u-1 Meta-Unlearn | 16.98 | 30.20 |
| UCE Unlearn | 17.59 | 31.01 |
| UCE Meta-Unlearn | 19.20 | 31.25 |

Meta-Unlearn achieves FID/CLIP scores close to those of the corresponding Unlearn methods, without significantly degrading benign generation quality.

### Ablation Study

**SDXL Copyright Removal (qualitative results in Fig.2)**:

| Scenario | Configuration | Outcome |
|------|------|------|
| SpongeBob/Snoopy | ESD-u-1 Unlearn + FT 100 steps | Copyright characters regenerated |
| SpongeBob/Snoopy | **ESD-u-1 Meta + FT 100 steps** | **Copyright characters still cannot be generated** |
| Thomas Kinkade/Kelly McKernan style | ESD-u-1 Unlearn + FT 100 steps | Style relearned |
| Thomas Kinkade/Kelly McKernan style | **ESD-u-1 Meta + FT 100 steps** | **Style still cannot be reproduced** |

**Gradient Angle Verification (Fig.6)**:

- Throughout the meta-unlearning training process, the regression line of $\nabla_\theta \mathcal{L}_{DM}(\theta; \mathcal{D}_{FT})^\top \nabla_\theta \mathcal{L}_{DM}(\theta; \mathcal{D}_{retain})$ exhibits a downward trend, confirming that the gradient angle progressively exceeds 90° and the self-destruction mechanism is activated.

### Key Findings

- **All six unlearning methods show substantially improved resistance to fine-tuning after incorporating the meta-objective**, with nudity scores after malicious fine-tuning reduced by approximately 30–60%.
- **Improved safety under benign fine-tuning (CLEAN data)**: Conventional unlearning methods may still produce harmful content after CLEAN fine-tuning (nudity scores of 9–25%), whereas meta-unlearning reduces this to only 2–6%.
- **Copyright and style protection are equally effective**: Concepts such as SpongeBob, Snoopy, and Thomas Kinkade remain irrecoverable after 100 steps of fine-tuning following meta-unlearning.
- **Generation quality is largely preserved**: Differences in FID and CLIP Score are minimal, and benign prompt generation quality is maintained.

## Highlights & Insights

- **Elegant application of meta-learning in the security domain** — MAML's notion of "learning to learn" is repurposed as "learning to prevent others from learning." The mathematical foundation lies in controlling gradient directions to create mutual opposition. This idea is concise, elegant, and highly generalizable.
- **Gradient-level interpretation of the self-destruction mechanism** — The first-order approximation reveals two roles of the meta-objective: reducing gradient norm (deceleration) and rotating gradient direction (self-destruction), providing clear theoretical intuition.
- **Plug-and-play design** — Only a few lines of meta-objective code need to be added on top of any existing unlearning method, without altering the original workflow, making it highly practical.

## Limitations & Future Work

- Meta-training requires simulating the fine-tuning process (inner loop), introducing additional computational overhead from second-order gradients.
- Evaluation is limited to at most 300 fine-tuning steps; whether resistance persists under longer fine-tuning remains to be verified.
- The improvement from meta-unlearning is relatively modest for RECE, suggesting limited meta-optimization headroom for closed-form methods.
- Adaptive attacks by adversaries who are aware of the meta-unlearning mechanism (e.g., restoring gradient alignment first) are not discussed.
- Hyperparameters in the meta-objective ($\zeta$, $\gamma_1$, $\gamma_2$, inner step count $M$) require tuning.

## Related Work & Insights

- **vs. ESD/SDD**: Standard unlearning methods can reach nudity scores of 20–57% after 100 fine-tuning steps, demonstrating that unlearning is not durable; Meta-Unlearn reduces these figures to 10–45%.
- **vs. AdvUnlearn**: AdvUnlearn enhances robustness via adversarial training but can still be circumvented by fine-tuning; Meta-Unlearn fundamentally reshapes the gradient landscape of the parameter space.
- **vs. RECE**: RECE iteratively constructs new erasure embeddings and is already among the strongest unlearning methods (pre-unlearning nudity score of only 4.93%); Meta-Unlearn further reduces this to 4.23%.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The combination of meta-learning and machine unlearning is highly novel; the gradient analysis of the self-destruction mechanism is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 6 unlearning baselines, 2 base models, 3 attack scenarios, and three task categories (copyright, style, and safety).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are clear, and the intuitive explanation in Fig.1 is excellent.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a core practical problem in the machine unlearning field with strong generalizability and deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] When Are Concepts Erased From Diffusion Models?](../../NeurIPS2025/image_generation/when_are_concepts_erased_from_diffusion_models.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](../../NeurIPS2025/image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[NeurIPS 2025\] Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning](../../NeurIPS2025/image_generation/towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](../../CVPR2026/image_generation/erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)

</div>

<!-- RELATED:END -->
