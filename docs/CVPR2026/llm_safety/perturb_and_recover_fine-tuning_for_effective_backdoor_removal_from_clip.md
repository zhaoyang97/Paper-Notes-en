---
title: >-
  [Paper Note] Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP
description: >-
  [CVPR 2026][LLM Safety][backdoor attack] This paper proposes PAR (Perturb and Recover), a simple yet effective backdoor cleansing method for CLIP: by explicitly pushing model embeddings away from the poisoned state (Perturb) while recovering clean performance via the standard CLIP loss (Recover), PAR achieves robust backdoor removal against arbitrary trigger types without relying on strong data augmentation, and remains effective even when using only synthetic data.
tags:
  - CVPR 2026
  - LLM Safety
  - backdoor attack
  - CLIP model cleansing
  - fine-tuning defense
  - structured triggers
  - synthetic data
date: 2026-05-08
content_hash: fcb26ed516365082
---

# Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP

**Conference**: CVPR 2026
**arXiv**: [2412.00727](https://arxiv.org/abs/2412.00727)
**Code**: [https://github.com/](https://github.com/) (available, as declared in the paper)
**Area**: AI Safety
**Keywords**: backdoor attack, CLIP model cleansing, fine-tuning defense, structured triggers, synthetic data

## TL;DR

This paper proposes PAR (Perturb and Recover), a simple yet effective backdoor cleansing method for CLIP: by explicitly pushing model embeddings away from the poisoned state (Perturb) while recovering clean performance via the standard CLIP loss (Recover), PAR achieves robust backdoor removal against arbitrary trigger types without relying on strong data augmentation, and remains effective even when using only synthetic data.

## Background & Motivation

1. **State of the Field**: Vision-language models such as CLIP are widely used for zero-shot classification, retrieval, and as visual encoders in large VLMs such as LLaVA. Because training data is scraped from the web (e.g., LAION-400M), these models are highly vulnerable to backdoor attacks—a poisoning rate as low as 0.01% is sufficient to successfully inject a backdoor.

2. **Limitations of Prior Work**: Existing cleansing methods such as CleanCLIP and RoCLIP rely heavily on **strong data augmentation** (e.g., AutoAugment) to break backdoor associations. This strategy carries an implicit assumption—the augmentation operations must overlap with the trigger distribution to be effective. For image-text pairs, strong augmentation also destroys annotation semantics (e.g., horizontal flipping alters positional descriptions, and color jitter invalidates color descriptions).

3. **Root Cause**: Augmentation-based cleansing is essentially "guessing" what the trigger might look like. If the trigger is random noise, noise-based augmentation is indeed effective; however, if the trigger is **structured** (stripes, triangles, watermark text, etc.), these structures survive augmentation intact, causing cleansing to fail entirely. In practice, defenders have no knowledge of the trigger type.

4. **Paper Goals**: To design a universal backdoor cleansing method that is **agnostic to trigger type** and operates via fine-tuning rather than retraining from scratch, keeping computational cost manageable.

5. **Starting Point**: The key insight is that instead of guessing and covering all possible trigger patterns, it is more effective to directly "push the model away" from the poisoned state. Once the model's embedding space diverges sufficiently from the poisoned model, the spurious associations introduced by the backdoor are naturally disrupted.

6. **Core Idea**: Apply an $\ell_2$ distance loss to actively push the poisoned embeddings away (Perturb), while simultaneously using the CLIP contrastive loss to maintain clean performance (Recover). The two objectives compete to achieve "forget the backdoor, retain the knowledge."

## Method

### Overall Architecture

PAR takes as input a backdoor-poisoned CLIP model and a set of clean image-text pairs (250K pairs, only 1/1600 of the training data). PAR cleanses the model via fine-tuning over approximately 10 epochs. The core idea is to simultaneously optimize two competing objectives: a "push-away" loss that forces the current model away from the poisoned snapshot, and a "recovery" loss that maintains clean CLIP performance.

### Key Designs

1. **Structured Triggers**:

    - *Function*: Demonstrate the vulnerability of existing cleansing methods and construct stronger attacks as evaluation benchmarks.
    - *Mechanism*: Four new structured triggers are proposed—BadNet-Stripes (1-pixel-wide colored stripe patches), Blended-Stripes (full-image overlaid stripes, $n_c=0.03$), Blended-Triangles (overlaid low-contrast triangles, $n_c=0.15$), and Blended-Text (red "Watermarked" watermark text, $n_c=0.5$). These triggers simulate real-world patterns such as watermarks.
    - *Design Motivation*: Random noise triggers happen to overlap with the noise augmentations in AutoAugment, which is why CleanCLIP can cleanse them effectively. However, structured patterns such as stripes and triangles survive all augmentation operations, exposing the fundamental flaw of augmentation-based methods. Experiments show that even increasing CleanCLIP's augmentation loss weight $\lambda$ to extreme values barely reduces the ASR for BadNet-Stripes.

2. **Perturbation Loss ($\mathcal{L}_{\text{PERT}}$)**:

    - *Function*: Actively push model embeddings away from the poisoned state to break backdoor associations.
    - *Mechanism*: Let the normalized embeddings of the frozen poisoned model be $\phi_P(\cdot), \psi_P(\cdot)$, and those of the current model being cleansed be $\phi(\cdot), \psi(\cdot)$. The mean $\ell_2$ distances for the visual and text encoders are computed per batch as $S_\phi = \frac{1}{|B|}\sum \|\phi(x_I^n) - \phi_P(x_I^n)\|_2^2$ and $S_\psi$. A threshold $\tau$ is introduced for truncation: $\mathcal{L}_{\text{PERT}} = \frac{1}{2}(\mathbb{I}[S_\phi \leq \tau] \cdot S_\phi + \mathbb{I}[S_\psi \leq \tau] \cdot S_\psi)$. Once both encoders have been pushed beyond $\tau$, the perturbation loss is automatically deactivated.
    - *Design Motivation*: This design elegantly achieves **controlled perturbation**—$\tau$ serves both as a knob controlling the degree of push-away and as a safety net preventing the model from being pushed so far that knowledge is lost. Since the $\ell_2$ distance between normalized embeddings is upper bounded by 4, the objective is well-suited for minimization. Equivalently, $S_\phi = 2 - \frac{2}{|B|}\sum \cos(\phi, \phi_P)$, meaning the loss simultaneously minimizes cosine similarity with the poisoned model.

3. **PAR Total Loss**:

    - *Function*: Simultaneously achieve "forgetting the backdoor" and "maintaining performance."
    - *Mechanism*: $\mathcal{L}_{\text{PAR}} = \mathcal{L}_{\text{CLIP}} - \mathcal{L}_{\text{PERT}}$. Minimizing this objective simultaneously maximizes contrastive learning performance and the distance from the poisoned model. In early training, $\mathcal{L}_{\text{PERT}}$ dominates and $\mathcal{L}_{\text{CLIP}}$ temporarily increases (performance degrades); once the push-away distance reaches the threshold, $\mathcal{L}_{\text{PERT}}$ is truncated to 0 and the model optimizes only $\mathcal{L}_{\text{CLIP}}$ to recover performance.
    - *Design Motivation*: The two-phase dynamics are fully adaptive with no manual scheduling required. During cleansing, only two light augmentations—Gaussian noise and small-patch CutOut—are used, which do not corrupt image-text semantics.

4. **Cleansing with Synthetic Data**:

    - *Function*: Cleanse models using synthetic image-text pairs when access to real clean data is unavailable.
    - *Mechanism*: SynthCLIP (image-text pairs generated by a text-to-image diffusion model) with 250K or 500K samples is used in place of real CC3M data. PAR remains effective at cleansing backdoors, though ImageNet zero-shot accuracy decreases slightly.
    - *Design Motivation*: In real-world scenarios, obtaining data that is guaranteed to be clean is costly. Synthetic data completely eliminates the risk of poisoning, and PAR's perturbation mechanism exhibits a degree of robustness to distribution shift.

### Loss & Training

The learning rate is linearly decayed from 3e-5 to 3e-6 (first half), then cosine-annealed to 1e-9 (second half). $\tau=2.15$ is tuned on BadNet-Stripes with RN50 and then fixed for all attacks and encoders. Cleansing runs for 10 epochs using 250K clean samples.

## Key Experimental Results

### Main Results (RN50 CLIP)

| Attack | Metric | PAR ASR | CleanCLIP ASR | RoCLIP ASR | PAR Clean |
|--------|--------|---------|---------------|------------|-----------|
| BadNet-Rand | ImageNet ASR | **6.3%** | 14.5% | 75.1% | 53.3% |
| BadNet-Stripes | ImageNet ASR | **42.4%** | 62.3% | 82.0% | 53.0% |
| Blended-Rand | ImageNet ASR | **0.0%** | 19.5% | 1.5% | 53.6% |
| Blended-Stripes | ImageNet ASR | **0.1%** | 61.8% | 7.0% | 53.5% |
| Blended-Triangles | ImageNet ASR | **10.3%** | 48.7% | 37.1% | 52.9% |
| Blended-Text | ImageNet ASR | **18.1%** | 42.4% | 59.1% | 53.4% |
| WaNet | ImageNet ASR | **0.0%** | 0.0% | 2.0% | 54.4% |
| BadCLIP | ImageNet ASR | **30.4%** | 40.1% | — | 53.4% |

PAR achieves the lowest ASR on nearly all attacks while maintaining the highest or comparable clean accuracy. CleanCLIP reaches ASR above 60% on structured triggers, rendering it essentially ineffective.

### ViT-B/32 Results

| Attack | PAR ASR | CleanCLIP ASR | Note |
|--------|---------|---------------|------|
| BadNet-Stripes | 0.1% | 86.8% | Largest gap; PAR achieves near-perfect cleansing |
| Blended-Stripes | 0.1% | 15.2% | PAR near-perfect |
| Blended-Triangles | 15.9% | 91.4% | CleanCLIP completely fails |
| Blended-Text | 37.3% | 62.9% | PAR significantly superior |

### Key Findings

- **CleanCLIP completely fails against structured triggers**—the most important finding. Even increasing its augmentation loss weight does not reduce ASR, demonstrating the fundamental flaw of augmentation-based methods.
- **PAR generalizes across architectures**—the same $\tau$ and training strategy transfer directly from RN50 to ViT-B/32 with consistently strong results. The appendix further validates ViT-L/14 and SigLip.
- **Synthetic data is nearly equally effective**—500K SynthCLIP samples reduce the ASR of BadNet-Stripes from 99.8% to 3.7%, making the approach fully viable in practical settings.
- **Adaptive attacks cannot bypass PAR**—re-poisoning a cleansed model and applying PAR again still results in complete backdoor removal; PAR does not "revert" to the original poisoned model.

## Highlights & Insights

- **The "perturb-then-recover" idea is remarkably clean and elegant**—it requires no assumption about trigger morphology, no complex augmentation strategy, and achieves universal cleansing using only an $\ell_2$ distance.
- **Threshold truncation is an elegant adaptive mechanism**—$\tau$ causes perturbation to stop automatically, after which the model naturally enters a pure recovery phase without any manual switching of the training strategy.
- **The introduction of structured triggers is itself an important contribution**—it exposes a fundamental blind spot in existing CLIP defenses and carries significant warning value for the security community.
- **Using synthetic data for model cleansing is a promising direction**—it lowers the barrier to backdoor defense and is particularly suitable for scenarios where access to the original training data is unavailable.

## Limitations & Future Work

- ASR does not reach 0% on certain attacks (e.g., BadNet-Stripes, BadCLIP)—while far superior to baselines, this may be insufficient in high-security settings.
- $\tau$ is tuned on a single attack type; in principle, different attacks and architectures may require different values of $\tau$. The authors claim good generalization but provide no theoretical guarantee.
- Evaluation is limited to classification and retrieval tasks; the effectiveness of PAR when CLIP serves as a visual encoder in VLMs (e.g., LLaVA) has not been verified.
- Cleansing still requires 250K samples and 10 epochs of fine-tuning; scalability to very large models (e.g., ViT-G) is unknown.

## Related Work & Insights

- **vs. CleanCLIP**: CleanCLIP uses $\mathcal{L}_{\text{CLIP}} + \lambda \mathcal{L}_{\text{UniAug}}$, relying on a unimodal self-supervised term with strong augmentation to break backdoors. PAR uses $\mathcal{L}_{\text{CLIP}} - \mathcal{L}_{\text{PERT}}$, breaking backdoors by explicitly pushing away from the poisoned model, with no dependence on augmentation. The essential distinction is that CleanCLIP is an "indirect strike" (hoping augmentation covers the trigger), whereas PAR is a "direct push" (regardless of trigger type, the model is simply forced to diverge).
- **vs. RoCLIP / SafeCLIP**: These methods require training from random initialization, incurring 2–3 orders of magnitude higher computational cost, and result in significantly reduced clean accuracy. PAR only fine-tunes, preserving near-original clean accuracy.
- **vs. unimodal defenses (ANP, SAU, etc.)**: These cannot be directly applied to multimodal models such as CLIP, as they require access to poisoned data, introduce additional parameters, or demand large amounts of training data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The perturb-then-recover idea is concise and effective, though perturbation-based fine-tuning is not an entirely novel concept.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 8 attack types, RN50/ViT-B/32/ViT-L/14/SigLip, classification and retrieval, and both real and synthetic data; extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated; t-SNE visualizations and training dynamics analysis are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses a practical security concern for foundational CLIP models; the synthetic data solution offers strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](../../NeurIPS2025/llm_safety/toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[CVPR 2026\] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[AAAI 2026\] FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA](../../AAAI2026/llm_safety/fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)

<!-- RELATED:END -->
