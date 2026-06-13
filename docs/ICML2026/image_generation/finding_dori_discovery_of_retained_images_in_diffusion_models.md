---
title: >-
  [Paper Note] Finding DoRI: Discovery of Retained Images in Diffusion Models
description: >-
  [ICML2026][Image Generation][Diffusion model memorization] The authors demonstrate via a simple adversarial text embedding optimization method (DoRI) that diffusion model memorization mitigation schemes based on "pruning…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Diffusion model memorization"
  - "adversarial embedding"
  - "pruning failure"
  - "adversarial fine-tuning"
  - "copyright and privacy"
date: 2026-05-08
content_hash: 6fa5323e905f1da0
---

# Finding DoRI: Discovery of Retained Images in Diffusion Models

**Conference**: ICML2026  
**arXiv**: [2507.16880](https://arxiv.org/abs/2507.16880)  
**Code**: https://github.com/sprintml/finding_dori  
**Area**: AI Safety / Diffusion Model Memorization / Training Data Privacy  
**Keywords**: Diffusion model memorization, adversarial embedding, pruning failure, adversarial fine-tuning, copyright and privacy

## TL;DR
The authors demonstrate via a simple adversarial text embedding optimization method (DoRI) that diffusion model memorization mitigation schemes based on "pruning and localizing memory neurons" (such as NeMo / Wanda) merely "hide" memories rather than truly erasing them. This occurs because memorization is not local at the embedding, activation, or weight levels. Furthermore, an adversarial fine-tuning scheme is proposed to truly extract training samples from the model.

## Background & Motivation

**Background**: Text-to-image diffusion models (DMs, e.g., Stable Diffusion v1.4) can reproduce training images verbatim, posing significant privacy and copyright risks. Current mainstream "permanent mitigation" routes, represented by NeMo (Hintersdorf et al., 2024) and Wanda (Chavhan et al., 2024), involve observing abnormal activations when a memorized prompt is triggered, localizing a small number of "memory neurons/weights," and pruning them. They claim to erase specific training samples without harming overall generation quality.

**Limitations of Prior Work**: All these pruning methods assume that memorization is **local**—that the "fingerprint" of a specific training image is hidden within only a few weights, making directional pruning sufficient. However, their effectiveness has only been verified using the **original memorized prompts**; no one has checked whether the same image can still be retrieved using different inputs.

**Key Challenge**: Pruning severs the "prompt $\to$ memorized image" mapping, but the memorized image may not follow only this single path. If other paths exist in the text embedding space that can trigger the same image, "pruning as forgetting" is merely superficial.

**Goal**: (1) Design a method to actively search for "retained triggers" in pruned models; (2) Systematically test whether the memorization locality hypothesis holds at the embedding, activation, and weight levels; (3) Provide a scheme that can truly erase memorized data upon rejecting locality.

**Key Insight**: The search for "retained triggers" is modeled as a continuous adversarial optimization over text embeddings—unconstrained by the discreteness of natural language. It directly uses diffusion loss gradient descent in the embedding space to find triggers. If an embedding capable of reproducing the original image can still be optimized on a pruned model, it indicates the memory was not erased.

**Core Idea**: Utilize DoRI (Discovery of Retained Images)—an adversarial embedding optimizer that repeatedly resamples noise and timesteps—as both a "memorization detection probe" and an "adversarial training sample generator" to expose the illusion of pruning and perform adversarial fine-tuning for true erasure.

## Method

### Overall Architecture
The paper is structured into three phases: "Probe + Diagnosis + Treatment":

1.  **Probe (DoRI)**: Given a known memorized training image $\bm{x}_{\mathit{mem}}$ and a model pruned by NeMo/Wanda ($\bm{\theta}_{N/W}$), search for $\bm{y}_{adv}$ in the text embedding space using diffusion loss gradient descent to re-generate $\bm{x}_{\mathit{mem}}$.
2.  **Triple Locality Diagnosis**: Use DoRI to generate a large number of adversarial embeddings and check them across (a) distribution in text embedding space, (b) discrepancies in activations across layers, and (c) the overlap rate of weight sets flagged by pruning methods. All three levels reject the "memorization = local minor weights" hypothesis.
3.  **Treatment (Adversarial Fine-tuning)**: Use DoRI as an inner loop and re-fit using surrogate images in the outer layer to "wash" memorized samples out of the model while maintaining diffusion loss on normal image-caption pairs to preserve generation quality.

The experiments use Stable Diffusion v1.4 with 500 known memorized prompts from LAION-5B (following standard benchmarks by Wen et al.), with generalization verification on SD v2.0.

### Key Designs

1.  **DoRI Adversarial Embedding Optimization (Probe)**:

    *   **Function**: Given a memorized image $\bm{x}_{\mathit{mem}}$ and a (possibly pruned) model, find a text embedding $\bm{y}_{adv}$ that can stably reproduce that image.
    *   **Mechanism**: Initialize $\bm{y}_{adv}$ as the embedding of the original prompt (or random Gaussian / non-memorized prompt) and iterate for 50 steps using Adam with $\eta=0.1$ and batch=8: $\bm{y}_{adv}^{(i+1)} = \bm{y}_{adv}^{(i)} - \eta \nabla_{\bm{y}_{adv}^{(i)}} \mathcal{L}(\bm{x}_{\mathit{mem}}, \bm{\epsilon}, \bm{y}_{adv}^{(i)}, t, \bm{\theta}_{N/W})$. In each step, **noise $\bm{\epsilon}\sim\mathcal{N}(0,I)$ and timestep $t\sim\mathcal{U}(1,T)$ are resampled**. This forces the discovered embedding to trigger reproduction under any initial noise, rather than memorizing a specific noise-timestep combination.
    *   **Design Motivation**: Continuous embeddings offer a much larger search space than discrete prompts, exposing "hidden channels" in pruned models. Resampling noise/timesteps is a critical anti-cheating design; otherwise, the model might merely overfit a specific sampling trajectory. The authors demonstrate in the appendix that for **non-memorized images**, the same setting requires $>500$ steps to force reproduction, so triggering implies memorization without false positives (Memorization Rate on non-memorized prompts is only 0.02).

2.  **Triple Locality Diagnosis (Diagnosis)**:

    *   **Function**: Use DoRI to quantify how dispersed memorization triggers actually are, falsifying locality from embedding, activation, and weight perspectives.
    *   **Mechanism**: (a) **Embedding Layer**: Run DoRI for 50 steps on 100 randomly initialized $\bm{y}_{adv}^{(0)}\sim\mathcal{N}(0,I)$ for the same memorized image. t-SNE shows that adversarial embeddings are as dispersed as the initial random points; pairwise L2 distances are even larger than those between non-memorized prompts. (b) **Activation Layer**: Define discrepancy as the mean pairwise $\ell_2$ distance of activations in the same layer under different embeddings (fixed noise). The activation discrepancy between 100 $\bm{y}_{adv}$ that trigger the same image is **comparable** to the discrepancy between 100 completely different memorized prompts. (c) **Weight Layer**: Define weight agreement as the Intersection over Union (IoU) of the "to-be-pruned weight sets" flagged by NeMo/Wanda under different embeddings. Wanda shows $<0.6$ in most layers. NeMo seemingly shows $>0.8$, but this is because it doesn't select weights in layers 2, 6, and 7 (agreement is forced to 1); the actual effective layer (Layer 1) shows only 0.6.
    *   **Design Motivation**: The legitimacy of pruning methods rests on the assumption that the same memorized image corresponds to the same small cluster of weights. These three experiments systematically negate this hypothesis. Notably, the weight agreement experiment uses the pruning methods' own localization logic to refute them—different embeddings for the same image lead to non-overlapping pruned weight sets, indicating that "memory neurons" are input-dependent pseudo-local solutions rather than objective internal storage locations.

3.  **Adversarial Fine-tuning (Treatment)**:

    *   **Function**: Use DoRI as an inner loop to actively generate adversarial embeddings and use them to fine-tune the full model, completely erasing memorized images from the weights.
    *   **Mechanism**: In each fine-tuning step, DoRI first collects a batch of $\bm{y}_{adv}$ for each memorized image to be erased. Then, a pre-prepared surrogate image $\widetilde{\bm{x}}$ (generated by the pruned model + original prompt, "semantically similar but pixel-different") is used as the target. The loss is $\mathcal{L}_{Adv}(\widetilde{\bm{x}}_0, \bm{\epsilon}, \bm{y}_{adv}, t, \bm{\theta}) = \|\bm{\epsilon} - \bm{\epsilon}_{\bm{\theta}}(\widetilde{\bm{x}}_t, t, \bm{y}_{adv})\|_2^2$, pulling the output of the adversarial embedding from the original image toward the surrogate. Simultaneously, a standard diffusion loss $\mathcal{L}_{\mathrm{DM}}$ on regular LAION image-caption pairs is added to prevent overall model collapse. Total loss $\mathcal{L} = \mathcal{L}_{\mathrm{DM}} + \mathcal{L}_{Adv}$. Full-model fine-tuning is used (LoRA failed, further supporting the need for global tuning).
    *   **Design Motivation**: Since memorization is distributed, the entire model must be tuned against multiple triggers simultaneously. Using a surrogate instead of a random image as a target avoids deleting "semantic content" (distinguishing this from concept unlearning) while preventing the introduction of new memorized samples.

### Loss / Training Strategy
Adversarial fine-tuning is performed for 5 epochs. A single epoch significantly reduces the memorization rate, while the standard diffusion loss maintains model versatility. The inner-loop DoRI uses 50 adversarial optimization iterations to generate new $\bm{y}_{adv}$. Hyperparameters tuned on SD v1.4 remain effective when transferred to SD v2.0.

## Key Experimental Results

### Main Results (DoRI Exposing Pruning + Adversarial Fine-tuning Effectiveness)

| Setup | SSCD$_{\mathrm{Orig}}$ ↓ | MR ↓ | Description |
| :--- | :--- | :--- | :--- |
| Original DM + Memorized prompt | 0.90 ± 0.04 | 0.98 | Memorization baseline |
| Non-memorized prompt + DoRI | 0.48 ± 0.06 | 0.00 | DoRI does not report false positives |
| NeMo (Pruning) | 0.33 ± 0.18 | 0.20 | Appears to successfully erase |
| NeMo + DoRI | **0.91 ± 0.03** | **0.99** | Adversarial embeddings recover almost everything |
| Wanda (Pruning) | 0.20 ± 0.08 | 0.00 | Appears more thorough |
| Wanda + DoRI | **0.76 ± 0.05** | **0.72** | 72% of memorized images still retrieved |
| ESD + DoRI | 0.90 ± 0.04 | 0.98 | Concept unlearning fails to protect |
| Concept Ablation + DoRI | 0.91 ± 0.04 | 0.97 | Same as above |
| SISS (Data Unlearning SOTA) + DoRI| 0.60 ± 0.22 | 0.39 | Better than pruning but misses 39% |
| **Ours (Adv. FT) + DoRI** | **0.36 ± 0.14** | **0.02** | Almost zeroed; only one highly repeated sample remains |

Conclusions are consistent on SD v2.0: NeMo+DoRI pulls MR from 0.06 back to 1.00, while the proposed method suppresses MR to 0.06; FID decreased from 14.44 to 13.61, indicating a slight improvement in generation quality.

### Ablation Study / Locality Diagnosis

| Dimension | Phenomenon | Locality Holds? |
| :--- | :--- | :--- |
| Embedding Distribution | Pairwise L2 of $\bm{y}_{adv}$ for the same image > distance between non-memorized prompts | No |
| Activation Discrepancy | Discrepancy of 100 adv. embeddings $\approx$ discrepancy between 100 different memorized prompts | No |
| Weight Agreement | Wanda <0.6 (most layers); NeMo 0.6 in L1, other "high" layers are artifacts of zero-flagging | No |
| LoRA Adv. FT | Failed | Further rejects locality |
| Wanda (10% Sparsity) | Resists DoRI but generation quality collapses | Locality patch not viable |

### Key Findings
- **Pruning $\neq$ Erasing**: Local pruning like NeMo/Wanda only cuts the shortest path in prompt space, turning memorization from "retrieve via doorbell" to "retrieve via password," which DoRI brute-forces.
- **Memorization is Distributed**: Three levels of evidence converge—the same image can be triggered by almost any point in the embedding space, and the corresponding activations and "important weights" do not overlap. This suggests DMs do not store images in specific neurons but scatter them throughout the network's geometry.
- **Adversarial Fine-tuning is Necessary and Sufficient**: One must tune globally (LoRA fails) and against multiple triggers (concept unlearning/SISS only target single prompts and miss $\ge39\%$). If these are done correctly, 5 epochs are sufficient without degrading FID.

## Highlights & Insights
- **Unifying "Memorization Detection" and "Adversarial Training"**: DoRI acts both as an auditing tool to expose the pseudo-security of pruning and as a training data generator to drive adversarial fine-tuning. This "common source for attack and defense" design is efficient and self-consistent—whatever can be found can be washed away in the same manner.
- **Noise/Timestep Resampling is the Critical Trick**: Many adversarial optimizations fix $\bm{\epsilon}, t$, resulting in the optimizer learning to "cheat the model on this specific trajectory." Resampling at every step ensures $\bm{y}_{adv}$ truly triggers the image stably, raising the "success" bar to be equivalent to "true residual memory," causing the 50-step threshold to clearly separate memorized vs. non-memorized images (MR 0.99 vs. 0.02).
- **Using "Pruning Methods' Own Localization Operators" for IoU**: The weight agreement evaluation cleverly refutes the opponent's core assumption using their own logic—if they could localize memory neurons, different triggers for the same image should point to the same weights. Measured IoU failing to reach 0.6 effectively forces pruning methods to admit their "localization" is an input-dependent random solution.
- **Transfer Inspirations**: The paradigm of "Adversarial optimization to expose hidden triggers $\to$ Triple-level evidence to deny locality $\to$ Adversarial fine-tuning for global erasure" is easily transferable to LLM training data memorization, unlearning evaluation, and backdoor detection. The core takeaway is to distrust any safety method evaluated only on "original triggers."

## Limitations & Future Work
- **Older Model Scale**: Experiments focus on SD v1.4 / v2.0 because only they have public, large-scale "known memorized prompt" sets (Wen 2024, Webster 2023). Newer models like SDXL / FLUX require community-built benchmarks due to training data deduplication.
- **Dependency on Known Memorized Images**: DoRI requires knowing which images were memorized beforehand. For "unknown memorized images," it must be paired with detection pipelines.
- **Computational Cost**: Adversarial fine-tuning with an inner-loop 50-step optimization for each image to be erased is expensive; the cost and engineering for large-scale erasure (e.g., tens of thousands of images) are not fully discussed.
- **Weak TM (Template Memorization) Evaluation**: The authors note that metrics like SSCD are insensitive to template-level memorization. This leaves a gap for future work to solve the objective measurement of "semantic rather than pixel-level" memorization.

## Related Work & Insights
- **vs. NeMo / Wanda**: This paper directly breaks these methods. While they assume memory resides in a few weights of cross-attention value or FFN layers, DoRI proves memory can be retrieved via detours, and increasing pruning intensity destroys generation quality.
- **vs. SISS (ICLR 2025)**: SISS is a data unlearning SOTA that prevents original prompts from triggering images but still misses 39% under DoRI. This method upgrades "passive forgetting" to "active forgetting" by feeding adversarial embeddings into the training loop, pressing MR to 2%.
- **vs. Concept Unlearning (ESD, Concept Ablation)**: Concept unlearning deletes categories (e.g., "cars"), not specific images. Experiments show these are ineffective against individual memorization (MR 0.97-0.98), emphasizing that "concept unlearning $\neq$ data unlearning" in DMs.
- **vs. UnlearnDiffAtk (Zhang et al., 2024c)**: UnlearnDiffAtk searches for triggers in concept unlearning; this paper shows it fails in verbatim reproduction tasks, justifying the need for DoRI.
- **Inheriting from Zhang et al., 2026**: Regarding the intuition that memorized images induce "spike activations," this paper provides a corollary—if internal states behave like attractors, there are naturally many "entrances" to this attractor in the embedding space, not restricted to the vicinity of the original prompt.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The discovery that "memorization is global rather than local" overturns the mainstream pruning mitigation route, and the same DoRI tool unifies attack and defense.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ SD v1.4 + v2.0 dual models, 500 prompts, triple locality diagnosis, and comprehensive comparison against four baseline types (pruning/concept/data/LoRA). Half a star deducted due to the lack of benchmarks for newer models.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Argumentation is clear (rebuttal $\to$ diagnosis $\to$ reconstruction). Each table directly serves the core thesis, and appendices cover all details (threshold selection, false positive prevention, strengthened pruning controls).
- **Value**: ⭐⭐⭐⭐⭐ Redefines evaluation standards for DM memorization mitigation (must be adversarial against multiple triggers) and provides a deployable adversarial fine-tuning solution, with direct engineering significance for privacy/copyright audits before releasing open-source models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](../../NeurIPS2025/image_generation/what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)
- [\[CVPR 2026\] SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images](../../CVPR2026/image_generation/simlbr_learning_to_detect_fake_images_by_learning_to_detect_real_images.md)
- [\[AAAI 2026\] Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images](../../AAAI2026/image_generation/beautiful_images_toxic_words_understanding_and_addressing_offensive_text_in_gene.md)
- [\[ICCV 2025\] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](../../ICCV2025/image_generation/attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)
- [\[ICML 2026\] Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior](initialization_is_half_the_battle_generating_diverse_images_from_a_guidance_pote.md)

</div>

<!-- RELATED:END -->
