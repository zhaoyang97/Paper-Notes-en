---
title: >-
  [Paper Note] You Don't Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models
description: >-
  [ICML2026][Image Generation][Text-to-Image Diffusion Models] This paper proposes GUARD, an inference-time framework for mitigating memorization in text-to-image diffusion models. By adding "repulsion" from original memor…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Text-to-Image Diffusion Models"
  - "Training Data Memorization"
  - "Inference-time Mitigation"
  - "cross-attention"
  - "GUARD"
date: 2026-05-08
content_hash: b0fb13487c437fe0
---

# You Don't Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models

**Conference**: ICML2026  
**arXiv**: [2603.00133](https://arxiv.org/abs/2603.00133)  
**Code**: https://github.com/kairanzhao/GUARD  
**Area**: Image Generation / Diffusion Models / Memorization Mitigation  
**Keywords**: Text-to-Image Diffusion Models, Training Data Memorization, Inference-time Mitigation, cross-attention, GUARD  

## TL;DR
This paper proposes GUARD, an inference-time framework for mitigating memorization in text-to-image diffusion models. By adding "repulsion" from original memorized prompts and "attraction" toward safe conditional predictions to the standard classifier-free guidance, and instantiating the positive target using dynamic cross-attention spike detection and attenuation, the method reduces training image replication while maintaining image quality and prompt alignment.

## Background & Motivation
**Background**: Text-to-image diffusion models have been shown to replicate certain images from the training set under specific prompts, posing privacy leakage and copyright risks. Existing mitigation methods are generally categorized into training-time prevention, finetuning/unlearning, and inference-time intervention.

**Limitations of Prior Work**: Training-time methods require control over the original training process, but many applications rely on open-source or commercial pre-trained models. Unlearning requires additional fine-tuning for the forget set, which is costly and potentially unstable. Existing inference-time methods often only scale conditional components, modify initial noise, or handle fixed tokens like EOT/padding, making it difficult to cover both verbatim memorization and template memorization simultaneously.

**Key Challenge**: Forcefully suppressing prompt conditional signals can reduce training sample replication but leads to a decline in image semantics and quality. Conversely, redistribution focused only on fixed token attention misses the actual tokens triggering memorization across different prompts. The goal is to "precisely suppress memorization triggers during inference while preserving prompt-related generation capabilities."

**Goal**: The authors aim to design an inference-time method that requires no weight changes, no retraining, and no prior knowledge of the training process. The method should dynamically locate memorization-related cross-attention spikes for each prompt and pull the generation trajectory away from memorized samples during the standard denoising process.

**Key Insight**: The paper re-analyzes cross-attention distributions of memorized vs. non-memorized prompts. It finds that in verbatim memorization, EOT often has strong spikes, but other tokens can also be sharper. In template memorization, EOT is not even necessarily the primary issue. Therefore, fixed EOT attenuation is unreliable, and per-prompt statistical detection is essential.

**Core Idea**: Inference-time guidance is formulated as attractive-repulsive dynamics: repulsion pulls away from conditional predictions of the original memorized prompt, while attraction points toward a safe conditional prediction subject to dynamic attention spike attenuation.

## Method
GUARD does not train a new diffusion model but rewrites the noise prediction combination at each denoising step. It retains the unconditional prediction and computes two additional conditional predictions: a standard conditional prediction of the original prompt (negative target) and a conditional prediction after attenuating prompt-specific attention spikes (positive target).

### Overall Architecture
Standard classifier-free guidance formulates noise prediction as the unconditional prediction plus the difference between prompt-conditional and unconditional predictions. GUARD adds two forces to this formula: using weight $s$ to attract toward the positive conditional prediction $\epsilon_\theta^+$, and weight $r$ to repel the original memorized conditional prediction $\epsilon_\theta^-$. Intuitively, if the original prompt pulls the trajectory toward a training image, repulsion subtracts this direction; however, simply moving away would degrade quality, so the positive target provides an alternative direction that remains prompt-aligned but less memorized.

The specific instantiation in this paper is called CA-in-GUARD. It first reads cross-attention maps from the original conditional pass to automatically identify the spike token set $S(p)$ for the current prompt. It then scales the attention logits of these tokens in the positive branch to obtain the spike-attenuated conditional prediction. Implementation-wise, the unconditional, original conditional, and spike-attenuated conditional predictions can be concatenated into a batch for a single U-Net forward pass to reduce overhead.

### Key Designs
1. **Attractive-repulsive guidance**:

	- Function: Simultaneously reduces memorization replication and maintains image quality at the generation trajectory level.
	- Mechanism: The final noise prediction is formulated as $\hat{\epsilon}=\epsilon_\theta(x_t,e_\phi)+s(\epsilon_\theta^+(x_t,e_p)-\epsilon_\theta(x_t,e_\phi))-r(\epsilon_\theta^-(x_t,e_p)-\epsilon_\theta(x_t,e_\phi))$. Here, $\epsilon_\theta^-$ is the standard prompt conditional prediction, and $\epsilon_\theta^+$ is the safety-processed positive target.
	- Design Motivation: Simply weakening prompt conditions causes generation degradation; simply adding an alternative target might still stay close to the memorization direction. Separating attraction and repulsion allows independent adjustment of memorization mitigation and quality preservation.

2. **Per-prompt cross-attention spike detector**:

	- Function: Dynamically locates token positions in each prompt most likely to trigger training image replication.
	- Mechanism: From the cross-attention distribution of the original conditional pass, the maximum attention mass $M_i$ across spatial queries is calculated for each token. Outlier detection is performed via $Z_i=(M_i-\mu)/\sigma$, and tokens exceeding a threshold $\tau$ are added to $S(p)$. This set can include EOT or any prompt-specific tokens.
	- Design Motivation: Trigger patterns for verbatim and template memorization differ; fixed EOT/padding handling is insufficient and can be counterproductive. The statistical outlier mechanism allows the method to adapt to prompts and denoising steps.

3. **Surgical CA-logit attenuation**:

	- Function: Constructs the positive target for GUARD, ensuring the model follows the prompt without over-relying on memorization-triggering tokens.
	- Mechanism: In selected U-Net cross-attention modules, multiplicative scaling $\ell'_{q,i}=\ell_{q,i}\cdot\alpha$ is applied to the attention logits of $i \in S(p)$ before the softmax. By default, this is processed in down/mid blocks to avoid quality degradation in late up blocks; all heads or only "hot heads" can be chosen, with all-heads found to be a robust default.
	- Design Motivation: Directly deleting tokens or zeroing out attention is too blunt and can destroy semantics. Logit attenuation is a finer intervention that suppresses abnormal spikes while preserving other normal cross-attention patterns.

### Loss & Training
This method involves no training loss and does not update model weights. The main hyperparameters are the attention spike threshold $\tau$, attenuation factor $\alpha$, and GUARD repulsion strength $r$. The authors performed grid searches for different architectures and memorization types, adopting a quality-constrained selection strategy: among configurations where CLIP degradation does not exceed 15% of the reference value, the setting with the lowest SSCD is prioritized.

## Key Experimental Results

### Main Results
Experiments used 500 memorized prompts identified by Webster, with models including SD v1.4 and SD v2.0. SD v1.4 was evaluated for both verbatim and template memorization, while SD v2.0 primarily focused on template memorization. Memorization levels were measured by SSCD, and quality by CLIP and FID.

| Setting | Method | SSCD↓ | CLIP↑ | FID↓ | Notes |
|------|------|-------|-------|------|------|
| SD v1.4 verbatim | No mitigation | 0.875 | 0.346 | 243.056 | High replication of training images in original model |
| SD v1.4 verbatim | Wen et al. | 0.115 | 0.267 | 162.848 | Strong baseline |
| SD v1.4 verbatim | Ren et al. | 0.113 | 0.258 | 164.638 | Fixed CA processing method |
| SD v1.4 verbatim | CA attenuation | 0.109 | 0.282 | 164.660 | Dynamic spike attenuation outperforms Ren et al. |
| SD v1.4 verbatim | CA-in-GUARD | 0.079 | 0.266 | 158.115 | Lowest SSCD and best FID |
| SD v1.4 template | Han et al. | 0.479 | 0.188 | 210.839 | Prior best in this setting |
| SD v1.4 template | CA-in-GUARD | 0.517 | 0.186 | 210.983 | Close to prior best, more stable overall |
| SD v2.0 template | Wen et al. | 0.260 | 0.183 | 188.914 | Strong baseline |
| SD v2.0 template | CA attenuation | 0.193 | 0.184 | 245.850 | Significant SSCD reduction but worse FID |
| SD v2.0 template | CA-in-GUARD | 0.193 | 0.183 | 212.727 | Maintains low SSCD while mitigating FID degradation |

### Ablation Study
Key analyses include the individual effect of CA attenuation, the combined effect of GUARD, and side effects on non-memorized prompts. The following table highlights the most illustrative comparisons.

| Analysis Item | Comparison | Key Metrics | Conclusion |
|--------|------|----------|------|
| Fixed EOT vs. Dynamic Spike | Ren et al. vs. CA attenuation on SD v2.0 template | SSCD 0.356 vs. 0.193 | Handling only EOT/padding misses trigger tokens in template memorization |
| Sufficiency of Positive Target | CA attenuation vs. CA-in-GUARD on SD v1.4 verbatim | SSCD 0.109 vs. 0.079, FID 164.660 vs. 158.115 | Repulsion and spike-attenuated attraction work synergistically |
| Quality-Memorization Trade-off | CA attenuation vs. CA-in-GUARD on SD v2.0 template | FID 245.850 vs. 212.727 | GUARD mitigates quality damage caused by pure attention attenuation |
| Robustness on Non-memorized Prompts | No mitigation vs. CA attenuation on SD v1.4 | SSCD 0.071 vs. 0.069, CLIP 0.299 vs. 0.298 | No significant negative impact on non-memorized prompts |
| Robustness on Non-memorized Prompts | No mitigation vs. CA attenuation on SD v2.0 | SSCD 0.074 vs. 0.072, CLIP 0.322 vs. 0.320 | Suggests the method does not strictly require knowing which prompts are memorized in advance |

### Key Findings
- Template memorization is more challenging than verbatim memorization. Many prior methods effective on SD v1.4 verbatim degrade significantly when moved to template settings or SD v2.0.
- CA spikes are not equivalent to EOT spikes. The attention analysis explains why fixed token rules like Ren et al. fail in template settings.
- The advantage of CA-in-GUARD is its stability across architectures, memorization types, and quality metrics, rather than just achieving the lowest SSCD. It also maintains strong performance across samplers, step counts, CFG scales, the DINO retrieval metric, and SD v3.0.

## Highlights & Insights
- The GUARD formula is highly interpretable: the memorized conditional prediction is the negative target, and the attenuated conditional prediction is the positive target. This clearly indicates what the generation trajectory should move away from and toward.
- The dynamic spike detector transforms manual rules from prior work into a prompt-level statistical test, making it particularly suitable for long-tail prompts and scenarios like template memorization where trigger tokens are not fixed.
- The refinement of evaluation protocols is crucial; focusing only on low-memorization examples overestimates safety. Reporting results separately for verbatim and template settings reveals the hidden weaknesses of many methods.

## Limitations & Future Work
- GUARD is an inference-time mitigation and does not delete memorized information from model weights. White-box attackers may still extract memorized data through other means.
- The method requires extra conditional branches and attention hooks. Although batched forward passes reduce overhead, it may be less straightforward than standard CFG when deployed on highly optimized or closed-source inference engines.
- Hyperparameter tuning depends on the architecture and memorization type; automatic selection of $\tau, \alpha, r$ for production systems remains an engineering challenge.
- Current experiments focus on the Stable Diffusion series and image similarity metrics. Future research is needed for larger multimodal models, video diffusion models, and stricter definitions of copyright/privacy risks.

## Related Work & Insights
- **vs. training-time memorization mitigation**: Training-time methods prevent memorized data from being written into weights but require control over training. GUARD accepts that weights may already contain memorized data and prevents its manifestation in the inference trajectory.
- **vs. diffusion unlearning**: Unlearning requires fine-tuning for the forget set and may be unstable. GUARD does not modify weights and is suitable for rapid deployment, though it cannot defend against white-box extraction.
- **vs. CA redistribution by Ren et al.**: Ren et al. handle EOT/padding/BOT in a fixed manner. CA-in-GUARD uses per-prompt spike detection to find actual anomalous positions, making it more robust for template memorization.
- **vs. initial noise adjustment by Han et al.**: Han et al. escape memorized basins from a sample-time perspective, whereas GUARD directly modifies the conditional direction at each step. The two could potentially be combined.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The combination of the GUARD framework and dynamic CA spike attenuation is clear and explains the instability of prior CA methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers SD v1.4/v2.0, verbatim/template settings, primary metrics, trade-offs, non-memorized prompts, and additional robustness analyses.
- Writing Quality: ⭐⭐⭐⭐☆ Motivations and mechanisms are well-explained; evaluation protocols are detailed. Due to the high volume of tables and appendices, full reproduction requires careful cross-referencing.
- Value: ⭐⭐⭐⭐☆ Directly relevant to copyright and privacy risks in T2I deployment, especially for scenarios where retraining or fine-tuning is unfeasible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](../../CVPR2026/image_generation/low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)
- [\[NeurIPS 2025\] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models](../../NeurIPS2025/image_generation/fairimagen_post-processing_for_bias_mitigation_in_text-to-image_models.md)
- [\[CVPR 2026\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](../../CVPR2026/image_generation/mitigating_memorization_in_texttoimage_diffusion_v.md)
- [\[NeurIPS 2025\] Aligning Text to Image in Diffusion Models is Easier Than You Think](../../NeurIPS2025/image_generation/aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)
- [\[ICML 2026\] Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec.md)

</div>

<!-- RELATED:END -->
