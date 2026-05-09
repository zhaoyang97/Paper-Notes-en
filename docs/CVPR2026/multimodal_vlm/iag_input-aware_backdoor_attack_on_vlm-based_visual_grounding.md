---
title: >-
  [Paper Note] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding
description: >-
  [CVPR 2026][Multimodal VLM][Backdoor attack] This paper proposes IAG, the first multi-target backdoor attack method against VLM-based visual grounding. By employing a text-conditioned U-Net to dynamically generate input-aware triggers, IAG embeds the semantic information of any attacker-specified target object into the visual input, achieving the highest attack success rate in 11 out of 12 evaluated settings.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Backdoor attack
  - visual grounding
  - multi-target attack
  - input-aware trigger
  - VLM security
date: 2026-05-08
content_hash: c0e2e396e2e2b097
---

# IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding

**Conference**: CVPR 2026
**arXiv**: [2508.09456](https://arxiv.org/abs/2508.09456)
**Code**: [https://github.com/lijunxian111/IAG](https://github.com/lijunxian111/IAG)
**Area**: Multimodal VLM
**Keywords**: Backdoor attack, visual grounding, multi-target attack, input-aware trigger, VLM security

## TL;DR
This paper proposes IAG, the first multi-target backdoor attack method against VLM-based visual grounding. By employing a text-conditioned U-Net to dynamically generate input-aware triggers, IAG embeds the semantic information of any attacker-specified target object into the visual input, achieving the highest attack success rate in 11 out of 12 evaluated settings.

## Background & Motivation
1. **State of the Field**: VLM-based visual grounding has been widely deployed in GUI agents, embodied AI, and related systems, where users specify target objects via natural language for the model to localize. Open model sharing platforms such as HuggingFace make the dissemination of malicious models a realistic threat.
2. **Limitations of Prior Work**: Existing VLM backdoor attacks (e.g., BadSem) predominantly rely on **static triggers and fixed targets**, restricting attacks to predefined single categories. However, in real-world visual grounding scenarios, object categories and descriptions vary substantially across images, rendering static approaches fundamentally insufficient.
3. **Root Cause**: Multi-target backdoor attacks require triggers capable of **dynamically encoding the semantic information of arbitrary target objects**, while simultaneously maintaining imperceptibility and preserving clean-sample performance—a significantly more challenging problem than single-target attacks.
4. **Paper Goals**: To realize the first multi-target backdoor attack on VLM-based visual grounding, allowing an attacker to designate **any object** in an image for the compromised VLM to localize, regardless of the user's actual query.
5. **Starting Point**: A text-conditioned U-Net is employed as the trigger generator to encode target object descriptions into imperceptible visual perturbations, training the VLM to associate such perturbation patterns with target localization.
6. **Core Idea**: Use a text-conditioned U-Net to dynamically generate imperceptible triggers that semantically encode the attack target.

## Method

### Overall Architecture
**Input**: A clean image $x$ and an attacker-specified target object description $o$. **Trigger generation**: A text-conditioned U-Net $\mathcal{G}_\phi$ generates a trigger $r$ conditioned on $x$ and the embedding $z_o$ of $o$, producing the triggered image $x \oplus r$. **Backdoor injection**: The U-Net and VLM are jointly trained so that the VLM behaves normally on clean inputs and localizes the attack target on triggered inputs. At inference time, the attacker need only supply triggered images (e.g., a webpage screenshot injected with an advertising link); regardless of the user's query, the VLM will localize the attack target.

### Key Designs
1. **Input-aware Trigger Generator**:
    - **Function**: Dynamically generates imperceptible backdoor triggers conditioned on the input image and the target object description.
    - **Mechanism**: A text-conditioned U-Net (3 downsampling blocks + 1 bottleneck block + 3 upsampling blocks) is adopted; cross-attention layers are appended after the bottleneck and each upsampling block to inject the target text embedding $z_o$, which is derived from the frozen clean VLM embedding layer. The output trigger shares the spatial dimensions of the input and is added directly to produce the triggered image.
    - **Design Motivation**: Linear mappings (e.g., Imperio) cannot model the complex nonlinear relationship between target objects and triggers; shallow conditional autoencoders (e.g., Marksman) suffer from information bottlenecks. The cross-modal conditional injection and skip connections of U-Net jointly capture global context and fine-grained visual details.

2. **Joint Training Objective**:
    - **Function**: Balances attack effectiveness, clean-sample performance, and trigger imperceptibility.
    - **Mechanism**: The total loss is $\mathcal{L} = \mathcal{L}_{LM} + \beta \cdot \mathcal{L}_{rec}$. $\mathcal{L}_{LM}$ comprises two components—standard grounding loss on clean samples and attack-target grounding loss on triggered samples (standard token-level cross-entropy). $\mathcal{L}_{rec} = \alpha_1 \mathcal{L}_{pix} + \alpha_2 \mathcal{L}_{LPIPS}$ enforces visual imperceptibility of the trigger (pixel-level L1 + perceptual LPIPS). Hyperparameters are set as $\alpha_1=1$, $\alpha_2=0.05$, $\beta=0.5$.
    - **Design Motivation**: The dual components of $\mathcal{L}_{LM}$ ensure correct model behavior under both input conditions; the LPIPS term in $\mathcal{L}_{rec}$ guarantees human-imperceptible triggers. Joint optimization enables the U-Net and VLM to be trained collaboratively.

3. **Attack Data Preparation**:
    - **Function**: Systematically constructs a small poisoned dataset from existing annotated data.
    - **Mechanism**: A subset is randomly sampled from a visual grounding dataset at poisoning rate $\alpha$. For each image, one annotated object is randomly selected as the attack target $o$; the user query $q$ is drawn from the description of a **non-target object**, and the answer $y^*$ is the bounding box of the attack target. A unified prompt template is used: Q: xxx \<object\>. A: \<object\>[\<bbox\>].
    - **Design Motivation**: Leveraging existing multi-object annotations incurs no additional labeling cost. Ensuring that the query and target correspond to different objects guarantees that the attack is semantically misleading rather than a normal grounding behavior.

### Loss & Training
LoRA fine-tuning is applied to LLaVA-v1.5-7B with a poisoning rate of $\alpha = 5\%$. The U-Net ($lr=5\times10^{-4}$) and VLM ($lr=2\times10^{-5}$) are jointly optimized. Proposition 1 provides a theoretical lower bound on the attack success rate, showing that the success probability increases with trigger norm $\varepsilon$ and text alignment degree $\gamma$.

## Key Experimental Results

### Main Results (12 VLM × Dataset Combinations)

| Setting | IAG ASR@0.5 | Strongest Baseline | Gain |
|---------|------------|-------------------|------|
| LLaVA + RefCOCO | 58.9% | Imperio 55.2% | +3.7% |
| LLaVA + F30k | 40.0% | Imperio 33.6% | +6.4% |
| InternVL + RefCOCO | 66.9% | Imperio 65.5% | +1.4% |
| InternVL + RefCOCO+ | 68.1% | Imperio 63.8% | +4.3% |
| Ferret + F30k | 53.8% | Imperio 48.1% | +5.7% |
| Ferret + RefCOCO | 48.9% | Imperio 35.6% | +13.3% |

Clean accuracy degradation: gap between backdoored accuracy (BA) and clean accuracy (CA) is < 3% (e.g., LLaVA-RefCOCO: BA 80.7% vs. CA 82.1%).

### Ablation Study

| Configuration | ASR | Note |
|---------------|-----|------|
| Full IAG | 58.9% | Complete model |
| w/o LPIPS loss | ASR increases but trigger becomes visible | Imperceptibility compromised |
| Fixed trigger (One-to-N) | 3.2% | Cannot perform multi-target attack |
| Shallow autoencoder (Marksman) | 8.5% | Limited by information bottleneck |
| Linear mapping (Imperio) | 55.2% | Competitive but cannot model complex relationships |

### Key Findings
- IAG achieves the highest ASR in 11 out of 12 settings; the sole exception is one setting where Imperio marginally outperforms.
- Compared to fixed triggers (One-to-N: 3–5%), input-aware triggers improve ASR by 10–50%+.
- The BA–CA gap is minimal (< 3%), indicating that the backdoored model is nearly unaffected on clean data, achieving high stealthiness.
- Transferability across datasets and model architectures is also validated, suggesting that IAG learns generalizable vulnerabilities.
- IAG remains robust against existing defenses (e.g., STRIP, Fine-pruning).

## Highlights & Insights
- **Formalization of multi-target backdoor attacks**: This work is the first to formally define the multi-target backdoor attack problem for VLM-based grounding—where the attacker can specify arbitrary objects rather than fixed categories—revealing a substantially more severe security threat than single-target attacks.
- **"Semantic injection" via text-conditioned triggers**: The trigger not only constitutes a perturbation but also carries the semantic information of the target object. This design allows the VLM's cross-attention mechanism to "perceive" target object features even when the target is not mentioned in the query. Proposition 1 provides a rigorous theoretical lower bound supporting this claim.
- **Security warning for GUI agents and embodied AI**: IAG is also effective in the ShowUI agent scenario (ASR 25–35%), demonstrating that malicious webpages can redirect VLM-powered agents to localize advertisements or malicious links instead of the user's intended target—a highly realistic threat.

## Limitations & Future Work
- Attack success rates remain relatively low in certain settings (e.g., ~47% on RefCOCOg, 25–35% on ShowUI), with limited effectiveness against complex expressions and dense UI elements; a substantial gap remains compared to near-100% ASR achieved by classification backdoors.
- Trigger generation requires access to the clean VLM's embedding layer; while a same-architecture open-source model can substitute, the approach is inapplicable when architectures differ substantially (e.g., mismatched embedding dimensions).
- The default poisoning rate of 5% may be impractical in scenarios requiring large amounts of clean data; attack performance under lower poisoning rates remains unexplored.
- The paper adopts a purely offensive perspective and proposes no effective defense. While the finding that all existing defenses fail serves as a strong warning, it lacks constructive contribution—future work should simultaneously investigate detection methods tailored to input-aware triggers.
- The U-Net trigger generator introduces additional model overhead (3 downsampling + 3 upsampling blocks + cross-attention), which may be impractical in deployment-constrained settings.
- The approach imposes a maximum length constraint on target object descriptions (determined by dataset settings); attack effectiveness under longer descriptions remains unknown.

## Related Work & Insights
- **vs. BadSem**: BadSem exploits semantic misalignment as a trigger but is limited to static targets; IAG's input-aware design supports arbitrary target switching. BadSem's design assumption of fixed attack categories is fundamentally misaligned with the open-vocabulary nature of visual grounding.
- **vs. Imperio (input-aware classification attack)**: Imperio is the strongest baseline (RefCOCO ASR: 55.2 vs. IAG: 58.9), but the gap widens substantially in complex scenarios such as ShowUI (16.0 vs. 32.3). Imperio's linear mapping is viable in simple settings but lacks the capacity to model complex target–trigger relationships.
- **vs. Marksman (multi-target classification attack)**: Marksman employs a shallow conditional autoencoder; the information bottleneck limits complex semantic control, yielding ASR of only 8–33%, far below IAG.
- **Defense implications**: The results suggest that more rigorous security auditing is required before VLM deployment. For fine-tuned models from unverified sources in particular, detection methods targeting input-aware triggers must be developed—existing spectral or statistical detection approaches are ineffective against input-adaptive perturbations.
- **Warning for GUI agent security**: Experimental results in the ShowUI scenario (ASR 25–35%) demonstrate that malicious webpages can redirect VLM-powered agents to localize advertisements or malicious links rather than user-intended targets—a highly realistic threat.
- **Implications for the open-source model ecosystem**: Fine-tuned models on platforms such as HuggingFace lack security auditing; IAG demonstrates that an effective backdoor can be injected with as little as 5% poisoned data, posing a significant challenge to trust mechanisms in the open-source model community.
- **Value of theoretical contribution**: Proposition 1 provides a mathematical explanation for why input-aware triggers outperform fixed triggers—text-conditioned subspaces align the perturbation direction with the grounding features of cross-attention, thereby increasing effective projection gain $m$ and alignment degree $\gamma$.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First multi-target backdoor attack on VLM-based visual grounding; both the problem formulation and the proposed solution are novel, filling an important gap in VLM security research.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 settings (3 models × 5 datasets), covering imperceptibility, defense robustness, input attack comparisons, and theoretical analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The threat model is clearly defined, theoretical and empirical analyses are well integrated, and the problem formalization is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ Reveals an important blind spot in VLM security with significant implications for the security community, particularly for GUI agent deployment scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching VLA Models](flowhijack_dynamics_aware_backdoor_attack_on_flow_matching_vla_models.md)
- [\[ICLR 2026\] BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning](../../ICLR2026/multimodal_vlm/beat_visual_backdoor_attacks_on_vlm-based_embodied_agents_via_contrastive_trigge.md)
- [\[CVPR 2026\] Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning](mastering_negation_boosting_grounding_models_via_g.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](docseeker_long_document_understanding.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)

<!-- RELATED:END -->
