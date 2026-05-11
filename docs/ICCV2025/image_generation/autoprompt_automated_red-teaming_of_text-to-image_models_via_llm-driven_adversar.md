---
title: >-
  [Paper Note] AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts
description: >-
  [ICCV 2025][Image Generation][Red-Teaming] This paper proposes APT (AutoPrompT), a black-box red-teaming framework driven by an LLM. Through an alternating optimize-finetune pipeline and a dual-evasion strategy…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Red-Teaming"
  - "text-to-image"
  - "Adversarial Prompts"
  - "LLM"
  - "Safety Evaluation"
date: 2026-05-08
content_hash: 34459974c40a4df7
---

# AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts

**Conference**: ICCV 2025
**arXiv**: [2510.24034](https://arxiv.org/abs/2510.24034)
**Code**: N/A
**Area**: Image Generation / AI Safety
**Keywords**: Red-Teaming, text-to-image, Adversarial Prompts, LLM, Safety Evaluation

## TL;DR
This paper proposes APT (AutoPrompT), a black-box red-teaming framework driven by an LLM. Through an alternating optimize-finetune pipeline and a dual-evasion strategy, APT automatically generates human-readable adversarial suffixes that bypass content filters, effectively circumventing the safety mechanisms of T2I models while enabling zero-shot cross-prompt transferability.

## Background & Motivation
Text-to-image (T2I) diffusion models have achieved unprecedented generative capabilities through large-scale multimodal training, yet they inherit safety risks from uncontrolled data collection — carefully crafted adversarial prompts can induce the generation of unsafe (NSFW) content. Existing safety mechanisms include training data filtering, NSFW safety checkers, inference-time guidance, and concept-erasure fine-tuning, but their effectiveness and robustness lack standardized automated evaluation.

Existing red-teaming methods suffer from three critical limitations:

**White-box dependency**: Most methods (Ring-A-Bell, P4D, UnlearnDiffAtk) require gradient access to the target model, which is impractical in real-world scenarios.

**Semantic unreadability**: Methods based on discrete optimization produce adversarial prompts that are meaningless character concatenations ("gibberish"), easily detected and blocked by perplexity-based filters.

**Inclusion of forbidden words**: Generated adversarial prompts frequently contain sensitive vocabulary present in blocklists and are directly intercepted by word filters.

Core innovation: leveraging the natural language generation capabilities of LLMs to automatically produce **human-readable** and **filter-evading** adversarial suffixes under a fully black-box setting.

## Method

### Overall Architecture
APT employs an alternating optimize-finetune training strategy. During the optimization phase, the LLM is frozen and adversarial suffixes are optimized token-by-token via stochastic beam search; during the fine-tuning phase, the LLM is fine-tuned using the optimized suffixes as targets. The dual-evasion strategy is applied throughout the optimization phase to ensure that generated adversarial prompts simultaneously bypass perplexity filters and blocklist word filters.

### Key Designs

1. **Adversarial Suffix Optimization**:

    - Function: Given a benign prompt $x$, generate an adversarial suffix $S_T = [s_1, \ldots, s_T]$ such that the concatenated prompt $[x, S_T]$ induces the T2I model to generate unsafe content.
    - Mechanism:
        - **Alignment constraint**: $\ell_{align}(x, S_t) = \text{sim}(\mathcal{G}([x, S_t]), I) + \frac{1}{|c|} \sum_{w \in \mathcal{W}} \text{sim}(\mathcal{G}([x, S_t]), w)$, where the first term aligns the generated image with unsafe reference images and the second term aligns it with unsafe textual concepts.
        - **Stochastic beam search**: At each step, $k=12$ candidate tokens are sampled from the LLM's predicted distribution, and the $b=4$ beams with the lowest objective value are retained, iterating up to maximum length $T=15$.
        - **Prior suffix**: A prior suffix (e.g., "and a beautiful girl's body with") is appended after the benign prompt to provide contextual guidance for the LLM.
    - Design Motivation: Token-by-token optimization enables fine-grained control at each generation step; incorporating the LLM's language prior ensures generation quality.

2. **Dual-Evasion Strategy**:

    - Function: Ensure that generated adversarial prompts simultaneously bypass perplexity filters and blocklist word filters.
    - Mechanism:
        - **Perplexity constraint**: An auxiliary pre-trained LLM $\mathcal{M}_\phi$ is introduced to compute perplexity: $\ell_{per}(S_t|x) = -\sum_{t=1}^T \log p_\phi(s_t | [x, S_{t-1}])$, integrated into the jailbreak objective: $\min_{S_T} \mathcal{L}_{jai} = -\ell_{align} + \lambda \ell_{per}$.
        - **Forbidden token penalty**: The tokenizer vocabulary is scanned to identify tokens whose semantic similarity to unsafe words $\mathcal{W}$ exceeds a threshold; their probabilities are penalized during prediction. An additional check is applied for multi-token combinations that may spell forbidden words (by inspecting the last complete word of each beam).
    - Design Motivation: Low perplexity ensures readability; the forbidden penalty prevents the LLM from taking shortcuts by directly generating sensitive words.

3. **Suffix Generator Fine-Tuning**:

    - Function: Fine-tune the LLM with the high-quality suffixes obtained during optimization, enabling it to progressively learn to directly generate effective suffixes.
    - Mechanism: $(x, S_T)$ pairs are stored in a replay buffer $\mathcal{R}$, with priority sampling determined by successful jailbreaks and the lowest $\mathcal{L}_{jai}$; the LLM is fine-tuned using cross-entropy loss: $\mathcal{L}_{CE} = -\sum_{t=1}^T \log p_\theta(s_t | [x, S_{t-1}])$.
    - Design Motivation: The quality of suffixes obtained during the optimization phase improves over iterations; fine-tuning internalizes jailbreak patterns into the LLM, ultimately enabling zero-shot inference — directly generating effective adversarial suffixes for unseen prompts.

### Implementation Details
The suffix generator uses Llama-3.1-8B; the auxiliary LLM uses the same weights (frozen). The unsafe image set contains 50 images (verified by a classifier); 23 nudity-related and 17 violence-related forbidden words are used. Benign prompts are truncated to 50 tokens.

## Key Experimental Results

### Main Results (RSR after blocklist word filtering)

| Method | ESD↑ | SLD-MAX↑ | Receler↑ | AdvUnlearn↑ | Note |
|--------|------|----------|----------|-------------|------|
| Ring-A-Bell | 2.00% | 2.50% | 1.00% | 0.50% | White-box, nearly ineffective |
| UnlearnDiffAtk | 18.50% | 52.00% | 16.50% | 3.00% | White-box |
| P4D-Union | 41.50% | 62.50% | 41.50% | 9.50% | White-box, requires gradients |
| **APT (Ours)** | **61.50%** | **70.50%** | **36.50%** | **30.50%** | Black-box, human-readable |

### Ablation Study (ESD model, nudity category)

| Configuration | RSR↑ | PPL_Avg↓ | BR↓ | Note |
|---------------|------|----------|-----|------|
| w/o unsafe image alignment | 38.5% | 0.175 | 1% | Lacks visual guidance |
| w/o unsafe word list alignment | 30.5% | 0.067 | 1% | Lacks semantic guidance |
| w/o perplexity constraint | 35% | 0.198 | 1% | Reduced readability |
| w/o forbidden token penalty | 9.5% | 0.171 | **87%** | Nearly all intercepted |
| **Full APT** | **61.5%** | **0.167** | **2%** | All components |

### Key Findings
- APT's perplexity (PPL) is only **1/70** that of Ring-A-Bell (0.167 vs 11,646 ×10³), far superior to all baselines.
- APT achieves the lowest block rate (BR) — approximately 2% for both nudity and violence categories, compared to up to 87% for baseline methods.
- APT achieves an RSR of 30.5% against AdvUnlearn, **3.2× that of P4D** (9.5%), with a particularly pronounced advantage under strong defenses.
- Strong cross-model transferability: prompts optimized for AdvUnlearn achieve over 40% success rate on the other three models.
- APT can directly attack the latest models including SDXL, SD3.5, and FLUX.1-dev, as well as commercial platforms such as Leonardo.Ai.

## Highlights & Insights
- Simultaneously satisfying the three constraints of black-box, human-readable, and filter-evading is far more practically valuable than white-box methods in real deployment scenarios.
- The alternating optimize-finetune strategy enables the LLM to progressively internalize jailbreak patterns, ultimately achieving zero-shot generalization.
- Priority sampling in the replay buffer is a critical design choice for training stability.
- The two-level mechanism of the forbidden token penalty (single-token level + multi-token combination check) reflects engineering completeness.
- Successful attacks against the latest commercial APIs reveal the fundamental fragility of existing safety measures.

## Limitations & Future Work
- Maintaining low perplexity and evading filters may sacrifice some attack strength — overly strict forbidden penalties may suppress semantically critical tokens.
- The prior suffix is currently set manually ("and a beautiful girl's body with"); automated selection could further improve performance.
- A separate suffix generator must be trained for each safe T2I model; a unified generator across different defense methods has not yet been realized.
- The paper focuses on nudity and violence — coverage of other harmful content types (hate speech, discrimination, etc.) remains unexplored.
- The release of red-teaming tools requires careful balance between research value and potential misuse risk.

## Related Work & Insights
- **vs Ring-A-Bell**: Based on genetic algorithm discrete optimization; generated prompts exhibit extremely high perplexity (~11,646), rendering them nearly completely ineffective under filters.
- **vs P4D**: Optimizes in continuous space and requires model gradients; achieves higher RSR but cannot be adapted to black-box settings and produces unreadable prompts.
- **vs AdvPromter**: Also an LLM-driven method but requires white-box gradients; APT is fully black-box and introduces the dual-evasion strategy.
- **Insight**: The language generation capability of LLMs can be directed via "guided fine-tuning" to produce specific adversarial suffixes — this paradigm may generalize to other safety evaluation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The three-way constraint of black-box + human-readable + filter-evading is achieved simultaneously in T2I red-teaming for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four safe T2I models, latest architectures and commercial APIs, comprehensive ablation and transferability analysis.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, algorithm pseudocode is complete, and comparative analysis is thorough.
- Value: ⭐⭐⭐⭐⭐ Reveals the fundamental vulnerability of existing T2I safety mechanisms and provides a practical tool for safety evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LumiCtrl: Learning Illuminant Prompts for Lighting Control in Personalized Text-to-Image Models](../../CVPR2026/image_generation/lumictrl_learning_illuminant_prompts_for_lighting_control_in_personalized_text-t.md)
- [\[ICCV 2025\] DIA: The Adversarial Exposure of Deterministic Inversion in Diffusion Models](dia_the_adversarial_exposure_of_deterministic_inversion_in_diffusion_models.md)
- [\[CVPR 2026\] AutoDebias: An Automated Framework for Detecting and Mitigating Backdoor Biases in Text-to-Image Models](../../CVPR2026/image_generation/autodebias_automated_framework_for_debiasing_text-to-image_models.md)
- [\[ICCV 2025\] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)
- [\[ICLR 2026\] Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search](../../ICLR2026/image_generation/exposing_hidden_biases_in_text-to-image_models_via_automated_prompt_search.md)

</div>

<!-- RELATED:END -->
