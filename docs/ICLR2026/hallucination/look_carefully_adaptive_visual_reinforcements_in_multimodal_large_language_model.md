---
title: >-
  [Paper Note] Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation
description: >-
  [ICLR 2026][Hallucination Detection][MLLM hallucination mitigation] This paper proposes AIR (Adaptive vIsual Reinforcement), a framework that reduces hallucinations in MLLMs at inference time without any training…
tags:
  - "ICLR 2026"
  - "Hallucination Detection"
  - "MLLM hallucination mitigation"
  - "visual reinforcement"
  - "optimal transport"
  - "token reduction"
  - "training-free inference"
date: 2026-05-08
content_hash: a8f53cce361a10df
---

# Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation

**Conference**: ICLR 2026
**arXiv**: [2602.24041](https://arxiv.org/abs/2602.24041)  
**Code**: Not yet released  
**Area**: Hallucination Detection
**Keywords**: MLLM hallucination mitigation, visual reinforcement, optimal transport, token reduction, training-free inference

## TL;DR
This paper proposes AIR (Adaptive vIsual Reinforcement), a framework that reduces hallucinations in MLLMs at inference time without any training, via prototype-distance-based token reduction combined with optimal-transport-guided selective patch reinforcement (LLaVA-1.5-7B CHAIR_S: 22→18.4, POPE accuracy +5.3%), while preserving general multimodal capabilities.

## Background & Motivation

**Background**: MLLMs (LLaVA, Qwen-VL, etc.) have achieved remarkable progress in vision-language reasoning, yet remain prone to "hallucinations"—generated text that is inconsistent with image content, such as describing non-existent objects or producing contradictions. Hallucination mitigation methods are broadly categorized into training-time (requiring additional annotations), post-processing (requiring external models), and inference-time approaches (e.g., contrastive decoding).

**Limitations of Prior Work**: Recent visual reinforcement methods (e.g., MemVR) attempt to re-inject visual tokens into FFN layers during decoding to strengthen visual signals, but suffer from a critical flaw—**all visual tokens are injected indiscriminately**, causing redundant signals from background regions to interfere with the model's focus on salient regions, potentially introducing new hallucinations.

**Key Challenge**: The large number of visual tokens (e.g., 576 in LLaVA) contains substantial background-redundant tokens. Full injection introduces noise, while no injection leads to visual signal attenuation—a balance must be struck between "reinforcing visual signals" and "avoiding background interference."

**Goal**: Design a selective visual reinforcement mechanism that injects only the visual patches most relevant to the current generation step into the decoding process, strengthening critical visual cues while avoiding redundant interference.

**Key Insight**: The authors observe that similarity between hidden states and different visual tokens varies substantially—tokens corresponding to salient object regions exhibit high similarity while background tokens exhibit low similarity—motivating an adaptive selection strategy.

**Core Idea**: Prototype-distance-based reduction is used to prune redundant visual tokens, and optimal transport is used to quantify patch–hidden-state alignment, with only highly aligned patches being injected.

## Method

### Overall Architecture
AIR operates at the FFN stage of each Transformer layer and consists of two sequential components: (1) Prototype-based Token Reduction, which compresses visual tokens into a compact subset; and (2) OT-guided Patch Reinforcement, which evaluates patch–hidden-state alignment via optimal transport and selectively injects highly aligned patches. The entire pipeline requires no training and can be plugged into any MLLM.

### Key Designs

1. **Prototype-based Token Reduction**:

    - **Function**: Compresses $K$ visual tokens into $Q$ tokens ($Q \ll K$), retaining the most informative ones.
    - **Mechanism**: Computes the mean of all visual tokens as a prototype $h_p$, ranks tokens by their L2 distance to the prototype, and retains the top-$Q$ most distant tokens—tokens farther from the prototype encode more distinctive visual information.
    - **Design Motivation**: Among the full 576 tokens, a large proportion are similar background tokens; retaining outlier tokens suppresses redundancy and reduces the computational cost of subsequent OT computation.

2. **OT-guided Patch Reinforcement**:

    - **Function**: Crops the image into $M$ patches, evaluates each patch's alignment with the current hidden state via optimal transport, and injects only patches with high alignment (low OT distance).
    - **Mechanism**: Models hidden states and patch embeddings as discrete distributions and efficiently solves for the OT distance $d_\text{OT}(m)$ using the Sinkhorn algorithm. A threshold $\tau$ is applied to select the patch set $\mathcal{M}$ where $d_\text{OT}(m) \leq \tau$; embeddings of selected patches are concatenated and injected into the FFN.
    - **Design Motivation**: OT distance captures global geometric structure and is more discriminative than pointwise cosine similarity. The paper provides a theoretical proof that OT's discriminative sensitivity is strictly higher than that of cosine distance.

3. **Selective Visual Grounding**:

    - **Function**: The final FFN output = original FFN output + an interactive reinforcement term between the reduced hidden states and selected patches.
    - **Mechanism**: $\text{FFN}(H|\tilde{Z}) = \phi(HW_1)W_2^\top + \phi(H'\tilde{Z}^\top)\tilde{Z}$, where $H'$ is the reduced hidden state and $\tilde{Z}$ is the OT-selected patch embeddings.
    - **Design Motivation**: Adding the reinforcement term as a residual preserves the model's original behavior and provides additional visual grounding only when highly aligned patches are present.

### Loss & Training
No training is required. Hyperparameters include: the number of retained tokens after reduction $Q$, OT threshold $\tau$, and number of patches $M$.

## Key Experimental Results

### Main Results — CHAIR Hallucination Benchmark (MSCOCO, max 64 tokens)

| Method | LLaVA-1.5-7B CHAIR_S↓ | CHAIR_I↓ | Qwen-VL CHAIR_S↓ | CHAIR_I↓ | GLM-4V CHAIR_S↓ | CHAIR_I↓ |
|--------|----------------------|----------|-------------------|----------|-----------------|----------|
| Vanilla | 22.0 | 6.7 | 20.0 | 6.2 | 13.0 | 5.6 |
| VCD | 24.6 | 7.3 | 19.2 | 5.7 | 14.8 | 6.5 |
| MemVR | 21.6 | 6.4 | 20.0 | 6.1 | 13.0 | 5.6 |
| VAF | 20.4 | 6.5 | 20.6 | 6.6 | 11.6 | 5.3 |
| **AIR** | **18.4** | **5.7** | **18.6** | **5.9** | **11.6** | **5.3** |

### POPE Benchmark (LLaVA-1.5-7B)

| Dataset | Setting | Vanilla Acc | MemVR Acc | AIR Acc | AIR F1 |
|---------|---------|-------------|-----------|---------|--------|
| MSCOCO | Random | 83.7 | 87.6 | **89.0** | **88.2** |
| MSCOCO | Popular | 78.2 | 86.0 | **87.1** | **86.4** |
| MSCOCO | Adversarial | 75.0 | 83.5 | **83.9** | **83.6** |
| A-OKVQA | Random | 83.4 | 89.0 | **89.0** | **88.5** |

### Ablation Study

| Component | CHAIR_S↓ | POPE Acc↑ |
|-----------|----------|-----------|
| Vanilla | 22.0 | 83.7 |
| +Token Reduction only | 20.1 | 86.8 |
| +Patch Reinforcement only | 19.5 | 87.2 |
| +Full AIR | **18.4** | **89.0** |
| OT replaced by Cosine | 19.8 | 87.5 |

### Key Findings
- AIR achieves state-of-the-art or tied best hallucination mitigation across three MLLMs with different architectures.
- OT-based selection outperforms cosine-similarity-based selection (CHAIR_S: 18.4 vs. 19.8), validating the theoretical analysis.
- Performance remains robust under the POPE adversarial setting, demonstrating that selective reinforcement is effective against adversarial prompts.
- General capabilities (LLaVA-Bench, MME, MMBench) do not degrade significantly, confirming that the method does not trade general performance for lower hallucination rates.

## Highlights & Insights
- **Theory and practice in perfect synergy**: The theoretical advantage of OT (strictly higher discriminative sensitivity than cosine distance) is formally proven and empirically validated.
- **Training-free, plug-and-play**: No annotations or fine-tuning are required; the method can be directly applied to arbitrary MLLMs including LLaVA, Qwen-VL, and GLM-4V.
- **"Less is more"**: Reduced tokens with selective reinforcement outperform full injection, demonstrating that the quality of visual reinforcement matters more than quantity.
- Attention heatmap visualizations clearly show that AIR concentrates attention on semantically salient regions.

## Limitations & Future Work
- Cropping the image into patches and encoding them separately introduces additional inference cost, which scales with image resolution.
- The OT threshold $\tau$ and reduction count $Q$ require tuning and may need different configurations for different models and datasets.
- Validation is currently limited to captioning and VQA settings; effectiveness in multi-turn dialogue, long-form generation, and other scenarios remains unknown.
- The prototype-distance ranking assumes "outliers = informative," which may not hold in specific cases (e.g., uniformly textured images).

## Related Work & Insights
- AIR directly improves upon MemVR (full visual token injection into FFN): AIR demonstrates that selective injection significantly outperforms full injection.
- AIR is complementary to VCD (visual contrastive decoding): VCD mitigates hallucinations via noise-based contrastive decoding, while AIR does so by reinforcing critical visual signals.
- The use of OT in VLMs opens a new direction for future work, such as attention allocation and token merging.

## Rating
- Novelty: ⭐⭐⭐⭐ The OT-guided selective visual reinforcement idea is novel, with theoretical proofs as an additional contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models, multiple hallucination benchmarks, general capability evaluation, and complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is thorough and motivating figures are clear.
- Value: ⭐⭐⭐⭐ A practical training-free solution for MLLM hallucination mitigation with high plug-and-play value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)
- [\[ICML 2026\] Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models](../../ICML2026/hallucination/adaptive_residual-update_steering_for_low-overhead_hallucination_mitigation_in_l.md)
- [\[ICLR 2026\] Copy-Paste to Mitigate Large Language Model Hallucinations](copy-paste_to_mitigate_large_language_model_hallucinations.md)
- [\[ICML 2026\] Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization](../../ICML2026/hallucination/learning_from_fine-grained_visual_discrepancies_mitigating_multimodal_hallucinat.md)
- [\[AAAI 2026\] Verb Mirage: Unveiling and Assessing Verb Concept Hallucinations in Multimodal Large Language Models](../../AAAI2026/hallucination/verb_mirage_unveiling_and_assessing_verb_concept_hallucinations_in_multimodal_la.md)

</div>

<!-- RELATED:END -->
