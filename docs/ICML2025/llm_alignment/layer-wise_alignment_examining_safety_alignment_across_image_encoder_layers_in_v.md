---
title: >-
  [Paper Note] Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models
description: >-
  [ICML 2025 Spotlight][LLM Alignment][VLM Safety Alignment] This work identifies the Image enCoder Early-exiT (ICET) vulnerability in VLMs, where skipping certain layers of the image encoder significantly increases the probability of generating harmful outputs. It proposes Layer-wise PPO (L-PPO), which modifies the Clipped-PPO algorithm to perform multimodal RLHF across different layers, leading to up to a 48% reduction in ASR and a 33.64% reduction in toxicity score.
tags:
  - "ICML 2025 Spotlight"
  - "LLM Alignment"
  - "VLM Safety Alignment"
  - "Image Encoder"
  - "Layer-wise Safety"
  - "Early-exit Vulnerability"
  - "RLHF"
date: 2026-05-08
content_hash: 3c77256e9a7e2956
---

# Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2411.04291](https://arxiv.org/abs/2411.04291)  
**Code**: None  
**Area**: LLM Alignment/RLHF  
**Keywords**: VLM Safety Alignment, Image Encoder, Layer-wise Safety, Early-exit Vulnerability, RLHF

## TL;DR
This work identifies the Image enCoder Early-exiT (ICET) vulnerability in VLMs, where skipping certain layers of the image encoder significantly increases the probability of generating harmful outputs. It proposes Layer-wise PPO (L-PPO), which modifies the Clipped-PPO algorithm to perform multimodal RLHF across different layers, leading to up to a 48% reduction in ASR and a 33.64% reduction in toxicity score.

## Background & Motivation
**Background**: VLMs (e.g., LLaVA-1.5, LLaVA-NeXT, Llama 3.2 Vision) have achieved great progress in multimodal understanding, but safety alignment remains a challenge. Existing safety training methods (SFT, RLHF, unlearning) are primary trained under default layer embeddings.

**Limitations of Prior Work**: Existing research indicates that specific layers of LLMs retain different types of information, and skipping specific layers affects the generation of harmful content. The multimodal architecture of VLMs further complicates this risk, as the intermediate layer embeddings of image encoders are never covered during safety training.

**Key Challenge**: Safety alignment training is only performed at the default layer of the image encoder (typically the penultimate layer). However, adversaries can utilize intermediate layer embeddings to bypass safety guards, as these intermediate embeddings represent out-of-distribution (OOD) scenarios.

**Goal**: (1) To systematically reveal the uneven layer-wise safety distribution of VLM image encoders; (2) to propose an effective defense method that extends safety alignment to cover different layers.

**Key Insight**: Approaching the problem from the perspective of neural network early exit, an efficiency optimization technique, this study reveals its catastrophic impact on VLM safety.

**Core Idea**: Utilizing embeddings from different image encoder layers to conduct layer-wise RLHF training (L-PPO), ensuring safety alignment is not confined to a single layer.

## Method

### Overall Architecture
1. **Vulnerability Discovery (ICET)**: Systematically test the impact of early exits in different image encoder layers on VLM safety outputs.
2. **Defense Method (L-PPO)**: Modify the Clipped-PPO algorithm to use embeddings from specific intermediate layers during training instead of default layer embeddings, extending safety alignment to potential vulnerability layers.

### Key Designs

1. **ICET Vulnerability (Image enCoder Early-exiT)**:

    - Function: To discover and systematically quantify the impact of image encoder early exits on VLM safety.
    - Key Finding: When using embeddings from layer 18 of LLaVA-1.5 (instead of the default penultimate layer), the VLM generates harmful responses even when the input image is safe and only the text is harmful.
    - Mechanism: Intermediate layer embeddings constitute OOD inputs. The language backbone interprets these embeddings differently, causing safety alignment to fail in this region.
    - Key Distinction: Although intermediate layer embeddings yield **coherent** outputs (semantically relative and logically consistent), the safety mechanism is breached.
    - Design Motivation: Early exit is a common optimization technique in neural networks that might be triggered unintentionally or intentionally in real-world deployment.

2. **Layer-wise PPO (L-PPO)**:

    - Function: Modifies the Clipped-PPO algorithm to execute RLHF on specific intermediate layer embeddings of the image encoder.
    - Mechanism: Since the vulnerability stems from intermediate layer embeddings whose distributions are not covered by safety training, safety alignment training is directly performed using embeddings from these layers.
    - Specific Modifications: During standard PPO training, the visual embeddings input to the VLM are replaced from the default layer to the targeted vulnerability layers.
    - The PPO objective function retains the standard form of Clipped-PPO: $L^{CLIP}(\theta) = \hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$
    - Difference from Standard PPO: Standard multimodal RLHF trains on default layer embeddings, whereas L-PPO specifically targets and trains on intermediate layer embeddings.
    - Theoretical Foundation: Provides theoretical proof for the effectiveness of L-PPO.

3. **Metrics System**:

    - ASR (Attack Success Rate): Uses Llama Guard to determine if a response is harmful.
    - TR (Total Rewards): Uses a specialized safety reward model to evaluate safety.
    - TS (Toxicity Score): Uses the Perspective API to evaluate toxicity.
    - These three metrics evaluate the safety of VLMs from different perspectives.

### Loss & Training
- Uses the Clipped-PPO objective function, with the core modification being the choice of input embedding layer.
- The reward model is trained for safety (high scores for safe responses, low scores for harmful responses).
- Training Data: Redteam 2K, minijailbreak-V28K, and other datasets containing safe images paired with harmful text.
- Validated across three VLMs: LLaVA-1.5, LLaVA-NeXT, and Llama 3.2.

## Key Experimental Results

### ICET Vulnerability Severity (LLaVA-1.5 Early Exit Across Different Layers)

| Encoder Layer | ASR ↓ | Toxicity Score ↓ | Description |
|---------|-------|-----------------|------|
| Default Layer (Penultimate) | Low | Low | Normal safe inference |
| Layer 18 | Significantly higher | Significantly higher | Safety alignment failure |
| Intermediate Layers | Varies | Varies | Uneven distribution of harmful information across layers |

### L-PPO Defense Effectiveness

| Model | Metric | Without L-PPO | With L-PPO | Gain |
|------|------|---------|---------|------|
| LLaVA-1.5 | ASR | High | Drastically reduced | Up to 48% |
| LLaVA-NeXT | ASR | High | Drastically reduced | Significant |
| Llama 3.2 | ASR | High | Drastically reduced | Significant |
| Cross-dataset | Toxicity | High | Drastically reduced | Up to 33.64% |

### Key Findings
- The ICET vulnerability is present across all three VLMs (LLaVA-1.5, LLaVA-NeXT, and Llama 3.2), indicating a structural issue.
- Harmful information is **unevenly distributed** across different layers of the image encoder, with certain layers being particularly vulnerable.
- The limited generalization of current safety training methods is the root cause: safety training covering only the default layer fails to generalize to intermediate layers.
- L-PPO can effectively mitigate ICET, but requires separate training targeted at each vulnerable layer.
- Outputs generated from intermediate layer embeddings remain semantically coherent, with only the safety being compromised—which makes the vulnerability more dangerous.

## Highlights & Insights
- **Uncovers a previously unknown safety vulnerability**: Image encoder early exit can compromise VLM safety alignment, which carries immediate practical warnings for VLM deployment security.
- The perspective of "safety hazards of efficiency optimization techniques" is highly novel—early exit, originally intended to accelerate inference, inadvertently becomes a safety vulnerability.
- Although the L-PPO method is straightforward, it directly addresses the core issue: insufficient coverage of safety alignment.
- Validation across three diverse VLMs strengthens the generalizability of the findings.
- Proposes an important safety principle: safety training should cover various configurations under which the model might be deployed, rather than being restricted to the default configuration.

## Limitations & Future Work
- L-PPO requires prior identification of which layers are vulnerable, potentially requiring re-analysis for new VLM architectures.
- Whether fixing a vulnerability in one layer introduces new vulnerabilities (the trade-offs in layer-wise safety alignment) has not been fully discussed.
- Future work could explore training methods that cover all layers simultaneously (such as multi-layer embedding mixture training or random layer sampling during training).
- The threat model definition for practical attack scenarios could be more precise (e.g., how an adversary accesses intermediate layer embeddings).
- The context/cache files are short (64 lines), and some experimental details (e.g., specific ASR values) were not fully captured.

## Related Work & Insights
- Echoes prior LLM layer-wise safety studies (e.g., Zhao et al. 2023 finding that skipping layers affects the generation of harmful content).
- Early-exit research typically focuses on the efficiency-accuracy trade-off; this work adds an "efficiency-safety" trade-off dimension.
- Provides a new red-teaming approach for VLM safety evaluation: testing not only from the input aspect but also through architectural variations.
- The core concept of L-PPO can be generalized: performing safety alignment on any potential OOD embedding space.
- Complements research on multimodal adversarial attacks—the attack in this work does not require adversarial tokens, only altering the selection of the embedding layer.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[ACL 2025\] LPOI: Listwise Preference Optimization for Vision Language Models](../../ACL2025/llm_alignment/lpoi_listwise_preference_optimization_for_vision_language_models.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](../../ICML2026/llm_alignment/towards_context-invariant_safety_alignment_for_large_language_models.md)
- [\[ACL 2025\] SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings](../../ACL2025/llm_alignment/sea_lowresource_safety_alignment_for_multimodal.md)
- [\[AAAI 2026\] EASE: Practical and Efficient Safety Alignment for Small Language Models](../../AAAI2026/llm_alignment/ease_practical_and_efficient_safety_alignment_for_small_language_models.md)

</div>

<!-- RELATED:END -->
