---
title: >-
  [Paper Note] Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation
description: >-
  [ECCV 2024][Multimodal VLM][MLLM safety] ECSO (Eyes Closed, Safety On) is proposed: a training-free MLLM defense method that detects the safety of its own responses and adaptively converts images in unsafe queries into text descriptions, thereby restoring the intrinsic safety mechanism of pre-aligned LLMs. It achieves up to a 71.3% safety improvement on MM-SafetyBench without compromising general performance.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "MLLM safety"
  - "Jailbreak attack defense"
  - "Image-to-text transformation"
  - "Training-free method"
  - "Safety alignment"
date: 2026-05-08
content_hash: fda910f87e5c860c
---

# Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation

**Conference**: ECCV 2024  
**arXiv**: [2403.09572](https://arxiv.org/abs/2403.09572)  
**Code**: [Project Page](https://gyhdog99.github.io/projects/ecso/)  
**Area**: Multimodal VLM  
**Keywords**: MLLM safety, Jailbreak attack defense, Image-to-text transformation, Training-free method, Safety alignment

## TL;DR

ECSO (Eyes Closed, Safety On) is proposed: a training-free MLLM defense method that detects the safety of its own responses and adaptively converts images in unsafe queries into text descriptions, thereby restoring the intrinsic safety mechanism of pre-aligned LLMs. It achieves up to a 71.3% safety improvement on MM-SafetyBench without compromising general performance.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) achieve strong multimodal conversation capabilities by combining visual encoders with pre-aligned LLMs. However, the introduction of image inputs makes these models vulnerable to malicious jailbreak attacks that induce harmful content generation.

**Limitations of Prior Work**: Traditional safety alignment strategies (such as SFT and RLHF) require designing a large number of red-teaming queries, which is even more difficult and expensive when image inputs are involved. Existing inference-time defense methods either rely on manual system prompts (hard to cover new attacks) or require externally trained detectors.

**Key Challenge**: MLLMs inherit safety mechanisms from LLMs, but the introduction of image features "overpowers" these mechanisms. Specifically, when images are removed, the model can reject malicious queries almost 100% of the time, but with images, the harmless rate drops sharply to about 20%.

**Core Problem**: How to transfer the safety mechanism of pre-aligned LLMs to MLLMs without requiring extra training?

**Key Insight**: Two key observations are discovered: (a) Although MLLMs are prone to generating harmful content, they can identify the safety of their own responses with high accuracy (>95% accuracy); (b) Removing the image restores the LLM's safety mechanism.

**Core Idea**: Let the MLLM first self-detect the safety of its response. If unsafe content is detected, the image is converted into a text description, and the model "closes its eyes" to regenerate a safe response using a text-only LLM.

## Method

### Overall Architecture

Input (image $v$, query $x$) $\rightarrow$ Step 1: Normally generate an initial response $\tilde{y}$ $\rightarrow$ Step 2: MLLM self-evaluates response safety $s$ $\rightarrow$ [If safe, return directly] $\rightarrow$ Step 3: If unsafe, convert the image into a query-aware text description $c$ $\rightarrow$ Step 4: Regenerate a safe response $y$ using text-only (no image).

### Key Designs

1. **Harmful Content Detection**:

    - **Function**: Enables the MLLM to judge whether its generated initial response is safe.
    - **Mechanism**: First, the initial response is normally generated: $\tilde{y} = F_{\theta}(v, x)$. Then, a detection prompt template $P_{\text{det}}$ is used to wrap the original query and the initial response, allowing the MLLM to self-evaluate: $s = F_{\theta}(v, P_{\text{det}}(x, \tilde{y}))$.
    - **Design Motivation**: Experiments reveal that MLLMs perform exceptionally well in discrimination tasks (achieving >95% accuracy on LLaVA-1.5-7B and ShareGPT4V-7B), and this discrimination accuracy is unaffected by the presence of images. The hypothesis that discrimination is easier than generation is supported by scalable oversight theory.

2. **Query-Aware I2T Transformation**:

    - **Function**: Converts the input image into a query-related text description.
    - **Mechanism**: Uses a prompt template $P_{\text{trans}}$ containing the original question to guide the MLLM in generating a query-aware image description: $c = F_{\theta}(v, P_{\text{trans}}(x))$.
    - **Design Motivation**: (a) By converting to text, harmful content in the image is either transformed into words or discarded; (b) query-awareness ensures that the description contains key information required to answer the question, avoiding information loss from irrelevant descriptions. Ablation studies prove that removing query-awareness GSM significantly reduces utility.

3. **Safe Response Generation Without Images**:

    - **Function**: Replaces the image with a text description, allowing the model to regenerate a response in text-only mode.
    - **Mechanism**: $y = F_{\theta}(\text{null}, P_{\text{gen}}(c, x))$, where null represents the absence of an image. At this point, the MLLM degrades to a text-only LLM, reactivating its pre-aligned safety mechanism.
    - **Design Motivation**: Experiments show that once the image is removed, the LLM becomes almost 100% harmless. Adding keywords like "HARMLESS and ETHICAL" in the prompt further reinforces safety priority, raising the harmless rate even above the Text-Only upper bound.

### Loss & Training

ECSO is a completely training-free inference-time method without any loss functions or training processes. Additionally, ECSO can serve as a data engine to automatically generate SFT safety alignment data: applying the ECSO pipeline to an unsupervised safety dataset $D = \{(v,x)\}$ yields $D' = \{(v,x,y)\}$ for fine-tuning.

## Key Experimental Results

### Main Results (MM-SafetyBench, LLaVA-1.5-7B)

| Attack Type | Metric (Harmless Rate %) | Direct | ECSO | Gain |
|----------|---------------------|--------|------|------|
| SD (Stable Diffusion) | Average | 85.0 | 95.4 | +10.4 |
| OCR | Average | 31.7 | 90.3 | +58.6 |
| SD+OCR | Average | 32.1 | 86.4 | +54.3 |
| SD+OCR - Illegal Activities | Harmless Rate | 25.8 | 92.8 | +67.0 |
| SD+OCR - Hate Speech | Harmless Rate | 51.5 | 90.2 | +38.7 |
| SD+OCR - Malware | Harmless Rate | 38.6 | 84.1 | +45.5 |
| VLSafe (across 5 MLLMs) | Harmless Rate | ~20% | ~90% | +71.3 (Max) |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Fully Featured ECSO | HR=86.4% (SD+OCR) | Baseline |
| Retain Image + Caption | HR drops significantly | Proves that removing the image is critical |
| Remove Query-Awareness | MMBench: 65.8 (-1.05%) | Query-awareness is indispensable to maintain utility |
| Direct Refusal (w/o Step 3 & 4) | MME: 1847 (vs 1865) | Steps 3 & 4 ensure normal response to benign queries |
| SFT data generated by ECSO | Outperforms manual annotation VLGuard | ECSO can serve as a data engine |

### Utility Preservation (False Positive Rate & Performance)

| Model | MME False Positive Rate | MMBench False Positive Rate | MME-P (Direct/ECSO) | MMBench (Direct/ECSO) |
|------|----------|-------------|-------------------|---------------------|
| LLaVA-1.5-7B | 0.50% | 1.23% | 1507.4/1507.4 | 64.6/64.2 |
| ShareGPT4V-7B | 1.93% | 4.24% | 1566.4/1567.1 | 66.5/66.1 |
| Qwen-VL-Chat | 1.26% | 2.88% | 1481.5/1481.5 | 59.7/59.1 |

### Key Findings

- The safety mechanism of MLLMs is not gone, but "overpowered" by image features—removing the image achieves a ~100% harmless rate for almost all models.
- MLLMs have an extremely strong ability to self-distinguish whether their responses are safe (>95%), which is unaffected by image inputs.
- OCR and SD+OCR attacks are more effective than pure SD attacks because they contain more direct malicious textual information.
- The safety alignment data generated by ECSO achieves quality comparable to, or even exceeding, manually annotated data.

## Highlights & Insights

- **The insight "discrimination is easier than generation" is highly valuable**: Utilizing the model's own discriminatory capability to cover safety loopholes during generation is an elegant self-bootstrapping safety strategy.
- **The training-free design** allows ECSO to be plug-and-play for any MLLM, making it highly practical.
- **The image-to-text modality conversion trick**: "Closing its eyes" to reduce a multimodal problem to a text-only problem cleverly leverages the pre-existing safety alignment in the LLM.
- **The query-aware captioning design** prevents information loss, a trick that is transferable to other scenarios requiring image-to-text translation.
- **Data engine byproduct**: ECSO not only provides inference-time protection but also automatically generates safety alignment data, creating a virtuous cycle.

## Limitations & Future Work

- ECSO relies on the safety capability of the underlying LLM itself; if the LLM has inherent safety flaws, ECSO will also fail.
- Information loss is inevitable during the image-to-text conversion process, which may affect response quality for queries heavily dependent on visual details.
- Multi-turn inference (generation $\rightarrow$ safety evaluation $\rightarrow$ conversion $\rightarrow$ final generation) increases inference latency.
- Using multimodality to shift from a "safety challenge" to a "safety advantage" (by leveraging rich multimodal context to build stronger safety mechanisms) remains unexplored.
- Advanced I2T methods (such as $V^*$ guided visual search) can be explored to improve information preservation.

## Related Work & Insights

- **vs MLLM-Protector [Pi et al.]**: MLLM-Protector requires training additional detectors and detoxicators, while ECSO is completely training-free and leverages the model's own capacity.
- **vs Self-Moderation [Chen et al.]**: Pure instruction-based self-moderation still fails when images are present, whereas ECSO fundamentally resolves this by removing images.
- **vs Safety Steering Vectors [Wang et al.]**: Steering vectors focus primarily on textual unsafe intent, potentially overlooking malicious content inside images.
- **vs VLGuard [Zong et al.]**: VLGuard aligns through SFT requiring annotated data, whereas ECSO can automatically generate equivalent or even superior alignment data.

## Rating

- Novelty: ⭐⭐⭐⭐ Observing the discrepancy between discrimination and generation capabilities and utilizing modality conversion to recover safety mechanisms is a clever idea, although the technical implementation is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 MLLMs, 3 safety benchmarks, 3 utility benchmarks, and detailed ablation studies make the experiments exceptionally thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain is highly clear: observation $\rightarrow$ insight $\rightarrow$ method $\rightarrow$ evaluation, with elegant and easy-to-understand diagrams.
- Value: ⭐⭐⭐⭐ Practical value is high as a training-free, plug-and-play solution, although it fundamentally "bypasses the problem" rather than "solving it".

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](../../ICCV2025/multimodal_vlm/multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)
- [\[ACL 2026\] DMN: A Compositional Framework for Jailbreaking Multimodal LLMs with Multi-Image Inputs](../../ACL2026/multimodal_vlm/dmn_a_compositional_framework_for_jailbreaking_multimodal_llms_with_multi-image_.md)
- [\[ECCV 2024\] Merlin: Empowering Multimodal LLMs with Foresight Minds](merlin_empowering_multimodal_llms_with_foresight_minds.md)
- [\[CVPR 2026\] Text-Printed Image: Bridging the Image-Text Modality Gap by "Printing" Text into Images](../../CVPR2026/multimodal_vlm/text-printed_image_bridging_the_image-text_modality_gap_for_text-centric_trainin.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](../../ACL2025/multimodal_vlm/cant_see_the_forest_for_the.md)

</div>

<!-- RELATED:END -->
