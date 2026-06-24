---
title: >-
  [Paper Note] Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval
description: >-
  [CVPR 2025][Reasoning][Composed Image Retrieval] This paper proposes OSrCIR, a training-free one-stage zero-shot composed image retrieval method. It utilizes multimodal large language models to directly process the reference image and modification text, and accurately understands the user's implicit intent through reflective Chain-of-Thought reasoning, outperforming existing training-free methods by 1.80% to 6.44% across multiple benchmarks.
tags:
  - "CVPR 2025"
  - "Reasoning"
  - "Composed Image Retrieval"
  - "Zero-Shot"
  - "Chain-of-Thought"
  - "Multimodal Large Language Model"
  - "Training-Free"
date: 2026-05-08
content_hash: 41ec850c8e51b9fd
---

# Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval

**Conference**: CVPR 2025  
**arXiv**: [2412.11077](https://arxiv.org/abs/2412.11077)  
**Code**: [https://github.com/Pter61/osrcir2024/](https://github.com/Pter61/osrcir2024/)  
**Area**: LLM Reasoning  
**Keywords**: Composed Image Retrieval, Zero-Shot, Chain-of-Thought, Multimodal Large Language Model, Training-Free

## TL;DR
This paper proposes OSrCIR, a training-free one-stage zero-shot composed image retrieval method. It utilizes multimodal large language models to directly process the reference image and modification text, and accurately understands the user's implicit intent through reflective Chain-of-Thought reasoning, outperforming existing training-free methods by 1.80% to 6.44% across multiple benchmarks.

## Background & Motivation
1. **Background**: Composed Image Retrieval (CIR) retrieves a target image using a reference image paired with a modification text. Zero-shot CIR (ZS-CIR) leverages pre-trained models such as CLIP, bypassing the need for large amounts of labeled triplet data.
2. **Limitations of Prior Work**: Current training-free ZS-CIR methods adopt a two-stage pipeline: first generating a textual caption of the reference image using an image captioner, and then reasoning the target description using an LLM. This leads to two issues: (1) the captioning stage is unaware of the modification intent, thereby missing crucial visual details; and (2) simple prompting limits the reasoning capabilities of the LLM.
3. **Key Challenge**: The separation of captioning and reasoning in two-stage methods causes information loss—key visual details are lost during the captioning stage, which subsequent LLM reasoning cannot salvage.
4. **Goal**: Design a one-stage reasoning approach that preserves complete visual information and fully exploits the reasoning capability of MLLMs.
5. **Key Insight**: Directing the MLLM to handle the reference image and the modification text simultaneously, thereby avoiding information loss in the intermediate captioning stage.
6. **Core Idea**: One-stage MLLM reasoning combined with Reflective Chain-of-Thought (Reflective CoT) guidance to precisely understand modification intent.

## Method

### Overall Architecture
Reference image $I_r$ + modification text $T_m$ → MLLM (with Reflective CoT prompt) → target image description $T_t$ → CLIP text encoding → cosine similarity matching with image database → retrieval results. All computations are conducted on a single NVIDIA A100 GPU using PyTorch.

### Key Designs

1. **One-Stage Inference Process**:

    - **Function**: Eliminates the information loss inherent in two-stage methods, directly reasoning the target description from the image and modification text.
    - **Mechanism**: $T_t = \Psi_M(p_c \circ I_r \circ T_m)$, where the CoT prompt, reference image, and modification text are concatenated as input to the MLLM to generate the target description in a single pass. Since the MLLM simultaneously "sees" both the image and the modification intent, it can preserve key visual details relevant to the modification.
    - **Design Motivation**: In two-stage methods, the captioner is unaware of the modification intent and cannot decide which details to retain. The one-stage approach allows complete visual information to participate in the reasoning process.

2. **Reflective Chain-of-Thought (Reflective CoT)**:

    - **Function**: Guides the MLLM to reason about the user's modification intent step-by-step, avoiding misunderstandings caused by simple prompting.
    - **Mechanism**: A four-step progressive reasoning process: (1) Reference image description: focusing on visual details related to the modification text; (2) Thoughts: analyzing the modification intent and the affected visual elements; (3) Reflections: filtering out incorrect intents and identifying the most relevant modified elements to mitigate hallucination; (4) Target image description: generating the final description based on the filtered elements. All steps are completed within a single prompt to ensure efficiency.
    - **Design Motivation**: User modification intents are often implicit (e.g., "without human" actually means retaining the puppy and the blurred human background), requiring multi-step reasoning and reflection to be accurately comprehended.

3. **Vision-by-Language Contextual Learning (Vision-by-Language ICL)**:

    - **Function**: Enables the MLLM to understand the expected output format of each CoT step while maintaining the zero-shot setting.
    - **Mechanism**: Providing example outputs in pure text format (without reference images) to guide the MLLM to generate properly formatted reasoning outputs at each step.
    - **Design Motivation**: Merely providing CoT guidelines is insufficient; the MLLM requires concrete examples to understand the expected behavior at each step.

### Loss & Training
This method is entirely training-free. Retrieval utilizes a frozen CLIP model, ranking candidate images based on the cosine similarity between their image encodings and the text encoding of the target description $T_t$. Evaluation is performed on hidden test sets via submission servers for CIRCO and CIRR, and directly evaluated for Fashion-IQ and GeneCIS.

## Key Experimental Results

### Main Results

| Method | CIRCO mAP@5 | CIRCO mAP@25 | CIRR R@1 | CIRR R@5 |
|------|------------|-------------|---------|---------|
| CIReVL (ViT-L/14) | 18.57 | 20.89 | 24.55 | 52.31 |
| CIReVL* (GPT-4o) | 18.92 | 21.15 | 24.83 | 52.68 |
| **OSrCIR** | **23.87** | **27.84** | **29.45** | **57.68** |
| LinCIR | 12.59 | 15.00 | 25.04 | 53.25 |

### Ablation Study

| Configuration | CIRCO mAP@5 | Description |
|------|------------|------|
| Full OSrCIR | 23.87 | Full Reflective CoT |
| w/o Reflections step | 21.42 | Contribution of the reflections step is +2.45 |
| Simple prompt (w/o CoT) | 19.85 | Overall contribution of the CoT framework is +4.02 |
| Two-stage + GPT-4o | 18.92 | One-stage outperforms two-stage |

### Key Findings
- OSrCIR significantly outperforms existing training-free methods across all CLIP architectures (ViT-B/32, ViT-L/14, ViT-G/14).
- The reflection step effectively filters out hallucinations from the Thoughts stage, improving reasoning accuracy.
- The improvement is particularly prominent on CIRCO (+5.30 mAP@5), which has a more accurate evaluation protocol, while the gain on CIRR is smaller but still significant.
- The inference efficiency of the one-stage method is comparable to that of the two-stage method, incurring no additional overhead.
- On Fashion-IQ, OSrCIR (ViT-L/14) achieves an average R@10 of 33.26%, which is +4.21 higher than CIReVL* and +5.46 higher than the best training-based method Context-I2W, with Shirt/Dress/Toptee reaching 33.17/29.70/36.92, respectively.
- Solely upgrading to a stronger MLLM (CIReVL $\rightarrow$ CIReVL*) yields marginal improvements (+0.50 mAP@5), indicating that the two-stage paradigm itself is the bottleneck.
- By default, GPT-4o (temperature=0) is used, and the results are averaged over three runs; it also supports GPT-4o-mini, GPT-4V, and open-source models like LLaVA and MiniGPT4.

## Highlights & Insights
- **Reason-before-Retrieve Paradigm**: OSrCIR reformulates CIR as an MLLM reasoning problem rather than a simple feature combination problem, aligning better with the human cognitive process of retrieval.
- **Critical Role of the Reflection Step**: The Reflections step resembles human "wait, let me think again" self-correction, effectively reducing reasoning hallucinations. Contributing +2.45 mAP@5, the reflection step is core to the performance gain.
- **Transferability to Other Multimodal Retrieval Tasks**: The design concept of Reflective CoT can be applied to any multimodal task that requires understanding implicit intents.
- **Importance of Information Integrity**: Experiments demonstrate that the one-stage method outperforms the two-stage approach with GPT-4o (+4.95 mAP@5), indicating that fully preserving visual information is more important than employing a stronger LLM.
- **Ingenious Design of Vision-by-Language ICL**: Using only pure text examples (sans images) is sufficient to guide the MLLM to understand the output format of each CoT step, maintaining a truly zero-shot setting.
- **Consistency Across CLIP Architectures**: From ViT-B/32 to ViT-G/14, OSrCIR consistently performs with a large margin, demonstrating that the method does not rely on a specific CLIP version. Qualitative analysis shows that OSrCIR can accurately capture key details missed by CIReVL (such as poster types, dog breeds, colors, etc.).

## Limitations & Future Work
- It relies heavily on the reasoning capacity of the MLLM, and the effectiveness may vary significantly across different MLLMs.
- The improvement on CIRR is relatively limited (+4.9 R@1), likely due to noisy annotations in this benchmark.
- The combination with training-based methods has not been explored; theoretically, integrating training could lead to further improvements.
- The retrieval still relies on the text-image alignment quality of CLIP, with CLIP's performance bottlenecking the final retrieval accuracy.
- The four-step reasoning of CoT increases the MLLM inference overhead, which demands efficiency considerations in large-scale retrieval scenarios.
- When the modification text is extremely simple (e.g., changing only the color), the reflection step might be redundant.
- It also significantly outperforms all adaptation methods on the GeneCIS benchmark (a more generalized composition retrieval), showcasing robust cross-benchmark generalization.

## Related Work & Insights
- **vs CIReVL**: CIReVL is a two-stage method where the separation of captioning and reasoning causes information loss; OSrCIR is a one-stage method that preserves complete visual information.
- **vs LinCIR/Pic2Word**: These methods require training a text inversion network, whereas OSrCIR is completely training-free and delivers superior performance (CIRCO mAP@5: OSrCIR 23.87% vs LinCIR 12.59% vs Context-I2W 13.04%).
- **vs LDRE**: LDRE uses diffusion model ensembles, which incur high computational overhead; OSrCIR is much more efficient.

## Rating

### Implementation Details
By default, the MLLM is GPT-4o with the API temperature set to 0 and all parameters left at their defaults. The retrieval module is built on PyTorch running on a single NVIDIA A100. CLIP variants utilize official weights (ViT-B/32, ViT-L/14), and ViT-G/14 uses OpenCLIP weights.
- Novelty: ⭐⭐⭐⭐ First application of one-stage + reflective CoT in ZS-CIR
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified across multiple benchmarks and architectures (CIRCO/CIRR/Fashion-IQ/GeneCIS) with detailed ablations
- Writing Quality: ⭐⭐⭐⭐ Clearly explained motivation and intuitive illustrations
- Value: ⭐⭐⭐⭐ Training-free method achieving new SOTA with high practical applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](../../ICLR2026/llm_reasoning/cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[ICLR 2026\] Retrieval-of-Thought: Efficient Reasoning via Reusing Thoughts](../../ICLR2026/llm_reasoning/retrieval-of-thought_efficient_reasoning_via_reusing_thoughts.md)
- [\[CVPR 2025\] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)
- [\[CVPR 2025\] Interleaved-Modal Chain-of-Thought](interleaved-modal_chain-of-thought.md)
- [\[CVPR 2025\] Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection](style_evolving_along_chain-of-thought_for_unknown-domain_object_detection.md)

</div>

<!-- RELATED:END -->
