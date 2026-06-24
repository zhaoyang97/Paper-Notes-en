---
title: >-
  [Paper Note] Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities
description: >-
  [ACL 2025][Multimodal VLM][Multimodal Jailbreaking] This paper proposes the Con Instruction method, which optimizes adversarial images or audio to align them with target malicious instructions in the embedding space. This achieves jailbreaking of multimodal large language models (MLLMs) without textual inputs, reaching an attack success rate of 86.6% on LLaVA-v1.5. Additionally, the ARC evaluation framework is introduced to simultaneously measure both the quality and relevanc…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multimodal Jailbreaking"
  - "Adversarial Attacks"
  - "Non-Textual Instructions"
  - "Embedding Space Alignment"
  - "Safety Mechanism Bypassing"
date: 2026-05-08
content_hash: 1c986fefaa8861e4
---

# Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities

**Conference**: ACL 2025  
**arXiv**: [2506.00548](https://arxiv.org/abs/2506.00548)  
**Code**: Yes (mentioned as public in the paper)  
**Area**: AI Safety / Multimodal VLMs  
**Keywords**: Multimodal Jailbreaking, Adversarial Attacks, Non-Textual Instructions, Embedding Space Alignment, Safety Mechanism Bypassing

## TL;DR

This paper proposes the Con Instruction method, which optimizes adversarial images or audio to align them with target malicious instructions in the embedding space. This achieves jailbreaking of multimodal large language models (MLLMs) without textual inputs, reaching an attack success rate of 86.6% on LLaVA-v1.5. Additionally, the ARC evaluation framework is introduced to simultaneously measure both the quality and relevance of attack responses.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) such as LLaVA, InternVL, Qwen-VL, and Qwen-Audio are capable of understanding and processing non-textual modalities like images and audio. Meanwhile, the safety defense mechanisms of these models are primarily designed for textual inputs, rejecting hazardous requests by detecting harmful intents within text.

**Limitations of Prior Work**: Existing MLLM jailbreak attacks (such as visual adversarial attacks, adversarial prompt injections, etc.) mainly operate in a "textual instruction + adversarial image assistance" manner—where the malicious intent is still conveyed via text, and the adversarial image only serves as an auxiliary bypass. Consequently, text safety filters can still detect the attack intent. Furthermore, many methods require training data or preprocessing of textual instructions, which increases attack complexity.

**Key Challenge**: MLLMs possess robust capabilities in understanding non-textual instructions (e.g., "reading" text within images, understanding semantics in audio), yet safety defense mechanisms primarily inspect the textual channel. This implies that if a malicious instruction is completely delivered through non-textual modalities, safety filters may fail to detect it entirely.

**Goal**: (1) To verify whether MLLMs can receive and execute malicious instructions solely through non-textual modalities (images/audio); (2) To develop a universal, training-free jailbreak method; (3) To design a more comprehensive attack evaluation framework.

**Key Insight**: Since MLLMs are trained to comprehend the semantics of non-textual inputs, can malicious instructions be "encoded" into images or audio, turning the model's multimodal understanding capability itself into a source of safety vulnerability?

**Core Idea**: To generate adversarial images/audio via gradient optimization, aligning them highly with target malicious instructions in the MLLM's embedding space, thereby achieving "non-textual modality as instruction" jailbreak attacks.

## Method

### Overall Architecture

The attack pipeline of Con Instruction consists of three steps: (1) **Target Instruction Encoding**: Mapping the malicious textual instruction into the embedding space using the MLLM's text encoder to obtain the target embedding vector; (2) **Adversarial Sample Optimization**: Initializing a random image/audio and optimizing its pixels/spectral values via gradient descent so that its embedding after passing through the MLLM's vision/audio encoder is as close as possible to the target embedding; (3) **Attack Execution**: Inputting the optimized adversarial image/audio either alone or combined with harmless text into the MLLM, allowing the model to "read" the malicious instruction embedded in the non-textual modality through multimodal fusion and execute it.

### Key Designs

1. **Embedding Space Alignment Optimization (Embedding Space Alignment)**:

    - **Function**: Encoding the semantics of malicious instructions into non-textual modalities.
    - **Mechanism**: Given a malicious textual instruction $t$, the target embedding $e_t = \text{TextEnc}(t)$ is obtained via the text encoder. For an image input $x$ (initially random noise), the embedding $e_x = \text{VisEnc}(x)$ is computed via the vision encoder. The optimization objective is to minimize the cosine distance $\mathcal{L} = 1 - \cos(e_x, e_t)$, iteratively updated in the pixel space using PGD (Projected Gradient Descent). The same applies to the audio modality. This process requires zero training data, relying solely on white-box forward/backward propagation through the model.
    - **Design Motivation**: The multimodal fusion mechanism of MLLMs maps different modalities into a shared embedding space, meaning that aligned images/audio in the embedding space will be decoded by the model as equivalent to the corresponding textual instructions. This leverages the model's own cross-modal understanding ability to convey malicious signals.

2. **Multi-Modal Amplification**:

    - **Function**: Significantly boosting attack success rates by combining non-textual adversarial samples with harmless text.
    - **Mechanism**: While using adversarial images/audio alone can already bypass safety mechanisms, the success rate is bounded by the precision of embedding alignment. By supplementing the textual channel with harmless text related to the malicious topic (e.g., "Please describe the content in the image" or related context), the model is assisted in more accurately "decoding" the hidden instructions in the non-textual modality, resulting in a substantial boost in attack success rate.
    - **Design Motivation**: Multimodal reasoning in MLLMs is collaborative—text provides context, while images/audio provide content. Utilizing this synergy helps overcome the precision bottleneck of single-modality embedding alignment.

3. **Attack Response Categorization (ARC)**:

    - **Function**: Comprehensively evaluating attack effectiveness and distinguishing different types of successes and failures.
    - **Mechanism**: Traditional evaluation focuses only on "whether harmful content is generated" (binary classification). ARC introduces two orthogonal dimensions: (a) response quality—the informativeness and completeness of the generated content; (b) response relevance—whether the generated content is relevant to the specific intent of the malicious instruction. This produces four quadrants: high-quality high-relevance (complete success), high-quality low-relevance (the model generated harmful content but not what was requested), low-quality high-relevance (the model understood the intent but provided an incomplete response), and low-quality low-relevance (complete failure).
    - **Design Motivation**: Existing ASR metrics are too coarse to distinguish between "the model generating irrelevant harmful content" and "the model precisely executing the malicious instruction." ARC provides a more fine-grained evaluation of attack effectiveness.

### Loss & Training

The core loss is the cosine distance loss in the embedding space $\mathcal{L} = 1 - \cos(e_x, e_t)$. PGD optimization is used, where the step size and number of iterations are hyperparameters. The $L_\infty$ norm constraint is applied to images to control perturbation size, and a similar spectral-domain constraint is applied to audio. No additional training data or fine-tuning of the target model is required.

## Key Experimental Results

### Main Results

**Vision-Language Model Attack Results (AdvBench + SafeBench):**

| Model | Method | AdvBench ASR | SafeBench ASR | Notes |
|------|------|-------------|-------------|------|
| LLaVA-v1.5 (7B) | Text-only Attack | 32.1% | 28.5% | Baseline |
| LLaVA-v1.5 (7B) | Con Instruction | **76.8%** | **79.2%** | Pure Image Attack |
| LLaVA-v1.5 (13B) | Con Instruction | **81.3%** | **86.6%** | Pure Image Attack |
| LLaVA-v1.5 (13B) | Con Inst. + Text Combination | **89.7%** | **92.1%** | Combined Attack |
| InternVL | Con Instruction | 68.4% | 71.2% | Pure Image Attack |
| Qwen-VL | Con Instruction | 65.7% | 69.8% | Pure Image Attack |

**Audio-Language Model Attack Results:**

| Model | Con Instruction ASR | Notes |
|------|-------------------|------|
| Qwen-Audio | 72.3% | Pure Audio Attack |
| Qwen-Audio + Text | 84.5% | Combined Attack |

### Ablation Study

| Configuration | ASR (LLaVA-v1.5-13B) | Notes |
|------|----------------------|------|
| Con Instruction (Full) | 81.3% | Image modality |
| Text-only attack (no image) | 32.1% | Safety filter effective |
| Random image + malicious text | 38.5% | Image yields no alignment effect |
| Con Inst. + harmless text | 89.7% | Combined attack significantly improved |
| Con Inst. (reduced optimization steps by 50%) | 62.4% | Decreased alignment precision |
| Con Inst. + adversarial training defense | 48.2% | Defense is effective but vulnerabilities remain |
| Con Inst. + input detection defense | 55.1% | Detection rate is limited |

### Key Findings

- Pure non-textual adversarial samples (without any malicious text) can reach an attack success rate of 81.3%, proving that safety mechanisms provide almost no protection for non-textual channels.
- The 13B model is more vulnerable to attacks than the 7B model (81.3% vs 76.8%). This is because larger models possess stronger cross-modal understanding capabilities, which ironically makes them easier to "decode" malicious instructions embedded in images.
- The non-textual + textual combined attack further boosts the success rate to 89.7%, demonstrating that multimodal synergetic effects can be exploited by attackers.
- Existing defense methods (adversarial training, input detection) show some efficacy but are far from sufficient, indicating a massive safety gap.
- The ARC evaluation framework reveals an important distinction missed by traditional ASR metrics—approximately 15% of the "successful" attacks actually generated harmful content irrelevant to the instructions.

## Highlights & Insights

- **Profound Insight of "Capability as a Vulnerability"**: The more powerful the MLLM (the better its cross-modal understanding), the easier it is to be attacked by Con Instruction. This reveals a fundamental paradox of multimodal AI safety—enhancing comprehension capabilities inevitably expands the attack surface.
- **Practicality of Zero-Data Attacks**: It requires no training data or target model fine-tuning, only white-box access for gradient computation. This significantly lowers the barrier to execution and implies that any open-source MLLM faces this threat.
- **Methodological Contribution of the ARC Evaluation Framework**: By employing a two-dimensional evaluation of quality × relevance, it provides a more accurate metric of attack effectiveness than traditional ASR, offering standardizing value for subsequent security research.

## Limitations & Future Work

- The attack requires white-box access (gradient computation) and is not directly applicable to closed-source models (e.g., GPT-4V). However, this can be partially addressed through transferability attacks (generating adversarial samples on open-source models and testing them on closed-source models).
- The adversarial images/audio generated here usually appear as meaningless noise to humans, making them easily identifiable by manual auditing. However, automated systems lack manual auditing phases.
- Testing on larger-scale models (such as LLaVA-Next-34B) is limited.
- Defense exploration is preliminary, and detection methods based on embedding space monitoring (e.g., detecting whether non-textual modality embeddings are abnormally close to harmful text embeddings) have not been investigated in-depth.
- The persistence of attacks across multi-turn conversation scenarios was not considered—specifically, whether the model can "remember" non-textual instructions in subsequent turns.

## Related Work & Insights

- **vs. Visual Adversarial Examples (Qi et al. 2024)**: The visual adversarial attacks by Qi et al. still rely on the textual channel to deliver parts of the malicious intent, whereas Con Instruction conveys instructions entirely through non-textual modalities, making it more thorough.
- **vs. Textual Jailbreaking such as GCG**: GCG manipulates text tokens and can be detected by perplexity filters. Con Instruction manipulates image pixels/audio specs, existing entirely outside the scope of text safety detection.
- **vs. Multimodal Prompt Injection**: Prompt injection typically embeds text inside images (e.g., as OCR carriers). Con Instruction encodes information within the embedding space rather than the pixel space, making it more stealthy.
- **Insights for Security Research**: MLLM safety requires "omni-modal" protection rather than relying solely on text filters. Embedding space monitoring could be an effective defense direction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic exploration of pure non-textual modalities as carriers of malicious instructions for jailbreak attacks, presenting a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both vision and audio modalities, multiple models, and two benchmarks, though testing on closed-source models is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions, and the ARC framework is rationally designed.
- Value: ⭐⭐⭐⭐⭐ Reveals key blind spots in multimodal AI safety, and the ARC framework can serve as a standardized evaluation tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](../../ICCV2025/multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[ACL 2025\] MEIT: Multimodal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation](meit_multimodal_electrocardiogram_instruction_tuning_on_large_language_models_fo.md)
- [\[ACL 2025\] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](hidellava_hierarchical_decoupling_for_continual_instruction.md)
- [\[ACL 2025\] Vision-Language Models Struggle to Align Entities across Modalities](vision-language_models_struggle_to_align_entities_across_modalities.md)
- [\[ICCV 2025\] IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves](../../ICCV2025/multimodal_vlm/ideator_jailbreaking_and_benchmarking_large_visionlanguage_m.md)

</div>

<!-- RELATED:END -->
