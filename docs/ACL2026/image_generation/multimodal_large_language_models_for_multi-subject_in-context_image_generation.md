---
title: >-
  [Paper Note] Multimodal Large Language Models for Multi-Subject In-Context Image Generation
description: >-
  [ACL2026][Image Generation][Multi-subject generation] This paper proposes MUSIC, which introduces the visual reasoning capabilities of multimodal large language models into multi-subject in-context image generation. By a…
tags:
  - "ACL2026"
  - "Image Generation"
  - "Multi-subject generation"
  - "Visual Chain-of-Thought"
  - "Spatial layout planning"
  - "Subject consistency"
  - "Test-time scaling"
date: 2026-05-08
content_hash: aa329142ef1022b1
---

# Multimodal Large Language Models for Multi-Subject In-Context Image Generation

**Conference**: ACL2026  
**arXiv**: [2604.07422](https://arxiv.org/abs/2604.07422)  
**Code**: Not disclosed  
**Area**: Multimodal VLM / Personalized Image Generation  
**Keywords**: Multi-subject generation, Visual Chain-of-Thought, Spatial layout planning, Subject consistency, Test-time scaling

## TL;DR
This paper proposes MUSIC, which introduces the visual reasoning capabilities of multimodal large language models into multi-subject in-context image generation. By automatically synthesizing training data, employing visual CoT, and utilizing semantic-driven spatial layout planning, it significantly mitigates problems such as subject omission, identity blending, and semantic drift during the simultaneous generation of multiple reference subjects.

## Background & Motivation
**Background**: Text-to-image models have become capable of generating stable scenes based on natural language. Personalized image generation further allows users to provide one or more reference subjects and requires the model to place these subjects into new contexts. Early methods often relied on per-subject optimization, while subsequent approaches like IP-Adapter, BLIP-Diffusion, ELITE, MS-Diffusion, and UNO have moved toward zero-shot or multi-subject generation, with the core goal of preserving reference subject identities while satisfying text instructions.

**Limitations of Prior Work**: When the number of reference subjects increases, the task becomes exponentially more difficult. The model must not only remember the visual identity of each subject but also understand their semantic relationships, spatial relationships, and roles within the scene. If multiple reference images are simply fed into a diffusion model as conditions, it often results in missing subjects, attribute bleeding between subjects, partially ignored text instructions, or images that preserve local appearances but fail in overall semantics.

**Key Challenge**: The difficulty of multi-subject generation is not just "more conditions," but rather the compositional reasoning between conditions. Diffusion-based subject-to-image methods are generally proficient at injecting visual features into the generation process but struggle with explicit planning regarding "where each subject should be, who they interact with, and which appearance belongs to whom." While MLLMs excel at contextual reasoning across images and text, making them truly serve image generation requires structured training signals and executable intermediate plans.

**Goal**: The authors aim to solve three sub-problems. First, how to construct large-scale multi-subject training data without manual annotation. Second, how to enable the model to understand the relationships between multiple subjects before generation rather than just concatenating conditions. Third, how to ground semantic relationships into 2D spatial layouts in complex scenes to reduce identity entanglement and subject loss.

**Key Insight**: A key observation is that multi-subject in-context generation is more akin to a process of "reading images, imagining the scene, and then drawing." Since MLLMs already possess certain visual understanding and linguistic reasoning abilities, multi-subject generation can be decomposed into explicit visual reasoning and spatial planning, using these intermediate results as conditions for image generation.

**Core Idea**: Use an MLLM to first produce a visual CoT and a semantic spatial layout plan, and then generate the target image based on the plan, thereby replacing black-box multi-subject condition fusion with interpretable intermediate reasoning.

## Method
The overall logic of MUSIC can be summarized in two layers: first, automatically generating training data, and then training a multimodal generative model capable of "seeing reference subjects, writing plans, and generating images." Instead of merely adding an adapter to an existing subject-driven model, it explicitly models multi-subject generation as a visual reasoning task.

### Overall Architecture
The input consists of a user instruction and several reference subject images; the output is a target image containing these subjects that satisfies the text requirements. To train this process, the authors first use foundation models to automatically construct data: an LLM selects semantically related categories from Object365 and writes scene captions, a T2I model generates target images containing multiple subjects, an open-vocabulary detector finds subject boxes, a VLM verifies the detection results, an I2I model transforms cropped subjects into new reference images, and finally SAM2 and CLIP assist in generating spatial layout descriptions.

During the training phase, each sample includes a set of reference subject images, a target image, a user instruction, a visual CoT, and a semantic spatial layout plan. MUSIC is based on SEED-X and fine-tuned via LoRA. The model first learns to predict the CoT and layout from the "instruction + subject images," and then learns to generate the target image from the "CoT + layout." During inference, the model follows this same pipeline: it first generates one or more candidate plans and then produces an image based on the plan. If test-time scaling is enabled, it generates multiple layout branches and uses CLIP to select the result that best matches the text.

### Key Designs
1. **Fully Automatic Multi-Subject Data Generation Pipeline**:

	- **Function**: Constructs training samples containing reference subjects, target images, user instructions, visual reasoning, and spatial layouts without manual annotation or real metadata.
	- **Mechanism**: Candidate categories are randomly sampled from Object365, and an LLM filters for semantically compatible combinations to generate scene descriptions. A T2I model generates the target image, GroundingDINO detects subjects, and a VLM checks for detection accuracy. Subjects are Затем cropped and transformed via a UNO-FLUX-style I2I model for viewpoint or appearance changes, resulting in reference subject images that are not identical to the target image crops.
	- **Design Motivation**: Multi-subject personalization lacks large-scale training sets, and manual annotation is costly. Furthermore, if subjects are cropped directly from the target image to serve as reference images, the model may learn a simple "copy-paste" shortcut. The automatic pipeline reduces data costs and forces the model to learn "identity preservation in new scenes" through I2I transformations.

2. **Visual CoT and Two-Stage Capability Decomposition**:

	- **Function**: Allows the model to explicitly analyze instructions, subject identities, and subject relationships before generating the image.
	- **Mechanism**: The VLM generates simulated user instructions based on the target image (with boxes and labels) and then generates a detailed CoT explaining each subject's role, relationships, and spatial arrangements. During training, MUSIC learns the mapping $f_1(T_{instr}, \{I_{subj_i}\}) \rightarrow (\hat{C}_{CoT}, \hat{L}_{spatial})$ and then learns $f_2(C_{CoT}, L_{spatial}) \rightarrow \hat{I}_{tgt}$. The total objective consists of a cross-entropy loss for reasoning/layout and a representation loss for image generation, expressed as $L = w_1 L_1 + w_2 L_2$.
	- **Design Motivation**: Multi-subject errors often stem from the model's inability to distinguish "who is who" and "where they should be." CoT converts implicit combinatorial relationships into intermediate text states, allowing the reasoning advantages of MLLMs to participate in generation rather than treating image features purely as low-level conditions.

3. **Semantic-Driven Spatial Layout Planning and Test-Time Scaling**:

	- **Function**: Binds subject semantics to 2D image regions to reduce identity entanglement between subjects and enhances quality through multi-candidate plans during inference.
	- **Mechanism**: The paper divides the target image into an $8 \times 8$ grid. SAM2 generates subject masks from detection boxes, and CLIP compares whether the mask or box region better fits the category text; if $Sim_{mask} > Sim_{unmask}$, the mask is used. The IoU between subject regions and each patch is calculated, and category labels are assigned to grid patches using a dynamic threshold $\tau = \lambda \cdot \frac{1}{K}\sum_{i=1}^{K} IoU(b_i, p_i)$. During inference, MUSIC generates $N$ candidate CoT/layout branches and uses CLIP to select the image with the highest text similarity.
	- **Design Motivation**: Pure text CoT may remain abstract and fail to precisely constrain image regions. Grid-based layouts ground semantic relations like "the flower is on the left" into spatial prompts for the generation module. Multi-branch reasoning transforms layout planning into a low-cost test-time search.

### Loss & Training
In implementation, the authors use Qwen-3 as the LLM, Qwen-2.5-VL as the VLM, FLUX-1.0-DEV as the T2I model, UNO-FLUX-1.0-DEV as the I2I model, GroundingDINO for open-vocabulary detection, SAM2 for segmentation, and CLIP ViT-L/14 for mask quality assessment and candidate selection.

The training data consists of 10,000 automatically generated samples, with the number of subjects ranging up to 12. Bounding box area filters are set to 0.01 of the target image area, using an $8 \times 8$ spatial grid and a dynamic IoU scaling factor $\lambda=0.05$. MUSIC is initialized from SEED-X and fine-tuned using LoRA with rank 64, $\alpha=64$, for 10 epochs at a learning rate of $1\times10^{-4}$ on 8 A100 GPUs. The loss weights are fixed at $w_1=0.5$ and $w_2=0.5$.

One notable training technique is "Subject Count Decimation Augmentation": the authors sort subjects by bounding box size and progressively remove smaller ones until only two remain, updating the IDs in the instructions accordingly. This provides training samples with varying subject counts from the same complex sample for free, ensuring the model encounters both simple and crowded scenes.

## Key Experimental Results

### Main Results
The paper evaluates MUSIC on two datasets. MSIC is a new multi-subject in-context generation benchmark compiled by the authors covering 1 to 12 subjects; DreamBench is used for single-subject personalization. Metrics include DINO, CLIP-I, and CLIP-T, measuring image/subject fidelity, consistency, and text alignment, respectively.

| Dataset | Metric | Best Result (Ours) | Strongest Baseline | Gain |
|--------|------|--------------|--------------|------|
| MSIC | DINO | MUSIC*: 0.631 | UNO: 0.541 | +0.090 |
| MSIC | CLIP-I | MUSIC*: 0.822 | UNO: 0.721 | +0.101 |
| MSIC | CLIP-T | MUSIC*: 0.330 | OmniGen: 0.300 | +0.030 |
| DreamBench | DINO | MUSIC*: 0.768 | UNO: 0.760 | +0.008 |
| DreamBench | CLIP-I | MUSIC*: 0.840 | UNO: 0.835 | +0.005 |
| DreamBench | CLIP-T | MUSIC*: 0.321 | RealCustom++: 0.318 | +0.003 |

The multi-subject MSIC highlights the value of the proposed method. Without test-time scaling, MUSIC achieves DINO 0.622, CLIP-I 0.812, and CLIP-T 0.322; MUSIC* further improves these to 0.631, 0.822, and 0.330. Compared to the strongest baseline UNO, subject consistency and image fidelity are significantly improved, indicating that explicit reasoning and layout planning mitigate degradation as subject counts increase.

| Method | MSIC DINO ↑ | MSIC CLIP-I ↑ | MSIC CLIP-T ↑ | Note |
|------|-------------|---------------|---------------|------|
| Subject Diffusion | 0.513 | 0.702 | 0.287 | Weak multi-subject extension |
| MIP-Adapter | 0.497 | 0.715 | 0.288 | Decent CLIP-I, low fidelity |
| MS-Diffusion | 0.532 | 0.714 | 0.290 | Includes layout but limited reasoning |
| OmniGen | 0.525 | 0.714 | 0.300 | Stronger text alignment than most diffusion baselines |
| UNO | 0.541 | 0.721 | 0.296 | Strongest subject-to-image baseline |
| MUSIC (Ours) | 0.622 | 0.812 | 0.322 | Significantly outperforms all baselines |
| MUSIC* (Ours) | 0.631 | 0.822 | 0.330 | Further gain from test-time layout scaling |

Gains on DreamBench are smaller but important: MUSIC is designed for multi-subject tasks but does not sacrifice single-subject performance. MUSIC* scores (DINO 0.768, CLIP-I 0.840, CLIP-T 0.321) surpass or match the strongest single-subject personalized methods, showing that two-stage reasoning is effective beyond complex scenes.

### Ablation Study
The paper does not provide a module-by-module ablation (e.g., removing CoT or layout separately). Therefore, quantitative results mainly reflect the gain of test-time scaling and overall performance compared to the strongest baseline.

| Configuration | Key Metrics | Description |
|------|---------|------|
| UNO on MSIC | DINO 0.541 / CLIP-I 0.721 / CLIP-T 0.296 | Strongest baseline, lacks explicit MLLM reasoning chain |
| MUSIC on MSIC | DINO 0.622 / CLIP-I 0.812 / CLIP-T 0.322 | Full model with auto-data, visual CoT, and spatial layout |
| MUSIC* on MSIC | DINO 0.631 / CLIP-I 0.822 / CLIP-T 0.330 | Multi-candidate layouts + CLIP selection; +0.009 / 0.010 / 0.008 over MUSIC |
| MUSIC on DreamBench | DINO 0.761 / CLIP-I 0.837 / CLIP-T 0.317 | Multi-subject training does not damage single-subject capability |
| MUSIC* on DreamBench | DINO 0.768 / CLIP-I 0.840 / CLIP-T 0.321 | Test-time scaling also provides small gains on single subjects |

### Key Findings
- In multi-subject scenarios, the advantages of MUSIC are primarily reflected in CLIP-I and DINO, suggesting it better preserves the visual identities of multiple reference subjects rather than just following text better.
- The gains from test-time scaling are stable but modest (around 0.01 across MSIC metrics), consistent with its role as a candidate layout search rather than model retraining.
- MUSIC* achieves top scores even on single-subject DreamBench, suggesting that complex data and reasoning-based training did not bias the model solely toward crowded scenes.
- A weakness of the paper is the lack of module-level ablations. While "Complete MUSIC" is effective, the specific contributions of visual CoT, semantic grids, I2I viewpoint transformation, and decimation augmentation remain unquantified.

## Highlights & Insights
- Defining multi-subject generation as a reasoning problem, rather than just a conditioning problem, is the core insight. Failure in multi-subject generation often stems from the model's inability to establish the "subject-identity-space-semantics" binding.
- The automatic data pipeline is highly practical. It chains LLM, T2I, OVD, VLM, I2I, SAM2, and CLIP into a closed loop, bypassing expensive manual labeling and ensuring training data contains natural transitions from reference to target scene.
- The semantic-driven $8 \times 8$ spatial layout is a transferable design. Similar ideas can be applied to multi-object editing, video planning, or robotic visual tasks.
- MUSIC* test-time scaling resembles using CoT sampling in text generation. Just as voting across multiple reasoning paths improves LLM outputs, searching across spatial plan candidates improves generation.
- The MLLM is not used as a simple captioner; it handles structured intermediate states. This is more meaningful than merely generating longer prompts, as grids and subject IDs provide actionable constraints for the generation module.

## Limitations & Future Work
- Lack of module-level ablation studies makes it difficult to judge the relative importance of visual CoT vs. spatial layout vs. I2I transformations.
- Training data is entirely synthetic. While scalable, it might inherit biases from the foundation models (OVD failures, VLM errors, or unnatural I2I transformations).
- The MSIC benchmark is derived from synthetic data; a domain gap might exist between this and real-world photos, complex portraits, branded products, or heavy occlusion.
- The $8 \times 8$ grid is suitable for coarse planning but may lack precision for thin objects, local interactions (hand-holding), or complex layering. Future work could explore hierarchical grids or instance mask tokens.
- Test-time scaling relies on CLIP, which is better at global matching than identifying fine-grained subject identity errors. A better verifier should check text, subject similarity, and spatial logic simultaneously.
- Training costs remain high, requiring multiple foundation models and 8 A100s for fine-tuning. Future work should investigate lighter data construction and path-count control.

## Related Work & Insights
- **vs DreamBooth / Textual Inversion**: These optimize dedicated tokens for single subjects. MUSIC uses an in-context approach, avoiding per-subject optimization and making it suitable for ad-hoc multi-subject prompts.
- **vs IP-Adapter / BLIP-Diffusion / ELITE**: These zero-shot methods use extra encoders for injection but lack explicit intermediate reasoning. MUSIC's advantage lies in establishing relationships/layouts via MLLM first.
- **vs MS-Diffusion**: MS-Diffusion also uses layout guidance but focuses on diffusion control. MUSIC derives spatial layouts from visual CoT, making categorical and spatial signals part of a readable planning signal.
- **vs UNO**: UNO is the primary baseline for visual ICL. MUSIC improves where UNO degrades (multi-subject counts) by introducing MLLM reasoning and test-time layout sampling.
- **Related Work**: Multimodal generation need not be a black-box end-to-end process. The "Understand-Plan-Execute-Verify" paradigm is highly applicable to long-form video, 3D scenes, and interactive content.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to explicitly use MLLMs as the reasoning and planning core for multi-subject in-context generation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Strong results on MSIC and DreamBench, but lacks key module ablations and real-world photo analysis.
- Writing Quality: ⭐⭐⭐⭐☆ Methodology and data construction are well-detailed; the experimental analysis could be more expansive.
- Value: ⭐⭐⭐⭐⭐ Multi-subject personalized generation is a practical need; MUSIC provides an extensible data-reasoning-planning framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](../../CVPR2026/image_generation/micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](../../CVPR2026/image_generation/psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)
- [\[ACL 2026\] From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons](from_ar_to_diffusion_efficiently_adapting_large_language_models_with_strictly_ca.md)
- [\[AAAI 2026\] Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](../../AAAI2026/image_generation/unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [\[ACL 2026\] MENTOR: Efficient Autoregressive Image Generation with Balanced Multimodal Control](mentor_efficient_multimodal-conditioned_tuning_for_autoregressive_vision_generat.md)

</div>

<!-- RELATED:END -->
