---
title: >-
  [Paper Note] "Jutters"
description: >-
  [NeurIPS 2025 (Creative AI Track)][AIGC Detection][AI-generated content] Through the metaphor of the Dutch tradition of *jutters* (beachcombers), this work constructs an immersive installation art piece that integrates r…
tags:
  - "NeurIPS 2025 (Creative AI Track)"
  - "AIGC Detection"
  - "AI-generated content"
  - "art installation"
  - "human-AI interaction"
  - "digital curation"
  - "AI imagery"
date: 2026-05-08
content_hash: cf421fdc35642178
---

# "Jutters"

**Conference**: NeurIPS 2025 (Creative AI Track)
**arXiv**: [2601.11532](https://arxiv.org/abs/2601.11532)
**Code**: None (art installation project)
**Area**: AIGC Detection
**Keywords**: AI-generated content, art installation, human-AI interaction, digital curation, AI imagery

## TL;DR
Through the metaphor of the Dutch tradition of *jutters* (beachcombers), this work constructs an immersive installation art piece that integrates real beach debris with AI-generated images and videos, guiding visitors to adopt a beachcomber's mindset in reflecting on how to engage with AI-generated content.

## Background & Motivation

**Background**: With the rapid advancement of generative AI, the proportion of AI-generated content in the digital landscape has surged, making it increasingly difficult for users to distinguish original from generated content.

**Limitations of Prior Work**: People tend to either passively accept or wholesale reject AI-generated content, lacking a deliberate *curation* mindset for filtering and interpreting such material.

**Key Challenge**: AI-generated content lacks the relational network—authorship, history, ownership, and intended use—inherent to traditional objects, making it difficult for people to form meaningful relationships with it.

**Goal**: How can people be guided to engage with AI-generated content in a more deliberate and proactive manner?

**Key Insight**: The authors map the Dutch coastal beachcombing tradition (*jutters*) onto the digital age. Jutters search for objects washed ashore after storms and endow them with new meaning; similarly, people can act as jutters by *curating* AI-generated content that floods their information streams.

**Core Idea**: By materializing AI-generated content within a physical installation space, the work invites visitors to actively filter, evaluate, and assign meaning as jutters.

## Method

### Overall Architecture
The project is an immersive installation of approximately 2×3 meters, simulating a small beach. Starting from 100+ photographs of real objects collected on Dutch beaches, the work uses AI models to generate images and videos, which are then embedded in the artificial beach space as physical prints and screen displays. Guided by an audio narrative, visitors traverse a sandy path and decide—like jutters—which "washed-up objects" are worth keeping.
The core interaction design of the installation concretizes the act of filtering digital content into a physical "beachcombing" experience. Printed photographs and video screens are interspersed with real shells and driftwood; visitors cannot determine which items are real and which are AI-generated, and this ambiguity is central to the design. An audio narrative system incorporating wave sounds and a narrator's voice helps visitors inhabit the jutter role and reflect on the value of each object.

### Key Designs

1. **AI Image Generation Pipeline**:

    - **Function**: Transform photographs of real beach objects into images that are plausible yet uncanny.
    - **Mechanism**: More than 100 object photographs were taken on Hoek van Holland beach; GPT-4 Omni then combined two or three photographs into prompts to generate new images. The selection criterion was "familiar but uncanny," designed to evoke visitors' desire to judge.
    - **Design Motivation**: By introducing ambiguity, visitors are compelled to think and judge actively rather than passively accept content.

2. **AI Video Generation Pipeline**:

    - **Function**: Generate surreal videos of objects morphing within ocean waves.
    - **Mechanism**: Real beach object videos were recorded at Zandvoort beach and subjected to video-to-video style transfer using WarpFusion (based on Stable Diffusion v1.4). An early diffusion model was deliberately chosen to achieve a more surreal aesthetic. Post-processing employed TopazLabs frame interpolation to improve frame rate.
    - **Design Motivation**: The videos metaphorically represent the erosion and transformation of objects by waves, mirroring how AI "processes" digital content.

3. **Immersive Space Design**:

    - **Function**: Create a hybrid physical–digital interactive space.
    - **Mechanism**: Alternating wet and dry sand simulates the intertidal zone; light-colored sand demarcates the walking path; real beach debris such as shells and driftwood is interspersed with 7 AI-printed images and 2 video screens. An audio narrative guides visitors into the jutter role.
    - **Design Motivation**: Embodied experience in physical space is more conducive to reflection and judgment than digital presentation on a screen.

### Loss & Training
This project involves no model training; it relies entirely on inference from pre-trained GPT-4 Omni and Stable Diffusion v1.4 models.

## Key Experimental Results

### Installation Parameters

| Element | Details |
|---------|---------|
| Spatial dimensions | Approx. 2×3 m artificial beach |
| AI-generated images | 7 printed works |
| AI videos | 2 screen displays |
| Real object photographs | 100+ |
| Image generation model | GPT-4 Omni |
| Video generation model | WarpFusion + SD v1.4 |

### Design Choice Analysis

| Design Decision | Choice | Rationale |
|----------------|--------|-----------|
| Image generation model | GPT-4 Omni | Capable of generating from combined multi-photo prompts; produces realistic yet uncanny results |
| Video base model | SD v1.4 (early version) | Early diffusion models produce more surreal outputs better suited to artistic requirements |
| Physical medium | Mixed prints and screens | Anchors digital content in physical space, enhancing embodied experience |

### Key Findings
- Selecting "familiar but uncanny" AI images elicits more active judgment from visitors than images that are obviously fake or fully photorealistic.
- Early diffusion models (SD v1.4) hold an advantage over newer models in artistic creation precisely because their "imperfections" carry a surreal aesthetic quality.
- The co-placement of real beach debris alongside AI-generated images blurs the boundary between the real and the generated, accurately mirroring conditions in everyday information streams.

## Highlights & Insights
- **Metaphor-Driven Design**: Mapping the jutter tradition onto AI content consumption creates a cross-domain metaphor that is more effective at provoking empathy and reflection than direct discussion of AI literacy.
- **Embodied Interaction over Screen Interaction**: Having visitors physically "walk through," "pick up," and "decide to keep or discard" cultivates critical awareness more effectively than scrolling through a screen.
- **Model Imperfection as Feature**: The deliberate choice of an early SD model's "imperfect" outputs demonstrates that in generative AI art, greater model capability is not always preferable.

## Limitations & Future Work
- **Absence of User Studies**: The paper reports no visitor behavioral data or feedback questionnaires, making it impossible to quantify the installation's actual impact on "AI literacy." Future work should incorporate pre/post-test surveys or behavioral tracking.
- **Limited Scale**: With only 7 images and 2 videos, the installation is too small to simulate the "information overload" of real information streams. Generating a larger corpus of AI images would create a richer "beach."
- **Unidirectional Narrative**: The audio guidance is relatively directive, constraining visitors' freedom to explore and potentially rendering the experience more of a "guided tour" than an act of "autonomous curation." Removing the audio to allow fully self-directed exploration is worth considering.
- **No Comparative Baseline**: The work is not compared with other AI literacy education methods such as lectures or online interactive tools.
- **No Multimodal Interaction**: Visitors can only observe visually; they cannot physically "pick up" or "take away" AI images, weakening the sense of jutter participation.
- **Poor Reproducibility**: The installation depends on a physical space and actual sand, making it difficult for other researchers to replicate or improve upon.

## Related Work & Insights
- **vs. Traditional AI Literacy Education**: Conventional approaches primarily transmit information (lectures, detection tools), whereas this work cultivates intuitive judgment through artistic experience, prioritizing attitude change over knowledge transfer.
- **vs. AI Art Exhibitions**: Most AI art exhibitions showcase AI's creative capabilities; this work inversely focuses on *human judgment of AI content*, centering people rather than AI as the subject.
- **vs. Digital Literacy Tools**: Tools such as Reality Defender and deepfake detectors prioritize technical accuracy, whereas this work focuses on cultivating human judgment capacity.
- The approach of using physicalization and embodiment to help people understand digital phenomena is transferable to contexts such as deepfake education and information literacy training.
- The jutter metaphor surfaces a deeper question: if the stance toward AI-generated content shifts from "detect and remove" to "curate and assign meaning," the entire framework of the AIGC detection field may need to be reconsidered.

## Rating
- Novelty: ⭐⭐⭐⭐ — Uniquely combines traditional cultural metaphor with critical reflection on AI-generated content.
- Experimental Thoroughness: ⭐⭐ — Lacks user studies and quantitative evaluation; amounts to an installation description.
- Writing Quality: ⭐⭐⭐⭐ — Fluent narrative and precise metaphors, balancing academic rigor with poetic sensibility.
- Value: ⭐⭐⭐ — Thought-provoking as a creative AI work, though of limited direct reference value to the technical community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)
- [\[NeurIPS 2025\] ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)
- [\[NeurIPS 2025\] Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving](reasoning_compiler_llm-guided_optimizations_for_efficient_model_serving.md)
- [\[NeurIPS 2025\] CLAWS: Creativity Detection for LLM-Generated Solutions Using Attention Window of Sections](clawscreativity_detection_for_llm-generated_solutions_using_attention_window_of_.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)

</div>

<!-- RELATED:END -->
