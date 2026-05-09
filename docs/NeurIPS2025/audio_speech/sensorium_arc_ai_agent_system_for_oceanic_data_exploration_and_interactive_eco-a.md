---
title: >-
  [Paper Note] Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art
description: >-
  [NeurIPS 2025][Audio & Speech][Ocean data visualization] This paper presents Sensorium Arc, a multimodal interactive AI agent system that personifies the ocean as a poetic "narrator" figure. Leveraging a multi-agent RAG architecture, the system integrates NASA ocean science data with eco-aesthetic texts, enabling users to explore complex marine environmental data through natural conversation while dynamically generating scientific visualizations and artistic audiovisual feedback—realizing a paradigm shift from "passive data observation" to "active ecological dialogue."
tags:
  - NeurIPS 2025
  - "Audio & Speech"
  - Ocean data visualization
  - RAG
  - multi-agent LLM
  - interactive eco-art
  - conversational AI
  - immersive media
date: 2026-05-08
content_hash: 4459264bf9053c9d
---

# Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art

**Conference**: NeurIPS 2025
**arXiv**: [2511.15997](https://arxiv.org/abs/2511.15997)
**Code**: None
**Area**: Human-Computer Interaction / Eco-Art / Multi-Agent Systems
**Keywords**: Ocean data visualization, RAG, multi-agent LLM, interactive eco-art, conversational AI, immersive media

## TL;DR

This paper presents Sensorium Arc, a multimodal interactive AI agent system that personifies the ocean as a poetic "narrator" figure. Leveraging a multi-agent RAG architecture, the system integrates NASA ocean science data with eco-aesthetic texts, enabling users to explore complex marine environmental data through natural conversation while dynamically generating scientific visualizations and artistic audiovisual feedback—realizing a paradigm shift from "passive data observation" to "active ecological dialogue."

## Background & Motivation

**Background**: The ocean is accompanied by vast publicly available environmental datasets—such as atmospheric CO₂ emissions, chlorophyll concentration, sea surface temperature, and ocean currents provided by NASA EarthData. However, these data have long been presented primarily through static visualizations or specialized query interfaces designed for researchers and policymakers, making it difficult for the general public—and even many domain experts—to intuitively and emotionally grasp the ecological significance behind the numbers. Meanwhile, immersive media technologies (e.g., VR/AR installations, interactive exhibitions) have been shown to enhance user empathy on environmental issues, knowledge retention, and action-oriented thinking. Representative works include *Rising Together*, *Dive*, and the *Sea Level Rise Explorer* series, which primarily help coastal communities understand the impacts of sea-level rise and adaptation strategies.

**Limitations of Prior Work**: Conventional environmental data visualization tools suffer from two fundamental problems. First, they present ocean data as "abstract datasets"—users see contour maps, heat maps, and time-series charts rather than a living narrative subject, a mode of presentation that struggles to inspire emotional resonance or deeper ecological awareness. Second, even immersive media installations (such as earlier VR and art installation versions of the Sensorium project), despite excelling in sensory engagement, elicited user feedback indicating that audiences desired the experience of "directly conversing with the ocean" rather than receiving information in a one-way flow. In other words, sensory immersion is important, but unidirectional presentation without dialogue or interactivity fails to satisfy users' deeper need for "co-creative ecological exploration."

**Key Challenge**: The essential tension is this: on one hand, the complexity and high dimensionality of scientific data demand precise information retrieval and multimodal synthesis; on the other, ecological awareness education requires emotional resonance, poetic expression, and embodied interaction to produce lasting impact. A gap exists between "data precision" and "emotional accessibility." Existing solutions either prioritize scientific accuracy at the expense of emotional power (traditional data visualization tools) or foreground artistic expression with weak scientific foundations (purely artistic installations), and no system has yet effectively bridged the two. Furthermore, when a single large language model is tasked simultaneously with retrieval, multimodal control, and persona-driven narration, prompt interference, persona confusion, and opaque errors arise, making fine-grained control difficult.

**Goal**: The authors decompose the core challenge into four concrete sub-problems: (1) How can an AI embody a scientifically reliable and artistically convincing "oceanic persona," maintaining a consistent ecological narrative style without domain-specific fine-tuning? (2) How can relevant scientific data visualizations and audiovisual feedback be dynamically triggered in response to natural-language dialogue? (3) How can multiple LLM agents be coordinated in a real-time system to achieve a modular, debuggable, and efficient workflow? (4) How can an embodied physical interaction interface be designed so that digital dialogue and ecological metaphor merge naturally in physical space?

**Key Insight**: The authors' core observation draws on the decades-long practice of eco-aesthetic pioneer Newton Harrison, who proposed that "the world ocean can be given a voice"—that is, the ocean is not merely a source of data but an ecological entity with "subjectivity" capable of dialogue with humans. On the technical side, the authors observe that LLMs have recently expanded from task-oriented dialogue to creative and embodied applications (e.g., ChatCam uses natural language to orchestrate visual workflows; USER-LLM achieves dynamic personalization via RAG and chain-of-thought reasoning), and that RAG has proven effective in biomedical QA, legal reasoning, and geospatial analysis (e.g., ClimateQA), making it technically feasible to construct a "dialogue-based ecological agent." The innovation of this angle lies in extending RAG from conventional "fact retrieval" to "poetic grounding"—using an eco-literary archive to shape the AI's narrative style rather than merely answering factual questions.

**Core Idea**: By personifying the ocean as a poetic narrator through a multi-agent RAG architecture, the system organically integrates NASA scientific data, eco-aesthetic texts, and immersive audiovisual feedback via a modular LLM pipeline, realizing a paradigm shift from "static data visualization" to "conversational ecological narration."

## Method

### Overall Architecture

Sensorium Arc adopts a modular four-layer design, integrating all components within the Unity engine. The system data flow proceeds as follows: the user whispers a question to the nautilus-shell interface → the microphone records audio and transcribes it via Whisper-tiny → the text enters the multi-agent LLM pipeline (passing sequentially through the Visualization Decider Agent, the Retrieval and Query Rewriter Agent, and the Responder Agent) → the pipeline produces a text response and visualization control tokens → the response processing system converts text to speech (Jets TTS), triggers keyword events, and indexes data visualizations → the audiovisual layer renders scientific data globe visualizations, plays video overlays, and displays synchronized captions. The input is the user's natural-language speech; the output is the ocean persona's voiced response together with dynamically linked scientific data visualizations and environmental art video.

The central design philosophy of this architecture is "separated modular dynamic layers"—each layer can be updated and extended independently, so new datasets, language support, or visual effects can be integrated without affecting other layers. Unity was chosen as the integration platform to leverage its mature capabilities in real-time rendering, physical interaction, and cross-platform deployment, ensuring the system can be deployed in a mobile form at public exhibition venues.

### Key Designs

1. **Input Processing Module — Embodied Nautilus Shell Interaction Interface**

    - **Function**: Serves as the physical interaction node between the user and the "ocean," converting a natural gesture (whispering) into a digital input signal.
    - **Mechanism**: The interaction module takes the nautilus (*Nautilus*) as its design archetype—the nautilus is often called a "living fossil connecting ancient and modern oceans," naturally carrying the metaphor of oceanic temporal memory. The hardware is Arduino-based, integrating a distance sensor, LED lighting, and a micro-motor, all housed within a nautilus shell. When a user approaches within 50 cm, the distance sensor triggers blue light ripples flowing across the shell surface, signaling that the system is ready; if the user remains within the threshold, the microphone system activates (with active noise cancellation to suppress ambient noise) and begins recording; when the user moves beyond 50 cm, recording ends. The recorded audio is then processed by the Whisper-tiny model for speech-to-text transcription, and the resulting text query is fed into the LLM pipeline.
    - **Design Motivation**: The gesture of "whispering into a shell" was chosen over conventional button or touchscreen interaction to establish—at the physical level—the metaphor of "conversing with the ocean": the user's bodily action itself becomes part of the ecological narrative. This embodied design anchors digital interaction in physical gesture, endowing an otherwise abstract human-machine dialogue with ritual weight and emotional significance. The progressive feedback of the distance sensor (blue light ripples) creates the perception that "the ocean is responding to your approach," deepening immersion.

2. **Multi-Agent LLM Pipeline — Three-Stage Collaborative Reasoning Architecture**

    - **Function**: Transforms the user's natural-language query into visualization selection tokens, retrieves relevant knowledge, and generates a final oceanic persona response—replacing the conventional single-model approach in which one large model handles all tasks.
    - **Mechanism**: The pipeline comprises three specialized agents executed in strict sequential order:

        **Stage 1: Visualization Decider Agent.** This is a small, stateless decision-maker using the Llama 3.2 3B model. It receives the user query and textual descriptions of all prepared visualization effects, and outputs a "visual selection token" specifying which scientific data visualization should be triggered for the current conversational turn. To ensure deterministic output format, the authors apply GBNF (Generalized Backus-Naur Form) grammar constraints, restricting the model's output space to tokens matching only the available options. Combined with few-shot examples embedded in the system prompt and a "reason before output" prompting strategy, the 3B-scale model reliably performs classification decisions. Placing visualization selection before retrieval and response generation ensures that the final Responder Agent knows what visualization is currently displayed, allowing the ocean's spoken response to maintain semantic consistency with the on-screen visual content.

        **Stage 2: Retrieval and Query Rewriter Agent.** This stage employs RAG for two simultaneous purposes: retrieving scientific facts relevant to the user's question, and providing a textual foundation for the oceanic persona's poetic expression. The knowledge base consists of the Harrison couple's eco-aesthetic archive, including *The Time of the Force Majeure* (a collected works spanning decades), "Apologia Mediterraneo" (a poetic dialogue with the Mediterranean), and the "Peninsula Europe I" detailed catalogue, among other unstructured texts. These documents are preprocessed via paragraph segmentation → sentence segmentation → embedding into a 384-dimensional vector space using the all-MiniLM-L12-v2 model → local lightweight vector database indexing. At retrieval time, the query is encoded with the same embedding model and an approximate nearest-neighbor (ANN) search implemented via the usearch library retrieves $k=2$ most similar sentence embeddings, returning the matched sentences along with their source paragraphs. The choice of $k=2$ rather than a larger value reflects early testing showing that retrieving more paragraphs filled the final LLM's context window and degraded response generation quality. Additionally, a query rewriting LLM agent (using the Qwen 8B model) is introduced prior to semantic search to rephrase the user's colloquial query into a more academic/literary formulation better matched to the textual knowledge base, while removing unnecessary pragmatic markers. This agent has prior awareness of the overall content and eco-philosophical framework of the Harrison texts; a basic chain-of-thought (CoT) prompting strategy encourages the model to reason about the conceptual core of the query before generating the rewrite. Qwen 8B was selected based on comparative testing across a representative query set, where it demonstrated the strongest rule-following behavior.

        **Stage 3: Responder Agent.** As the final step of the LLM pipeline, this agent receives three streams of contextual input: a textual description of the selected central visualization (from Stage 1), two retrieved passages (from Stage 2), and the original user query. It outputs the ocean's final response, which must satisfy both scientific accuracy and poetic register. Tuning this agent's system prompt proved to be the most challenging aspect of the entire project, as it required establishing multiple complex requirements within a single prompt: definition of the ocean persona's character, the manner of engagement with retrieved content, introductory knowledge of key ecological concepts, and instructions for using the dynamic context provided with each invocation. The authors found that carefully constructed zero-shot prompts performed well in most cases, but could not guarantee consistency of response characteristics such as length and format.

    - **Design Motivation**: Compared to the single-model approach explored in early iterations, the multi-agent architecture offers three key advantages: (1) *Debuggability*—each agent's behavior can be inspected and adjusted independently, enabling rapid localization of errors; (2) *Avoidance of prompt interference*—when a single model handles multiple tasks, different instructions tend to conflict with each other; separation allows each agent to focus on a single well-defined sub-task; (3) *Flexibility*—the most appropriate model scale can be selected for each sub-task (e.g., a 3B model for decisions, an 8B model for query rewriting, and a larger model for final response), balancing performance and efficiency. The first two agents are kept stateless and lightweight as much as possible, while more intensive reasoning and complex context management are concentrated in the final response generation step.

3. **RAG Knowledge Base and Retrieval Strategy — Dual-Purpose Semantic Retrieval System**

    - **Function**: Provides the system with a factual foundation and a poetic grounding foundation, ensuring that the oceanic persona's responses are both scientifically grounded and literarily deep.
    - **Mechanism**: Unlike conventional RAG systems used solely for fact retrieval, Sensorium Arc's RAG serves a dual purpose. The first is conventional fact retrieval—when users ask scientific questions about ocean acidification or temperature change, the system retrieves relevant scientific descriptions from the knowledge base. The second, more distinctive purpose is "poetic grounding"—by retrieving the Harrisons' eco-aesthetic texts, the system shapes the narrative style and register of the oceanic persona. This means the underlying corpus can be updated or re-weighted without retraining the model. The specific indexing strategy is: unstructured documents → paragraph segmentation → sentence segmentation → all-MiniLM-L12-v2 embeddings (384-dimensional) → local ANN index (usearch library, integrated via the LLMUnity plugin). At query time, the query vector is encoded with the same model, $k=2$ nearest neighbors are retrieved, and matched sentences with their source paragraphs are returned. The query rewriting agent converts informal queries into more academically styled formulations better matched to the document style before the search.
    - **Design Motivation**: The core value of RAG lies in enabling "persona customization without fine-tuning"—specific narrative style and knowledge context are granted to the AI through dynamic retrieval rather than parameter updates. This allows the oceanic persona's mode of expression to be altered simply by replacing or augmenting the knowledge base, dramatically reducing maintenance and iteration costs. The ultra-low retrieval count of $k=2$ reflects a practical engineering trade-off: within a limited context window, more retrieved passages are not always better, and may in fact overwhelm core persona instructions and visualization descriptions.

### Response Processing and Audiovisual Layer

The response processing system bridges the LLM pipeline output and the user-perceptible experience. It performs three key operations:

**Keyword Event Triggering**: The text output of the ocean's response passes through a modular keyword matcher that triggers corresponding audiovisual layer changes based on detected geographic locations, time periods, and thematic keywords. For example, when the response mentions "Arctic," the globe view automatically rotates to the Arctic region; when "coral bleaching" is mentioned, the associated video is triggered. Although simple, this keyword-based event system is highly extensible—new keyword-event mappings can be added to support new datasets and visualization effects.

**Visualization Selection Indexing**: The selection token from the Visualization Decider Agent is used to index a pre-prepared mapping table of data visualizations and video overlay layers, activating the corresponding global scientific data view.

**Text-to-Speech**: The final text response is converted to spoken audio via the Jets TTS model integrated in Unity Sentis, with synchronized captions generated simultaneously to ensure accessibility.

**Scientific Data Visualization Layer**: The system preprocesses multiple scientific datasets from NASA EarthData and earthaccess, including atmospheric CO₂ emissions, chlorophyll concentration, sea surface temperature, ocean currents, and hyperspectral diffuse attenuation coefficient (Kd) data from the NASA PACE satellite (2024). These data are converted to a "globe-ready" format that maintains visual clarity even when multiple datasets are overlaid. The paper presents four example scenarios: (a) the default globe view before user interaction; (b) chlorophyll concentration data visualized on the globe surface; (c) water clarity rendered using 16 stacked layers of hyperspectral Kd data; and (d) a combined visualization of atmospheric CO₂ levels and ocean surface wind fields.

**Video Playback Layer**: Pre-rendered videos include phytoplankton blooms, ocean acidification, plastic waste dispersion, and sea-level rise. Due to available computational resource constraints, all videos are pre-rendered in Unity to ensure performance efficiency; however, the authors plan to replace video playback with real-time rendering in future versions to support direct user interaction with narratively driven scenes.

### Hardware and Runtime Environment

System development and testing were conducted in the following environment: Unity 6000.0.24f1, running on Windows 11 with an Intel Core i9-13900HX CPU, 64 GB RAM, and an NVIDIA GeForce RTX 4090 Laptop GPU. CUDA acceleration enables a configurable number of GPU layers to be offloaded from CPU to GPU, significantly improving LLM inference performance and achieving an average end-to-end response latency of under 4 seconds following user input. Additional testing on macOS used Vulkan for GPU acceleration. The choice of local hardware (rather than cloud-based deployment) was made to ensure the system can be deployed in a mobile form suitable for public exhibitions. The software supports both Windows and macOS installations, and the system is planned for educational installations in schools and public institutions.

## Key Experimental Results

### System Architecture and Technology Stack

| Module | Implementation | Model / Tool | Key Parameters |
|--------|---------------|-------------|----------------|
| Speech Recognition | Speech-to-text | Whisper-tiny | Real-time transcription, active noise cancellation |
| Visualization Decision | Stateless classification | Llama 3.2 3B | Few-shot + GBNF grammar constraints |
| Query Rewriting | Query optimization | Qwen 8B | Chain-of-Thought prompting |
| Text Embedding | Semantic vectorization | all-MiniLM-L12-v2 | 384-dimensional vector space |
| Semantic Retrieval | Approximate nearest-neighbor search | usearch (ANN) | $k=2$, returns sentence + paragraph |
| Response Generation | Oceanic persona narration | Large LLM | Zero-shot prompting |
| Speech Synthesis | Text-to-speech | Jets (Unity Sentis) | Real-time audio + synchronized captions |
| Rendering Engine | Real-time rendering and integration | Unity 6000.0.24f1 | Cross-platform (Win/macOS) |
| GPU Acceleration | LLM inference acceleration | CUDA / Vulkan | RTX 4090 Laptop |

### System Performance and User Experience

| Performance Metric | Value / Result | Notes |
|-------------------|---------------|-------|
| End-to-end response latency | < 4 seconds | From user speech input to complete system response |
| Hardware configuration | i9-13900HX + 64 GB + RTX 4090 | Mobile exhibition-grade hardware |
| Deployment platforms | Windows + macOS | Dual-platform support |
| RAG retrieval count | $k=2$ | More passages cause context window overflow |
| Interaction distance threshold | 50 cm | Nautilus shell distance sensor trigger range |
| Number of datasets | 5+ | CO₂, chlorophyll, sea temperature, currents, Kd |
| Visualization layers | 16 | Stacked rendering of PACE satellite Kd data |

### Exhibition Observations and Qualitative Feedback

| Evaluation Dimension | Observed Result | Specific Evidence |
|---------------------|----------------|-------------------|
| Sensory engagement | High | Immersive nautilus shell interface and dynamic globe visualizations effectively attract users |
| Conversational willingness | Strong | Earlier version feedback: audiences actively sought direct dialogue with the ocean |
| Emotional resonance | Notable | Combination of scientific data and poetic narration perceived as enhancing environmental empathy |
| Multi-agent advantage | Clear | Provides better debuggability and fine-grained control compared to single-model approaches |
| Embodied interaction | Effective | Whispering gesture transforms passive observation into active co-creative dialogue |

### Ablation and Design Choice Analysis

| Design Choice | Adopted Approach | Alternative | Rationale |
|--------------|-----------------|-------------|-----------|
| Architecture pattern | Multi-agent (3 specialized agents) | Single large model | Single model exhibited prompt interference, persona confusion, and opaque errors |
| Retrieval count | $k=2$ | $k>2$ | More passages fill context window and degrade response quality |
| Visualization decision model | Llama 3.2 3B | Larger model | 3B + GBNF constraints sufficiently reliable; keeps pipeline lightweight |
| Query rewriting model | Qwen 8B | Other models | Demonstrated strongest rule-following behavior in comparative testing |
| Responder prompting strategy | Zero-shot | Few-shot | Carefully constructed zero-shot performs well in most cases |
| Deployment mode | Local hardware | Cloud inference | Public exhibitions require mobile deployment capability |
| Video approach | Pre-rendered playback | Real-time rendering | Performance compromise under current hardware resource constraints |

### Key Findings

- **Multi-Agent vs. Single Model**: Decomposing complex workflows into role-driven specialized agents significantly improves performance, debuggability, and adaptability compared to monolithic designs. Each agent uses customized prompts, context management, and communication protocols, focusing optimization on its specific sub-task while reducing prompt interference and persona confusion.
- **RAG Retrieval Count Sensitivity**: $k=2$ is the optimal retrieval count for the current architecture. Increasing $k$ actually degrades response quality, as excessive retrieved passages occupy the final LLM's context window space, displacing the effective attention weight available for persona instructions and visualization descriptions.
- **Effective Utilization of Small Models**: With clear constraints (e.g., GBNF grammar) and carefully designed few-shot examples, a 3B-scale model is sufficient to reliably perform classification decisions without requiring large model deployment for every sub-task.
- **Irreplaceability of Embodied Interaction**: Iterative experience from early VR versions to the current nautilus shell interface demonstrates that the ritual weight and emotional significance created by physical gestures (such as whispering) are difficult to replicate with purely digital interfaces.
- **GPU Resource Competition Between LLM Inference and Rendering**: This is the core hardware bottleneck of the current system. LLM inference and Unity real-time rendering share the same GPU, creating resource contention that limits the model scale and rendering complexity that can operate concurrently.

## Highlights & Insights

- **RAG as "Poetic Grounding" — A Novel Application**: Extending RAG from conventional fact retrieval to the dynamic shaping of persona and register is an elegant design concept. By retrieving the Harrisons' eco-aesthetic archive to endow the AI with a specific narrative style—rather than hard-coding these requirements into the system prompt—the system achieves "style updatability without retraining." This idea is transferable to any scenario requiring an AI to inhabit a specific knowledge-grounded role: for instance, having an AI narrate historical events in the register of a particular historian requires only replacing the RAG knowledge base.

- **GBNF Grammar Constraints + Small Models — An Efficient Decision Paradigm**: When classification decisions must be made from a finite set of options, applying grammar constraints to force the model to output tokens matching only available options—combined with few-shot prompting—enables a 3B model to achieve sufficiently reliable performance. This "constraints + small model" combination outperforms "large model + free generation + post-processing" in both latency and reliability, and is particularly suited for decision-making steps embedded in real-time systems.

- **Placing Visualization Selection Before Response Generation**: Positioning the visualization decision as the first step in the LLM pipeline rather than the last allows the final Responder Agent to organize its language around the already-selected visualization content. This "decide what to show, then decide what to say" information flow ensures semantic consistency across multimodal outputs, preventing misalignment between spoken content and visual display.

- **The Complete Metaphorical Chain from Nautilus to Ocean Narrative**: The physical interface design is not merely functional (detecting distance, recording speech) but narrative—the nautilus as a symbol of a "living fossil," blue light ripples simulating the ocean's response, and the ritual quality of whispering together constitute a complete metaphorical chain that places users within the narrative frame of "conversing with the ocean" from the very first moment of contact. This "interface as narrative" design philosophy is generalizable to other AI interaction systems: the choice of physical interaction form affects not only usability but also the user's cognitive framing of and emotional attitude toward the system.

## Limitations & Future Work

- **Absence of Quantitative User Studies**: This is the paper's most significant weakness. System performance relies on qualitative exhibition observations and retrospective design decisions, with no controlled experiments quantifying knowledge retention, changes in environmental empathy, or behavioral intention shifts. The authors themselves acknowledge the need for such research. For a system aiming to "transform public environmental awareness," claims of impact remain speculative without supporting data.

- **Overly Simple RAG Retrieval Strategy**: The current approach—sentence-level ANN search, a fixed $k=2$, and single-pass query rewriting—is practically sufficient from an engineering standpoint but has clear upper limits in retrieval diversity and precision. Improvement directions mentioned by the authors—such as query expansion (generating multiple semantically distinct rewrites), document type annotation and classification (distinguishing artistic text, scientific data, and activist narrative), and "Read the Doc before Rewriting" (R&R)—have not yet been implemented. On complex ambiguous queries or cross-domain questions, the current RAG may retrieve irrelevant passages.

- **Response Consistency Cannot Be Guaranteed**: The authors explicitly acknowledge that while zero-shot prompting performs well in most cases, it cannot guarantee consistent response characteristics such as length and format. Over extended exhibition periods, this may result in uneven experiential quality for some users, and no automated response quality evaluation or feedback mechanism is in place.

- **GPU Resource Competition as a Hard Bottleneck**: LLM inference and Unity real-time rendering share the GPU, limiting the usable model scale and rendering complexity. Although the authors suggest offloading inference to the cloud in future work, this conflicts with the "mobile local deployment for public exhibitions" design goal, as such venues may not have stable network connectivity.

- **Pre-Rendered Rather Than Real-Time Video**: All environmental videos (phytoplankton blooms, ocean acidification, etc.) are currently pre-rendered, precluding user interaction with video content. This limits the system's immersive depth and users' autonomous exploration space. Real-time rendering is a planned direction but requires resolving the fundamental GPU resource allocation problem.

- **Lack of Multilingual Support and Autonomous Knowledge Base Updates**: The system operates entirely in English and cannot be adapted for exhibition deployments in non-English-speaking contexts. The knowledge base also lacks autonomous update capability—new scientific findings or ecological events must be manually embedded as new documents before the system can reference them.

- **Inherent Multi-Agent System Challenges Remain Unresolved**: The authors cite open challenges in the multi-agent LLM literature—ensuring clarity of role definitions, maintaining inter-agent alignment, and validating outputs—but propose no concrete solutions for the current system. As system complexity grows, these challenges may amplify.

## Related Work & Insights

- **vs. ClimateQA / NASA EarthData Interfaces**: These systems similarly use RAG to allow users to access environmental data through natural language, but their purpose is "answering factual questions" via conventional text input/output. Sensorium Arc's distinctiveness lies in not only retrieving facts but also using RAG to shape the "oceanic persona's" narrative style, transforming scientific Q&A into poetic dialogue. However, ClimateQA may be more reliable in scientific accuracy and verifiability; Sensorium's balance mechanism between poetic expression and factual accuracy is not sufficiently transparent.

- **vs. ChatCam / USER-LLM**: ChatCam uses natural language to orchestrate visual workflows; USER-LLM achieves user-perception-aware personalization through RAG and CoT. Sensorium Arc resembles these systems in its direction of "LLM as creative co-creator," but adds a physical interaction dimension (the nautilus shell interface) and depth of ecological narration. Its multi-agent design is also more extensible than these single-model systems, at the cost of increased system complexity and latency.

- **vs. Rising Together / Sea Level Rise Explorer**: These are representative tools for immersive climate engagement, focused on helping coastal communities understand sea-level rise. They provide quantitative user study evidence (enhanced empathy, knowledge retention), which Sensorium Arc currently lacks. Sensorium's advantage lies in stronger interactivity (conversational rather than presentational), but it lags in demonstrable impact.

- **vs. General Multi-Agent LLM Systems**: Sensorium Arc's multi-agent design follows the recent paradigm of "decomposing complex workflows into role-driven specialized agents," but its application context (art installation rather than production system) means that its requirements for latency, reliability, and scalability differ from those of typical multi-agent systems. The system provides valuable engineering reference for deploying multi-agent architectures on resource-constrained edge devices.

- This paper represents a leading example of the emerging interdisciplinary field of "AI + eco-art." Its core contribution lies not in the depth of innovation of any single technology, but in demonstrating how existing technologies (multi-agent LLM, RAG, TTS, Unity rendering, sensor interaction) can be creatively integrated into a meaningful experiential system. Its proposition that "data is not meant to be observed but conversed with" carries inspirational value for environmental education and science communication.

- **Broader Implications for LLM Applications**: The Sensorium Arc case demonstrates that in AI-driven creative systems, "what to retrieve" and "how to retrieve" are equally important—selecting the Harrisons' eco-aesthetic texts as the knowledge base rather than a general environmental encyclopedia is itself an artistic decision. This perspective of treating knowledge base selection as a creative design choice rather than a purely technical decision has important implications for future AI-assisted creative workflows. The system also demonstrates the feasibility and engineering challenges of deploying multi-agent LLMs in real-time interactive contexts, providing a reusable technical blueprint for analogous settings such as museum exhibitions, science centers, and educational installations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Using RAG as "poetic grounding" and personifying the ocean as a conversational agent is original within the AI × eco-art intersection, though the individual technical components are not novel in themselves.
- **Experimental Thoroughness**: ⭐⭐ — Entirely lacks quantitative experiments, user studies, and baseline comparisons; only technical descriptions and qualitative observations are provided.
- **Writing Quality**: ⭐⭐⭐⭐ — Intellectually rich with substantive depth in eco-philosophical exposition, though the paper lacks the experimental rigor expected of a technical paper.
- **Value**: ⭐⭐⭐⭐ — Proposes a valuable paradigm for AI-enabled environmental education and science communication; the multi-agent integration approach offers useful engineering reference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era](from_generation_to_attribution_music_ai_agent_architectures_for_the_post-streami.md)
- [\[NeurIPS 2025\] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)
- [\[NeurIPS 2025\] The Impact of Scaling Training Data on Adversarial Robustness](the_impact_of_scaling_training_data_on_adversarial_robustness.md)
- [\[AAAI 2026\] Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](../../AAAI2026/audio_speech/thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)
- [\[NeurIPS 2025\] LUMIA: A Handheld Vision-to-Music System for Real-Time, Embodied Composition](lumia_a_handheld_vision-to-music_system_for_real-time_embodied_composition.md)

</div>

<!-- RELATED:END -->
