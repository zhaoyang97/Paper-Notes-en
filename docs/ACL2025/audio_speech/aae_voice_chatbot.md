---
title: >-
  [Paper Note] Finding A Voice: Exploring the Potential of African American Dialect and Voice Generation for Chatbots
description: >-
  [ACL 2025][Audio & Speech][Speech Dialogue] This work presents a systematic study on integrating African American English (AAE) into chatbots across text and speech modalities. It reveals that while text-based AAE hurts the user experience, speech-based chatbots paired with an African American accent are favored by AAE speakers, highlighting the crucial role of modality choice in linguistic personalization.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Speech Dialogue"
  - "Dialect Generation"
  - "Chatbots"
  - "Personalization"
  - "African American Vernacular English (AAVE)"
date: 2026-05-08
content_hash: fb7a9f4c247b5aaf
---

# Finding A Voice: Exploring the Potential of African American Dialect and Voice Generation for Chatbots

**Conference**: ACL 2025  
**arXiv**: [2501.03441](https://arxiv.org/abs/2501.03441)  
**Code**: [https://github.com/emorynlp/AAVE-Chat](https://github.com/emorynlp/AAVE-Chat)  
**Area**: Audio & Speech / Dialogue Systems  
**Keywords**: Speech Dialogue, Dialect Generation, Chatbots, Personalization, African American Vernacular English (AAVE)

## TL;DR
This work presents a systematic study on integrating African American English (AAE) into chatbots across text and speech modalities. It reveals that while text-based AAE hurts the user experience, speech-based chatbots paired with an African American accent are favored by AAE speakers, highlighting the crucial role of modality choice in linguistic personalization.

## Background & Motivation

**Background**: Chatbot personalization is key to enhancing user trust and engagement. While visual similarity (avatar skin tone matching) has proven effective, and linguistic similarity (code-switching, multilingualism) has seen preliminary exploration, dialect-level personalization remains extremely under-explored.

**Limitations of Prior Work**:
   - Approximately 80% of African Americans use AAE in daily life, but existing chatbots are entirely based on Standard American English (SAE), leading to a lack of linguistic representation.
   - AAE has long been marginalized in NLP (e.g., Twitter UD parsing, ASR bias), where technical biases undermine community trust.
   - Existing AAE generation research is limited to tweet-style text, leaving multi-turn dialogue scenarios unexplored.
   - The impact of accents on user perception has been scarcely studied within native speaker communities.

**Key Challenge**: Intuitively, linguistic similarity should enhance rapport, but existing text-based AAE studies show mixed results. The interaction effects among dialect intensity, modality (text vs. speech), and accent remain unclear.

**Goal**: (1) Systematically evaluate LLMs' ability to generate AAE text of varying intensities. (2) Compare the impact of AAE on user experience across text and speech modalities. (3) Explore the optimal combination of African American accents and dialect intensities.

**Key Insight**: Decouple dialect expression from response generation (generating SAE responses first, then translating them into AAE). Control dialect intensity across three levels (Low/Medium/High) while introducing F5-TTS to generate African American-accented speech, followed by dual-channel evaluation of text and speech.

**Core Idea**: By decoupling dialect translation and response generation, this work systematically compares the impact of AAE on actual users in text vs. speech modalities. It finds that accents are more effective than dialects in enhancing personalization.

## Method

### Overall Architecture
SODA multi-turn dialogue dataset → SAE response generation → LLM dialect translation (SAE→AAE with three levels: Low/Med/High) → Text chatbot evaluation + F5-TTS African American accent synthesis → Speech chatbot evaluation → Likert scale evaluations by 12 (text) / 8 (speech) AAE speakers.

### Key Designs

1. **Decoupled Dialect Translation Strategy**:
    - Separation of response generation and dialect expression: standard responses are generated first using an LLM, and then translated into AAE using a separate prompt.
    - Translation function E(I, SAE, AAE) → O, with three prompt levels controlling the density of AAE features.
    - **Design Motivation**: Prevent the dialect from directly altering the response content (keeping semantics invariant while modifying only surface style), thereby eliminating the confounding factor of content bias.

2. **Automatic AAE Linguistic Feature Labeling System**:
    - Use Claude-Sonnet-3.5 to automatically identify and label AAE linguistic features in the generated text.
    - Coverage of over 30 AAE features: phonological (e.g., final consonant cluster reduction), morphological (e.g., habitual "be"), syntactic (e.g., multiple negation), and lexical (lexical items).
    - Test set: 90 AAE text segments, 136 feature labels → Claude achieved 91% accuracy.
    - **Design Motivation**: Quantitatively analyze the distribution of AAE features across different LLMs at various dialect intensities.

3. **African American Accented Speech Synthesis**:
    - Implement F5-TTS (Diffusion Transformer + ConvNeXt V2) for voice cloning.
    - Reference audio source: Real African American Vernacular English speakers from the CORAAL corpus.
    - Preprocessing: Number/symbol to text normalization → sentence segmentation using spaCy → sentence-by-sentence synthesis → concatenation + pauses.
    - **Design Motivation**: Independently control the two dimensions of dialect (text) and accent (speech) to analyze their respective contributions.

### Evaluation System

| Dimension | Metrics (15 in total) | Type |
|------|---------------|------|
| Text + Speech Common | Comprehension, Rapport, Non-offensiveness, Trustworthiness, Self-similarity, Communication Comfort, Persona Appropriateness, Interaction Preference | Attributes |
| Text-only | Dialect Expression, Faithfulness, Grammaticality, Persona Consistency | Ratings |
| Speech-only | Naturalness, Clarity, Voice Persona Consistency | Ratings |

## Key Experimental Results

### Experimental Scale

| Dimension | Quantity |
|------|------|
| Text Chatbot Configurations | 9 (3 LLMs × 3 dialect intensities) + 1 SAE baseline |
| Speech Chatbot Configurations | 4 (SAE/Low/Med/High × AA accent) + 1 SA baseline |
| Number of Dialogues | 100 (5 domains × 20 dialogues, 10 turns per dialogue) |
| Annotators (Text) | 12 AAE speakers |
| Annotators (Speech) | 8 AAE speakers |
| Evaluation Dimensions | 15 Likert scales |

### Text Chatbot: AAE Feature Distribution (Average Features per Turn)

| LLM | Dialect Intensity | Phonological Features | Morphological Features | Syntactic Features | Lexical Features |
|-----|---------|---------|---------|---------|---------|
| Claude | High | ~3.0 | ~1.2 | ~2.0 | ~0.4 |
| Claude | Low | ~0.8 | ~0.5 | ~0.8 | ~0.1 |
| Llama | High | >3.0 | ~1.0 | ~1.0 | ~0.1 |
| GPT-4o | High | ~2.5 | ~0.8 | ~1.5 | ~0.3 |

### Core Conclusions Comparison

| Metric | Text AAE vs. SAE Baseline | Speech AA Accent + SAE vs. SA Baseline |
|------|-------------------|--------------------------|
| Comprehension | ↓ Decrease | **↑ Gain** |
| Rapport | ↓ Decrease | **↑ Gain** |
| Trustworthiness | ↓ Significant Decrease | ≈ Comparable |
| Self-similarity | ≈ Comparable or ↓ | **↑ Gain** |
| Interaction Preference | ↓ Decrease | **↑ Gain** |
| Non-offensiveness | ≈ (Low/Med), ↓ (High) | ≈ Comparable |

### Key Findings
- **Complete failure of text AAE**: The SAE baseline outperforms AAE across almost all metrics at all dialect intensities, with High AAE performing particularly poorly.
- **Speech accent as the winning factor**: The combination of AA Accent + SAE dialect outperforms the baseline across all dimensions, representing the optimal configuration.
- **The root cause of High AAE issues**: Primarily due to the over-expression of phonological features (3+ spelling alterations per turn), which makes the text resemble caricature or mocking of AAE.
- **Claude is the most balanced in AAE generation**: It achieves the best syntactic expression and is the only LLM that maintains a good balance across Low/Med/High settings.
- **Modality as a key moderating variable**: The same dialect content is perceived negatively in text, but becomes positive in speech when paired with an appropriate accent—modality fundamentally alters the direction of language personalization effects.

## Highlights & Insights
- **First study to systematically evaluate AAE chatbots among actual AAE speakers**: Instead of asking general users "what do you think of AAE", this work invites daily AAE speakers to evaluate whether "the chatbot sounds like me".
- **The "Accent > Dialect" finding carries profound practical value**: When designing chatbots for dialects or community languages, prioritizing investment in speech synthesis rather than text style transfer may yield higher efficiency.
- **Decoupled design eliminates content confounding**: Dialect affects expression style without changing response content, presenting a significant methodological improvement for dialect research.

## Limitations & Future Work
- **Offline Evaluation**: Annotators evaluated dialogues as "spectators" rather than interacting directly, which may not fully capture emotional responses during real-time interaction.
- **Limited Annotator Demographics**: Participants were limited to college-student AAE speakers, failing to reflect diversity across ages, regions, and educational backgrounds.
- **SAE Bias in TTS Models**: F5-TTS was predominantly trained on SAE data, which may fail to perfectly reproduce the nuanced characteristics of AA accents.
- **Dynamic Dialect Adaptation Unexplored**: Since real AAE speakers dynamically adjust dialect intensity based on context, a fixed intensity may sound unnatural.

## Related Work & Insights
- **vs. Deas et al. (2023)**: Both works evaluate LLMs' AAE generation, but Deas et al. is limited to tweet text; this work extends the evaluation to multi-turn dialogues and speech modalities.
- **vs. Pinhanez et al. (2024)**: Pinhanez et al. trained an AA voice model but did not test it in chatbot scenarios; this work completes the full loop from speech generation to user evaluation.
- **vs. Obremski et al. (2022)**: Obremski et al. found that accent adaptation had a negative impact on cross-lingual conversational agents; this work finds that within-dialect accent adaptation is positive—the critical distinction lies in the distance between dialect vs. language.

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models](does_your_voice_assistant_remember_analyzing_conversational_context_recall_and_u.md)
- [\[ACL 2025\] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)
- [\[ACL 2025\] Distilling an End-to-End Voice Assistant Without Instruction Training Data](distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)
- [\[ACL 2025\] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis](tcsinger_2_customizable_multilingual_zero-shot_singing_voice_synthesis.md)
- [\[ACL 2025\] SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](speechiq_speechagentic_intelligence_quotient_across_cognitive.md)

</div>

<!-- RELATED:END -->
