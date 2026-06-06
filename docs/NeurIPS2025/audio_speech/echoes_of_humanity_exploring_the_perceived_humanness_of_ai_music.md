---
title: >-
  [Paper Note] Echoes of Humanity: Exploring the Perceived Humanness of AI Music
description: >-
  [NeurIPS 2025][Audio & Speech][AI music perception] Through a randomized controlled crossover trial (RCCT) and mixed-methods content analysis…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "AI music perception"
  - "Turing test"
  - "randomized controlled crossover trial"
  - "mixed-methods content analysis"
  - "human perception"
date: 2026-05-08
content_hash: 217dbdca3fc8c1c3
---

# Echoes of Humanity: Exploring the Perceived Humanness of AI Music

**Conference**: NeurIPS 2025
**arXiv**: [2509.25601](https://arxiv.org/abs/2509.25601)  
**Code**: [GitHub](https://github.com/uai-ufmg/hp-study)  
**Area**: Audio & Speech
**Keywords**: AI music perception, Turing test, randomized controlled crossover trial, mixed-methods content analysis, human perception

## TL;DR
Through a randomized controlled crossover trial (RCCT) and mixed-methods content analysis, this paper systematically investigates listeners' ability to distinguish AI-generated music (AIM) from human-created music. Results show that listeners perform at chance level under random pairing (accuracy ≈ random guessing), but accuracy rises significantly to 66% under similar pairing. Vocal, sound, and technical cues are identified as the key factors enabling successful discrimination.

## Background & Motivation

1. **Industry disruption by AI music**: Text-to-music services such as Suno and Udio are reshaping music creation, production, and consumption. Fully AI-generated projects (e.g., The Velvet Sundown) have amassed millions of streams, and 10% of newly uploaded tracks on Deezer are AI-generated.
2. **Gap in perception research**: Existing studies predominantly use AI music perception experiments to evaluate generative model quality (higher confusion = better model), rather than systematically studying listeners themselves—when they can distinguish AIM, how they do so, and what cues they rely on.
3. **Dataset limitations**: Prior work largely employs researcher-generated symbolic music, lacking real-user audio data generated via commercial models, and offers no control over pairing similarity.

## Core Problem

Do human listeners rely on specific contextual cues (e.g., repetitive structure, synthetic vocals) to judge whether music is AI- or human-created? This study addresses the question along two dimensions:
- **When**: A causal RCCT analysis quantifies the effect of pairing similarity on discrimination ability.
- **How**: Mixed-methods content analysis reveals the perceptual cues listeners actually employ.

## Method

### Overall Architecture
A three-stage design: (1) construction of an in-the-wild dataset; (2) a randomized controlled crossover trial (RCCT); (3) mixed-methods content analysis of free-text feedback.

### Key Design 1: Dataset Construction
- **AIM source**: Reddit community r/SunoAI was scraped (July 2023–February 2025), yielding 33,626 posts, from which Suno links (4,059 tracks) and YouTube links (8,315 tracks) were downloaded. Meme songs were excluded, retaining authentic user-generated content not controlled by the researchers.
- **Human music source**: Independent artist tracks from the MTG-Jamendo dataset, collected in 2019—prior to the commercial rise of AIM (Suno launched in 2023)—thereby precluding AI contamination.
- **Genre control**: The Essentia classifier was used to assign genre labels to AIM tracks with a confidence threshold of 0.4 (far exceeding the 0.011 chance level for 87-class classification), retaining only the top quartile.

### Key Design 2: Pairing Strategy
- **Random Set**: One AIM track and one human track were randomly selected from each of five genres—pop, rock, hip-hop, electronic, and metal—and randomly paired per participant. Requirements: English lyrics or no lyrics; duration 1.5–4 minutes.
- **Similar Set**: CLAP embeddings were generated for AIM and human tracks within the same genre; cosine similarity was computed, and pairs exceeding 0.8 were retained (only 2.5% met this threshold). Ten most similar pairs satisfying duration and language constraints were manually selected. Final distribution: 3 electronic, 2 rock, and 1 each of classical/ambient/hip-hop/pop/metal.

### Key Design 3: Experimental Procedure
- Each participant evaluated 5 track pairs: the first 4 presented randomly (2 from the Random Set, 2 from the Similar Set); the 5th served as a gold-standard trap (Beethoven's Fifth Symphony vs. an obviously AI-generated track).
- Within each pair, track order was randomized; titles were not displayed; skipping or revising answers was not permitted.
- Participants selected one of five options: A is AI / B is AI / Neither is AI / Both are AI / Cannot tell.
- Optional free-text feedback was invited to explain each judgment.
- A post-experiment demographic questionnaire (optional) collected age, native language, music education, playing experience, and familiarity with AIM.

### Key Design 4: Participant Recruitment
- **Volunteer group**: Recruitment began within the Computer Science and Music departments at UFMG (Brazil) and expanded via social media and the university website.
- **Crowdsourced group**: 100 native English speakers recruited via Prolific, each compensated £2.
- The two groups provide diverse demographics and distinct motivations (intrinsic interest vs. extrinsic reward).

## Key Experimental Results

### Participant Filtering
653 total logins → 337 correctly identified the Beethoven trap → 308 with no prior familiarity with any of the first 4 pairs → 1,232 valid responses
- 290 participants completed the demographic survey: 73% Portuguese-speaking, 22% English-speaking; 50% had no playing experience; 67% had no formal music education; 34% were familiar with AIM.
- Mean age: 31 years (SD = 13); average listening time per pair: 2.98 minutes.

### RCCT Core Results

| Pairing Type | Accuracy | Comparison to Chance (0.5) |
|---|---|---|
| All pairs | 0.60 | — |
| Random Set | 0.53 | p > 0.05 (not significant) |
| Similar Set | **0.66** | **p < 10⁻⁹** (highly significant) |
| Similar Set – with lyrics | 0.75 | p < 10⁻⁶ |

- **Treatment effect**: Similar pairing improves accuracy by 13 percentage points over random pairing (0.66 − 0.53).
- Conclusions hold after excluding classical/ambient genres or restricting to the lyrics subset, ruling out genre and lyrics confounds.

### Key Factors from Mixed-Effects Logistic Regression

| Factor | Direction | Significance |
|---|---|---|
| Similar pairing | ↑ Positive | p = 0.10 (*) |
| Playing experience > 10 years | ↑ Positive | p = 0.0009 (***) |
| Familiarity with AIM | ↑ Positive | p = 0.00005 (***) |
| Age | ↓ Negative | p = 0.0009 (***) |
| Formal education 5–10 years | ↓ Negative | p = 0.009 (***) |

- Model McFadden R² = 0.44, indicating acceptable explanatory power.
- The negative effect of formal education disappears when playing experience is excluded, suggesting high collinearity between the two variables.

### Content Analysis Results
- 317 free-text responses were collected from 140 participants and coded by three annotators over two rounds.
- Seven major themes identified: vocals-related, sound-related, technical aspects, humanness-related, modifiers, genre, and lyrics-related.
- Top-7 high-frequency labels: vocals (369), lyrics (247), negative (231), artificial (224), generic (174), human (130), robotic (112).
- **Key finding**: Significant differences exist between correct and incorrect responses (χ² test); participants who correctly identified AIM more frequently cited sound, technical, and vocal cues.

## Highlights & Insights
1. **Causal inference design**: The first AIM perception study to employ an RCCT, enabling causal—rather than merely correlational—evidence that pairing similarity enhances discrimination ability.
2. **In-the-wild dataset**: The first use of commercially generated AIM from real users (Reddit r/SunoAI) rather than researcher-controlled data, avoiding self-generation bias.
3. **Mixed-methods analysis**: Quantitative RCCT combined with qualitative grounded-theory content coding provides a dual-perspective account of perceptual mechanisms.
4. **Counterintuitive finding**: Similar pairing is actually easier to discriminate—when AI and human tracks are stylistically close, subtle synthetic artifacts become more salient.

## Limitations & Future Work
1. **WEIRD bias**: Both tracks and participants skew toward Western, educated, industrialized, rich, and democratic (WEIRD) backgrounds; non-Western musical traditions are not represented.
2. **Single AIM model**: AIM sources are limited to Reddit r/SunoAI, excluding other commercial models such as Udio.
3. **30-second excerpt limitation**: The experimental platform played only excerpts, leaving perception of full-length tracks untested.
4. **Limited genre coverage**: The Similar Set covers only 10 pairs across specific genre combinations; performance differences across a broader range of genres remain underexplored.
5. **Temporal validity**: Rapid evolution of AIM models raises questions about the generalizability of these findings to future model versions.

## Related Work & Insights
- **vs. Sarmento et al. (ISMIR 2024)**: That study analyzes symbolic AIM in rock and progressive metal using researcher-generated data; the present study uses audio AIM generated by real users.
- **vs. Grötschla et al. (ICASSP 2025)**: That work focuses on user preference (finding a preference for AI), whereas this study focuses on perceptual discrimination ability and employs a distinct pairing strategy.
- **vs. Donahue / Hernandez**: Those studies use Turing test paradigms to evaluate model quality; the present work takes listeners—rather than models—as the primary object of study.
- **vs. Noll (1966)**: A classical precursor in the visual domain; the present study extends this framework to music and incorporates causal inference and mixed methods.

### Broader Implications
- Directly informs AI detection literacy: training users to attend to vocal quality and technical details (rather than melody or lyrics) can improve identification rates.
- The finding that similar pairing yields higher discrimination suggests an "uncanny valley" effect in AI music: the closer an AI track mimics human creation, the more conspicuous its subtle flaws become.
- The methodological approach (RCCT + mixed-methods content analysis) is transferable to perception research on AI-generated images, video, and text.
- Offers guidance for AIM model improvement: vocal synthesis and low-level technical details (e.g., audio compression artifacts) represent the current weakest links.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First study combining RCCT, in-the-wild data, and mixed methods for AIM perception research.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 653 participants, dual recruitment groups, multi-model comparisons, and content coding—though genre coverage remains limited.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear, methodologically rigorous, and well-illustrated.
- **Value**: ⭐⭐⭐⭐ Offers practical guidance for both AI music detection literacy and model improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Ethics Statements in AI Music Papers: The Effective and the Ineffective](ethics_statements_in_ai_music_papers_the_effective_and_the_ineffective.md)
- [\[NeurIPS 2025\] From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era](from_generation_to_attribution_music_ai_agent_architectures_for_the_post-streami.md)
- [\[NeurIPS 2025\] Accelerate Creation of Product Claims Using Generative AI](accelerate_creation_of_product_claims_using_generative_ai.md)
- [\[ICML 2026\] MusicDET: Zero-Shot AI-Generated Music Detection](../../ICML2026/audio_speech/musicdet_zero-shot_ai-generated_music_detection.md)
- [\[AAAI 2026\] Aligning Generative Music AI with Human Preferences: Methods and Challenges](../../AAAI2026/audio_speech/aligning_generative_music_ai_with_human_preferences_methods_and_challenges.md)

</div>

<!-- RELATED:END -->
