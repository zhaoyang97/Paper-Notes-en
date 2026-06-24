---
title: >-
  [Paper Note] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art
description: >-
  [ACL 2025][Multimodal VLM][Video Comment Art] GODBench presents the first benchmark designed to systematically evaluate the video bullet-screen/comment-art generation capabilities of Multimodal Large Language Models (MLLMs), defining 5 creative dimensions and 25 subcategories. It introduces Ripple of Thought (RoT), a multi-step reasoning framework inspired by physical wave propagation, to enhance models' creative generation performance.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Video Comment Art"
  - "Multimodal Large Language Models"
  - "Creative Evaluation Benchmark"
  - "Ripple of Thought"
  - "Video Understanding"
date: 2026-05-08
content_hash: ae63e1aa62a2d34a
---

# GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art

**Conference**: ACL 2025  
**arXiv**: [2505.11436](https://arxiv.org/abs/2505.11436)  
**Code**: [https://github.com/stan-lei/GODBench-ACL2025](https://github.com/stan-lei/GODBench-ACL2025)  
**Area**: Multimodal VLM / Creative Generation  
**Keywords**: Video Comment Art, Multimodal Large Language Models, Creative Evaluation Benchmark, Ripple of Thought, Video Understanding

## TL;DR

GODBench presents the first benchmark designed to systematically evaluate the video bullet-screen/comment-art generation capabilities of Multimodal Large Language Models (MLLMs), defining 5 creative dimensions and 25 subcategories. It introduces Ripple of Thought (RoT), a multi-step reasoning framework inspired by physical wave propagation, to enhance models' creative generation performance.

## Background & Motivation

**Background**: Video bullet-screen/comment art is a highly vibrant form of creative expression in internet culture. High-quality video comments require a deep understanding of video context as well as higher-order creative abilities such as humor, sarcasm, and empathy. Despite the rapid progress of Multimodal Large Language Models (such as GPT-4o and Gemini) demonstrating powerful reasoning in STEM tasks, their performance in the domain of creative expression has not been systematically evaluated.

**Limitations of Prior Work**: Existing creative evaluation benchmarks suffer from two core limitations: (1) Restricted Modalities—most are confined to text-only or image-text data, lacking benchmarks for multimodal video-text creativity; (2) Insufficient Categories—previous benchmarks usually encompass only a small scope of creative categories (e.g., joke generation), failing to systematically evaluate the holistic capability of "comprehending video contexts and formulating creative responses."

**Key Challenge**: Creative generation demands deep cultural understanding, situational awareness, and "unexpectedness" (highly surprising yet logical associations), which fundamentally differs from the structured, pattern-based reasoning MLLMs excel at. While existing Chain-of-Thought (CoT) methods boost logical reasoning, they provide limited aid to creative divergence—overly linear reasoning chains can even suppress creative cognitive leaps.

**Goal**: (1) Construct a video comment benchmark covering multi-dimensional creative capabilities; (2) Design a reasoning framework tailored for creative generation to replace linear CoT.

**Key Insight**: The authors draw inspiration from the propagation pattern of physical ripples—when a stone is dropped into water, ripples diffuse outward layer by layer, with each layer holding the potential to spark new expansions. Analogously, in creative thinking, an initial trigger point can trigger multi-layered, multi-directional associative diffusions.

**Core Idea**: Propose the Ripple of Thought (RoT) framework, representing creative thinking in five successive stages: "Trigger → Propagation → Resonance → Superposition → Convergence". This guides MLLMs to expand associations incrementally like ripples and eventually produce highly creative video comments.

## Method

### Overall Architecture

GODBench delivers two core contributions: (1) A Benchmark Dataset—comprising video-comment pairs classified into 5 creative dimensions (Resonant Thinking, Divergent Association, Witty Twist, Imaginary Completion, Emotional Resonance), with 5 subcategories under each dimension (totaling 25 categories), covering both discriminative (selection, ranking, classification, explanation) and generative (creative writing) task modalities; (2) The RoT Reasoning Framework—replacing standard CoT to guide MLLM creative reasoning through five propagation stages.

### Key Designs

1. **Five-Dimensional Creative Classification System**:

    - Function: Systematically define and cover diverse creative capacities in video comment generation.
    - Mechanism: Based on analyzing vast video comments and expert annotations, comment creativity is classified into five dimensions:
        - **Resonant Thinking (RT)**: Comments evoking deep emotional resonance in viewers.
        - **Divergent Association (DA)**: Ingeniously associating video content with seemingly unrelated concepts.
        - **Witty Twist (WT)**: Utilizing unexpected turns to craft humorous effects.
        - **Imaginary Completion (IV)** (Original notation kept): Creatively filling in elements not shown in the video.
        - **Emotional Resonance (ER)**: Capturing and expressing the precise emotional undertone of the video.
    - Design Motivation: A single metric cannot comprehensively evaluate creative capacity. The five-dimensional classification guarantees evaluation systematicity and completeness, while helping researchers identify specific dimensions where models are stronger or weaker.

2. **Multi-Task Evaluation Design**:

    - Function: Comprehensively assess MLLMs' creative capacities from both discriminative and generative standpoints.
    - Mechanism: Design five task styles—selection (SEL, choosing the best candidate comment), ranking (RNK, ordering comments by creativity), classification (CLS, identifying the comment's creative dimension), explanation (EXP, clarifying why a comment is creative), and creation (CRE, generating comments directly based on the video). The first four are discriminative tasks evaluated using Exact Match Accuracy; the last is a generative task assessed via GPT-4o scoring and human pairwise voting.
    - Design Motivation: Creative proficiency is multi-layered—"recognizing a good comment" and "writing a good comment" require distinct cognitive layers. The multi-task design allows for a more precise diagnostic of the gap between a model's creative understanding vs. creative generation.

3. **Ripple of Thought (RoT) Reasoning Framework**:

    - Function: Replace traditional CoT and maximize the creative reasoning capabilities of MLLMs.
    - Mechanism: Inspired by physical wave propagation, RoT models creative reasoning into five stages:
        - **Trigger**: Identifying core information and emotional hooks in the video.
        - **Propagation**: Expanding outward from the trigger point to yield multiple association chains.
        - **Resonance**: Detecting mutually reinforcing resonance points within the distinct association chains.
        - **Superposition**: Overlapping and integrating multiple resonance points to form complex, multi-layered meanings.
        - **Convergence**: Distilling the superimposed and rich thoughts down to a single, concise creative comment.
    - Design Motivation: The linear reasoning of standard CoT fits logical deductions but hinders creative expansion. RoT's "diverge-then-converge" strategy mimics actual human creative workflows—expanding associations broadly before focusing and refining. The explicit decoupling of these five stages enables the model to focus on unique cognitive operations at each step.

### Loss & Training

The RoT framework is primarily executed through carefully engineered prompt strategies and requires no training. For specific experiments, the authors fine-tune several open-source MLLMs on the GODBench training partition using LoRA to gauge adaptation outcomes.

## Key Experimental Results

### Main Results

Evaluating discriminative tasks using Exact Match Accuracy (EMA) across multiple MLLMs:

| Model | Size | SEL | RNK | CLS | EXP | Average |
|------|------|-----|-----|-----|-----|------|
| GPT-4o | - | Top Tier | Top Tier | Top Tier | Top Tier | Best |
| Gemini 1.5 Pro | - | High | High | Med-High | High | Second Tier |
| InternVL2 | 26B | Med | Med | Med | Med | Moderate |
| VideoLLaMA2 | 7B | Low | Low | Low | Low | Poor |
| LLaVA-Video | 7B | Low | Low | Low | Low | Poor |

On generative tasks (comment creation), comparing RoT with standard CoT and direct generation:

| Method | GPT-4o Score | Human Pairwise Preference | Description |
|------|-----------|-------------|------|
| Direct (Direct Gen.) | Baseline | Baseline | No reasoning guidance |
| Standard CoT | Marginal Gain | Slight Preference | Linear reasoning offers limited support for creativity |
| RoT (Ours) | Significant Gain | Strong Preference | Diverge-then-converge strategy effectively sparks creativity |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| RoT (Full 5 Stages) | Best | All five stages make necessary contributions |
| w/o Resonance | Decline | The Resonance stage helps screen out low-quality associations |
| w/o Propagation | Significant Decline | Flaring out associations is the essence of creative generation |
| w/o Convergence | Quality Decline | Lack of convergence yields overly loose and disjointed comments |
| Fewer Propagation Branches | Decline | More association branches -> higher creative diversity |

### Key Findings

- **MLLMs fall significantly short in creative understanding/generation compared to STEM reasoning**: Even GPT-4o struggles on specific creative dimensions, notably in Witty Twist and Divergent Association which demand deep vernacular and cultural insights.
- **Closed-source models far exceed open-source equivalents**: GPT-4o and Gemini perform substantially better on creative tasks than 7B-26B open-source models, indicating creative ability is deeply bound to model capacity and data quality.
- **RoT offers authentic help for creative generation**: Compared with standard CoT, RoT-generated comments receive significantly higher preferences in human evaluations, primarily driven by the broad multi-directional associations established in the Propagation stage.
- **LoRA fine-tuning provides benefits on discriminative tasks but yields limited help on generative tasks**: This suggests creative generation is highly demanding and hard to acquire simply through small-scale parameters adjustment, possibly requiring larger baseline capability upgrades.
- **Models perform fairly well on Imaginary Completion but worst on Witty Twist**: The former depends heavily on narrative completion (where MLLMs excel), whereas the latter demands true humor and surprise elements.

## Highlights & Insights

- The **five-dimensional creative classification system** is designed systematically, offering both theoretical structure and robust practical operability. This classification framework can be readily generalized to other creative appraisal scenarios (e.g., ad copywriting, social media content generation).
- The **"physical wave" analogy** for RoT is intuitive—equating creative thoughts to ripple dispersion: first expanding, then contracting. This paradigm can pivot to other tasks needing divergent-convergent schemas, such as brainstorming, storytelling, or product ideation.
- Featuring a **69-page paper alongside 66 figures**, this represents a huge workload, offering exceptionally solid dataset construction and human evaluation practices.

## Limitations & Future Work

- **High Cultural Dependency**: Bullet-screen/comment cultures are deeply rooted in specific online platforms (e.g., Bilibili, YouTube); hence, safety, humor norms, and rating criteria might lack universal cross-cultural validity.
- **Evaluations Still Rely on Humans or GPT-4o**: Assessing creativity is inherently subjective, and the validity and consistency of automated indicators require further validation.
- **RoT Increases Reasoning Overheads and Token Usage**: The five-step reasoning structure demands long context prompts and multi-turn inferences, meaning operational efficiency must be solved for actual deployment.
- **Dataset Dominated by Chinese Videos (Presumed)**: This might hinder the fair and direct appraisal of non-Chinese MLLMs.
- **Future Directions**: Converting RoT's structural steps into trainable layers rather than pure prompting pathways, or exploring RLHF with human creative commentary datasets to directly reinforce model creative expression.

## Related Work & Insights

- **vs CreativeBench / HumorBench**: Prior creative benchmarks worked mostly with pure text or single images. GODBench is the first to introduce the video modality and provides more complex, multi-dimensional creative criteria.
- **vs Chain-of-Thought (CoT)**: Standard CoT leverages a linear deductive model suitable for logic problems but constrains creative expansion; RoT's "diverge-then-converge" blueprint is intentionally engineered for creative exploration over linear logic.
- **vs Tree-of-Thought (ToT)**: While ToT utilizes branch exploration, it is intended to search for a singular optimal path rather than prompt creative diversity. The Resonance and Superposition stages of RoT are completely absent in ToT.

## Rating

- Novelty: ⭐⭐⭐⭐ First video-based comment creativity benchmark; the RoT framework is highly creative, and the five-dimensional classification is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Very comprehensive, featuring diverse models, both discriminative and generative metrics, ablations, and human evaluations.
- Writing Quality: ⭐⭐⭐⭐ The 69-page text is extremely thorough with vast classification charts and cases, though its immense length might lower readability.
- Value: ⭐⭐⭐⭐ Fills a crucial void in video creative comprehension and generation benchmarks; the RoT concept inspires prompt design for creative tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ACL 2025\] COSMMIC: Comment-Sensitive Multimodal Multilingual Indian Corpus](cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md)
- [\[ACL 2025\] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](speaking_beyond_language.md)
- [\[ACL 2025\] Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences?](burn_after_reading_do_multimodal_large_language_models_truly_capture_order_of_ev.md)
- [\[ACL 2025\] COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models](coling-unia_at_scivqa_2025_few-shot_example_retrieval_and_confidence-informed_en.md)

</div>

<!-- RELATED:END -->
