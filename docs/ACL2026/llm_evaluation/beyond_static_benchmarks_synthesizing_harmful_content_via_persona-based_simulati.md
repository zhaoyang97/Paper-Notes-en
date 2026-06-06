---
title: >-
  [Paper Note] Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation
description: >-
  [ACL 2026][LLM Evaluation][Persona Simulation] The authors drive LLM agents with "two-dimensional personas" (intrinsic identity + extrinsic strategy) to role-play users writing harmful comments on real Reddit posts. This…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Persona Simulation"
  - "Harmful Content Synthesis"
  - "Safety Classifier Evaluation"
  - "Reddit Data"
  - "Diversity Metrics"
date: 2026-05-08
content_hash: a08898616f6d28af
---

# Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation

**Conference**: ACL 2026  
**arXiv**: [2604.17020](https://arxiv.org/abs/2604.17020)  
**Code**: https://github.com/huijelee/synthesizing_harmful_content (Available)  
**Area**: LLM Safety Evaluation / Harmful Content Detection  
**Keywords**: Persona Simulation, Harmful Content Synthesis, Safety Classifier Evaluation, Reddit Data, Diversity Metrics

## TL;DR
The authors drive LLM agents with "two-dimensional personas" (intrinsic identity + extrinsic strategy) to role-play users writing harmful comments on real Reddit posts. This synthesizes a harmful content evaluation set that is more challenging, diverse, and broader in coverage than traditional static benchmarks, reducing the accuracy of four mainstream safety classifiers to 13–31% (vs. 60–94% on static sets), exposing the fact that existing benchmarks have been "saturated."

## Background & Motivation

**Background**: Current toxic/hate speech/trolling detection systems (OpenAI Moderation, Perspective API, LlamaGuard) almost exclusively report performance on **static human-annotated benchmarks** such as Qian-Gab, CONAN, and ELF22. These benchmarks, either manually curated or crawled from platforms, have been the de facto standards for the past several years.

**Limitations of Prior Work**: The authors point out three specific flaws in static benchmarks—(1) **Poor scalability** of manual curation, failing to keep pace with LLM evolution; (2) **Insufficient topic/style diversity**, failing to cover emerging social issues or subtle expressions; (3) **Pre-training data contamination**, where models have already encountered these test samples during pre-training. Consequently, classifiers show inflated performance (90%+) on static benchmarks but fail in real-world scenarios.

**Key Challenge**: While existing work on "synthesizing harmful data" (ToxiGen, Toxicraft) addresses scalability, content generated purely via prompting tends to have **stereotyped styles and repetitive structures**, essentially failing to escape a few "template-based" malicious patterns. This prevents them from truly testing the blind spots of classifiers. The reason is that simple prompt control cannot inject the complexity of "real users"—real trolls possess stable identities/interests and switch attack strategies based on the context.

**Goal**: Synthesize a harmful content collection that is (a) highly harmful, (b) difficult to detect, and (c) possesses style/topic diversity approaching human-authored datasets for stress-testing existing safety classifiers.

**Key Insight**: Drawing from social psychological observations that real users exhibit **"constant identity + context-dependent behavior,"** the persona is decoupled into two orthogonal dimensions: "intrinsic" and "extrinsic." These are randomly paired to produce a large variety of agents with distinct styles.

**Core Idea**: **Feed "intrinsic identity + extrinsic strategy" two-dimensional personas to LLM agents and let them role-play users writing malicious comments in real Reddit posts**, thereby generating high-diversity, highly-concealed harmful content in a controllable manner.

## Method

### Overall Architecture
The input consists of real Reddit posts $x$ (including subreddit name, title, original post, and comments) crawled from Pushshift. The pipeline consists of two steps:

1.  **Persona Synthesis**: An LLM $\mathcal{M}_{in}$ (GPT-4o) generates an intrinsic persona $a_{in}$ based on seed posts + user types + subreddits of interest. Simultaneously, an extrinsic persona $a_{ex}=(h,d,e)$ is sampled from ELF-HP (6 trolling strategies) or CADD (4 abusive categories).
2.  **Agent Simulation**: The $(a_{in},a_{ex})$ pair is injected into a backbone LLM $\mathcal{M}$ to instantiate a harmful agent $A_H \leftarrow \mathcal{M}(a_{in},a_{ex})$. After reading the original post $x$, the agent produces a harmful comment $h=A_H(x)$ as the final evaluation sample.

The output is a collection of 3,000 synthesized harmful comments per model, which are then fed into 4 safety classifiers to measure accuracy.

### Key Designs

1.  **Two-dimensional Persona (Intrinsic + Extrinsic)**:
    *   **Function**: Decouples "who the user is" from "how the user attacks," generating diverse agents through combinatorial explosion.
    *   **Mechanism**: The intrinsic persona $a_{in}=\mathcal{M}_{in}(th,u,s_{top},s_{recent})$ includes "profile items" such as username, account age, bio, interest categories, frequently visited subreddits, knowledge background, and typical comment length. The extrinsic persona $a_{ex}=(h,d,e)$ includes strategy type $h$ (e.g., "Antipathy: subtly introduces provocative topics"), natural language description $d$, and example $e$. The paper randomly combines 30,472 subreddit names, 6 trolling strategies, and 3 user types (newcomer / regular / longtime), making every agent portrait unique.
    *   **Design Motivation**: A single dimension (only demographics or only strategies) results in content that is either "monotonous identity" or "monotonous attack patterns." By making the dimensions orthogonal, **the same trolling strategy can be expressed in completely different tones by a military enthusiast, a toy collector, or a teenage gamer**, which is the primary difference from single-dimension synthesis like ToxiGen or Shin et al. 2023.

2.  **Contextual Grounding based on Real Community Posts**:
    *   **Function**: "Embeds" synthesized harmful comments into real Reddit conversation contexts rather than generating them in a vacuum.
    *   **Mechanism**: Each $h$ is generated based on a specific $x$ (metadata, original post, existing comments). During generation, the agent "observes" the topic of the original post and uses the persona's interests to "improvise." This explains why, for the same K-Pop post in Table 6, a toy collector mocks "K-pop idol names are as flashy as their plastic surgeries," while a military enthusiast says "this naming is like North Korean hackers"—the same post yields vastly different and highly contextualized attacks.
    *   **Design Motivation**: Prompt-based synthesis is often "rootless," with concentrated attack topics decoupled from real conversations, making it easy for classifiers to recognize patterns. Contextual grounding allows attacks to be **hidden within plausible conversations**, targeting the blind spots of classifiers.

3.  **Multi-dimensional Evaluation Protocol (Harmfulness + Challenge Level + Diversity)**:
    *   **Function**: Provides a 3D evaluation suite to validate the "synthetic benchmark" itself.
    *   **Mechanism**: (a) **Harmfulness**: Double-blind judgment using GPT-4o + Claude-3.5 and manual labeling of 100 samples by 5 humans (Fleiss $\kappa$=0.70); (b) **Challenge**: Measuring accuracy of 4 classifiers under a strict threshold of 0.2 (lower indicates a harder benchmark); (c) **Diversity**: Using Sentence-BERT embeddings to calculate convex hull area + pairwise cosine distance; using Self-BLEU / TTR / Vocab Size for linguistic diversity; and Shannon entropy via GPT-4o classification for topic diversity.
    *   **Design Motivation**: Synthetic data is often criticized for "looking right but not actually being harmful/difficult/diverse." This protocol maps every concern to a quantitative metric, providing a paradigm transferable to other synthetic benchmark works.

### Loss & Training
**No models are trained** in this paper; it relies entirely on prompting and sampling. Backbone agents use Llama-3.1 70B / DeepSeek-Llama 70B / GPT-4o, with $temperature=0.7$, $top-p=1.0$, and $max\_tokens=1024$. Each agent model generates 3,000 harmful comments.

## Key Experimental Results

### Main Results: Classifier Detection Rates vs. 8 Static Benchmarks

| Benchmark / Setting | LlamaGuard-1 | LlamaGuard-2 | OpenAI Mod | Perspective | Average |
|---|---|---|---|---|---|
| Qian-Gab | 91.77 | 75.84 | 99.06 | 97.34 | **91.00** |
| CONAN | 98.47 | 86.65 | 95.29 | 96.97 | **94.35** |
| COVID-HATE | 58.56 | 34.83 | 87.89 | 96.40 | 69.42 |
| CADD | 50.25 | 43.82 | 68.41 | 90.19 | 63.17 |
| **Ours (CADD Strategy)** | **20.83** | **0.77** | **37.17** | **65.55** | **31.08** |
| ELF22 | 15.09 | 12.07 | 25.85 | 43.96 | 24.24 |
| ELF-HP | 21.60 | 13.94 | 30.63 | 48.57 | 28.69 |
| **Ours (trolling Strategy)** | **5.65** | **10.20** | **18.25** | **19.88** | **13.50** |

→ Lower detection rates indicate a more challenging benchmark. The synthesized set in this paper pulls the average accuracy of 4 classifiers down from 60-90+% to **13.5–31%**, with LlamaGuard-2 dropping as low as 0.77%.

### Ablation Study: Impact of Persona on Generation Diversity (Trolling Setting)

| Model | Persona | Self-BLEU ↓ | TTR ↑ | Vocab ↑ | Shannon Entropy ↑ |
|---|---|---|---|---|---|
| Llama-3.1 70B | w/o | 3.877 | 0.039 | 4,044 | 2.251 |
| Llama-3.1 70B | w/ | **1.699** | **0.051** | **6,776** | **2.699** |
| DeepSeek-Llama 70B | w/o | 1.750 | 0.065 | 4,394 | 2.251 |
| DeepSeek-Llama 70B | w/ | **1.208** | **0.076** | **6,890** | **2.765** |
| GPT-4o | w/o | 2.259 | 0.078 | 4,707 | 2.485 |
| GPT-4o | w/ | **1.522** | 0.066 | **6,902** | **2.766** |

→ Diversity metrics across all 4 categories improved consistently after adding personas, with vocabulary size increasing by ~50%. In the CADD setting, GPT-4o's improvement was particularly extreme—the baseline consisted almost entirely of "refusal templates" (vocab only 152), which recovered to 2,152 with personas.

### Key Findings
- **Both persona dimensions are indispensable**: Ablations using only intrinsic or only extrinsic features caused t-SNE embeddings to cluster; combined, they cover the entire semantic space (Fig 3).
- **LLM-judge harmfulness rate 90.40% → 96.80%**: Adding personas increased the proportion of content judged as harmful by both GPT-4o and Claude-3.5 by 6.4 pp, with DeepSeek and GPT-4o showing the most significant gains (+7~14 pp).
- **Hard-to-detect $\neq$ far from known harmful clusters**: t-SNE in Fig 1 shows many "missed" samples are adjacent to known harmful ones, suggesting classifier blind spots stem from **subtle shifts in intent/context** rather than out-of-distribution expressions.
- **Human Eval Fleiss $\kappa$=0.70**: Five annotators reached substantial agreement on harmfulness, with a majority-vote accuracy of 96%, indicating stable and reliable synthesis quality.

## Highlights & Insights
- **Cognitive modeling of "constant identity + context-dependent behavior" is clever**: Abstracting social psychological descriptions of real trolls into orthogonal two-dimensional personas is a rare "theory $\rightarrow$ engineering" mapping, offering more interpretability than simple prompt stacking.
- **Threshold=0.2 instead of 0.5**: The authors voluntarily tightened the detection threshold for classifiers, yet performance remained poor, indicating the problem is not "classifier conservatism" but that synthesized content enters genuine semantic blind spots.
- **Diversity $\neq$ Difficulty**: Table 3 shows that while ELF-HP's hull area is close to Ours (135.35 vs 151.99), its detection rate is 15+ pp higher, suggesting the additional "difficulty" of this work comes from contextual grounding rather than mere distributional spread—a decoupling worth noting.
- **"Persona $\times$ Thread" combination is transferable**: This paradigm can be directly applied to jailbreaking, bias evaluation, and toxic red-teaming by swapping the extrinsic persona library.

## Limitations & Future Work
- The authors admit: (1) The current persona library only covers 6 trolling + 4 abusive strategies; finer-grained harm types (gaslighting, implicit bias) are missing. (2) Conducted only on English Reddit; multi-lingual scenarios are not validated.
- Observation: Using GPT-4o/Claude as judges overlaps with the generation models (GPT-4o both generates and judges), potentially leading to **self-evaluation bias** and overestimating harmfulness. Switching to less common judges (e.g., fine-tuned small models) might change the numbers.
- Low classifier scores on "our synthetic set" **do not necessarily mean the classifiers are bad**—it might mean the benchmark over-simulates certain edge cases. Validation of detection rate improvements on real toxic traffic in online A/B tests would be more convincing.
- Future improvements: (a) Expand the extrinsic library into a full taxonomy (like OpenAI's 13 safety categories) for automated risk-slicing; (b) Add multi-turn simulation to cover escalation, gaslighting, and group harassment.

## Related Work & Insights
- **vs. ToxiGen (Hartvigsen et al., 2022)**: ToxiGen uses demonstration-based prompting with keywords; this work uses persona-driven agent simulation. ToxiGen focuses on coverage through templates, while this work uses identity+strategy variables to break patterns, resulting in higher diversity and detection difficulty at the cost of a more complex pipeline.
- **vs. Toxicraft (Hui et al., 2024b)**: They refine topics/contexts from seed samples; this work generates from real Reddit threads. Toxicraft is "seed+transformation," while this work is "character+context"—the latter's diversity stems from personality combinations rather than topic diffusion.
- **vs. ELF-HP (Lee et al., 2024)**: ELF-HP provides 6 trolling strategies, which this paper reuses as the "extrinsic persona library." This serves as a model for inserting existing taxonomies as modular resources into synthesis pipelines.
- **Insight**: This "two-dimensional persona $\times$ real context" paradigm can be transferred to (a) jailbreak prompt synthesis, (b) multi-modal deepfake evaluation, and (c) red-team stress testing of LLM agents by identifying two decoupled dimensions of "identity" and "context."

## Rating
- Novelty: ⭐⭐⭐⭐ The decoupling of two-dimensional personas has precedents in social simulation but is systematically applied here for harmful content synthesis and stress testing for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 classifiers $\times$ 8 static benchmarks $\times$ 3 backbones $\times$ 5-person human annotation + multi-dimensional diversity metrics.
- Writing Quality: ⭐⭐⭐⭐ The 3D evaluation framework is clear, and the case study (Table 6) is intuitive; the math section is light, and persona examples could be integrated better into the body.
- Value: ⭐⭐⭐⭐ Directly reveals that "existing safety benchmarks have been saturated" and provides a reusable stress-testing paradigm; significant impact for the safety research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation](beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)
- [\[ACL 2026\] SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks](spence_a_syntactic_probe_for_detecting_contamination_in_nl2sql_benchmarks.md)
- [\[ACL 2026\] BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks](benchmarker_an_education-inspired_toolkit_for_highlighting_flaws_in_multiple-cho.md)
- [\[ICLR 2026\] Same Content, Different Representations: A Controlled Study for Table QA](../../ICLR2026/llm_evaluation/same_content_different_representations_a_controlled_study_for_t.md)
- [\[ACL 2026\] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)

</div>

<!-- RELATED:END -->
