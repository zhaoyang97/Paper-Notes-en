---
title: >-
  [Paper Note] Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety
description: >-
  [ACL 2026][Text Generation][Children's Reading] The authors generated 2,580 stories based on the UFLI K–2 English reading curriculum using GPT-4o / Llama-3.3-70B. They evaluated four SFT designs (baseline, Good Stories…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Children's Reading"
  - "Controllable Difficulty"
  - "Compact LLMs"
  - "Rewarded SFT"
  - "QLoRA"
date: 2026-05-08
content_hash: d1f134925ae25ca8
---

# Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety

**Conference**: ACL 2026  
**arXiv**: [2605.13709](https://arxiv.org/abs/2605.13709)  
**Code**: None (Data generated based on UFLI K–2 curriculum + GPT-4o/Llama-3.3)  
**Area**: Text Generation / Education  
**Keywords**: Children's Reading, Controllable Difficulty, Compact LLMs, Rewarded SFT, QLoRA

## TL;DR
The authors generated 2,580 stories based on the UFLI K–2 English reading curriculum using GPT-4o / Llama-3.3-70B. They evaluated four SFT designs (baseline, Good Stories, Rewarded SFT, and simulated children's mispronunciations) across three 8B models (Llama 3, Granite 3.3, and Apertus). The results demonstrate that **compact models with appropriate SFT strategies** can **outperform zero-shot GPT-4o and Llama-3.3-70B** on key K-2 metrics such as Spache readability, syntactic complexity, and toxicity. Among these, Rewarded SFT proved the most stable and nearly hallucination-free.

## Background & Motivation

**Background**: Generative AI has seen a surge in interest for producing children's educational content, with systems like AIStory, StoryPrompt, and Storiza. A prior study by the same team (Leite et al. 2025) used zero-shot prompting with GPT-4o / Llama-3.3-70B to generate 2,580 stories following the UFLI K–2 curriculum.

**Limitations of Prior Work**: (1) Even high-end closed-source models like GPT-4o struggle to strictly adhere to K-2 curriculum constraints, often exceeding phoneme ranges or using vocabulary and syntax beyond second-grade levels, which undermines readability goals. (2) High costs of closed-source APIs and the 80GB+ VRAM required for local 70B deployments are impractical for most classrooms and homes. (3) While compact models (<10B) are affordable, they are prone to "mode collapse or logical fragmentation" under multi-dimensional educational constraints (strict vocabulary + simplified syntax + safety guardrails); standard SFT is insufficient.

**Key Challenge**: Compact models (sub-10B) face a severe "creativity vs. strict rule-following" trade-off. Tightening constraints leads to repetitive, formulaic stories, while maintaining creativity results in constraint violations. The authors term this the "controllability gap."

**Goal**: (RQ1) Which SFT strategy best enables sub-10B models to generate children's stories that satisfy both K-2 readability and narrative coherence? (RQ2) Can these fine-tuned compact models achieve content safety levels comparable to zero-shot 70B models?

**Key Insight**: Shift "educational constraints" from the prompt level into model parameters through SFT to internalize curriculum structures. This study systematically compares three enhancements: (a) training on a quality-filtered "Good Stories" subset; (b) Rewarded SFT (embedding multi-metric rewards into loss weights); and (c) input-side augmentation injecting simulated children's reading errors.

**Core Idea**: Compact models do not necessarily need more parameters; they require "reward-aware data + pedagogical domain internalization." This simplifies RL into "weighted SFT" to bypass the lack of sample volume typically required for RLHF.

## Method

### Overall Architecture
Infrastructure: Three 8B models (Llama-3-8B / Apertus-8B-instruct-2509 / Granite-3.3-8B-instruct); QLoRA (NF4 4-bit + LoRA r=32, $\alpha=64$, dropout=0.1, lr=1e-4, batch=4, epochs=5). Training data: 129 lessons from the UFLI K–2 curriculum × 20 stories (10 GPT-4o + 10 Llama-3.3) = 2,580 stories. Evaluation: 5 metrics (Spache Readability / GPT-2 LM-PPL / Coherence via shared NER / Syntactic Complexity = avg MDD + avg NSC / Detoxify toxicity) + two Self-BLEU redundancy metrics. Inference: nucleus sampling top-p=0.9, T=0.8. Each (experiment, model) combination generated 1,290 stories (129 lessons × 10). Metric analysis was performed after manually filtering non-story outputs.

### Key Designs

1.  **Rewarded SFT—Re-weighting SFT loss with multi-metric scalar rewards**:
    *   **Function**: Simplifies RL into weighted supervised learning when RLHF samples are scarce, allowing 8B models to perceive "what constitutes a good story" during training.
    *   **Mechanism**: For each training sample $i$, raw scores are computed for 5 metrics. For "lower-is-better" metrics (e.g., Spache, PPL), inverted min-max normalization is used: $\tilde{m}_i = \max(0, \min(1, (b_m - m_i)/b_m))$. For "higher-is-better" metrics, standard min-max is applied. A scalar reward $r_i = \frac{1}{5}\sum_k \tilde{m}_i^{(k)}$ is then derived from the **unweighted average** of the 5 normalized metrics. The cross-entropy loss of the SFT Trainer is re-weighted by $r_i$, forcing the model to learn more effectively from "high-reward stories." This aligns with reward-weighted regression (Peters 2006) and offline alignment (Mukherjee et al. 2025) but without training a separate reward model.
    *   **Design Motivation**: (a) The team had only 2,580 stories, insufficient for a stable RLHF reward model. (b) Compact models need clear signals on "which samples are worth learning" under strict constraints to avoid mode collapse. (c) Using pre-calculated automatic metrics as "cheap rewards" is computationally efficient as it only modifies loss weights without increasing inference cost.

2.  **Good Stories—Quality-filtered subset SFT**:
    *   **Function**: Evaluates the relative value of "more data vs. better data" for K-2 story tasks.
    *   **Mechanism**: Stories where **all 5 metrics were above the corpus mean** were retained, resulting in 996 stories (~38% of the data), followed by standard SFT on this subset.
    *   **Design Motivation**: Aligns with research like AlpaGasus / LIMA emphasizing "less is more" for quality data, while testing if simple filtering exceeds reward weighting.

3.  **SFT with simulated children's reading errors—Input augmentation with phoneme errors**:
    *   **Function**: Exposes the model during training to mispronunciations real early readers make, teaching the output to specifically reinforce target phonemes.
    *   **Mechanism**: GPT-OSS-120B was used with few-shot prompts of real children's mispronunciations to generate 3–8 "simulated misread phonemes" per story. During training, the input was concatenated as "Original Lesson Phonemes + Simulated Misread Phonemes." This follows the Self-Instruct and STaR bootstrapping logic.
    *   **Design Motivation**: Data on real children's reading errors is scarce and restricted by IRB, but explicitly teaching models to target specific phonetic difficulties has significant application value. Since the target output remains unchanged, it places nearly zero burden on lexical/syntax control.

## Key Experimental Results

### Main Results: Baseline vs. Rewarded SFT (Selected from Table 1)

| Metric | Baseline-Llama3 | Baseline-Granite | Baseline-Apertus | Rewarded-Llama3 | Rewarded-Granite | Rewarded-Apertus (Ours) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Coherence ↑ | 0.02 | 0.07 | 0.09 | 0.12 | **0.18** | 0.13 |
| Syntactic Complexity ↓ | 4.63 | 3.72 | 3.41 | 3.38 | 3.12 | **2.96** |
| Spache Readability ↓ | 4.05 | 3.52 | 2.83 | 2.71 | 2.56 | **2.34** |
| Toxicity ↓ | 0.01 | 0.06 | 0.00 | 0.01 | 0.02 | 0.02 |
| LM-PPL ↓ | 23.16 | 24.91 | 16.49 | 16.86 | **14.55** | 19.73 |
| Redundancy (Lesson) ↓ | 0.03 | 0.11 | 0.12 | 0.11 | 0.12 | 0.09 |

Comparison with zero-shot large models (Prev. SOTA in Table 2): Llama-3.3-70B original Spache=2.54, Syn=2.81, PPL=26.71; GPT-4o original Spache=3.31, Syn=3.94, PPL=28.08. **Ours** (Rewarded-Apertus: Spache=2.34, Syn=2.96, PPL=19.73) matched or outperformed Llama-3.3-70B in readability and syntax, and was **significantly superior to GPT-4o**.

### Ablation Study: Four SFT Strategies × Three Models (Table 2 Take-aways)

| Experimental Design | Best Model | Spache | Syn | PPL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o (zero-shot) | – | 3.31 | 3.94 | 28.08 | Closed-source baseline |
| Llama-3.3-70B (zero-shot) | – | 2.54 | 2.81 | 26.71 | Open-source baseline |
| **Baseline SFT** | Apertus | 2.83 | 3.41 | 16.49 | Standard SFT significantly lowers PPL |
| **Good Stories** | Apertus | 2.63 | 3.14 | 23.64 | Spache ↓, but PPL ↑ (due to less data) |
| **Rewarded SFT** | Apertus | **2.34** | **2.96** | 19.73 | **Near-optimal across metrics + Stable** |
| **SFT + Simulated Errors** | Apertus | 2.51 | 2.41 | 31.32 | Lowest Syn but high PPL (short stories) |

Welch's t-tests indicate the differences between Rewarded SFT and GPT-4o/Llama-3.3-70B in coherence, syntax, Spache, and PPL are **all $p < 0.001$ with Cohen's $d > 0.8$** (large effect size).

### Key Findings
- **8B models can outperform zero-shot 70B models at K-2 difficulty**: Rewarded-Apertus (Spache 2.34, PPL 19.73) beat both Llama-3.3-70B and GPT-4o, providing strong evidence for "small model + fine-tuning" in vertical educational scenarios.
- **Rewarded SFT is consistently superior and most stable**: Almost all 1,290 stories generated by Rewarded SFT were free of hallucinations, garbled text, or non-story content, whereas other designs had <10% unusable outputs.
- **Good Stories is not strictly better than Rewarded SFT**: Reduced data quantity slightly lowered Spache but increased PPL, showing "less is more" is **not absolute** for this task—reward shaping is more reliable than filtering.
- **Simulated errors reduce complexity but have side effects**: While Apertus achieved the lowest Syn (2.41) here, some stories became too short, leading to PPL spikes and occasional genre shifts.
- **Toxicity remains low (0–0.06)**: Out of 1,290 stories, only a few contained mild inappropriate language (e.g., "too fat"), but the mean toxicity remained very low.

## Highlights & Insights
- **"Rewarded SFT = Sample-weighted SFT" is a minimalist yet effective alignment paradigm**: It bypasses the complexity of full RLHF while providing a viable solution for teams with small datasets but cheap automatic metrics.
- **8B models + QLoRA + cheap reward shaping can surpass GPT-4o in K-2 content generation**: This supports the "cost-democratization" narrative for resource-constrained educational settings.
- **Warning regarding "Metric-Reward" overlap**: The authors acknowledge in Limitations that using the same metrics for both reward and evaluation risks "over-optimization," a candid self-critique rare in educational AI literature.

## Limitations & Future Work
- Compute constraints (3x L4 GPUs) limited the number of epochs and SFT hyperparameter exploration.
- Small sample size (2,580) prohibited training a proper RLHF reward model.
- Absence of human evaluation from children, parents, or teachers; results rely on NLP metrics which may diverge from pedagogical judgment.
- Limited to a single curriculum (UFLI).
- **Goal**: Future work aims to collect real user feedback for full RLHF, distill models to <8B for mobile deployment, and decouple rewards from metrics using expert scoring.

## Related Work & Insights
- **vs. Leite et al. 2025 (Storiza)**: Directly follows up by showing 8B SFT can beat 70B zero-shot.
- **vs. AlpaGasus / LIMA**: Challenges the "less is more" approach for this specific task, finding reward shaping more effective than simple data filtering.
- **Insight**: Any vertical application should attempt cheap reward-weighted SFT before committing to expensive RLHF. Controllability, not parameter count, is the real bottleneck in educational content generation.

## Rating
- **Novelty**: ⭐⭐⭐ (Individual methods are known, but the combination for phoneme-controlled story generation is a novel systematic comparison.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (3 models × 4 strategies × 7 metrics with statistical tests.)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear argumentation and honest assessment of limitations.)
- **Value**: ⭐⭐⭐⭐ (High practical value for democratizing cost-effective educational AI.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Difficulty-Controllable Cloze Question Distractor Generation](difficulty-controllable_cloze_question_distractor_generation.md)
- [\[ACL 2026\] Investigating the Representation of Backchannels and Fillers in Fine-tuned Language Models](investigating_the_representation_of_backchannels_and_fillers_in_fine-tuned_langu.md)
- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[ACL 2026\] XtraGPT: Context-Aware and Controllable Academic Paper Revision via Human-AI Collaboration](xtragpt_context-aware_and_controllable_academic_paper_revision_via_human-ai_coll.md)
- [\[ACL 2026\] In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis](in-depth_research_impact_summarization_through_fine-grained_temporal_citation_an.md)

</div>

<!-- RELATED:END -->
