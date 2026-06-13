---
title: >-
  [Paper Note] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories
description: >-
  [ACL 2026][AIGC Detection][Temporal Flattening] By constructing a longitudinal writing dataset spanning 12 years, this paper discovers a "temporal flattening" phenomenon in LLM-generated text—while lexical diversity is h…
tags:
  - "ACL 2026"
  - "AIGC Detection"
  - "Temporal Flattening"
  - "LLM-generated Text Detection"
  - "Longitudinal Writing Analysis"
  - "Cognitive-Emotional Features"
  - "Synthetic Data"
date: 2026-05-08
content_hash: 7337d7b361603267
---

# Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.12097](https://arxiv.org/abs/2604.12097)  
**Code**: [GitHub](https://github.com/yjkim717/Cognitive-Emotional-Trajectories)  
**Area**: AIGC Detection  
**Keywords**: Temporal Flattening, LLM-generated Text Detection, Longitudinal Writing Analysis, Cognitive-Emotional Features, Synthetic Data

## TL;DR

By constructing a longitudinal writing dataset spanning 12 years, this paper discovers a "temporal flattening" phenomenon in LLM-generated text—while lexical diversity is high, temporal drift in semantic and cognitive-emotional dimensions is significantly lower than in humans. Distinguishing human and LLM texts solely based on temporal variation patterns achieves 94% accuracy.

## Background & Motivation

**Background**: Large Language Models (LLMs) are widely used for content generation, dialogue systems, and synthetic training data production. The current LLM deployment paradigm is stateless—each generation is a standalone response, retaining no historical memory.

**Limitations of Prior Work**: Human writing is inherently longitudinal; the style, cognitive state, and emotional expression of an author evolve naturally over time. However, existing LLM generation paradigms assume that independently sampled documents sufficiently approximate the distribution of human writing, an assumption that has never been systematically verified.

**Key Challenge**: LLMs perform excellently on static quality metrics (e.g., semantic retention, fluency) but may systematically lose the longitudinal structure inherent in human writing across the temporal dimension. This loss poses a potential risk for downstream applications requiring temporal consistency (e.g., synthetic training data, authorship attribution, mental health trajectory modeling).

**Goal**: This work answers two core questions: (1) Can LLMs reproduce human temporal structures over long time spans? (2) If discrepancies exist, which temporal dynamics of human writing are systematically lost or flattened?

**Key Insight**: Drawing from psycholinguistics and computational stylometry, the authors treat writing as a longitudinal process, quantifying temporal structure through two complementary metrics—drift and variance—across three representation spaces: semantic, lexical, and cognitive-emotional.

**Core Idea**: LLM-generated texts exhibit "temporal flattening"—showing high diversity in lexical space but significantly lower temporal drift and fluctuation in semantic and cognitive-emotional spaces compared to humans. This discrepancy persists under both instance-wise and history-augmented conditions.

## Method

### Overall Architecture

The study constructs a longitudinal dataset containing 412 human authors and 6,086 documents (2012–2024), covering academic abstracts, blogs, and news. For each human author, matching writing trajectories were generated using three LLMs: DeepSeek V1, GPT-4o mini, and Claude 3.5 Haiku, totaling 103,459 LLM documents. The temporal dynamic differences between humans and LLMs are quantified by calculating drift and variance metrics across three representation spaces (lexical, semantic, cognitive-emotional), followed by validation via statistical testing and classifiers.

### Key Designs

1. **Three-Dimensional Feature Trajectory Construction**:
    - **Function**: Captures the temporal evolution of writing from multiple complementary dimensions.
    - **Mechanism**: For each author and year, three representations are computed: Lexical (TF-IDF + SVD reduced to 10D), Semantic (SBERT 384D embeddings), and Cog-Emo features (20 interpretable psychological traits, including Big Five personality proxies, sentiment, readability, etc.). Annual feature vectors are concatenated to form a temporal trajectory $\mathcal{T}(e) = (\mathbf{x}_1^{(e)}, \ldots, \mathbf{x}_T^{(e)})$.
    - **Design Motivation**: Lexical features capture surface-level word usage changes, semantics capture deep meaning migration, and cognitive-emotional features capture psychological and stylistic evolution.

2. **Drift-Variance Dual Metric System**:
    - **Function**: Quantifies temporal evolution from both global geometric shifts and local fluctuations.
    - **Mechanism**: Drift measures the total displacement between adjacent yearly feature vectors via L2 distance: $\text{drift}_t^{(e)} = \|\mathbf{x}_{t+1}^{(e)} - \mathbf{x}_t^{(e)}\|_2$. Variance quantifies the irregularity of inter-annual fluctuations using the Coefficient of Variation (CV): $\mathrm{CV}(f) = \frac{\mathrm{std}(\Delta f_{1:T})}{\mathrm{mean}(\Delta f_{1:T})}$.
    - **Design Motivation**: Drift reflects "how far" the writing has moved in the representation space, while variance reflects "how irregularly" it has moved; together, they characterize temporal dynamics.

3. **Paired Statistical Testing and Predictive Probes**:
    - **Function**: Rigorously validates systematic differences in temporal dynamics between humans and LLMs.
    - **Mechanism**: A binomial test is performed for each human-LLM author pair to check if the rate of human evolution exceeding LLM evolution is significantly higher than 50%. A Random Forest classifier is trained using 20D Cog-Emo CV features, using GroupKFold cross-validation to prevent data leakage.
    - **Design Motivation**: Statistical tests ensure reliability, while the classifier verifies the predictability of the differences.

### Loss & Training

This is an analytical study and does not involve model training. The classifier uses 5-fold GroupKFold cross-validation (grouped by `author_id`), ensuring that trajectories from the same author do not appear in both training and testing sets. Benjamini-Hochberg FDR correction ($q < 0.05$) is applied for multiple comparisons.

## Key Experimental Results

### Main Results

| Representation Space | Metric | Human "Win" Rate Range | p-value | Meaning |
|----------------------|--------|------------------------|---------|---------|
| TF-IDF (Lexical)     | Drift  | 0.20-0.33             | p=1.0   | LLM lexical drift is larger |
| SBERT (Semantic)     | Drift  | 0.75-0.86             | p<0.0001| Human semantic drift far exceeds LLM |
| Cog-Emo              | Drift  | 0.76-0.99             | p<0.0001| Human Cog-Emo drift is significantly higher |

### Ablation Study

| Configuration | Accuracy | AUC | F1 | Description |
|---------------|----------|-----|----|-------------|
| Pooled (IW)   | 0.936    | 0.977 | 0.863 | Instance-wise condition |
| Pooled (Hist) | 0.933    | 0.977 | 0.856 | History-augmented condition |
| Balanced-Claude 3.5 (IW) | 0.97 | 1.00 | 0.97 | Claude is most distinguishable |
| Balanced-GPT-4o mini (IW) | 1.00 | 1.00 | 1.00 | GPT-4o mini is nearly perfectly distinguished |

### Key Findings

- LLMs exhibit "Asymmetric Optimization": Lexical diversity is high, but semantic and cognitive-emotional drift is low, suggesting LLMs optimize for surface-level variation but fail to reproduce deep evolution.
- GPT-4o mini shows the most severe temporal flattening: in 97% of author pairs, human Cog-Emo drift was larger, rising to 99% under historical conditions.
- Temporal flattening persists in both instance-wise and history-augmented conditions, indicating this is a systematic property of current deployment paradigms.
- Most discriminative features: Average sentence length (18.7%), Agreeableness (15.9%), and Neuroticism (9.3%).

## Highlights & Insights

- Proposes the new concept of "temporal flattening," revealing a systematic flaw in LLM-generated text along the temporal dimension.
- Achieves 94% classification accuracy and 0.98 AUC using only 20D Cog-Emo CV features, providing a novel approach for AIGC detection.
- Publicly releases a longitudinal dataset of 412 authors to facilitate future research.
- Discovers that history augmentation does not fix temporal flattening, suggesting the root cause lies in model architecture or training paradigms.

## Limitations & Future Work

- The dataset primarily covers English writing; cross-linguistic temporal flattening remains to be verified.
- Only three commercial LLMs were tested; whether open-source or fine-tuned models behave differently remains unexplored.
- The study focuses on identifying the phenomenon rather than proposing solutions; how to generate LLM text with realistic temporal structures remains an open problem.
- Cognitive-emotional features rely on tools like LIWC; their generalization across different domains requires further validation.

## Related Work & Insights

- **vs. Synthetic Data Quality (Chim et al.)**: Prior studies focus on static quality; this work reveals systematic flaws in the temporal dimension.
- **vs. Model Collapse (Shumailov et al.)**: Model collapse focuses on distribution decay in iterative training; this work provides a complementary view from the perspective of temporal consistency.
- **vs. Human-LLM Co-evolution (Geng & Trotta)**: Existing work mostly analyzes the lexical level; this work extends to semantic and cognitive-emotional dimensions.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of "Temporal Flattening" is novel, providing a fresh longitudinal perspective on LLM text.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive design across three domains, three models, and two conditions with rigorous statistical testing.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured and driven by clear research questions.
- Value: ⭐⭐⭐⭐ Provides direct insights for AIGC detection, synthetic data generation, and longitudinal text modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ExaGPT: Example-Based Machine-Generated Text Detection for Human Interpretability](exagpt_example-based_machine-generated_text_detection_for_human_interpretability.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[ACL 2026\] DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection](detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec.md)
- [\[ACL 2026\] GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization](gigacheck_detecting_llm-generated_content_via_object-centric_span_localization.md)
- [\[ACL 2026\] Can AI-Generated Persuasion Be Detected? Persuaficial Benchmark and AI vs. Human Linguistic Differences](can_ai-generated_persuasion_be_detected_persuaficial_benchmark_and_ai_vs_human_l.md)

</div>

<!-- RELATED:END -->
