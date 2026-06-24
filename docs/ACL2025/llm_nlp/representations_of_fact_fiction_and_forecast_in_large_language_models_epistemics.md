---
title: >-
  [Paper Note] Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes
description: >-
  [ACL 2025][LLM (Other)][epistemic modality] By evaluating the semantic knowledge of epistemic modality (such as *may*/*must*, *know*/*believe*/*doubt*) in 8 open-source LLMs through a controlled storyboard task, this paper reveals that LLMs exhibit limited and non-robust capabilities in generating appropriate epistemic expressions—necessity (*must*) consistently outperforms possibility (*may*), and factual statements outperform belief statements.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "epistemic modality"
  - "modal auxiliaries"
  - "attitude verbs"
  - "uncertainty expression"
  - "Theory of Mind"
  - "linguistic knowledge"
date: 2026-05-08
content_hash: e2be09dd550e5621
---

# Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes

**Conference**: ACL 2025  
**arXiv**: [2506.01512](https://arxiv.org/abs/2506.01512)  
**Code**: [https://github.com/limengnlp/llm-fff](https://github.com/limengnlp/llm-fff)  
**Area**: LLM/NLP  
**Keywords**: epistemic modality, modal auxiliaries, attitude verbs, uncertainty expression, Theory of Mind, linguistic knowledge

## TL;DR
By evaluating the semantic knowledge of epistemic modality (such as *may*/*must*, *know*/*believe*/*doubt*) in 8 open-source LLMs through a controlled storyboard task, this paper reveals that LLMs exhibit limited and non-robust capabilities in generating appropriate epistemic expressions—necessity (*must*) consistently outperforms possibility (*may*), and factual statements outperform belief statements.

## Background & Motivation

**Background**: LLMs are increasingly deployed in real-world scenarios where expressing uncertainty is essential. Consequently, substantial research has targeted confidence estimation and calibration in LLMs (employing methods like verbalized uncertainty and logit-based indicators).

**Limitations of Prior Work**: Existing literature implicitly assumes that LLMs have already acquired the linguistic knowledge required to express uncertainty, treating the problem merely as an alignment task of "when to trigger" these expressions. However, this assumption **has never been systematically verified**—do LLMs truly understand the semantic differences among *may*, *must*, *know*, *believe*, and *doubt*?

**Key Challenge**: If LLMs lack adequate semantic knowledge of epistemic modals, the reliability of current calibration methods based on verbalized uncertainty is fundamentally questioned.

**Goal**: To systematically evaluate the semantic knowledge of epistemic modality expressions in LLMs leveraging a typological framework.

**Key Insight**: Inspired by research in child language acquisition (such as the hidden object task) and linguistic fieldwork (using targeted storyboards), this study designs a controlled story task. By minimizing reasoning complexity and carefully controlling types of evidence and degrees of certainty, it observes whether LLMs can select appropriate modal expressions.

**Core Idea**: The insufficient semantic knowledge of epistemic modality in LLMs remains a potential root cause of their inability to express uncertainty reliably. Designing uncertainty-aware LLMs requires strengthening their modal semantic representations.

## Method

### Overall Architecture
Based on the typological framework of epistemic modality by Boye (2012) (which centers on evidence and commitment), two experiments are developed: Experiment 1 tests modal auxiliaries (*may*/*might* vs. *must*/*have to*), and Experiment 2 evaluates attitude verbs (*know*/*believe*/*doubt*). Controlled stories are generated via templates and manual construction, tested using greedy decoding across 8 open-source instruction-tuned LLMs, and analyzed using logistic regression to evaluate the effects of parameter scale, modal conditions, prompt formats, and story types on predictive accuracy.

### Key Designs 1: Modal Auxiliaries Experiment (Experiment 1)
- **Function**: Generates 150 controlled stories (5 templates × 3 story types). Each story features a necessity condition (N-modal: *must*/*have to*) and a possibility condition (P-modal: *may*/*might*), assessing whether LLMs can select the correct modal verb based on the sufficiency of the evidence.
- **Mechanism**: Utilizes the "hidden object" paradigm—a toy is hidden in one of three boxes. If two boxes are eliminated, it "*must* be in the blue box" (necessity); if only one is eliminated, it "*may* be in the blue box" (possibility). Three QA formats (direct slot / indirect slot / indirect sentence) are designed to verify robustness, and three story types (base / 1-shot / counterfactual) assess sensitivity to auxiliary context. The 1-shot setup provides in-context demonstrations, while the counterfactual setup tests counterfactual reasoning by altering verified premises.
- **Design Motivation**: Adopting developmental psychology paradigms minimizes reasoning complexity, ensuring the evaluation isolates semantic knowledge. If this knowledge is robust, performance should remain consistent across QA formats and story types. Paired accuracy requires correct responses under both N and P conditions for a given story, representing a more rigorous benchmark than individual condition accuracy. To prevent LLMs from copying 1-shot examples verbatim, lexical variations (different colors, names, etc.) are introduced.

### Key Designs 2: Attitude Verbs Experiment (Experiment 2)
- **Function**: Extracts 30 Theory-of-Mind (ToM) stories from the ToMi dataset to construct 8 statements (2 × pre-fact + 2 × current-fact + 2 × agent0's belief + 2 × agent1's belief). Each pair of statements contrasts low vs. non-low certainty, testing whether LLMs select *know* for facts and *believe*/*doubt* for beliefs.
- **Mechanism**: Leverages information asymmetry in ToM scenarios—since agent0 is unaware that agent1 has moved the object, agent0's output constitutes a "belief" rather than a "fact". LLMs must distinguish factual assertions (using *know*) from belief assertions (using *believe*/*doubt*), adapting their choice to the degree of certainty.
- **Design Motivation**: Attitude verbs represent linguistic formulations of metacognition, requiring an understanding of the differing mental states of multiple entities. Joint accuracy (correctly predicting all 8 statements within a single story) serves as the most stringent evaluation metric.

### Key Designs 3: Statistical Analysis Framework
- **Function**: Fits logistic regression models for each model family (Qwen2/2.5, Llama3/3.1) to systematically evaluate the modulating effects of parameter scale, modal conditions, statement types, and QA formats on accuracy.
- **Mechanism**: Rather than focusing solely on raw averages, regression coefficients are utilized to quantify the direction and significance of individual factors, identifying systemic patterns (e.g., verifying that the superiority of necessity over possibility is a statistically significant effect across models, rather than a random variation).
- **Design Motivation**: Standard behavioral study paradigms demand rigorous statistical inference. The regression modeling prevents descriptive statistics from masking key nuances and identifies interaction effects among variables.

## Key Experimental Results

### Main Results — Experiment 1: Accuracy of Modal Auxiliaries

| Model | Parameters | Accuracy | Paired Accuracy |
|------|--------|--------|-------------|
| Qwen2-7B | 7B | 51.2% | 2.4% |
| Qwen2.5-7B | 7B | 72.3% | 45.1% |
| Llama3-8B | 8B | 55.1% | 11.8% |
| Llama3.1-8B | 8B | 55.2% | 11.8% |
| Qwen2-72B | 72B | 82.3% | 64.7% |
| Qwen2.5-72B | 72B | **95.3%** | **90.7%** |
| Llama3-70B | 70B | 79.6% | 59.1% |
| Llama3.1-70B | 70B | 91.1% | 82.2% |

### Experiment 2: Accuracy of Attitude Verbs

| Model | Parameters | Accuracy | Paired Accuracy | Joint Accuracy |
|------|--------|--------|-------------|-------------|
| Qwen2-7B | 7B | 56.9% | 38.6% | 0% |
| Qwen2.5-72B | 72B | 60.8% | 46.4% | 5.6% |
| Llama3.1-70B | 70B | **72.8%** | **54.2%** | 0% |
| Llama3-70B | 70B | 62.1% | 40.3% | 0% |

### Key Findings
- **Necessity > Possibility**: Across all models, LLMs demonstrate significantly higher accuracy in necessity scenarios (where there is a unique conclusion) than in possibility scenarios (where multiple outcomes remain open), showing they handle unambiguous contexts much more effectively.
- **Parameter scale has a highly significant effect**: 70B+ models (79.6%–95.3%) systematically outperform 7-8B models (51.2%–72.3%) with highly significant regression coefficients ($p < .001$).
- **Factual statements >> Belief statements**: In Experiment 2, the models' accuracy on factual statements is systematically higher than on belief statements (regression coefficient $b = 2.74 \sim 4.79$, $p < .001$).
- **Extremely low joint accuracy**: Even the top-performing 72B model achieves a joint accuracy of only 5.6% on the ToM task, implying that LLMs struggle with consistency when selecting differing attitude verbs within the same scenario.
- **Minor but inconsistent impact of QA formats**: The prompt formatting yields varying directional effects across different model families, suggesting that the underlying modal knowledge is fragile.
- **Reversal phenomenon in Llama3/3.1-70B**: Medium-sized Llama models show surprisingly higher accuracy under low-certainty (*doubt*) conditions, which contrasts with the Qwen series and likely stems from instruction-tuning data distributions.
- **1-shot examples yield no consistent improvement**: Unexpectedly, inserting in-context examples failed to consistently boost performance across models, pointing to semantic knowledge gaps rather than task comprehension failures as the primary bottleneck.
- **Counterfactual stories do not drastically reduce performance**: Stories featuring counterfactual conditions did not suffer from significantly decreased accuracy as expected, suggesting the models' capability to navigate counterfactual prompts exceeds anticipation, though performance remains inconsistent across models.
- **Extremely low paired accuracy in small models**: Qwen2-7B displays a paired accuracy of only 2.4%, showing that 7B models can rarely resolve necessity and possibility concurrently.
- **Significant progress in the Qwen2.5 series**: Qwen2.5-72B (95.3% accuracy) achieves a notable gain compared to Qwen2-72B (82.3%), showing that iterative updates to training data and optimization strategies directly enhance modal competence.

## Highlights & Insights
- **First systematic evaluation of LLMs' modal semantic knowledge from a linguistic typology perspective**—reframing the problem of "whether LLMs can reliably express uncertainty" from calibration and alignment to the semantic representation layer, thereby identifying an overlooked fundamental bottleneck.
- **Exquisitely designed controlled experiments**: Adapting experimental paradigms from developmental psychology (such as the hidden object and ToM tasks) reduces reasoning overhead, successfully isolating the metric to purely measure semantic linguistic knowledge.
- **Paired and Joint accuracy metrics** expose the true performance deficits masked by standard average accuracy—e.g., a seemingly competent average of 72.8% on attitude verbs drops to 0% under strict "all-correct" joint analysis.
- **Insightful human-AI comparative perspective**: Children aged 4–5 can already master basic epistemic modality, and human adults reach near-perfect accuracy (97%–100%), which contrasts with the clear gaps still observed in modern LLMs.
- **Decoupling linguistic uncertainty from cognitive uncertainty (aleatoric/epistemic)**: Demonstrates that the reliability of verbalized uncertainty is fundamentally restricted by linguistic competence, bringing needed clarity to the taxonomy of research in this area.
- **Cross-model consistency analysis**: Systematic logistic regression outlines which effects are invariant across models (such as parameter scale and necessity vs. possibility) versus those that are unstable (such as prompt formatting), proposing an actionable benchmark for future development.

## Limitations & Future Work
- The evaluation focuses exclusively on English and overlooks morphological marking of epistemic modality in low-resource languages (such as suffixes, particle markers, etc.).
- There is no direct, parallel evaluation of human participants in the same setup; the work relies on historical child and adult performance data from literature.
- The experiments do not leverage logit probabilities to formulate intrinsic uncertainty baselines.
- The study excludes the impact of negation on modal semantics (like "not certain" versus "certain that not"), simplifying the scope of the language.
- Only 8 open-source models are benchmarked, leaving closed-source models (such as GPT-4 and Claude) unexamined.
- The story templates are limited (5 templates × 30 stories), suggesting that generalizability requires validation across more extensive, diverse benchmarks.
- The work does not investigate whether supervised fine-tuning or RLHF directly improves modal knowledge, testing only zero-shot/few-shot capabilities of native checkpoints.
- It does not address the consistency of modal expressions in multi-turn dialogues, which is crucial for real-world interactions where LLMs must preserve epistemic states over time.
- Tests are confined to greedy decoding without exploring how sampling strategies alter modal selections.
- The focus is restricted to only three syntactic classifications of epistemic forms in English, without addressing modal markers common in other languages (such as Japanese evidential particles or Turkish suffix options).

## Related Work & Insights
- **vs Xiong et al. (2024) verbalized uncertainty**: Xiong et al. assumed LLMs naturally possess the linguistic capability to express uncertainty and focused heavily on calibrating "when" to speak. This paper questions that assumption, showing that LLMs lack sufficient underlying semantic knowledge of the modal verbs themselves—implying that learning "how to express" is a prerequisite to calibration.
- **vs Holliday et al. (2024) conditionals and modal reasoning**: Holliday evaluated logical fallacies in conditional and modal reasoning, emphasizing reasoning capabilities. In contrast, this study deliberately minimizes reasoning difficulty to focus purely on semantic knowledge, presenting a highly complementary perspective.
- **vs Zhou et al. (2024) LLM uncertainty expression behavior**: Zhou noted that LLMs are hesitant to express uncertainty when providing incorrect responses. This paper presents a deeper explanation: it is more likely not a matter of "reluctance," but rather a fundamental ignorance of which modal verb is appropriate.
- **vs Sileo & Moens (2023) understanding of probability terms**: Sileo discovered that language models struggle to parse probability adverbs/adjectives. This work extends the scope to modal auxiliaries and attitude verbs, tracing a complete representation profile for epistemic modality.
- **Insights**: The developmental psychology angle proposed in this work (learning attitude verbs from pragmatically rich interactive situations) offers inspiration for building LLM training corpora. Since pre-training datasets likely lack explicit modal-semantic alignment, constructing targeted SFT data to bolster modal expression seems highly promising.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First study evaluating LLMs' modal semantic knowledge through a linguistic typology lens; highly innovative and fills a key research gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 models tested under strict statistical models and multi-variable controls, although lacking native human controls and closed-source model comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Masterfully integrates linguistic theory with empirical NLP evaluations, showing high clarity and a brilliant interdisciplinary perspective.
- Value: ⭐⭐⭐⭐ pinpoints a foundational bottleneck in uncertainty expression, providing directional guidance for constructing uncertainty-aware models and verbalized calibration methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts](cross_model_transferability_sv.md)
- [\[ACL 2025\] On the Acquisition of Shared Grammatical Representations in Bilingual Language Models](on_the_acquisition_of_shared_grammatical_representations_in_bilingual_language_m.md)
- [\[ACL 2025\] Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](language_codec_bridging_discrete_codec_speech_language_models.md)
- [\[ACL 2025\] Exploring Graph Representations of Logical Forms for Language Modeling](exploring_graph_representations_of_logical_forms_for_language_modeling.md)
- [\[ACL 2025\] Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)

</div>

<!-- RELATED:END -->
