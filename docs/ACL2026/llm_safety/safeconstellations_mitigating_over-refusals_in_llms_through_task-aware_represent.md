---
title: >-
  [Paper Note] SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering
description: >-
  [ACL2026][LLM Safety][Over-refusal] SafeConstellations discovers that representations in the middle-to-late layers of LLMs form stable "constellation trajectories" according to tasks. It significantly reduces over-refusa…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "Over-refusal"
  - "task-aware steering"
  - "representation trajectories"
  - "safety alignment"
  - "inference-time intervention"
date: 2026-05-08
content_hash: e6e725430ab47115
---

# SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering

**Conference**: ACL2026  
**arXiv**: [2508.11290](https://arxiv.org/abs/2508.11290)  
**Code**: https://github.com/Sakonii/SafeConstellations/  
**Area**: Interpretability / LLM Safety / Representation Intervention  
**Keywords**: Over-refusal, task-aware steering, representation trajectories, safety alignment, inference-time intervention

## TL;DR
SafeConstellations discovers that representations in the middle-to-late layers of LLMs form stable "constellation trajectories" according to tasks. It significantly reduces over-refusal without compromising general capabilities by lightweight steering of representations from refusal trajectories toward non-refusal trajectories on high-confidence benign tasks.

## Background & Motivation
**Background**: LLM safety alignment typically prevents harmful requests through refusal strategies. However, in practical applications, many tasks involve classifying, translating, transcribing, or retrieving information from sensitive text without requiring the model to generate harmful content. If safety systems only detect sensitive keywords or dangerous contexts, they easily misidentify benign tasks as harmful intents.

**Limitations of Prior Work**: Existing research on over-refusal often defines the problem as "toxic inputs being incorrectly rejected" without fully distinguishing the specific task requested by the user. For instance, the same segment of sensitive text should have different safety boundaries under tasks like "translation," "sentiment analysis," "paraphrasing," and "offensive generation." Task-agnostic refusal corrections affect both valid safety rejections and helpful responses.

**Key Challenge**: Safety alignment requires models to maintain refusal capabilities for dangerous intentions, while utility requires models to complete benign analytical tasks. Using a global steering direction to correct refusals is too coarse; conversely, no intervention results in high mis-refusal rates in scenarios like low-resource translation, encrypted text parsing, or sensitive sentence analysis.

**Goal**: The authors aim to answer three questions: whether tasks themselves form separable structures in the hidden representation space; whether refusal and non-refusal can be distinguished within the same task trajectory; and whether inference-time interventions can be applied only to specific benign tasks to reduce over-refusal while preserving rejections of genuinely harmful requests.

**Key Insight**: Starting from representation geometry, the paper assumes that each task category forms a relatively stable trajectory across Transformer layers, termed a "constellation pattern." Compared to observing output text, hidden state trajectories can expose earlier whether the model is "executing a task" or "sliding toward refusal."

**Core Idea**: Replace global refusal vectors with task-conditioned representation centers and hierarchical trajectories, performing minor steering only when the model's internal trajectory approaches the over-refusal region of that specific task.

## Method

### Overall Architecture
SafeConstellations is an inference-time intervention method that does not require retraining the base model. In the offline phase, the frozen LLM is run on task-labeled data to collect hidden vectors of the last input token at each layer. Responses are categorized into target behaviors and refusal behaviors using an LLM-as-a-judge. Target centers, refusal centers, and steering directions are then established for each task and layer.

In the online phase, given a new prompt, the method calculates its inter-layer representation trajectory to identify the most likely known task and estimates task confidence. Only if the task belongs to a developer-defined set of benign tasks and the confidence is sufficiently high does the system select a few layers most in need of correction to nudge the hidden states toward the non-refusal direction of that task; otherwise, the original model behavior is fully preserved.

### Key Designs

1.  **Task Constellation Embedding Library**:
    - **Function**: Saves "target response trajectories" and "over-refusal trajectories" for each task as geometric references for identification and intervention.
    - **Mechanism**: Layer-wise centers $c_{t,tar}^{(l)}$ and $c_{t,ref}^{(l)}$ are calculated for target and refusal samples within the same task. Their difference forms a task-specific steering vector $v_t^{(l)}$. Layer effectiveness is determined by both center distance and intra-cluster variance; layers with well-separated and tight clusters are preferred for intervention.
    - **Design Motivation**: Over-refusal is not a uniform direction; the reasonable output forms of different tasks vary greatly. Modeling tasks separately avoids blending the refusal boundaries of translation, sentiment analysis, and RAG-QA into a coarse safety vector.

2.  **Task-Aware Gating**:
    - **Function**: Decides when to leave the model untouched and when to allow the steering process.
    - **Mechanism**: The system calculates task scores using the current hidden trajectory and task centers, selecting the task with the highest score. If confidence is below 0.85 or the predicted task is not in the benign task set, the base model response is returned. The paper defines benign tasks as sentiment analysis, translation, cryptanalysis, and RAG-QA, excluding paraphrasing due to its more ambiguous intent.
    - **Design Motivation**: Safety interventions risk "over-correction." This gating limits the method to developer-approved benign tasks, ensuring the goal is correcting mis-refusals caused by task identification failure rather than relaxing safety boundaries.

3.  **Dynamic Layer Selection & Adaptive Intensity**:
    - **Function**: Performs minor representation corrections only on a few layers closest to the refusal manifold.
    - **Mechanism**: Relative distances from the current hidden state to the target and refusal centers are computed for each layer to select layers with the highest steering potential. Layer alignment metrics are then used to judge how close the layer already is to the target trajectory, adjusting intervention intensity accordingly. The hidden state is updated by moving a small step along the normalized task steering vector.
    - **Design Motivation**: Intervention in fixed layers often leads to degraded output quality. Dynamic selection concentrates intervention on layers where the "sample indeed biases toward refusal," reducing side effects on natural language capability and safety behavior.

### Loss & Training
The method itself does not train the base LLM or introduce new refusal classifier training objectives. During the offline construction phase, a 75% training split is used to estimate task embeddings. During online inference, a trajectory analysis and a few activation steerings are performed. The paper reports an average increase of approximately 0.2 seconds for short answers, while long answers are primarily determined by decoding length. For the LLaMA-3.1-8B task set, task embeddings require about 847MB of storage, growing linearly with the number of tasks and stored layers.

## Key Experimental Results

### Main Results

The authors constructed an over-refusal benchmark of 1,047 samples covering sentiment analysis, translation, paraphrasing, cryptanalysis, and RAG-QA. Base texts were sourced from Alpaca, XSTest, JailbreakBench, SaladBench, and self-built RAG-QA. Evaluations included refusal types, safety types, and MMLU utility.

| Model / Configuration | Over-Refusal Rate ↓ | Gain (Relative) ↑ | MMLU ↑ | Description |
| :--- | :--- | :--- | :--- | :--- |
| LLaMA3.1-8B Baseline | 17.77% | - | 46.57 | Unintervened |
| LLaMA3.1-8B + SafeConstellations | 4.81% | 72.92% | 46.57 | Dynamic layers + task-specific trajectories + alignment |
| Qwen1.5-7B Baseline | 8.15% | - | 28.42 | Unintervened |
| Qwen1.5-7B + SafeConstellations | 2.96% | 63.64% | 28.42 | Also maintains MMLU |
| LLaMA + Fixed Aggressive Steering | 7.03% | 60.42% | 43.66 | Lower refusal but damages general capability |
| LLaMA + Fixed Layers [15,20,25,30] | 16.66% | 6.25% | 39.20 | Weak intervention and significant utility drop |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full model | LLaMA Over-refusal 4.81%, 72.92% Gain | Dynamic selection, task-specific steering, and alignment used together |
| w/o dynamic selection: late layers | 6.29%, 64.58% Gain | Fixed middle-late layers are effective but inferior to dynamic selection |
| w/o dynamic selection: final layer only | 5.92%, 66.67% Gain | Final layer has strong signals but still loses some controllability |
| w/o trajectory alignment | 6.64%, 62.50% Gain | Task-specific steering alone is not precise enough |
| w/o task-specific steering | MMLU drops from 46.57 to 43.66 or 39.20 | Global/fixed intervention easily sacrifices output quality |

### Key Findings
- Over-refusal is most prominent in LLaMA on benign tasks, Claude is more cautious but has fewer mis-refusals, and GPT-4o's mis-refusal is concentrated in low-resource translation, suggesting over-refusal is affected by both model family and task type.
- UMAP and separability analysis show that hidden states are organized more by "task" than by "text sensitivity" or "final response type"; in layers L12-L19, silhouette scores for sentiment analysis and translation are significantly higher than in mixed task settings.
- When performing targeted mitigation on the most refusal-prone tasks, translation over-refusal dropped from 46.7% to 8.9% (81.0% Gain); sentiment analysis dropped from 36.4% to 18.2% (50.0% Gain).
- Aggressive fixed-layer intervention can stop the model from refusing but results in gibberish or repetitive tokens; this indicates that "reducing refusal rate" is not the sole goal—one must also ensure responses retain task semantics.

## Highlights & Insights
- The paper redefines over-refusal as "task identification failure" rather than just "sensitive word trigger failure," which is a crucial perspective. It explains why the same text should be handled differently under translation, sentiment analysis, and dangerous generation.
- Task constellation is an interpretable intermediate object: it provides both a visualization of task trajectories and layer-wise intervention directions, making it easier to diagnose than black-box prompts or coarse refusal thresholds.
- The gating design is conservative: steering only occurs for high-confidence benign tasks, falling back to the base model otherwise. This "selective intervention" is more suitable for safety scenarios than blindly reducing refusals.
- Experiments evaluated more than just refusal rates, including MMLU and qualitative output quality, revealing the side effects of fixed aggressive interventions and clarifying the practical boundaries of the method.

## Limitations & Future Work
- The method requires access to internal hidden states, making it difficult to apply directly to closed-source APIs or services exposing only text interfaces.
- Task embeddings are static and model-specific; centers must be recalculated or continuously updated when changing models, domains, or when task distributions shift.
- The set of benign tasks is pre-defined by developers; generalization to unseen tasks is insufficient, especially in scenarios where task semantics are strongly entangled with safety intentions.
- Utility is primarily measured by MMLU and has not yet covered finer-grained quality dimensions such as factuality, long-context consistency, conversational coherence, or calibration.
- Thresholds, layer selection, and steering intensity still involve heuristic components; future work could investigate more robust confidence estimation and automatic intensity calibration.

## Related Work & Insights
- **vs Traditional Over-refusal Mitigation**: Traditional methods often adjust refusal tendencies at the output layer or through general safety classifiers. SafeConstellations restricts refusal correction to task-conditioned representation trajectories, offering finer granularity but requiring internal states and offline embedding libraries.
- **vs Global Activation Steering**: Global steering usually assumes a unified "refusal direction." This work shows that task differences change trajectory geometry, meaning global directions may be neither sufficient nor stable.
- **vs Prompt-level Safety Calibration**: Prompt-based methods are easily influenced by surface text. By observing middle layers directly, this work can detect earlier whether the model understands a task as a benign analytical one.
- **Insights**: The task constellation concept could be transferred to hallucination suppression, format following, and tool-call error correction by treating "incorrect behaviors" as trajectory deviations within a task.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Explains over-refusal through task trajectories with a clear geometric perspective, though built on existing activation steering concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes self-built benchmarks, cross-model tests, ablations, separability analysis, and utility checks, though task sets and safety scenarios remain limited.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation, method, and visualization are closely linked, though some formulas and narratives are slightly crowded.
- Value: ⭐⭐⭐⭐☆ High value for safety utility, particularly for open-source model deployments where internal states are accessible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DART: Mitigating Harm Drift in Difference-Aware LLMs via Distill-Audit-Repair Training](dart_mitigating_harm_drift_in_difference-aware_llms_via_distill-audit-repair_tra.md)
- [\[ACL 2026\] Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding](please_refuse_to_answer_me_mitigating_over-refusal_in_large_language_models_via_.md)
- [\[ICCV 2025\] Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation](../../ICCV2025/llm_safety/forgetting_through_transforming_enabling_federated_unlearning_via_class-aware_re.md)
- [\[ACL 2026\] SWAN: Semantic Watermarking with Abstract Meaning Representation](swan_semantic_watermarking_with_abstract_meaning_representation.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)

</div>

<!-- RELATED:END -->
