---
title: >-
  [Paper Note] LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models
description: >-
  [ACL 2026][LLM Reasoning][Legal Reasoning] LegalDrill utilizes an Audit Agent to diagnose specific error patterns in 0.6B/1.7B SLMs during legal reasoning. A strong teacher (GPT-4o / Qwen3-30B) generates preference pairs…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Legal Reasoning"
  - "SLM Distillation"
  - "Diagnosis-Driven Synthesis"
  - "Difficulty Score"
  - "Iterative DPO"
date: 2026-05-08
content_hash: c10420b44f29a376
---

# LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.23809](https://arxiv.org/abs/2604.23809)  
**Code**: TBD  
**Area**: Law / Small Language Models (SLMs) / Knowledge Distillation / DPO  
**Keywords**: Legal Reasoning, SLM Distillation, Diagnosis-Driven Synthesis, Difficulty Score, Iterative DPO

## TL;DR
LegalDrill utilizes an Audit Agent to diagnose specific error patterns in 0.6B/1.7B SLMs during legal reasoning. A strong teacher (GPT-4o / Qwen3-30B) generates preference pairs through "deliberate reproduction + correction" based on these error instructions. Samples are then filtered using a Difficulty Score derived from the student’s forced-choice probabilities. After iterative SFT+DPO, the 1.7B student approaches the performance of the 30B teacher on several LegalBench subsets.

## Background & Motivation
**Background**: There is a strong demand for legal LLMs in tasks such as judgment prediction, contract QA, and privacy policy entailment. However, legal documents are inherently sensitive, precluding the use of external APIs (GPT/Gemini) or cloud-based RAG. Consequently, local deployment is necessary. Since hosting 30B+ open-source LLMs is cost-prohibitive, **SLMs (<3B)** like Qwen3-0.6B/1.7B are the practical choice for the industry.

**Limitations of Prior Work**: SLMs often demonstrate "the writing style of a lawyer but the logic of a novice" in legal reasoning—frequently misinterpreting statutes or performing logical leaps, leading to incorrect yes/no decisions. Directly applying teacher CoT trajectories for SFT is often ineffective because strong models (especially RL-aligned ones like o1/DeepSeek-R1) generate long, self-reflective, and exploratory paths that exceed the capacity of SLMs to learn effectively.

**Key Challenge**: Legal SFT data is expensive to acquire (requiring professional lawyer annotations), and standard rejection sampling (selection based on final verdict correctness) is too coarse-grained. It identifies "which answer is right" without explaining "why it is wrong," and fails to generate the concise reasoning chains that SLMs can actually internalize. Essentially, the teacher's behavioral distribution does not match the student's learnable distribution.

**Goal**: (1) Transform the teacher's implicit knowledge into concise, error-correcting reasoning chains within SLM capacity; (2) Focus the training budget on samples where the SLM **actually fails**, rather than wasting it on samples it already masters; (3) Eliminate the need for human legal expert annotations throughout the process.

**Key Insight**: Rather than allowing the teacher to generate content freely, an Audit Agent first **diagnoses the student's specific current errors** (e.g., "statute misinterpretation" or "logical leap"). These diagnoses are abstracted into **context-decoupled error instructions**. The teacher then follows these instructions to "deliberately commit errors + simultaneously correct them." The resulting preference pairs serve as "targeted training data" directly addressing the student's blind spots.

**Core Idea**: Diagnosis → Abstract Error Patterns → Targeted Synthesis of Preference Pairs → Filter Trivial Samples via Student Probabilities → Iterative DPO.

## Method

### Overall Architecture
LegalDrill follows a teacher–student iterative framework, with three steps in each round $t$:

- **Input**: $N$ legal queries $x_i = (c_i, q_i)$ (context + question), current student $\pi_{\theta_t}$, teacher $\pi_{\text{teach}}$, and Audit Agent $\pi_{\text{audit}}$.
- **Stage 1 Exploration + Diagnosis**: The student generates responses $\hat{y}_i$ using a CoT system prompt. The Audit Agent examines $(x_i, \hat{y}_i)$ to produce **context-agnostic** error instructions $\mathcal{I}^{(i)}$ (e.g., "ignoring the statute of limitations"). These are aggregated into an Error Instruction Bank $\Phi_{\text{err}} = \{\mathcal{I}^{(1)}, ..., \mathcal{I}^{(N)}\}$.
- **Stage 2 Targeted Generation**: For each sample $x$, $K$ error instructions are sampled from $\Phi_{\text{err}}$. The teacher first intentionally generates a rejected response $y_-^{(k)} \sim \pi_{\text{teach}}(\cdot \mid x, \mathcal{I}_k)$ based on the instruction, and then generates a chosen response $y_+^{(k)} \sim \pi_{\text{teach}}(\cdot \mid x, \mathcal{I}_k, y_-^{(k)})$ conditioned on the mistake.
- **Stage 3 Self-Reflective Verification**: A Difficulty Score is calculated using the student model to filter out trivial samples. The remaining $\mathcal{D}_{\text{train}}^t$ is used for SFT (initial cold-start) + DPO to update the student.
- **Iteration**: $\pi_{\theta_{t+1}}$ enters the next round for re-diagnosis, with the reference model updated as $\pi_{\text{ref}} \leftarrow \pi_{\theta_t}$.

### Key Designs

1.  **Diagnosis-Driven Error Instruction Synthesis (Audit Agent + Context Decoupling)**:
    - **Function**: Abstracts errors from specific student responses into reusable "error instruction templates" calibrated against common legal error taxonomies.
    - **Mechanism**: After reviewing $(x_i, \hat{y}_i)$, the Audit Agent is **forbidden from referencing specific case details**, outputting only context-agnostic descriptions such as "ignoring time windows during limitation period calculations." This allows every instruction in $\Phi_{\text{err}}$ to be recombined with any context to generate new preference pairs, expanding the data volume from $|\mathcal{D}|$ to $K \cdot |\mathcal{D}|$. This also acts as a strong regularizer: since chosen/rejected pairs share the same context and error type, the student cannot rely on surface shortcuts (like length or vocabulary) and must learn the logic of reasoning.
    - **Design Motivation**: Simply allowing the teacher to generate arbitrary rejected responses often results in chosen/rejected pairs with excessive superficial differences. The model then learns irrelevant features like "chosen is longer." Decoupling error patterns from context forces the model to focus on "logical rigor."

2.  **Targeted Two-Step Preference Generation (Error-First, Correction-Following)**:
    - **Function**: Generates the most targeted chosen-rejected pairs under a fixed error instruction.
    - **Mechanism**: This involves two steps: first, the teacher deliberately commits an error $\mathcal{I}_k$ to generate $y_-^{(k)}$; second, $y_-^{(k)}$ is used as additional input, requiring the teacher to generate $y_+^{(k)}$ by specifically identifying and correcting that error. This "correction-after-error" approach produces a more precise contrastive signal than independent generation, as the chosen response is not just correct but is a **specific counter-example to the error in the rejected response**.
    - **Design Motivation**: Standard DPO where chosen and rejected are sampled independently results in differences across multiple dimensions (length, style, path), leading to noisy signals. Conditional generation constrains the difference strictly to the specified logical error.

3.  **Self-Reflective Difficulty Score Filtering (Student-Led)**:
    - **Function**: Uses the student's own probability distribution to filter out "trivial" preference pairs it can already distinguish, focusing the training budget on genuine blind spots.
    - **Mechanism**: Instead of calculating the likelihood of the entire sequence $\pi(y \mid x)$ (which is prone to length/vocabulary interference), a binary forced-choice verification prompt $\mathcal{P}_{\text{ver}}(c, q, y)$ is constructed. The student outputs over $\{\texttt{correct}, \texttt{incorrect}\}$, and the normalized score $s_{\theta_t}(y \mid x) = \pi_{\theta_t}(\texttt{correct} \mid \mathcal{P}_{\text{ver}}) / [\pi_{\theta_t}(\texttt{correct}) + \pi_{\theta_t}(\texttt{incorrect})]$ is obtained. The Difficulty Score $\mathrm{DS} = s_{\theta_t}(y_-^{(k)} \mid x) - s_{\theta_t}(y_+^{(k)} \mid x)$ measures the extent to which the student is deceived by the incorrect response. Only samples with $\mathrm{DS} > \tau$ are retained.
    - **Design Motivation**: While teacher-synthesized pairs are objectively high-quality, many are already distinguishable by the student. Training on these is wasteful and risks degradation. Using the student's confidence gap as a threshold significantly improves data efficiency.

### Loss & Training
Two-stage optimization:
- **Cold-start SFT ($t=0$ only)**: $\mathcal{L}_{\text{SFT}}(\theta_0) = -\mathbb{E}_{(x, y_+) \sim \mathcal{D}_{\text{train}}^0}[\log \pi_{\theta_0}(y_+ \mid x)]$, providing a stable starting point for DPO.
- **Iterative DPO**: $\mathcal{L}_{\text{DPO}}(\theta_{t+1}) = -\mathbb{E}[\log \sigma(\beta(\log\frac{\pi_{\theta_{t+1}}(y_+ \mid x)}{\pi_{\theta_t}(y_+ \mid x)} - \log\frac{\pi_{\theta_{t+1}}(y_- \mid x)}{\pi_{\theta_t}(y_- \mid x)}))]$. A key technique is $\pi_{\text{ref}} = \pi_{\theta_t}$ (the current strategy serves as the reference for the next round), achieving progressive improvement in an online-DPO style.
- **Hyperparameters**: 1-3 epochs, learning rate $1 \times 10^{-4}$. $K$ (instructions per sample) and $\tau$ (DS threshold) are tuned per dataset.

## Key Experimental Results

### Main Results
Evaluation was conducted on four subsets of LegalBench (Cos. QA / Con. QA / Sara Ent. / Priv. Ent.) and two real-world financial legal document datasets (Real-World POA / Trust). Metrics include Accuracy, F1, and Judge Accuracy (LLM-as-Judge for reasoning quality).

| Model | Cos. QA Acc | Con. QA Acc | Sara Ent. Acc | Priv. Ent. Acc | RW POA Acc | RW Trust Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-0.6B (base) | 0.69 | 0.83 | 0.59 | 0.30 | 0.76 | 0.74 |
| Qwen3-1.7B (base) | 0.79 | 0.87 | 0.66 | 0.47 | 0.78 | 0.79 |
| Qwen3-30B-A3B (teacher) | 0.98 | 0.96 | 0.86 | 0.83 | — | — |
| GPT-4o (teacher) | 0.98 | 0.92 | 0.83 | 0.67 | 0.91 | 0.89 |
| **LegalDrill-0.6B (Qwen3-30B teach)** | **0.84** | **0.91** | **0.74** | **0.81** | — | — |
| **LegalDrill-1.7B (Qwen3-30B teach)** | **0.96** | **0.93** | **0.73** | **0.85** | — | — |
| **LegalDrill-0.6B (GPT-4o teach)** | 0.86 | 0.95 | 0.75 | 0.59 | **0.87** | **0.86** |
| **LegalDrill-1.7B (GPT-4o teach)** | 0.94 | 0.97 | 0.75 | 0.60 | **0.92** | **0.90** |

Notable Findings: LegalDrill-1.7B improved from 0.47 to 0.85 (+0.38 Gain) on Priv. Ent., **surpassing the 30B teacher** (0.83). On Real-World POA, the 1.7B student (0.92) roughly equaled the GPT-4o teacher (0.91). On Con. QA, the 1.7B student distilled from GPT-4o reached 0.97, **outperforming GPT-4o** (0.92).

### Ablation Study

| Configuration | Trend | Description |
| :--- | :--- | :--- |
| Full (SFT + DPO) | Optimal | Complete LegalDrill framework. |
| SFT Only (No DPO) | Consistent decline | The contrastive signal of chosen/rejected in DPO is key to gains. |
| No Difficulty Score | Increased compute, less gain | Trivial samples dilute critical gradients and risk degradation. |
| No context-agnostic constraint | Lower robustness | Student learns shortcuts; decoupling is essential for anti-shortcut training. |
| Increased iterations | Monotonic diminishing returns | Student blind spots are progressively filled. |

### Key Findings
- **DPO > SFT-only across nearly all settings**: This confirms that in legal reasoning, "seeing counter-examples to understand mistakes" is more effective than "seeing only positive examples," mirroring the intuition of using "error logs" in professional legal training.
- **Gain for 1.7B > 0.6B**: Qwen3-1.7B with LegalDrill approaches or exceeds the 30B teacher on multiple tasks, whereas the 0.6B model has a lower ceiling. This suggests a minimum capacity threshold for SLMs to master complex reasoning (modeling $\leq$ 0.5B is not recommended).
- **GPT-4o is not a universal teacher**: On Priv. Ent., GPT-4o scores only 0.67, and the 1.7B student reaches 0.60. On the same task, Qwen3-30B teaches the student to 0.85—showing the teacher's domain competence sets the student's upper bound.
- **Instruction reusability enables $K$-fold data expansion**: Decoupling context ensures the model does not overfit to specific cases while expanding the dataset from $|\mathcal{D}|$ to $K \cdot |\mathcal{D}|$.

## Highlights & Insights
- **"Context-agnostic error instructions" as an anti-shortcut mechanism**: Since chosen and rejected share the same context and error type, DPO is forced to learn the "reasoning rigor" axis rather than superficial traits like length. This approach is transferable to any reasoning distillation task (math, code, medicine).
- **Difficulty Score via binary forced-choice**: This avoids the bias where longer chosen responses naturally have lower likelihoods, serving as an elegant verification reward model more robust than PPL-based filtering.
- **Two-step conditional generation (chosen|rejected)**: Requiring the teacher to correct its own deliberate mistake produces a cleaner pair signal than independent sampling, representing a best practice for DPO data synthesis.
- **Iterative reference model**: Updating $\pi_{\text{ref}} = \pi_{\theta_t}$ rather than using a fixed base model is a standard online-DPO technique, but here it creates a complete "diagnosis → data → training → re-diagnosis" flywheel.
- **Real-world industrial validation**: Testing on financial POA/Trust datasets provides evidence for the industrial feasibility of achieving 0.9+ accuracy on locally deployed 1.7B models using GPT-4o as a teacher.

## Limitations & Future Work
- **Heavy dependence on teacher's domain capability**: If the teacher is weak (e.g., GPT-4o on Priv. Ent.), the student cannot exceed it easily; no strategy for "weak teacher" scenarios is provided.
- **Diagnosis also relies on the teacher**: Using the same teacher for the Audit Agent may result in systematic omissions of errors that the teacher itself cannot identify.
- **Hard hyperparameters**: The DS threshold $\tau$ and $K$ are manually tuned per dataset; an adaptive mechanism is lacking.
- **Limited to binary yes/no tasks**: Open-ended tasks like judgment generation or contract drafting are not covered, and the "final verdict" filtering is optimized for binary outcomes.
- **Lower bound of SLMs**: 0.6B models remain weak on certain tasks, indicating that sub-billion models may be insufficient for complex legal reasoning.
- **Future Directions**: Implementing multi-agent debate (red-teaming) for the Audit Agent; generalizing forced-choice DS to multi-class labels; and packaging the framework into a toolkit for other high-stakes domains like medicine and compliance.

## Related Work & Insights
- **vs Standard Rejection Sampling**: LegalDrill provides "precision-guided" data by using diagnosis to create concise, targeted chains rather than the potentially over-long chains produced by standard sampling.
- **vs Reasoning Compression (Zhao et al. 2025, Zhang et al. 2025)**: Instead of pruning the teacher's long CoTs, LegalDrill re-synthesizes targeted chains from scratch to fit the SLM's behavioral distribution.
- **vs SMART (Kim et al. 2025)**: LegalDrill bakes knowledge entirely into the SLM parameters, allowing for zero-dependency local deployment, unlike SMART which requires external LLM calls at inference.
- **vs UniLaw-R1 / Legal PRMs**: Instead of complex RL with step-wise or validity rewards, LegalDrill uses DPO to bypass reward model training while making the "reward criteria" explicit and readable via error instructions.
- **vs Iterative DPO (Pang et al. 2024, Xu et al. 2025)**: LegalDrill adds a diagnosis-synthesis loop to the data generation phase, providing higher blind-spot specificity than simple iterative sampling.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of diagnosis-driven synthesis, context-agnostic instructions, and DS filtering is quite novel for SLM distillation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 6 datasets (4 public, 2 industrial) across 2 teachers and 2 students with robust ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow from motivation to method to experiments; formulas are concise.
- Value: ⭐⭐⭐⭐⭐ High industrial demand for private legal SLMs; methodology is directly applicable to other high-sensitivity domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Accurate Legal Reasoning at Scale: Neuro-Symbolic Offloading and Structural Auditability for Robust Legal Adjudication](accurate_legal_reasoning_at_scale_neuro-symbolic_offloading_and_structural_audit.md)
- [\[ACL 2026\] TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models](trigreason_trigger-based_collaboration_between_small_and_large_reasoning_models.md)
- [\[ACL 2026\] AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)
- [\[ICML 2026\] DenseSteer: Steering Small Language Models towards Dense Math Reasoning](../../ICML2026/llm_reasoning/densesteer_steering_small_language_models_towards_dense_math_reasoning.md)
- [\[ACL 2026\] RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners](rsat_structured_attribution_makes_small_language_models_faithful_table_reasoners.md)

</div>

<!-- RELATED:END -->
