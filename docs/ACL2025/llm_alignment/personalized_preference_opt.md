---
title: >-
  [Paper Note] Whose Boat Does it Float? Improving Personalization in Preference Tuning via Inferred User Personas
description: >-
  [ACL 2025][Alignment & RLHF][personalization] Academic paper note for Whose Boat Does it Float? Improving Personalization in Preference Tuning via Inferred User Personas.
tags:
  - ACL 2025
  - Alignment & RLHF
  - personalization
  - DPO
  - persona inference
  - abductive reasoning
  - preference tuning
date: 2026-05-08
content_hash: 0a4c54f0e0ad3b5c
---
# Whose Boat Does it Float? Improving Personalization in Preference Tuning via Inferred User Personas

**Conference**: ACL 2025  
**arXiv**: [2501.11549](https://arxiv.org/abs/2501.11549)  
**Code**: [Pinafore/alignment-personalization](https://github.com/Pinafore/alignment-personalization)  
**Area**: LLM Alignment  
**Keywords**: personalization, DPO, persona inference, abductive reasoning, preference tuning

## Background & Motivation

Standard preference learning assumes a strong default hypothesis: if response A is selected and response B is rejected under the same prompt, then A is better than B.
While this assumption is convenient for general alignment, it begins to distort in personalization scenarios.
In reality, "better" typically depends on who the user is, what they prefer, and what they care about.
For example, some users prefer concise answers, while others prefer complete steps and abundant details.
In such cases, the rejected response is not necessarily "bad" but may just be more suitable for a minority of users.
The authors argue that what current DPO-like methods truly discard is not the score, but the explanatory information of "why someone would prefer this response."
Standard preference data only contain the prompt, chosen, and rejected responses, without recording the user personas behind the preferences.
Consequently, the model learns to "mimic the response selected by the majority" rather than "provide different responses based on different personas."
This leads the model to still reply with an averaged answer even when faced with explicit personalization requirements.
The example given at the beginning of the paper is highly representative: when a user only wants a short list, the DPO model provides 10 points; when a user explicitly states they quit alcohol, the model still suggests hiring a bartender.
These errors do not occur because the model cannot perform the task itself, but because it has not learned to align responses with user personas.
Therefore, this paper proposes a stronger perspective on preference learning: preferences should not only answer "which one is better," but also "for whom is it better, why is it better, and under what conditions is it better."
The authors borrow the concept of abductive reasoning, viewing the persona as a hidden context explaining the preference outcome.
Given a prompt and two candidate responses, if a reasonable persona can be abduced, it indicates that this persona acts as an explanatory variable for the preference difference.
Once these personas can be automatically inferred from existing preference data, they can in turn enhance the training set, training models to generate responses of different styles and contents according to different personas.
In other words, the motivation of this study is not to build a stronger judge, but to advance preference learning from "population-average optimal" to "conditional individual optimal."

## Method

The proposed method is divided into two sequential stages: Persona Inference (PI) and Persona Tailoring (PT).
The goal of PI is to infer personas from existing preference data.
The goal of PT is to utilize these personas as additional conditions to train the model to output customized responses based on the personas.

First, consider PI.
Let $p$ be the prompt, and let $r_1$ and $r_2$ be two candidate responses.
The objective of PI is to generate a persona $\mathcal{P}_1$ such that a user with this persona would prefer $r_1$ over $r_2$.
Setting $r_1$ as the chosen response and $r_2$ as the rejected response yields the chosen persona $\mathcal{P}_C$.
Swapping their order yields the rejected persona $\mathcal{P}_R$.
Although this process seems simple, it elevates key preference explanations from "response quality differences" to "user variation modeling."
The authors constrain the persona format to consistently adopt the sentence structure: "The user is [attribute] and prefers [explanation of preference]".
Additionally, the persona is required to describe only high-level characteristics (e.g., information needs, interests, personality) and avoid protected attributes like race to prevent stereotyping and ethical risks.
Inference is conducted using 5-shot prompting, encouraging the model to output a short persona description.
In the experiments, 9 models from the Claude, GPT, and LLaMA-3.1 series were compared, with LLaMA-405B demonstrating the best performance.

The clever aspect of PI is that it infers personas not only for the chosen response but also for the rejected response.
Traditional preference learning treats the rejected response almost as pure noise or a negative example, but this work argues that "uncommon but reasonable" demands may be buried within the rejected responses.
Therefore, $\mathcal{P}_R$ is not waste material in training, but rather a source of hard examples used to evaluate the personalization capabilities of the model.

Next, consider PT.
PT addresses whether a model can generate a customized response $r$ that matches the persona $\mathcal{P}$, given a prompt $p$ and a persona $\mathcal{P}$.
The authors first run PI with LLaMA-405B on the training set to augment existing preference data into a version with personas.
Then, LLaMA-8B is used as the student model to evaluate three distinct utilization methods.

The first is PT_fs, i.e., few-shot prompting.
This approach directly concatenates persona-augmented examples into the prompt, expecting the model to mimic this personalized behavior in-context. This method requires no training and represents the lightest deployment, but its stability is limited.

The second is PT_sft.
The persona and prompt are taken as inputs, and the chosen response is treated as the supervision target for SFT. This essentially learns a conditional generative model: the input includes both the task and the user persona.

The third is PT_dpo.
The initial model $\pi_0$ is first obtained based on PT_sft, and then further optimized using DPO.
At this point, the input becomes $x = \langle p \cdot \mathcal{P}_C \rangle$, and the model aims to increase the conditional probability of the chosen response relative to the rejected response.
The concept corresponding to this step is crucial: the model does not simply learn "how to answer this prompt," but rather learns "for this persona, why this response is more appropriate than the other."

During training, the authors only use the chosen persona $\mathcal{P}_C$ and the chosen response $r_C$ to supervise SFT, and use $r_C$ against $r_R$ for preference optimization in DPO.
Why not train with the rejected persona and rejected response simultaneously?
The paper argues that $\mathcal{P}_R$ only explains why someone might select the rejected response, which does not imply that the rejected response itself represents the optimal personalized response.
Empirically, training directly with $\mathcal{P}_R$ and $r_R$ as positive samples yields little benefit because the average quality of $r_R$ remains lower.
But $\mathcal{P}_R$ remains important during inference and evaluation, as it represents a minority but reasonable preference type.

To avoid information leakage, the authors also designed a test-time persona acquisition method.
Directly using the gold persona of the current sample might amount to cheating.
Thus, a more realistic approach is to retrieve similar prompts from the training set using ColBERT, and take their personas as the retrieved persona ($\mathcal{P}_{retr}$).
The experiments report results using both $\mathcal{P}_{gold}$ and $\mathcal{P}_{retr}$, where the former reflects the upper bound and the latter is closer to real-world usage.

The entire methodological pipeline can be summarized as follows:
First, a large language model is used to explain "preferences" as "user profile differences" via abductive reasoning.
These profiles are then attached to the preference data.
Finally, a smaller model is trained using persona-aware prompting, SFT, or DPO, enabling it to provide more customized answers when presented with a persona.
This represents a route of "distilling the explanatory capabilities of large models into the personalization capabilities of smaller models."

| Stage | Input | Output | Key Role |
|------|------|------|----------|
| Persona Inference | prompt + two responses | chosen/rejected persona | Explains why a user prefers a certain response |
| Persona Tailoring-FS | prompt + persona + few-shot examples | Personalized response | Fast validation of persona value with zero training |
| Persona Tailoring-SFT | prompt + persona | chosen response | Explicitly learns conditional generation on personas |
| Persona Tailoring-DPO | prompt + persona + chosen/rejected pair | Better personalized model | Optimizes preferences conditioned on personas |

| Method Design Point | Author's Choice | Implication |
|------------|----------|------|
| Persona Source | Imputed by LLaMA-405B | Utilizing a strong model to provide explanatory supervision |
| Persona Format | Single-sentence high-level description | Easy to parse, reduces overfitting to surface-level text |
| Training Signals | Only train on $\mathcal{P}_C$ and $r_C$ | Maintains response quality, avoids treating low-quality rejected responses as positive targets |
| Test Persona | Report both $\mathcal{P}_{retr}$ and $\mathcal{P}_{gold}$ | Formulates a distinction between realistic scenarios and the ideal upper bound |

## Key Experimental Results

The PI stage covers four datasets: BeaverTails, SHP, Anthropic HHH, and Mnemonic, spanning three types of scenarios: question answering, dialogue, and education.
The authors sample 300 instances from each dataset, forming a total of 600 PI inputs, and use GPT-4o as a judge to verify if the personas indeed explain user preferences.
Results show that the persona inference accuracy of LLaMA-405B reaches 91%, with an agreement rate with human judgment of approximately 90%.
More importantly, the quality gap between the chosen persona and the rejected persona is not large, with the difference in accuracy for the best model being only 0.06.
This indicates that the rejected response indeed may correspond to a real, albeit less mainstream, user demand.

| PI Evaluation Metric | Results | Implication |
|-----------|------|------|
| Best PI Model | LLaMA-405B | Most stable among open-source models |
| PI Accuracy | 91% | Personas correctly explain the direction of preferences |
| Human vs GPT-4o Agreement | 90% | The judge results are credible |
| Difference in Chosen/Rejected Persona Accuracy | 0.06 | Rejected personas possess equivalent explanatory power |
| Qualitative Human Conclusion | Rejected personas are rarer but reasonable | Supports the idea that "minority preferences should also be served" |

Entering the PT stage, the authors constructed persona-augmented preference data using three datasets: BeaverTails, Anthropic HHH, and Mnemonic.
The training set sizes are 2449, 1059, and 328 instances, respectively.
The main evaluation uses Prometheus as a pairwise judge, comparing Response Quality and Personalization separately, and defines a comprehensive metric $\Delta PQ$ to balance quality and personalization.

The main results are highly clear: whether using few-shot, SFT, or DPO, introducing personas significantly improves personalization.
Among them, PT_dpo is consistently the strongest.
When using retrieved personas, $\Delta PQ = +36.8$ on BeaverTails, +8.4 on Anthropic HHH, and +28.6 on Mnemonic.
If using gold personas, the upper bound is higher, reaching +41.6 on BeaverTails, and +23.0 on Anthropic HHH.

| Method | BeaverTails $\Delta PQ$ | Anthropic HHH $\Delta PQ$ | Mnemonic $\Delta PQ$ |
|------|-------------------------|----------------------------|-----------------------|
| PT_fs + $\mathcal{P}_{retr}$ | +46.3 | +2.5 | +20.3 |
| PT_sft + $\mathcal{P}_{retr}$ | +12.3 | +9.3 | +20.5 |
| **PT_dpo + $\mathcal{P}_{retr}$** | **+36.8** | **+8.4** | **+28.6** |
| PT_fs + $\mathcal{P}_{gold}$ | +55.0 | +12.5 | - |
| PT_sft + $\mathcal{P}_{gold}$ | +23.0 | +27.8 | - |
| **PT_dpo + $\mathcal{P}_{gold}$** | **+41.6** | **+23.0** | - |

The authors then conducted a more critical experiment: comparing "standard DPO unseen to personas during training" with "PT_dpo seen to personas during training" to see if standard DPO could automatically learn personalization simply by appending the persona to the prompt during testing.
The answer is negative.
PT_dpo already outperforms DPO on the chosen persona but shows an even larger advantage on rejected personas (representing minority demands).
On average, PT_dpo relative to DPO achieves a $\Delta PQ$ of 23.7 on rejected personas, compared to 13.4 on chosen personas.
This demonstrates that while standard DPO may implicitly learn mainstream preferences, it struggles to support uncommon personas.

| Comparative Setting | Average Results | Explanation |
|----------|----------|------|
| PT_dpo vs DPO on chosen persona | $\Delta PQ = 13.4$ | Standard DPO has limited adaptation to mainstream personas |
| PT_dpo vs DPO on rejected persona | $\Delta PQ = 23.7$ | Minority personas require explicit training |
| BeaverTails rejected + $\mathcal{P}_{gold}$ | +21.3 | Performance gain of PT is more pronounced on hard cases |
| HHH rejected + $\mathcal{P}_{retr}$ | +35.6 | Minority preferences yield greater benefits in dialogue scenarios |

The paper provides two additional validations.
The first is the human evaluation of persona quality.
Three PhD students rated 80 personas across four axes: plausibility, applicability, harmfulness, and overfitting, indicating that both chosen and rejected personas are highly plausible, and rejected personas are simply applicable to fewer people.
The second is real-user evaluation.
Eight student users handwritten 144 personas across BeaverTails and HHH scenarios and rated PT_dpo and DPO outputs from 1 to 5 based on Answerability and Personalization.
The results show that, especially on BeaverTails, the personalization score of PT_dpo is significantly higher while answerability is barely compromised.

Taken together, the experiments in this paper do not just prove "personas are useful" but progressively demonstrate three things: personas can be inferred, persona quality is reasonable, and persona-aware training indeed improves the model's capability to support diverse user needs.

## Highlights & Insights

The biggest highlight of this paper is redefining the problem formulation of preference learning.
Instead of treating chosen and rejected responses as a simplistic binary of "good and bad," it views them as "conditional preferences of different user demands under the same prompt."
The second highlight is that the value of the rejected response is re-established.
In the past, it was merely treated as a negative sample, but this work converts it into a source for discovering minority personas and constructing hard evaluations.
The third highlight is that the combination of PI and PT is highly lightweight.
Existing preference data do not need to be re-collected; running persona inference once with a strong model upgrades the original data into a persona-aware training set.
The fourth highlight is that the authors use personas not only for training but also for data bias analysis.
For example, in BeaverTails, chosen personas often contain "meticulous" and "multiple", while rejected personas often contain "to-the-point" and "concise", indicating a verbosity bias in the original preference data.
This endows personas with dual value as both training signals and data diagnostic signals.
The most important insight is: if the training objective only aligns with majority preferences, the model ultimately learns an "average persona" rather than "conditional adaptability."

## Limitations & Future Work

First, PI relies on powerful LLMs, especially large models like LLaMA-405B; the quality of personas inferred by smaller models drops significantly.
Second, personas are currently compressed into a single-sentence high-level description. While concise enough, this may be insufficient for complex, multi-dimensional, and dynamically changing user demands.
Third, PT assumes personas are benign, which introduces the risk of sycophancy, as malicious personas could induce biased, dangerous, or irrelevant outputs from the model.
Fourth, although the experiments cover three fields, they remain at the static single-turn prompt level, leaving the issue of persona evolution in long-term interactions unmodeled.
Possible future improvements include:
First, expanding personas from single sentences to structured profiles, such as including fields for information needs, detail tolerance level, and safety boundaries.
Second, integrating PI into reward model training to perform persona-aware reward modeling directly, rather than just persona-aware generation.
Third, inferring personas by combining long-term history rather than relying on a single preference pair each time.
Fourth, introducing persona filtering, refusal training, and system-level safety prompts at the safety level to mitigate risks from adversarial personas.

## Related Work & Insights

Compared with standard DPO, the distinction of this work is not changing the optimizer, but supplying the missing variable of "user conditions" back to DPO.
Compared with existing personalization work, this method does not require collecting large amounts of real-world persona annotations beforehand, but automatically gold-mines them from existing preference data using abductive reasoning, drastically lowering data costs.
Compared with methods that treat personas solely as additional prompts during inference, this paper demonstrates that without persona-aware training, simply adding "what kind of person I am" during inference is insufficient.
Compared with safety alignment works, this paper also offers a perspective worth drawing upon: many so-called "rejected answers" should not be viewed simply as low-quality answers; some merely deviate from mainstream preferences.
At least three insights can be derived for subsequent research.
First, rejected personas can be used as a source of hard examples in personalization benchmarks to test whether models only serve the majority.
Second, personas can be employed as analyzers to retrospectively examine whether preference data are biased toward specific styles like verbose, conservative, or sycophantic.
Third, in scenarios with strong individual variation—such as education, healthcare, and legal consulting—persona-aware preference tuning is likely far more critical than uniform alignment, because the "most helpful answer" inherently varies from person to person.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Bringing abductive reasoning into preference learning to explicitly recover personas represents a strong conceptual advancement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ PI accuracy, human sanity checks, Prometheus evaluations, hard-case analysis of rejected personas, and real-user evaluations are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐☆ The narrative is fluent, the relationship between PI and PT is well-explained, and the experiments are clearly structured.
- Value: ⭐⭐⭐⭐⭐ Highly inspiring for personalized alignment, data analysis, and constructing fairer preference evaluations.
- Overall Evaluation: 9.2/10. It truly addresses a long-overlooked issue in preference learning: just because the majority prefers something does not mean everyone should be served with the exact same answer.

## Related Papers

- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ACL 2026\] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback](../../ACL2026/llm_alignment/wildfeedback_aligning_llms_with_in-situ_user_interactions_and_feedback.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ICLR 2026\] Why DPO is a Misspecified Estimator and How to Fix It](../../ICLR2026/llm_alignment/why_dpo_is_misspecified_estimator.md)
- [\[ACL 2025\] IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization](iopo_input_output_preference.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ICLR 2026\] Why DPO is a Misspecified Estimator and How to Fix It](../../ICLR2026/llm_alignment/why_dpo_is_misspecified_estimator.md)
- [\[ACL 2026\] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback](../../ACL2026/llm_alignment/wildfeedback_aligning_llms_with_in-situ_user_interactions_and_feedback.md)
- [\[NeurIPS 2025\] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay](../../NeurIPS2025/llm_alignment/improving_data_efficiency_for_llm_reinforcement_fine-tuning_through_difficulty-t.md)
- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)

</div>

<!-- RELATED:END -->
