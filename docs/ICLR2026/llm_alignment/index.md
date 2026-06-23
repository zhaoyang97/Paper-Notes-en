---
title: >-
  ICLR2026 Alignment & RLHF Papers · 102 Notes
description: >-
  102 ICLR2026 papers in the Alignment & RLHF area, covering Alignment/RLHF, LLM, Adversarial Robustness, Reinforcement Learning, Few-/Zero-Shot Learning, Reasoning and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ICLR2026"
  - "Alignment & RLHF"
  - "AI paper notes"
  - "paper summaries"
  - "Alignment/RLHF"
  - "LLM"
  - "Adversarial Robustness"
  - "Reinforcement Learning"
  - "Few-/Zero-Shot Learning"
  - "Reasoning"
item_list:
  - u: "a2d_any-order_any-step_safety_alignment_for_diffusion_language_models/"
    t: "A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models"
  - u: "activedpo_active_direct_preference_optimization_for_sample-efficient_alignment/"
    t: "ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment"
  - u: "align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf/"
    t: "Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment"
  - u: "aligner_diagnose_thyself_a_meta-learning_paradigm_for_fusing_intrinsic_feedback_/"
    t: "Aligner, Diagnose Thyself: A Meta-Learning Paradigm for Fusing Intrinsic Feedback in Preference Alignment"
  - u: "aligning_deep_implicit_preferences_by_learning_to_reason_defensively/"
    t: "Aligning Deep Implicit Preferences by Learning to Reason Defensively"
  - u: "alignment-weighted_dpo_a_principled_reasoning_approach_to_improve_safety_alignme/"
    t: "Alignment-Weighted DPO: A Principled Reasoning Approach to Improve Safety Alignment"
  - u: "alphaalign_incentivizing_safety_alignment_with_extremely_simplified_reinforcemen/"
    t: "AlphaAlign: Incentivizing Safety Alignment with Extremely Simplified Reinforcement Learning"
  - u: "alphasteer_learning_refusal_steering_with_principled_null-space_constraint/"
    t: "AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint"
  - u: "anchored_supervised_fine-tuning/"
    t: "Anchored Supervised Fine-Tuning"
  - u: "annotation-efficient_honesty_alignment_via_confidence_elicitation_and_calibratio/"
    t: "Annotation-Efficient Honesty Alignment via Confidence Elicitation and Calibration"
  - u: "beyond_binary_preferences_a_principled_framework_for_reward_modeling_with_ordina/"
    t: "Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback"
  - u: "beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling/"
    t: "Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling"
  - u: "beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew/"
    t: "Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework"
  - u: "bird_behavior_induction_via_representation-structure_distillation/"
    t: "BIRD: Behavior Induction via Representation-structure Distillation"
  - u: "bradley-terry_and_multi-objective_reward_modeling_are_complementary/"
    t: "Bradley–Terry and Multi-Objective Reward Modeling Are Complementary"
  - u: "cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation/"
    t: "CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation"
  - u: "capability-based_scaling_trends_for_llm-based_red-teaming/"
    t: "Capability-Based Scaling Trends for LLM-Based Red-Teaming"
  - u: "chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model/"
    t: "Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training"
  - u: "cognitive_models_can_reveal_interpretable_value_trade-offs_in_language_models/"
    t: "Cognitive models can reveal interpretable value trade-offs in language models"
  - u: "comal_a_convergent_meta-algorithm_for_aligning_llms_with_general_preferences/"
    t: "COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences"
  - u: "contextif_enhancing_instruction-following_through_context_reward/"
    t: "ContextIF: Enhancing Instruction-Following through Context Reward"
  - u: "cultivating_pluralism_in_algorithmic_monoculture_the_community_alignment_dataset/"
    t: "Cultivating Pluralism In Algorithmic Monoculture: The Community Alignment Dataset"
  - u: "data_selection_for_llm_alignment_using_fine-grained_preferences/"
    t: "Data Selection for LLM Alignment Using Fine-Grained Preferences"
  - u: "disentangling_length_bias_in_preference_learning_via_response-conditioned_modeli/"
    t: "Disentangling Length Bias in Preference Learning via Response-Conditioned Modeling"
  - u: "displacement-resistant_extensions_of_dpo_with_nonconvex_f-divergences/"
    t: "Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences"
  - u: "dont_throw_away_your_pretrained_model/"
    t: "Don't Throw Away Your Pretrained Model"
  - u: "eigenbench_a_comparative_behavioral_measure_of_value_alignment/"
    t: "EigenBench: A Comparative Behavioral Measure of Value Alignment"
  - u: "elephant_measuring_and_understanding_social_sycophancy_in_llms/"
    t: "ELEPHANT: Measuring and Understanding Social Sycophancy in LLMs"
  - u: "eliminating_inductive_bias_in_reward_models_with_information-theoretic_guidance/"
    t: "Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance"
  - u: "enforcing_axioms_for_ai_alignment_under_loss-based_rules/"
    t: "Enforcing Axioms for AI Alignment under Loss-Based Rules"
item_total: 102
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# ⚖️ Alignment & RLHF

**🔬 ICLR2026** · **102** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (12)](../../CVPR2026/llm_alignment/index.md) · [💬 ACL2026 (38)](../../ACL2026/llm_alignment/index.md) · [🧪 ICML2026 (37)](../../ICML2026/llm_alignment/index.md) · [🤖 AAAI2026 (17)](../../AAAI2026/llm_alignment/index.md) · [🧠 NeurIPS2025 (36)](../../NeurIPS2025/llm_alignment/index.md) · [📹 ICCV2025 (2)](../../ICCV2025/llm_alignment/index.md)

🔥 **Top topics:** Alignment/RLHF ×47 · LLM ×22 · Adversarial Robustness ×8 · Reinforcement Learning ×4 · Few-/Zero-Shot Learning ×2

**[A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)**

:   The authors propose A2D, a token-level safety alignment method for diffusion language models (dLLMs). By training the model to output the `[EOS]` token at masked positions when encountering harmful content, it achieves safety defense across any decoding order and any decoding step. This reduces the DIJA template attack success rate from 80%+ to near zero (1.3%/0.0%) and supports early rejection for 19.3x acceleration.

**[ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment](activedpo_active_direct_preference_optimization_for_sample-efficient_alignment.md)**

:   ActiveDPO utilizes the "aligned LLM itself" as a reward model. Based on the gradient of its implicit reward, it derives a theoretically guaranteed uncertainty criterion to actively select the most valuable preference triplets for annotation. This allows the LLM to reach higher alignment levels using fewer human preference labels under a fixed annotation budget.

**[Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)**

:   The authors propose the Multi-Lingual Consistency (MLC) auxiliary loss. By using SVD to manipulate the singular values of the multilingual representation matrix toward rank-1 (i.e., making multilingual representations collinear), safety alignment effects from a single language can be consistently transferred to all languages using only multilingual prompt translations (without needing target language responses).

**[Aligner, Diagnose Thyself: A Meta-Learning Paradigm for Fusing Intrinsic Feedback in Preference Alignment](aligner_diagnose_thyself_a_meta-learning_paradigm_for_fusing_intrinsic_feedback_.md)**

:   To address the issue where "mislabeled preference pairs" in preference datasets degrade DPO alignment, this paper moves beyond single heuristics like perplexity differences. It allows the model to "self-diagnose"—constructing a diagnostic vector from three intrinsic signals: consistency, learning difficulty, and generation confidence. A small network is then trained via meta-learning to fuse these signals and adaptively weight each sample, significantly outperforming existing robust alignment methods across various noise ratios.

**[Aligning Deep Implicit Preferences by Learning to Reason Defensively](aligning_deep_implicit_preferences_by_learning_to_reason_defensively.md)**

:   Addressing the problem in LLM personalization where models "merely parrot explicit user preferences while failing to infer deep intentions or proactively avoid risks," this paper reformulates alignment from scalar reward matching into a structured reasoning process. It first constructs DeepPref, a reasoning chain dataset with step-by-step critique annotations using a "Multi-role Cognitive Committee." Then, it trains Pers-GenPRM, a generative process reward model that "critiques before scoring." Finally, a token-level online RL strategy (CDPA) is employed to integrate numerical and natural language feedback, achieving SOTA results in both deep preference understanding and defensive reasoning.

**[Alignment-Weighted DPO: A Principled Reasoning Approach to Improve Safety Alignment](alignment-weighted_dpo_a_principled_reasoning_approach_to_improve_safety_alignme.md)**

:   The authors first use causal intervention to prove that "current safety alignment is shallow and unrelated to deep reasoning," then release an open-source CoT safety fine-tuning dataset to teach models to "refuse with reasoning." Finally, they propose **Alignment-Weighted DPO**: decomposing responses into a "reasoning segment" and a "response segment" with different weights, applying heavier preference updates to the segment that is more harmful in failed jailbreaks. This significantly improves robustness against various jailbreak attacks while preserving utility.

**[AlphaAlign: Incentivizing Safety Alignment with Extremely Simplified Reinforcement Learning](alphaalign_incentivizing_safety_alignment_with_extremely_simplified_reinforcemen.md)**

:   AlphaAlign utilizes an extremely simplified pure reinforcement learning framework—requiring only binary "harmful/benign" labels and fewer than 200 RL steps—to incentivize the "latent safety self-awareness" embedded in large models during pre-training. By requiring the model to generate a safety rationale before answering and employing a dual-reward system (verifiable safety reward + normalized helpfulness reward), it breaks the "safety-utility" trade-off.

**[AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint](alphasteer_learning_refusal_steering_with_principled_null-space_constraint.md)**

:   AlphaSteer is proposed to dynamically construct steering vectors by learning a transformation matrix subject to null-space constraints. It generates near-zero vectors for benign inputs (preserving utility) and reconstructs refusal direction vectors for malicious inputs (enhancing safety), providing a theoretical guarantee for the decoupling of safety and utility.

**[Anchored Supervised Fine-Tuning](anchored_supervised_fine-tuning.md)**

:   This paper provides a rigorous interpretation of the nature of Dynamic Fine-Tuning (DFT) being "tighter but prone to drift" using the reward-weighted regression (RWR) framework. It proposes ASFT, which superimposes a lightweight KL anchoring term onto the DFT reweighting objective, achieving stable gains in both reasoning and knowledge tasks with SFT-level computational costs.

**[Annotation-Efficient Honesty Alignment via Confidence Elicitation and Calibration](annotation-efficient_honesty_alignment_via_confidence_elicitation_and_calibratio.md)**

:   This paper decomposes "honesty alignment" (enabling LLMs to accurately state their confidence before answering) into an "Elicitation-then-Calibration" two-stage paradigm: first, the model is taught to externalize its internal confidence using annotation-free self-consistency signals; second, this elicited confidence is calibrated to actual accuracy using a minimal amount of correctness labels (~1k samples, approximately 0.18% of the full set). The authors release HonestyBench with 560k training samples, demonstrating that using only 1k labels achieves 98% of the performance of full supervision.

**[Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback](beyond_binary_preferences_a_principled_framework_for_reward_modeling_with_ordina.md)**

:   This paper points out that existing reward models (RMs) only utilize binary preferences ("A is better than B"). When faced with human Likert-scale feedback ("significantly better/better/slightly better"), they rely on ad-hoc heuristic patches like manual margins or weighting factors. The authors reformulate reward modeling as a **discrete ordinal regression** problem. From the ordered logit model, they naturally derive two principled losses (NLL and All-Threshold), allowing "thresholds" that separate preference levels to be learned directly from data. This approach consistently matches or outperforms heuristic baselines on RewardBench / RM-Bench and reduces error severity by 87%.

**[Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)**

:   The RCPO framework is proposed to extend LLM alignment from pairwise preferences to ranked choice modeling. It unifies utility models (MNL) and ranking models (Mallows-RMJ) via MLE, outperforming DPO and its variants in both single-best and top-k feedback formats.

**[Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)**

:   This paper proposes a preference learning framework based on social choice theory axioms. It infers a feasibility set of evaluator population distributions from pairwise comparison data and constructs policies satisfying Population-Proportional Alignment (PPA) and Population Bounded Manipulability (PBM) axioms.

**[BIRD: Behavior Induction via Representation-structure Distillation](bird_behavior_induction_via_representation-structure_distillation.md)**

:   BIRD transfers "alignment behaviors" such as robustness and safety from a heterogeneous teacher to a student by matching the internal representation **structure** (the geometry of pairwise similarities within a batch, measured via CKA) of the student to that of an aligned teacher. The teacher and student can differ entirely in tasks, data, architectures, and output spaces. In image OOD robustness transfer, BIRD achieves up to 18% higher robust accuracy than fine-tuning, transfer learning, or continual learning, and enables weak-to-strong transfer from a teacher $25\times$ smaller than the student.

**[Bradley–Terry and Multi-Objective Reward Modeling Are Complementary](bradley-terry_and_multi-objective_reward_modeling_are_complementary.md)**

:   This paper proposes SMORM, which jointly trains a Bradley–Terry (BT) single-objective reward head and a multi-objective regression head on a shared embedding. The authors theoretically prove that the two are complementary: the regression head helps the single-objective head resist reward hacking under OOD conditions, while the BT head "lifts" the weaker multi-objective head. Consequently, a 7B model outperforms a 70B baseline.

**[CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)**

:   The CAGE framework is proposed, which decouples the adversarial structure of red-teaming prompts from cultural content via a Semantic Mold. This allows for the systematic adaptation of English red-teaming benchmarks to diverse cultural contexts, generating culturally grounded prompts that achieve significantly higher ASR (Attack Success Rate) than direct translation.

**[Capability-Based Scaling Trends for LLM-Based Red-Teaming](capability-based_scaling_trends_for_llm-based_red-teaming.md)**

:   Four jailbreak methods were systematically evaluated on over 600 attacker-target LLM pairs. The study found that the Attack Success Rate (ASR) follows a sigmoid scaling law ($R^2=0.83$) relative to the attacker-target capability gap, which can be quantified using the logit-transformed MMLU-Pro scores.

**[Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)**

:   The authors theoretically prove that reward over-optimization primarily stems from reward model misspecification in the high-reward tail regions. They propose a rubric-based reward modeling method: utilizing off-policy data (excellent responses generated by strong models) to construct scoring rubrics and refining them through progressive "great vs. greater" differentiation to effectively mitigate over-optimization.

**[Cognitive models can reveal interpretable value trade-offs in language models](cognitive_models_can_reveal_interpretable_value_trade-offs_in_language_models.md)**

:   This paper employs the Rational Speech Act (RSA) cognitive model of "polite speech" as a probe to fit weights for three utilities (informational, social, and presentational) in a truth-versus-face-saving dilemma task. It translates "invisible low-level decisions" such as inference budget, system prompts, and RLHF training dynamics into a set of interpretable parameters representing value trade-offs.

**[COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences](comal_a_convergent_meta-algorithm_for_aligning_llms_with_general_preferences.md)**

:   COMAL models "aligning to general human preferences" as an original (unregularized) two-player zero-sum game. Using the Conceptual Prox meta-algorithm derived from game theory—which solves a KL-regularized sub-game in each round and then advances the reference policy to the current solution—it proves for the first time that the algorithm achieves **last-iterate convergence to the exact Nash Equilibrium of the original game**. This guarantees a $\ge 50\%$ win rate against any opponent strategy. It can be implemented on top of existing methods like DPO/IPO/INPO with minimal changes, maintaining a $>60.2\%$ win rate against all baseline algorithms on Llama-3-8B-Instruct.

**[ContextIF: Enhancing Instruction-Following through Context Reward](contextif_enhancing_instruction-following_through_context_reward.md)**

:   ContextIF trains a "context generator" using reinforcement learning to automatically produce constraint summaries and parallel demonstrations for each instruction. This generated context is then fed into a frozen target model for In-Context Learning (ICL). Guided by a composite "Context Reward" that evaluates both structure and semantics, it improves an 8B model's IFEval score from 77.11 to 83.35 while maintaining or even enhancing general capabilities.

**[Cultivating Pluralism In Algorithmic Monoculture: The Community Alignment Dataset](cultivating_pluralism_in_algorithmic_monoculture_the_community_alignment_dataset.md)**

:   Based on representative human surveys of 15,000 individuals across 5 countries, the authors demonstrate that 21 SOTA LLM responses align with only 41% of human preferences ("algorithmic monoculture"). Existing preference datasets fail to learn this diversity because candidate responses are too homogeneous. To address this, "Negative Correlation (NC) sampling" is proposed—using a single prompt to generate four **deliberately divergent** responses at once. This significantly improves the ability of alignment methods to learn heterogeneous preferences. Consequently, the authors open-source Community Alignment, the largest and most representative multilingual multi-turn preference dataset to date (233,319 comparisons).

**[Data Selection for LLM Alignment Using Fine-Grained Preferences](data_selection_for_llm_alignment_using_fine-grained_preferences.md)**

:   Addressing the issue that training DPO on aggregated aspect-specific preferences is hindered by preference conflicts, this paper proposes Preference Divergence (PD) to quantify the degree of conflict between a sample and other preferences. It proves that "selecting only the samples with the most negative PD for standard DPO" achieves optimal upper and lower bounds for the loss. Consequently, using only 30% of data on UltraFeedback / HelpSteer consistently outperforms full-data alignment.

**[Disentangling Length Bias in Preference Learning via Response-Conditioned Modeling](disentangling_length_bias_in_preference_learning_via_response-conditioned_modeli.md)**

:   This paper transforms the implicit "length bias" in reward models into explicit "length instruction understanding." It proposes the Response-conditioned Bradley-Terry (Rc-BT) model—fixing the response and comparing different prompts—to simultaneously eliminate length cheating and enable the model to follow length instructions. This approach integrates seamlessly with Reward Modeling (Rc-RM) and DPO (Rc-DPO).

**[Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences](displacement-resistant_extensions_of_dpo_with_nonconvex_f-divergences.md)**

:   It is discovered that the solvability of f-DPO does not require $f$ to be convex (only $\lim_{t\to 0^+} f'(t) = -\infty$). Furthermore, it is proven that $\arg\min f(t) \geq 1$ is a necessary condition to resist probability displacement. Based on this, SquaredPO ($f(t) = \frac{1}{2}(\log t)^2$, non-convex) is proposed, which significantly alleviates the decline in winner probability while maintaining performance.

**[Don't Throw Away Your Pretrained Model](dont_throw_away_your_pretrained_model.md)**

:   The paper proposes SWITCH GENERATION: training a small "switcher" LM to dynamically select between pre-trained, fine-tuned, and aligned checkpoints as "speakers" for token fragments during a single response generation. This allows the complementarity of base capabilities lost during alignment (creativity, calibration, diversity) and capabilities gained through alignment (reasoning, instruction following), achieving a 31% average improvement over single models across 18 datasets and a 12.9% further gain over 8 types of collaboration baselines.

**[EigenBench: A Comparative Behavioral Measure of Value Alignment](eigenbench_a_comparative_behavioral_measure_of_value_alignment.md)**

:   EigenBench proposes a black-box, ground-truth-free value alignment measurement method: a population of language models evaluates each other's responses under a given "constitution" (value criteria). EigenTrust is used to aggregate these pairwise evaluations into a consensus score vector, where "more aligned models receive higher evaluative weight," ultimately outputting an Elo ranking of alignment for each model relative to that value system.

**[ELEPHANT: Measuring and Understanding Social Sycophancy in LLMs](elephant_measuring_and_understanding_social_sycophancy_in_llms.md)**

:   This paper extends LLM sycophancy from "agreeing with false facts" to "excessively maintaining user face," proposing a social sycophancy theoretical framework. It constructs the ELEPHANT benchmark to evaluate 11 major LLMs, finding they are on average 47 percentage points more sycophantic than humans in daily advice queries. The study reveals that sycophantic tendencies are rewarded in preference datasets and provides mitigation strategies such as prompt rewriting and DPO.

**[Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance](eliminating_inductive_bias_in_reward_models_with_information-theoretic_guidance.md)**

:   DIR formalizes reward model debiasing as an information-theoretic optimization problem—maximizing the mutual information between "reward prediction ↔ human preference" while minimizing it between "reward latent representation ↔ bias attributes." Using Barber-Agakov (BA) lower bounds and CLUB upper bounds for variational estimation, it unifiedly handles non-linear inductive biases such as length, sycophancy, and formatting.

**[Enforcing Axioms for AI Alignment under Loss-Based Rules](enforcing_axioms_for_ai_alignment_under_loss-based_rules.md)**

:   Under a linear social choice framework, loss-based reward models (including polynomial rewards) fail to guarantee Pareto Optimality (PO), but PO can be recovered in the limit when training data uniformly covers the embedding space—offering a provable data design for constitutional-style alignment.

**[Enhancing Trustworthiness of Fine-Tuned LLMs via Regularized Subset Selection](enhancing_trustworthiness_of_fine-tuned_llms_via_regularized_subset_selection.md)**

:   Addressing the decline in LLM trustworthiness caused by Supervised Fine-Tuning (SFT), this paper proposes a two-stage repair framework: first identifying "harmful training samples" using DPP-regularized subset selection, and then repairing the model via Proximal Bregman Response Function (PBRF) gradient ascent. This approach achieves up to a 21% improvement in trustworthiness at a cost of $\le 1\%$ perplexity.

**[Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment](evaluating_and_improving_cultural_awareness_of_reward_models_for_llm_alignment.md)**

:   This paper introduces the CARB (Cultural Awareness Reward Model) benchmark to systematically evaluate the preference judgment capabilities of reward models across 10 cultures and 4 cultural domains. Furthermore, it proposes "Think-as-Locals," which mandates generative reward models to first produce local cultural evaluation criteria before making a judgment. Optimized via RLVR/GRPO, this approach reduces spurious correlations caused by surface linguistic cues.

**[Fluent Alignment with Disfluent Judges: Post-training for Lower-Resource Languages](fluent_alignment_with_disfluent_judges_post-training_for_lower-resource_language.md)**

:   This paper proposes a post-training methodology for low-resource languages: it avoids target-language instruction data entirely, relying solely on on-policy reinforcement learning to learn from the model's own sampled responses. This enables the training of a linguistically authentic aligned model even with a "disfluent" judge—the core principle being "never exposing the model to translationese during training."

**[FSPO: Few-Shot Optimization of Synthetic Preferences Effectively Personalizes to Real Users](fspo_few-shot_optimization_of_synthetic_preferences_effectively_personalizes_to_.md)**

:   Reward modeling is reformulated as a "user-as-task" black-box meta-learning problem. LLMs use few-shot in-context preferences to rapidly infer personalized reward functions. Combined with a million-scale synthetic preference dataset (emphasizing diversity and structure), the approach enables Sim2Real transfer to real users, achieving a 70% win rate against humans in open-ended QA.

**[General Exploratory Bonus for Optimistic Exploration in RLHF](general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)**

:   It is theoretically proven that existing RLHF exploratory bonuses under KL and α-divergence regularization actually guide the policy toward high-probability regions of the reference model (contradicting the principle of optimism). This paper proposes the General Exploratory Bonus (GEB) framework, which counteracts the conservative bias of divergence regularization through reference-model-dependent reward adjustment and is provably optimistic.

**[Group-Normalized Implicit Value Optimization for Language Models](group-normalized_implicit_value_optimization_for_language_models.md)**

:   GN-IVO treats LLM generation as a step-by-step decision process. It constructs a normalized reward distribution from a group of candidate responses under the same prompt and then matches this distribution using the prefix probability ratio of the current policy relative to the old policy. This provides fine-grained value signals for tokens or reasoning steps without training an explicit critic or value network.

**[Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)**

:   By constructing a KL-regularized surrogate objective and deriving the pairwise consistency condition, this work proves from first principles that group-relative REINFORCE (GRPO) is naturally an off-policy algorithm. Furthermore, through component isolation experiments, it finds that clipping is the key to training stability while importance sampling can be entirely removed. Under this unified framework, it re-interprets several seemingly independent algorithms such as Kimi OPMD and Meta AsymRE.

**[GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)**

:   This paper proposes GuardAlign, a training-free inference-time safety defense framework for Large Vision-Language Models (LVLMs). It utilizes Optimal Transport (OT) to precisely detect and mask unsafe regions in images and applies cross-modal attention calibration to prevent the influence of safety prefixes from decaying. GuardAlign reduces the unsafe response rate by up to 39% across six LVLMs while maintaining or even enhancing general capabilities.

**[GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods](guidedbench_measuring_and_mitigating_the_evaluation_discrepancies_of_in-the-wild.md)**

:   By systematically measuring 37 jailbreak studies, this paper reveals that existing jailbreak evaluations are severely distorted due to a "lack of case-specific standards." It proposes GuidedBench—an evaluation system with per-question scoring guidelines that transforms the subjective judgment of "whether a jailbreak succeeded" into an objective check of "whether guideline points were hit," reducing inter-evaluator variance by at least 76.03%.

**[Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)**

:   This paper reveals the "historical context inconsistency" problem in stepwise group-based RL (such as GRPO/GiGPO), where steps within the same group may have different historical contexts, leading to biased advantage estimation. HGPO is proposed to achieve low-bias, balanced-variance advantage estimation through hierarchical grouping and adaptive weighting, achieving significant improvements on ALFWorld and WebShop with minimal extra overhead (<0.001%).

**[Holdout-Loss-Based Data Selection for LLM Finetuning via In-Context Learning](holdout-loss-based_data_selection_for_llm_finetuning_via_in-context_learning.md)**

:   By using In-Context Learning (treating the holdout set as in-context examples) to approximate "the holdout loss brought by training on a specific sample," the proposed method scores and dynamically reweights each fine-tuning sample without needing a reference model or retraining. This consistently improves alignment for SFT/DPO/SimPO with an additional overhead of only approximately 1.5%.

**[Humanline: Online Alignment as Perceptual Loss](humanline_online_alignment_as_perceptual_loss.md)**

:   This paper explains "why online alignment is superior to offline alignment" using **Prospect Theory** from behavioral economics—online on-policy sampling is closer to the subjective human perception distribution of model outputs. Furthermore, the clipping mechanism in PPO/GRPO implicitly recovers this perceptual bias, making them essentially "perceptual losses." Based on this, a design paradigm (humanline variants) is proposed to explicitly inject perceptual distortion into DPO/KTO/GRPO, matching online performance with offline data while training up to 6× faster.

**[IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment](ideal_data_equilibrium_adaptation_for_multi-capability_language_model_alignment.md)**

:   IDEAL models the problem of "how much data to allocate for each SFT domain" as a bilevel optimization problem. It utilizes second-order (Hessian) gradient information to determine whether each domain's data should be upsampled or downsampled. Iterating for two rounds results in a balanced overall improvement of approximately 7% across four capabilities: Math, Code, Reasoning, and Instruction Following.

**[Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment](inverse_reinforcement_learning_with_dynamic_reward_scaling_for_llm_alignment.md)**

:   DR-IRL utilizes Inverse Reinforcement Learning (IRL) to train category-specific shadow reward models from "balanced safety demonstration data." It then scales the advantage function in GRPO with a dynamic coefficient determined by both "data difficulty" and "model responsiveness." This concentrates optimization efforts on long-tail, high-difficulty harmful samples, significantly enhancing safety alignment without sacrificing (and even improving) general capabilities.

**[Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)**

:   This paper challenges the consensus that "on-policy data is always better." It discovers that the alignment process is divided into two stages: preference injection (requiring high-diversity off-policy data) and preference fine-tuning (requiring high-quality on-policy data). The optimal data type varies by model and stage. A boundary determination algorithm with only 3.2% computational overhead is proposed and validated across 5 models and 55 configurations.

**[JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)**

:   This paper proposes JailNewsBench, the first multilingual and multi-regional benchmark to evaluate the robustness of LLMs against fake news generation under jailbreak attacks. Covering 34 regions and 22 languages with approximately 300,000 instances, it reveals attack success rates up to 86.3% and uncovers a safety imbalance where defenses for English/US topics are significantly weaker than those for other regions.

**[JULI: Jailbreak Large Language Models by Self-Introspection](juli_jailbreak_large_language_models_by_self-introspection.md)**

:   This paper reveals the knowledge leakage issue where top-k token log probabilities of aligned LLMs still contain harmful information. It proposes JULI—a BiasNet plugin with less than 1% of the target model's parameters that manipulates logit bias. In API scenarios with access to only top-5 token probabilities, it successfully jailbreaks Gemini-2.5-Pro (Harmful Info Score 4.19/5), achieving a 140x speedup over LINT while doubling the harmfulness.

**[Keep the Best, Forget the Rest: Reliable Alignment with Order-Aware Preference Optimization](keep_the_best_forget_the_rest_reliable_alignment_with_order-aware_preference_opt.md)**

:   RAPPO uses the reference policy to assign "credibility" scores to samples within each batch, temporarily excluding high-loss preference pairs where the reference model itself is misaligned and the samples are the hardest to learn. By modifying DPO with just a few lines of code, it consistently outperforms baselines like SimPO/DPO in sentiment, detoxification, summarization, and safety alignment, while providing a tighter generalization bound.

**[Learning from Noisy Preferences: A Semi-Supervised Learning Approach to Direct Preference Optimization](learning_from_noisy_preferences_a_semi-supervised_learning_approach_to_direct_pr.md)**

:   Addressing the gradient conflict problem in Diffusion-DPO caused by multi-dimensional human visual preferences being compressed into a binary label, this paper proposes Semi-DPO. By treating samples agreed upon by multiple reward models as clean labels and dimensionally conflicting samples as noisy unlabeled data, the method uses the diffusion model itself as an implicit classifier to generate pseudo-labels across different timesteps for iterative self-training. It achieves SOTA alignment performance without introducing additional human annotations or explicit reward models.

**[Learning More with Less: A Dynamic Dual-Level Down-Sampling Framework for Efficient Policy Optimization](learning_more_with_less_a_dynamic_dual-level_down-sampling_framework_for_efficie.md)**

:   The authors propose the **D3S** (Dynamic Dual-Level Down-Sampling) framework, which maximizes advantage variance at the sample level and prioritizes high-entropy + high-advantage tokens at the token level. Combined with a dynamic scheduling strategy, it achieves faster convergence and superior performance using fewer than 20% of tokens.

**[Learning Ordinal Probabilistic Reward from Preferences (OPRM)](learning_ordinal_probabilistic_reward_from_preferences.md)**

:   This paper proposes the Ordinal Probabilistic Reward Model (OPRM), which discretizes response quality into ordinal levels 1-9 and learns the full probability distribution. Combined with Region Flooding Tuning (RgFT), it achieves data-efficient training. It reaches 89.3% on RewardBench, an improvement of 2.9%-7.4% over existing RMs, while providing uncertainty estimation and label disagreement detection.

**[Learning to Summarize User Information for Personalized RLHF (PLUS)](learning_to_summarize_user_information_for_personalized_reinforcement_learning_f.md)**

:   PLUS utilizes RL (PPO) to train a "user summarizer" that compresses each user's preferences, characteristics, and conversation history into a **natural language summary** $z$. This summary conditions the reward model, and both components undergo online co-adaptation. This approach improves reward model accuracy by 11–77% relative to Bradley-Terry without assuming "identical preferences for all users."

**[Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)**

:   This paper proposes NSPO, which projects safety alignment policy gradients into the null space of general task representations. This ensures, from a geometric perspective, that safety optimization does not damage general capabilities. Using only 40% of safety data, it achieves SOTA on 7 safety benchmarks with almost no performance loss in mathematics, coding, or instruction following.

**[Multi-objective Large Language Model Alignment with Hierarchical Experts](multi-objective_large_language_model_alignment_with_hierarchical_experts.md)**

:   HoE decomposes multi-objective alignment into a series of "single-preference subproblems" using a three-layer Mixture-of-Experts consisting of training-free extracted LoRA experts, lightweight routing experts, and parameter-free preference routing. It covers the entire Pareto front in a plug-and-play manner without retraining the backbone, responding to arbitrary user preference weights.

**[Multiplayer Nash Preference Optimization](multiplayer_nash_preference_optimization.md)**

:   This work generalizes Nash learning from human feedback (NLHF) from "two-player games" to "n-player games," allowing a policy to simultaneously compete against an entire population of opponents (historical checkpoints or multiple heterogeneous reward models). By using multiplicative weights updates to find approximate Nash equilibria, the method more stably and comprehensively captures non-transitive and heterogeneous human preferences in the real world.

**[No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping](no_prompt_left_behind_exploiting_zero-variance_prompts_in_llm_reinforcement_lear.md)**

:   It is discovered that many "zero-variance prompts" (where all sampled responses are either all correct or all incorrect) are discarded during GRPO training. The RL-ZVP algorithm is proposed to extract learning signals from these prompts via entropy-guided advantage shaping, achieving improvements of up to 8.61 accuracy points and 7.77 pass rate points across six mathematical reasoning benchmarks compared to GRPO.

**[Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search](obscure_but_effective_classical_chinese_jailbreak_prompt_optimization_via_bio-in.md)**

:   Ours proposes the CC-BOS framework, which leverages the semantic compression and ambiguity of Classical Chinese, combined with the Fruit Fly Optimization Algorithm (FOA), to search for optimal jailbreak prompts within an eight-dimensional strategy space, achieving near 100% attack success rates across six mainstream LLMs.

**[Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences](omni-reward_towards_generalist_omni-modal_reward_modeling_with_free-form_prefere.md)**

:   This paper proposes Omni-Reward, which extends reward modeling from "text/image only with fixed binary preferences" to "covering text/image/video/audio/3D with dynamic scoring based on free-form text preferences." It introduces a unified benchmark (Omni-RewardBench, 5 modalities, 9 tasks), a large-scale preference dataset (Omni-RewardData, 248K general + 69K instruction-tuned pairs), and two reward models (Discriminative BT version + Generative R1 version). Omni-Reward achieves a 20% gain over its base model on its own benchmark and reaches or exceeds SOTA on public leaderboards like VL-RewardBench.

**[On the Shelf Life of Fine-Tuned LLM-Judges: Future-Proofing, Backward-Compatibility, and Question Generalization](on_the_shelf_life_of_fine-tuned_llm-judges_future-proofing_backward-compatibilit.md)**

:   This paper formalizes the problem of "how long a fine-tuned LLM judge remains effective" as a **dual-distribution (question distribution $\times$ response distribution) shift** problem. Through systematic experiments on two reasoning datasets, three training recipes, and three backbones, it finds that judges struggle with "future-proofing" (significant performance drop on responses from stronger new models) but achieve "backward-compatibility" relatively easily (minimal drop on weaker legacy responses). Continual learning achieves a more balanced adaptation between old and new distributions, whereas all judges generalize poorly to new questions unseen during training.

**[Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks](optimal_sparsity_of_mixture-of-experts_language_models_for_reasoning_tasks.md)**

:   This study systematically investigates how the sparsity of Mixture-of-Experts (MoE) language models affects memory-intensive and reasoning-intensive tasks differently: memory tasks prefer higher sparsity (more total parameters), whereas reasoning tasks reach optimality near $\text{TPP} \approx 20$. This trend remains invariant even after GRPO post-training and increased test-time compute.

**[OrthAlign: Orthogonal Subspace Decomposition for Non-Interfering Multi-Objective Alignment](orthalign_orthogonal_subspace_decomposition_for_non-interfering_multi-objective_.md)**

:   Addressing the multi-objective alignment dilemma where "improving one preference harms another," OrthAlign constrains parameter updates of different preferences into mutually orthogonal subspaces. This ensures that optimization directions for each preference are mathematically non-interfering, achieving simultaneous alignment of helpful/harmless/truthful without sacrificing individual performance. It achieves a maximum single-item improvement of 50.89% and an average overall reward increase of 13.96%.

**[PALC: Preference Alignment via Logit Calibration](palc_preference_alignment_via_logit_calibration.md)**

:   PALC attaches a minimal "calibration module" to a frozen LLM, moving the alignment intervention from the entangled latent space to the naturally decoupled vocabulary logit space. By treating hidden states as read-only context to generate position-dependent logit offsets, it achieves adjustable preference alignment at test time with only 0.002%–0.13% extra parameters and almost no inference overhead.

**[Pretrain Value, Not Reward: Decoupled Value Policy Optimization](pretrain_value_not_reward_decoupled_value_policy_optimization.md)**

:   The authors argue that under fixed preference data, "training a reward model followed by online critic learning" is informationally equivalent to "directly pretraining a value model." Consequently, they propose DVPO: pretraining a Global Value Model (GVM) offline and freezing it as a universal critic to guide policy optimization. This eliminates online critic training, matches or exceeds mainstream RLHF methods on MT-Bench / Alpaca-Eval / Arena-Hard, while saving 30–40% VRAM and 30–45% training time.

**[Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead](quagmires_in_sft-rl_post-training_when_high_sft_scores_mislead_and_what_to_use_i.md)**

:   This paper demonstrates, through experiments involving over 100 models and 1 million GPU hours, that the common assumption in reasoning LLM post-training—"higher SFT scores lead to better RL performance"—is a widespread fallacy. It proposes **Validation Set Generalization Loss** and **Pass@large k** as reliable indicators to predict final RL performance, improving prediction accuracy ($R^2$, Spearman rank correlation) by up to 0.5 (approximately 2x) compared to using SFT scores directly.

**[RE-PO: Robust Enhanced Policy Optimization as a General Framework for LLM Alignment](re-po_robust_enhanced_policy_optimization_as_a_general_framework_for_llm_alignme.md)**

:   RE-PO treats the "correctness" of each preference label as a latent variable, utilizing the EM algorithm during training to update the policy while simultaneously inferring the confidence of each data point to perform adaptive downweighting of noisy preference data. It unifies a broad class of preference losses (DPO, IPO, SimPO, CPO, etc.) into the same probabilistic framework, enabling them to be "robustified," yielding improvements of up to 7.0 percentage points on AlpacaEval 2.

**[Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)**

:   This paper proposes the "Answer-Then-Check" strategy: the model first generates an intended answer summary within its chain-of-thought, performs a safety analysis based on safety policies, and finally decides whether to output or refuse. After training on the constructed 80K ReSA dataset, the defense rate reaches 99.3% (RL version) against 7 types of jailbreak attacks, with 500 samples being sufficient to achieve performance comparable to the full dataset.

**[RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data](recast_expanding_the_boundaries_of_llms_complex_instruction_following_with_multi.md)**

:   RECAST performs reverse mining of verifiable constraints from real "instruction-response" pairs and reassembles them into high-complexity training data (RECAST-30K, 30K samples / 19 constraint types), with more than ten constraints per instruction, supported by a dual-track rule and model-based verifier. SFT using this data enables small models to surpass much larger ones in complex instruction following; further applying Reinforcement Learning with Verifiable Constraints (RLVC) using "constraint satisfaction rate" as a reward provides additional gains without damaging general capabilities.

**[Reward Model Routing in Alignment](reward_model_routing_in_alignment.md)**

:   This paper proposes **BayesianRouter**, a hybrid routing framework that **selects the most suitable reward model for each preference pair** during RLHF/Online DPO training. In the offline phase, a multi-task router is trained on preference data to learn the expertise areas of various reward models (RMs). During the online phase, Bayesian Thompson Sampling is used to select models per query, injecting offline-learned strengths as Gaussian priors. This allows the router to adapt to policy distributions through linear updates during alignment. The method consistently outperforms single RMs, RM ensembles, and the existing LASER routing method on instruction-following and reasoning benchmarks.

**[Reward Models Inherit Value Biases from Pretraining](reward_models_inherit_value_biases_from_pretraining.md)**

:   This paper employs an interpretability method of "exhaustive token search + psycholinguistic corpora" to systematically examine 10 mainstream open-source Reward Models (RMs). It finds that RM preferences across multiple human value dimensions—such as "agency vs. communion"—highly depend on the base LLM (Llama series prefers agency; Gemma series prefers communion). These biases are traced back to the log-probabilities of the base models, proving that they are difficult to "wash away" during the preference fine-tuning process.

**[RewardBench 2: Advancing Reward Model Evaluation](rewardbench_2_advancing_reward_model_evaluation.md)**

:   This paper introduces RewardBench 2, a reward model evaluation benchmark utilizing completely new, unseen human prompts and transitioning from a "1-vs-1" to a "1-vs-3 (1 positive, 3 negatives)" format. Covering six major domains (including new areas like Ties, Precise IF, and Factuality), it is on average 20 points more difficult than the original RewardBench and exhibits significantly stronger correlation with downstream applications such as best-of-N sampling and PPO training.

**[RLBFF: Binary Flexible Feedback to Bridge Between Human Feedback & Verifiable Rewards](rlbff_binary_flexible_feedback_to_bridge_between_human_feedback_verifiable_rewar.md)**

:   This paper proposes RLBFF (Reinforcement Learning with Binary Flexible Feedback), which extracts "binary-answerable principles" from natural language feedback (e.g., "Information accuracy: Yes", "Code readability: No"). It reformulates reward model training as an entailment task—determining whether a response satisfies a specific principle—thereby achieving the broad coverage of RLHF and the interpretability/reward-hacking resistance of RLVR. The resulting scalar reward model outperforms Bradley-Terry models on RM-Bench (83.6) and JudgeBench (76.3). A GenRM further pushes RM-Bench/JudgeBench to 86.2/81.4 (SOTA), and is used to align Qwen3-32B to a level comparable to o3-mini/DeepSeek R1 with less than 5% of the inference cost.

**[Robust Preference Alignment via Directional Neighborhood Consensus](robust_preference_alignment_via_directional_neighborhood_consensus.md)**

:   The authors propose Robust Preference Selection (RPS), a training-free inference-time method for enhancing preference alignment. By sampling multiple candidate directions from the local neighborhood of the target preference to generate responses and selecting the optimal one according to the original preference, RPS achieves up to a 69% win rate over baselines on OOD preferences.

**[Robust Reward Modeling via Causal Rubrics](robust_reward_modeling_via_causal_rubrics.md)**

:   Addressing the issue where reward models (RM) exploit spurious features like length and format, CROME utilizes an Oracle LLM to list "causal rubrics" that determine true quality for each prompt. It then synthesizes two types of counterfactual data: "causal augmentation" (upgrading/degrading along a single causal attribute) and "neutral augmentation" (pairing answers with irrelevant questions). Combined with a composite loss, this makes the RM sensitive to causal attributes and invariant to unknown spurious ones, achieving an average improvement of 5.3% on RewardBench (+12.4% in Safety, +7.1% in Reasoning).

**[ROSETTA: Constructing Code-Based Reward from Unconstrained Language Preference](rosetta_constructing_code-based_reward_from_unconstrained_language_preference.md)**

:   ROSETTA decomposes spontaneous, time-varying natural language preferences in robot interactions into three steps: "preference grounding, reward staging, and code generation/verification," generating online trainable code reward functions that achieve an 87% success rate and 86% human satisfaction across 116 preferences.

**[SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](safedpo_preference_optimization_safety.md)**

:   By revisiting the safety-constrained RLHF objective and proving it possesses a closed-form optimal policy, this work derives an equivalent tractable objective, SafeDPO. It requires only safety-aware data transformation and a safety margin term (one additional hyperparameter) on top of standard DPO. Without needing reward/cost models, it achieves a 96.87% harmless rate on PKU-SafeRLHF-30K while maintaining competitive helpfulness, with training speeds 25x faster than SafeRLHF.

**[Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study](safety_subspaces_are_not_linearly_distinct_a_fine-tuning_case_study.md)**

:   This paper comprehensively validates a key finding across five open-source LLMs through four systematic experiments (parallel projection, orthogonal projection, subspace overlap, and activation space analysis): safety alignment behaviors are highly entangled with general learning in both the weight space and activation space. There is no linearly separable independent safety subspace, indicating that defense strategies based on subspace projection/filtering face fundamental limitations.

**[SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)**

:   The SEMA framework is proposed, which trains an attacker capable of automatically generating multi-turn jailbreak attacks through two-stage training: prefilling self-tuning and RL with intent-drift-aware rewards. Without requiring any existing attack strategies or external data, it achieves an average ASR@1 of 80.1% across three victim models on AdvBench, surpassing SOTA by 33.9%.

**[Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](semantic-aware_wasserstein_policy_regularization_for_large_language_model_alignm.md)**

:   This paper identifies that standard KL divergence regularization in RLHF only compares token probabilities at the same index while ignoring semantic similarity. It proposes Semantic-aware Wasserstein Policy Regularization (WPR) based on entropy-regularized Wasserstein distance. By leveraging a dual formulation, the regularization is transformed into a token-level penalty term, which consistently outperforms KL and various f-divergence baselines in dialogue generation and summarization tasks.

**[Semi-Supervised Preference Optimization with Limited Feedback](semi-supervised_preference_optimization_with_limited_feedback.md)**

:   SSPO reformulates preference optimization as a probabilistic classification problem. It learns a reward threshold from a small amount of paired preference labels that can reliably separate "winning" and "losing" responses. This threshold is then used to assign pseudo-labels to a massive amount of unpaired samples (e.g., SFT data), which are jointly trained using a curriculum scheduling. Using only 1% of UltraFeedback, SSPO consistently outperforms strong baselines trained on 10% of the data.

**[Sharpness-Aware Minimization in Logit Space Efficiently Enhances Direct Preference Optimization](sharpness-aware_minimization_in_logit_space_efficiently_enhances_direct_preferen.md)**

:   This paper explains the "squeezing effect" in DPO training (where the probability of preferred responses paradoxically decreases) from the perspective of logit space dynamics, identifying that negative gradients cause residuals to expand wildly along high-curvature directions. The authors prove that the curvature regularization of SAM effectively suppresses this expansion and implement a near-zero-overhead "logits-SAM" by perturbing only the output layer, providing consistent improvements for DPO and its variants on Pythia-2.8B, Mistral-7B, and Gemma-2B-IT.

**[Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)**

:   A two-stage Human-AI synergistic preference data curation pipeline is proposed: Phase 1 accumulates approximately 1M preference pairs through 8 iterations of human verification, error-driven adaptive retrieval, and preference-guided LLM annotation; Phase 2 scales the data to 26M pairs using dual-RM consistency filtering. The resulting Skywork-Reward-V2 8B model achieves 97.8% on RewardBench and an average of 88.6% across seven major benchmarks, outperforming all open-source 70B reward models.

**[Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)**

:   The authors propose Spectrum Tuning, a post-training method that improves in-context steerability, output space coverage, and distributional alignment by training on a distribution-fitting dataset across 90+ tasks. The work reveals that current instruction tuning impairs the in-context steerability of language models.

**[Stackelberg Learning from Human Feedback: Preference Optimization as a Sequential Game](stackelberg_learning_from_human_feedback_preference_optimization_as_a_sequential.md)**

:   Ours remodels LLM preference alignment as a "Leader-Follower" **sequential game** (SLHF): the Leader first commits to a response, and the Follower provides an improved version after observing this response. This naturally yields a deterministic equilibrium robust to non-transitive preferences and supports **training-free iterative self-refinement at inference time**, consistently outperforming RLHF (RLOO) and NLHF (Nash-MD-PG) baselines on 0.5B–8B models.

**[StoryAlign: Evaluating and Training Reward Models for Story Generation](storyalign_evaluating_and_training_reward_models_for_story_generation.md)**

:   This paper demonstrates that existing reward models struggle to identify human-preferred stories (even the strongest GPT-4o achieves only 66.3% accuracy). Consequently, the authors construct the first story preference evaluation benchmark, STORYRMB (1133 human-verified instances), and train a specialized reward model, STORYREWARD, using approximately 100,000 automatically constructed preference pairs. At an 8B scale, STORYREWARD achieves a SoTA accuracy of 75.0% on the benchmark and significantly outperforms other reward models in Best-of-N test-time scaling.

**[Superficial Safety Alignment Hypothesis](superficial_safety_alignment_hypothesis.md)**

:   Proposes the "Superficial Safety Alignment Hypothesis" (SSAH): safety alignment essentially teaches the model to perform an implicit binary classification task (execute vs. refuse). Only ~1.3% of neurons are required to establish safety guardrails; freezing these safety-critical units maintains safety during fine-tuning, and utilizing redundant units as an "alignment budget" can eliminate the alignment tax.

**[Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)**

:   Proposes Sysformer, a lightweight Transformer module pluggable to the front-end of any frozen LLM. It adaptively transforms system prompts in the embedding space based on user input, enabling the model to reject harmful requests while responding normally to safe ones without modifying LLM parameters or filtering user inputs.

**[Test-Time Alignment for Large Language Models via Textual Model Predictive Control](test-time_alignment_for_large_language_models_via_textual_model_predictive_contr.md)**

:   This paper reformulates the test-time preference alignment of LLMs as a trajectory optimization problem, utilizing Model Predictive Control (MPC) from control theory for "planning while moving." By employing **hindsight subgoal identification** to extract high-reward segments from generated rollouts as waypoints and performing **conditioned regeneration**, the method achieves rolling approximation of the optimum. It stably improves performance across machine translation, long-form response, and code generation tasks without modifying model parameters.

**[Text2Grad: Reinforcement Learning from Natural Language Feedback](text2grad_reinforcement_learning_from_natural_language_feedback.md)**

:   The paper aligns free-form textual criticism with output token segments, converts them into token-level pseudo-rewards, and constructs "Natural Language Gradients" to drive PPO updates. This approach ensures the model modifies only the "criticized tokens" rather than making global haphazard adjustments. It outperforms scalar reward RL and pure prompting-based reflection baselines across summarization, code generation, and question-answering tasks.

**[The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives](the_alignment_auditor_a_bayesian_framework_for_verifying_and_refining_llm_object.md)**

:   The study reformulates "recovering implicit LLM rewards via Inverse Reinforcement Learning (IRL)" from a one-off point estimation into a Bayesian auditing workflow. It first recovers the **posterior distribution** of rewards rather than a single point using variational inference, then shrinks the posterior round-by-round via sequential Bayesian updates. Epistemic uncertainty is employed to diagnose shortcuts and out-of-distribution inputs. Finally, the study demonstrates that the shrunken, low-uncertainty reward can be fed back into RLHF to replicate the alignment effects of an oracle reward (showing nearly identical toxicity reduction curves).

**[Token-Importance Guided Direct Preference Optimization (TI-DPO)](token-importance_guided_direct_preference_optimization.md)**

:   This paper proposes TI-DPO, which precisely quantifies the contribution of each token to preferences through a hybrid weighting mechanism (gradient attribution + Gaussian prior), combined with a triplet loss to guide optimization in continuous semantic space. It achieves SOTA with an average score of 62.3 across 6 benchmarks and provides interpretable token-level control.

**[Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)**

:   The authors propose UltraBreak, which utilizes semantic adversarial objectives (replacing cross-entropy with cosine similarity to optimize for a smooth loss landscape) combined with input space constraints (random transformations + TV regularization to generate transform-invariant features). By training a single universal adversarial image, jailbreaks can be achieved across more than 6 VLM architectures and commercial models, reaching a black-box average ASR of 71% (SafeBench), significantly surpassing prior methods.

**[Towards Self-Robust LLMs: Intrinsic Prompt Noise Resistance via CoIPO](towards_self-robust_llms_intrinsic_prompt_noise_resistance_via_coipo.md)**

:   This paper proposes CoIPO (Contrastive Learning + Inverse DPO) to ensure LLMs produce outputs consistent with clean prompts when facing noisy prompts (typos, word substitutions, syntactic perturbations). Without relying on external pre-processing tools, it embeds intrinsic robustness into the model during training, outperforming the current SOTA (CoIN) by an average accuracy of 3.64% on the self-constructed NoisyPromptBench.

**[Towards Understanding Valuable Preference Data for Large Language Model Alignment](towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)**

:   This paper investigates preference data quality from a model-dependent perspective. It proposes the Truncated Influence Function (TIF), revealing that medium-IF data is the most valuable (contrary to the classical view favoring high-IF data). Two lightweight proxy metrics, LossDiff and IRM, are designed to approximate TIF. Their combination, the LossDiff-IRM selector, achieves an average WinRate improvement of 13.58% using only 50-64% of the data, demonstrating effectiveness across multiple LLM families and alignment benchmarks.

**[Truthful or Fabricated? Using Causal Attribution to Mitigate Reward Hacking in Explanations](truthful_or_fabricated_using_causal_attribution_to_mitigate_reward_hacking_in_ex.md)**

:   The paper argues that preference optimization (DPO/RLHF) incentivizes LLMs to "clandestinely use" forbidden input cues while failing to acknowledge them, leading to unfaithful Chain-of-Thought (CoT) explanations. The authors detect this cue dependency via counterfactual causal attribution and inject this signal into the reward model input as "disclaimers," significantly reducing the incidence of CoT hacking in two controlled settings.

**[TS²: Sparsemax+ for Training and Softmax for Testing for Accurate and Diverse LLM Fine-tuning](ts2_training_with_sparsemax_testing_with_softmax_for_accurate_and_diverse_llm_fi.md)**

:   Addressing the issue where Cross-Entropy (CE) supervised fine-tuning collapses probability distributions into one-hot vectors and crushes output diversity, this paper proposes TS²: employing a Sparsemax+ loss with tail suppression (sparse support + explicit tail pruning) during training while reverting to softmax decoding during inference. This approach enhances accuracy and diversity for Llama-3.1-8B / Qwen-2.5-7B across chat, code, and open-ended generation without altering model architecture.

**[Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](unifying_stable_optimization_and_reference_regularization_in_rlhf.md)**

:   The authors propose DAR (Dual-regularized Advantage Regression). They observe that in standard RLHF, reference model regularization (to prevent reward hacking) and policy stability constraints (to prevent collapse) progressively conflict, leading to an overly restricted optimization space. By defining a dual KL objective that interpolates the reference policy in log-space and applying a regression transformation to eliminate policy-ratio instability, DAR achieves a 92.42% average win rate in direct AI alignment and standard RLHF settings, outperforming GRPO by 7.27%.

**[Verification and Co-Alignment via Heterogeneous Consistency for Preference-Aligned LLM Annotations](verification_and_co-alignment_via_heterogeneous_consistency_for_preference-align.md)**

:   This paper proposes Heterogeneous-Consistency Co-Alignment (HCC), which utilizes the consistency/inconsistency relationship between LLMs and task-specific embedding models to verify the reliability of LLM annotations in reference-free semi-supervised NLU scenarios. It further rectifies preference-inconsistent samples through two rounds of co-alignment based on nearest neighbor voting.

**[Weak-to-Strong Generalization with Failure Trajectories](weak-to-strong_generalization_with_failure_trajectories.md)**

:   This paper extends "Weak-to-Strong Generalization" (W2SG) from binary classification to multi-step interactive decision-making tasks. A weak model explores numerous action trajectories containing both successes and failures, which are merged into a "Trajectory Tree" based on common prefixes. Structured preference pairs (TreeDPO) or offline MCTS path search are then used to fine-tune a strong model. The resulting strong model not only outperforms the SFT weak model across three Agent environments but even surpasses the SFT strong model trained on expert data.

**[What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data](whats_in_my_human_feedback_learning_interpretable_descriptions_of_preference_dat.md)**

:   WIMHF uses Sparse Autoencoders (SAEs) on the "embedding difference between two candidate responses" to learn a small set of human-readable features. It then quantifies the impact of each feature on preference labels using logistic regression. This process automatically characterizes what a preference dataset "can measure" and what "annotators actually prefer" without pre-defined hypotheses, providing controllable levers for data sanitation and personalization.

**[When Data Is the Algorithm: A Systematic Study and Curation of Preference Optimization Datasets](when_data_is_the_algorithm_a_systematic_study_and_curation_of_preference_optimiz.md)**

:   This paper conducts the first systematic "sample-level" horizontal audit of 5 commonly used open-source DPO preference datasets. By using Magpie to annotate task category/difficulty/input quality and an independent reward model to assign "preference reward" scores to each pair, the authors find that 20–30% of samples contain "chosen responses that are actually inferior to the rejected ones." Based on these diagnostic signals, a curated mixture set, UltraMix, is designed, which is 30% smaller than the strongest single dataset yet achieves superior performance.

**[When Weak LLMs Speak with Confidence, Preference Alignment Gets Stronger](when_weak_llms_speak_with_confidence_preference_alignment_gets_stronger.md)**

:   Using a weak LLM of less than 0.5B parameters as a preference annotator and weighting the preference optimization target per sample based on its "confidence" (CW-PO) allows the method to surpass DPO trained with 100% human labels on multiple datasets using only 20%~30% of human annotations. It is compatible with various objectives such as DPO, IPO, and rDPO.

**[Why DPO is a Misspecified Estimator and How to Fix It](why_dpo_is_misspecified_estimator.md)**

:   This paper proves from an information geometry perspective that DPO is essentially a misspecified statistical estimation problem under parameterized (non-tabular) policy classes. It demonstrates that DPO projects the true reward function onto an implicit reward manifold via KL divergence, which leads to preference reversal and reward degradation when the reward is unachievable. The authors propose AuxDPO to fix this issue by introducing auxiliary variables in the null space.
