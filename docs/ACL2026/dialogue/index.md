---
title: >-
  ACL2026 Dialogue Systems Papers · 26 Notes
description: >-
  26 ACL2026 papers in the Dialogue Systems area, covering Dialogue, Reasoning, Sentiment Analysis, Agents, LLM, Alignment/RLHF and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ACL2026"
  - "Dialogue Systems"
  - "AI paper notes"
  - "paper summaries"
  - "Dialogue"
  - "Reasoning"
  - "Sentiment Analysis"
  - "Agents"
  - "LLM"
  - "Alignment/RLHF"
item_list:
  - u: "apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co/"
    t: "APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI"
  - u: "author-in-the-loop_response_generation_and_evaluation_integrating_author_experti/"
    t: "Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review"
  - u: "codebook-injected_dialogue_segmentation_for_multi-utterance_constructs_annotatio/"
    t: "Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation"
  - u: "codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm/"
    t: "CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment"
  - u: "cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti/"
    t: "Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation"
  - u: "context-agent_dynamic_discourse_trees_for_non-linear_dialogue/"
    t: "Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue"
  - u: "disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis/"
    t: "Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky"
  - u: "discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog/"
    t: "Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation"
  - u: "dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_/"
    t: "Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents"
  - u: "ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d/"
    t: "ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue"
  - u: "frame_of_reference_addressing_the_challenges_of_common_ground_representation_in_/"
    t: "Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue"
  - u: "genesisfunc_multi-agent_data_generation_for_accurate_and_generalizable_function-/"
    t: "GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling"
  - u: "locket_robust_feature-locking_technique_for_language_models/"
    t: "LOCKET: Robust Feature-Locking Technique for Language Models"
  - u: "ma2p_a_meta-cognitive_autonomous_intelligent_agents_framework_for_complex_persua/"
    t: "MA$^2$P: A Meta-Cognitive Autonomous Intelligent Agents Framework for Complex Persuasion"
  - u: "metro_towards_strategy_induction_from_expert_dialogue_transcripts_for_non-collab/"
    t: "Metro: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues"
  - u: "preference_learning_unlocks_llms_psycho-counseling_skills/"
    t: "Preference Learning Unlocks LLMs' Psycho-Counseling Skills"
  - u: "reactod_bounded_neuro-symbolic_agentic_nlu_for_zero-shot_dialogue_state_tracking/"
    t: "ReacTOD: Bounded Neuro-Symbolic Agentic NLU for Zero-Shot Dialogue State Tracking"
  - u: "reasoning_gets_harder_for_llms_inside_a_dialogue/"
    t: "Reasoning Gets Harder for LLMs Inside A Dialogue"
  - u: "simulated_students_in_tutoring_dialogues_substance_or_illusion/"
    t: "Simulated Students in Tutoring Dialogues: Substance or Illusion?"
  - u: "spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation/"
    t: "SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation"
  - u: "stress-testing_emotional_support_models_moving_from_homogeneous_to_diverse_help_/"
    t: "Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers"
  - u: "stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog/"
    t: "STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems"
  - u: "surprisal_minimisation_over_goal-directed_alternatives_predicts_production_choic/"
    t: "Surprisal Minimisation over Goal-directed Alternatives Predicts Production Choice in Dialogue"
  - u: "template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd/"
    t: "Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings"
  - u: "towards_proactive_information_probing_customer_service_chatbots_harvesting_value/"
    t: "Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation"
  - u: "your_students_dont_use_llms_like_you_wish_they_did/"
    t: "Your Students Don't Use LLMs Like You Wish They Did"
item_total: 26
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 🗣️ Dialogue Systems

**💬 ACL2026** · **26** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (1)](../../CVPR2026/dialogue/index.md) · [🔬 ICLR2026 (10)](../../ICLR2026/dialogue/index.md) · [🧪 ICML2026 (5)](../../ICML2026/dialogue/index.md) · [🤖 AAAI2026 (5)](../../AAAI2026/dialogue/index.md) · [🧠 NeurIPS2025 (8)](../../NeurIPS2025/dialogue/index.md)

🔥 **Top topics:** Dialogue ×18 · Reasoning ×3 · Sentiment Analysis ×3 · Agents ×3 · LLM ×2

**[APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)**

:   The proposed system constructs long-term conversational memory using a trio of "domain-agnostic ontology-supported property graphs + append-only event storage + ReAct multi-tool retrieval agents." By never overwriting during construction and resolving temporal conflicts only at retrieval, it achieves 88.88% on LOCOMO (3.5% higher than MIRIX) and 86.2% on LongMemEval (13.7% higher than the strongest RAG baseline).

**[Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)**

:   This work redefines academic author response (rebuttal) generation as an "Author-in-the-Loop" task, introducing the Re3Align dataset (3.4K papers, 440K sentence-level edit annotations, 15K review-response-revision triplets), the REspGen controllable generation framework, and the REspEval evaluation suite with 20+ metrics. The approach systematically validates the effects of author input, controllability, and evaluation-guided refinement across 5 state-of-the-art LLMs.

**[Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation](codebook-injected_dialogue_segmentation_for_multi-utterance_constructs_annotatio.md)**

:   The paper reformulates dialogue act annotation as a two-step "segment-then-label" problem. It proposes two approaches: codebook-injected LLM segmentation (System 1) and Dial-Start with DA-aware retrieval augmentation (System 2). It further introduces three categories of evaluation metrics that do not require gold boundaries (within-segment consistency, adjacent segment divergence, and human-AI distribution alignment). Experiments on TalkMoves and CLASS-annotated educational dialogues demonstrate that DA-aware prompting enables LLMs to produce more homogeneous segments, though coherence-based baselines and LLMs excel in different evaluation dimensions with no single optimal solution.

**[CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)**

:   Ours proposes CoDial, a framework that converts predefined dialogue flows (task schemas) into structured heterogeneous graphs and automatically generates LLM guardrail code (such as Colang). It achieves interpretable and controllable task-oriented dialogue policies during inference, reaching SOTA on the STAR benchmark without requiring training data.

**[Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)**

:   The CoPoLLM framework is proposed, which constructs the first Emotional Support Conversation (ESC) dataset with cognitive distortion labels, CogBiasESC. By combining a Cognitive Policy Reinforcement Learning (CPRL) engine and Dual-Stream Condition Optimization (DSCO), the LLM can diagnose 8 types of cognitive distortions and generate policy-aware intervention responses, consistently outperforming 15 SOTA baselines.

**[Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)**

:   The authors propose Context-Agent, which models multi-turn dialogue history as a "forest of discourse trees" (where each tree represents an independent topic and each branch represent an instruction refinement/fork). Nodes are organized by navigational intent rather than semantic similarity. Accompanying the model is the NTM benchmark for evaluating non-linear long-range dialogues, demonstrating improved task completion rates and reduced token consumption across various LLMs.

**[Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)**

:   The DiaFORGE framework is proposed, featuring a disambiguation-centric synthetic data generation pipeline, reasoning-chain finetuning, and a dynamic evaluation system. This allows open-source LLMs to achieve a tool-calling success rate 27 percentage points higher than GPT-4o and 49 percentage points higher than Claude-3.5-Sonnet when facing near-duplicate enterprise APIs.

**[Discourse Coherence and Response-Guided Context Rewriting for Multi-Party Dialogue Generation](discourse_coherence_and_response-guided_context_rewriting_for_multi-party_dialog.md)**

:   This paper proposes DRCR, the first framework to introduce context rewriting into multi-party dialogue generation. It utilizes dual feedback signals—discourse coherence and response quality—to construct preference data, enabling the rewriter and responder to mutually enhance each other through iterative dynamic self-evolution.

**[Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)**

:   The authors define "Inquisitive Dialogue"—where an AI actively questions an uncooperative interlocutor, exemplified by U.S. Supreme Court justices questioning attorneys—and propose a Dual Hierarchical RL framework. This framework consists of an Appraisal Agent that scores attorney responses in real-time across 9 appraisal categories, and a Hierarchical Dialogue Agent that performs DDQN in a three-layer (act/subtype/utterance) Poincaré action space. Combined with triple rewards (goal-relevance, novelty, and conciseness) and a conservative regularization term, the method improves Probing Effectiveness (PES) from a baseline of 4.22 to 4.47 on the Oyez Supreme Court dataset, achieving the highest Coverage and MR in multi-turn scenarios.

**[ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)**

:   ETHICMIND proposes an inference-time risk-aware alignment framework that jointly analyzes ethical risks and user emotions in each turn of a multi-turn dialogue. It plans high-level response strategies to generate replies that balance ethical guidance and emotional resonance, achieving consistent alignment in high-risk and morally ambiguous scenarios without additional training.

**[Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue](frame_of_reference_addressing_the_challenges_of_common_ground_representation_in_.md)**

:   Ours proposes the IndiRef benchmark to evaluate the ability of dialogue systems to establish and utilize persistent common ground through "relational reference" (e.g., "the cafe near the park we went to yesterday"). It finds that existing LLMs do not exceed 50% accuracy under full-context conditions and improves performance by 15-20% through synthetic data + GRPO reinforcement learning training.

**[GenesisFunc: Multi-Agent Data Generation for Accurate and Generalizable Function-Calling](genesisfunc_multi-agent_data_generation_for_accurate_and_generalizable_function-.md)**

:   GenesisFunc automatically constructs high-quality function-calling training data using a reliable tool pool, multi-agent dialogue generation, and multi-stage quality control. After fine-tuning Qwen3-8B, it outperforms open-source function-calling models of the same scale on BFCL, API-Bank, and ACEBench, demonstrating potential for scaling to more tools and multi-turn RL training.

**[LOCKET: Robust Feature-Locking Technique for Language Models](locket_robust_feature-locking_technique_for_language_models.md)**

:   LOCKET is a password-less, scalable, and jailbreak-resistant feature-locking scheme designed for the "pay-to-unlock" business model of LLMs. It trains a LoRA adapter for each feature to be locked (using LAT for adversarial reinforcement of refusals). When merging multiple adapters, it applies per-layer spectral norm clipping to prevent "over-refusal" collapse. Across 3 models and 4 features (Math/SQL/Summarize/MMLU), LOCKET achieves a 100% refusal rate, $\leq$ 7% utility loss, and $\leq$ 5% jailbreak attack success rate, significantly outperforming password-locking baselines.

**[MA$^2$P: A Meta-Cognitive Autonomous Intelligent Agents Framework for Complex Persuasion](ma2p_a_meta-cognitive_autonomous_intelligent_agents_framework_for_complex_persua.md)**

:   MA$^2$P decomposes complex persuasive dialogue into a closed loop of "Meta-strategy selection - Task-level multi-agent persuasion - Post-hoc knowledge update". Without training the base LLM, it transforms the persuadee's beliefs, desires, and concerns into specific strategic actions, significantly improving the persuasion success rate of various LLMs on CToMPersu.

**[Metro: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues](metro_towards_strategy_induction_from_expert_dialogue_transcripts_for_non-collab.md)**

:   Metro automatically induces expert dialogue transcripts into a "Strategy Forest"—a collection of trees rooted in K-Means clustered dialogue states. Each node represents an LLM-expanded micro-principle action, and branches represent complete action trajectories pruned by Wilson confidence lower bounds and MCTS-style value backpropagation. During inference, it retrieves a specific tree to extract short-term (breadth) and long-term (depth) recommendations in parallel. Without any training, it outperforms baselines such as PRINCIPLES, PPDPP, and GDP-Zero by approximately 10% on P4G and CB non-collaborative dialogue tasks.

**[Preference Learning Unlocks LLMs' Psycho-Counseling Skills](preference_learning_unlocks_llms_psycho-counseling_skills.md)**

:   This paper constructs the PsyCoPref preference dataset for psycho-counseling response quality and employs reward models, DPO, and iterative preference learning to train LLMs. The resulting 8B model achieves an 87.0% win rate against GPT-4o in psycho-counseling responses.

**[ReacTOD: Bounded Neuro-Symbolic Agentic NLU for Zero-Shot Dialogue State Tracking](reactod_bounded_neuro-symbolic_agentic_nlu_for_zero-shot_dialogue_state_tracking.md)**

:   ReacTOD decomposes task-oriented Dialogue State Tracking (DST) into bounded tool calls and uses a deterministic symbolic validator to intercept and provide feedback on LLM errors. This allows 8B to 32B-scale models to achieve Joint Goal Accuracy (JGA) on zero-shot MultiWOZ and SGD that surpasses previous large-scale LLM prompting methods.

**[Reasoning Gets Harder for LLMs Inside A Dialogue](reasoning_gets_harder_for_llms_inside_a_dialogue.md)**

:   This paper introduces the Boulder dynamic benchmark, demonstrating that while LLMs perform well on isolated reasoning problems, their performance significantly degrades when the same problems are embedded in task-oriented dialogues. This is primarily attributed to multi-turn context, dialogue role constraints, and the burden of tool calling.

**[Simulated Students in Tutoring Dialogues: Substance or Illusion?](simulated_students_in_tutoring_dialogues_substance_or_illusion.md)**

:   This paper proposes a evaluation framework for simulated students in mathematics tutoring dialogues. It finds that simple prompting often produces "students who seem to know how to answer," whereas SFT and DPO align more closely with real student behavior, though error replication and modeling of individual differences remain largely unresolved.

**[SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation.md)**

:   This paper proposes SPASM, a stability-centric persona-driven multi-turn dialogue simulation framework. Through modular persona generation, Egocentric Context Projection (ECP), and termination detection, it significantly reduces character drift and "echo" effects in LLM-LLM dialogues, constructing a high-quality dataset of 45,000 multi-turn dialogues.

**[Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers](stress-testing_emotional_support_models_moving_from_homogeneous_to_diverse_help_.md)**

:   This paper constructs nine-dimensional seeker profiles using Reddit emotional support dialogues and trains a controllable seeker simulator using LoRA-MoE with behavioral routing. This enables interactive stress-testing of emotional support models on more realistic, difficult, and diverse seeker populations.

**[STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)**

:   This paper proposes the STRIDE-ED framework, which constructs a comprehensive empathetic strategy system covering positive/neutral/negative emotions. By designing task-aligned multi-stage cognitive CoT reasoning combined with strategy-aware data refinement and a two-stage SFT+PPO training paradigm, it achieves SOTA performance in empathetic dialogue across multiple open-source LLMs, reaching an emotion accuracy of 57.25% and a BLEU-4 of 4.67.

**[Surprisal Minimisation over Goal-directed Alternatives Predicts Production Choice in Dialogue](surprisal_minimisation_over_goal-directed_alternatives_predicts_production_choic.md)**

:   This paper models utterance generation in natural dialogue as a cost-sensitive choice among contextual alternatives. It finds that minimizing surprisal relative to "goal-directed alternatives" (sharing the same communicative goal) best predicts actual human continuations.

**[Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)**

:   The TaDSE framework is proposed to utilize existing template information in dialogues as auxiliary anchors. Through three stages—template-aware data augmentation, paired contrastive training, and semantic compression inference—it significantly improves the quality of task-oriented dialogue sentence embeddings in unsupervised settings, outperforming previous SOTA and even surpassing supervised commercial embedding models across five benchmarks.

**[Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation](towards_proactive_information_probing_customer_service_chatbots_harvesting_value.md)**

:   This paper proposes ProChatIP, a framework that transforms customer service chatbots from passive responding tools into proactive information collection engines. Through a specialized dialogue policy module, it learns "when to probe" users for pre-defined target information while minimizing dialogue turns and user friction.

**[Your Students Don't Use LLMs Like You Wish They Did](your_students_dont_use_llms_like_you_wish_they_did.md)**

:   This paper proposes six computational behavioral metrics for educational AI dialogues and reveals, across 500 real student-AI conversations, that students frequently utilize LLM tools as answer extractors instead of learning aids. Furthermore, the mode of deployment influences this misalignment more significantly than system design or student preference.
