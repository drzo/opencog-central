# Tutorial: Implementing a Simple CogPrime-Based Agent

This tutorial guides you through the conceptual implementation of a very simple agent based on the principles of the CogPrime architecture, as detailed in Dr. Ben Goertzel's paper, "[CogPrime: An Integrative Architecture for Embodied Artificial General Intelligence](../docs/CogPrime_Integrative_Architecture_AGI.md)". (Context: `CogPrime - An Integrative Architecture for Embodied Artificial General Intelligence.md`)

We will focus on the "Build me something with blocks that I haven't seen before" example discussed in Section 10 of the paper. This tutorial simplifies many complex aspects of CogPrime to illustrate the core ideas of component interaction and cognitive synergy.

## 1. Overview of the Example

This example demonstrates:
*   How different CogPrime components (AtomSpace, PLN, MOSES, ECAN, Goal System, Perception, Action) can interact.
*   A simplified cognitive cycle.
*   The concept of cognitive synergy in achieving a creative task.
*   How an agent might process a natural language command, learn, and respond.

The agent will be purely conceptual, with simplified data structures and logic.

## 2. Setup Requirements (Conceptual)

To build this agent, we conceptually need:
*   **AtomSpace:** A simplified knowledge base to store Atoms (nodes and links representing concepts, procedures, goals, etc.).
*   **PLN (Probabilistic Logic Networks):** A simplified reasoning engine.
*   **MOSES (Meta-Optimizing Semantic Evolutionary Search):** A simplified procedure learning/selection mechanism.
*   **ECAN (Economic Attention Networks):** A basic attention allocation mechanism (STI - ShortTermImportance, LTI - LongTermImportance).
*   **Goal System:** To manage the agent's goals.
*   **Perception Module:** To receive input (e.g., teacher's commands).
*   **Action Module:** To perform actions (e.g., "build" blocks, respond verbally).

## 3. Basic Agent Implementation Outline (Conceptual Python-like Pseudocode)

```python
class SimpleCogPrimeAgent:
    def __init__(self):
        self.atomspace = SimplifiedAtomSpace()
        self.pln = SimplifiedPLN(self.atomspace)
        self.moses = SimplifiedMOSES(self.atomspace, self.pln) # MOSES might use PLN for evaluation
        self.ecan = SimplifiedECAN(self.atomspace)
        self.goal_system = SimplifiedGoalSystem(self.atomspace, self.pln, self.moses, self.ecan)
        self.perception = SimplifiedPerception()
        self.action = SimplifiedAction()
        self.is_running = True

    def cognitive_cycle(self):
        print("\n--- New Cognitive Cycle ---")
        # 1. Perception: Get input from the environment
        percepts = self.perception.get_percepts()
        if percepts:
            print(f"PERCEPTION: Received '{percepts['text']}'")
            self.atomspace.add_percept_to_atomspace(percepts)
            self.ecan.update_attention_from_percept(percepts['id']) # Focus on new percept

        # 2. Goal Processing: Update and select goals
        self.goal_system.derive_goals_from_percepts()
        active_goal_atom = self.goal_system.get_active_goal()

        if not active_goal_atom:
            print("GOAL_SYSTEM: No active goal.")
            # Potentially, agent could enter an 'idle' or 'exploratory' state
            return

        print(f"GOAL_SYSTEM: Active goal is '{active_goal_atom.name}' (STI: {active_goal_atom.sti:.2f})")

        # 3. Reasoning, Planning, Procedure Learning/Selection
        #    (Cognitive Schematics: Context -> Procedure -> Goal)
        current_context_atoms = self.atomspace.get_current_context(active_goal_atom)
        
        # Try to find or learn a procedure to achieve the goal
        procedure_atom = self.goal_system.find_or_learn_procedure_for_goal(active_goal_atom, current_context_atoms)

        if procedure_atom:
            print(f"PROCEDURE_SELECTION: Selected/Learned procedure '{procedure_atom.name}' for goal '{active_goal_atom.name}'")
            # 4. Action Execution
            action_result = self.action.execute_procedure(procedure_atom, self.atomspace)
            self.atomspace.record_action_outcome(active_goal_atom, procedure_atom, action_result)
            
            # Update goal status based on action_result
            if action_result.get("status") == "success": # Simplified
                print(f"GOAL_SYSTEM: Goal '{active_goal_atom.name}' achieved.")
                self.goal_system.mark_goal_achieved(active_goal_atom)
            else:
                print(f"GOAL_SYSTEM: Goal '{active_goal_atom.name}' not yet achieved or failed.")
                # Agent might try again, or replan, or give up on this specific procedure
        else:
            print(f"PROCEDURE_SELECTION: Could not find or learn a procedure for goal '{active_goal_atom.name}'. Agent might explore or ask for help.")
            self.action.perform_exploratory_action(f"Exploring options for goal {active_goal_atom.name}")


        # 5. Internal "Housekeeping"
        self.ecan.propagate_attention_and_decay()
        self.atomspace.perform_forgetting_if_needed()
        # self.atomspace.attempt_map_formation() # Conceptual step for pattern recognition

    def run_agent(self, num_cycles=5):
        for i in range(num_cycles):
            if not self.is_running:
                break
            self.cognitive_cycle()
            if not self.goal_system.has_active_goals() and not self.perception.has_pending_percepts():
                 print("AGENT: No active goals or pending percepts. Pausing.")
                 # In a real system, might wait for new percepts or self-generate goals
                 break 
```

## 4. Core Component Implementations (Simplified)

These are highly simplified for clarity.

*   **SimplifiedAtomSpace:**
    *   `atoms`: A dictionary to store Atoms. An Atom could be `{'id': 'unique_id', 'name': 'car', 'type': 'ConceptNode', 'tv': (0.8, 0.9), 'sti': 0.1, 'lti': 0.05, 'links_to': [], 'linked_from': []}`.
    *   `add_atom(atom_data)`: Adds an atom.
    *   `get_atom(atom_id)`: Retrieves an atom.
    *   `link_atoms(source_id, target_id, link_type, tv=(1.0,1.0))`: Creates a typed link (another Atom).
    *   `add_percept_to_atomspace(percept)`: Converts percept into Atoms. E.g., "Build novel blocks" -> `PerceptAtom`, `CommandAtom("build")`, `PropertyAtom("novel")`, `ObjectAtom("blocks")`. Links them.
    *   `get_current_context(active_goal_atom)`: Returns Atoms highly relevant to the goal (high STI, linked to goal).
    *   `record_action_outcome(goal_atom, procedure_atom, result)`: If `result` is positive, strengthen the TV of a `CognitiveSchematicLink` connecting context, procedure, and goal.
    *   `perform_forgetting_if_needed()`: Conceptually remove Atoms with very low LTI.
    *   `get_atoms_linked_to(atom_id, link_type_filter=None)`: Find atoms linked from a given atom.

*   **SimplifiedPLN:**
    *   `infer_novelty(item_atom, teacher_atom)`: Checks if `TeacherSeenLink(teacher_atom, item_atom)` has a low truth value (TV). Returns a TV. (Conceptual: CogPrime paper, Sec 10.1)
    *   `reason_about_procedure_applicability(procedure_atom, context_atoms)`: Returns a TV indicating if a procedure is suitable for the context.

*   **SimplifiedMOSES:**
    *   `find_existing_procedure(goal_atom, context_atoms)`: Searches AtomSpace for `ProcedureAtoms` linked to similar goals/contexts with high TV `CognitiveSchematicLinks`.
    *   `learn_procedure_for_novelty(goal_atom, context_atoms, teacher_atom)`:
        *   Conceptually "tries" combinations of known building procedures (e.g., `build_tower_proc`, `build_car_proc`).
        *   For each candidate structure, uses `pln.infer_novelty(candidate_structure_atom, teacher_atom)`.
        *   Selects a procedure that results in a high novelty TV.
        *   Example: Might generate a `ProcedureAtom("build_car_man_proc")`. (Conceptual: CogPrime paper, Sec 10.1, "car-man" example)
        *   Adds the new `ProcedureAtom` to AtomSpace.

*   **SimplifiedECAN:**
    *   `update_attention_from_percept(percept_atom_id)`: Increases STI of `percept_atom_id` and closely linked Atoms.
    *   `propagate_attention_and_decay()`:
        *   Spreads some STI to linked Atoms (Hebbian-like).
        *   Decays all STI slightly.
        *   Slowly converts persistent STI into LTI.
    *   `get_highest_sti_goal_candidate()`: Used by GoalSystem.

*   **SimplifiedGoalSystem:**
    *   `active_goals`: A list of `GoalAtom` IDs.
    *   `derive_goals_from_percepts()`: If a `CommandAtom("build")` with `PropertyAtom("novel")` exists and has high STI, create/activate `GoalAtom("build_novel_structure")`.
    *   `get_active_goal()`: Returns the `GoalAtom` with the highest STI from `active_goals`.
    *   `find_or_learn_procedure_for_goal(goal_atom, context_atoms)`:
        1.  Calls `moses.find_existing_procedure`.
        2.  If not found, or low TV, calls `moses.learn_procedure_for_novelty` (if goal is novelty related).
        3.  Uses `pln.reason_about_procedure_applicability` to filter/rank.
    *   `mark_goal_achieved(goal_atom)`: Removes from `active_goals`, maybe sets a `GoalStateAtom("achieved")` linked to `goal_atom`.

*   **SimplifiedPerception:**
    *   `queue`: Stores pending commands.
    *   `add_command_to_queue(text)`: Adds a command.
    *   `get_percepts()`: Returns the next command from queue as `{'id': 'percept_XYZ', 'type': 'command', 'text': '...'}`.

*   **SimplifiedAction:**
    *   `execute_procedure(procedure_atom, atomspace)`:
        *   Prints a message like: `ACTION: Executing procedure '{procedure_atom.name}'`.
        *   If `procedure_atom.name == "build_car_man_proc"`, it conceptually creates `ObjectAtom("car_man_structure_1")` in AtomSpace.
        *   Returns `{'status': 'success', 'created_item_id': 'car_man_structure_1'}`.
    *   `perform_exploratory_action(message)`: Prints the message.

## 5. Example Cognitive Cycle Walkthrough: "Build me something with blocks that I haven't seen before."

(Ref: CogPrime Paper, Section 10.1)

**Initial State:** Agent is idle. Teacher gives the command.
`agent.perception.add_command_to_queue("Build me something with blocks that I haven't seen before.")`

**Cycle 1:**
1.  **Perception:** `get_percepts()` returns the command.
    *   `AtomSpace` creates `PerceptAtom("cmd1")`, `CommandAtom("build_cmd")`, `PropertyAtom("novel_prop")`, `ObjectAtom("blocks_obj")`. Links them to `cmd1`.
    *   `ECAN` boosts STI of `cmd1` and its linked atoms.
2.  **Goal Processing:**
    *   `GoalSystem.derive_goals_from_percepts()` sees `build_cmd` and `novel_prop` with high STI. Creates/activates `GoalAtom("goal_build_novel_blocks")`.
    *   `get_active_goal()` returns `goal_build_novel_blocks`.
3.  **Procedure Selection/Learning:**
    *   `GoalSystem.find_or_learn_procedure_for_goal(...)` is called.
    *   `MOSES.find_existing_procedure` might not find a suitable procedure for "novelty" initially.
    *   `MOSES.learn_procedure_for_novelty` is invoked.
        *   It conceptually generates candidate procedures (e.g., "build tower", "build arch", "combine car and man parts").
        *   For each, it asks `PLN.infer_novelty(candidate_structure, teacher_atom)`. (Assume `teacher_atom` exists and has `SeenLink`s to past structures).
        *   Let's say "combine car and man parts" (leading to `ProcedureAtom("proc_build_car_man")`) gets a high novelty TV from PLN. This procedure is selected.
4.  **Action:**
    *   `Action.execute_procedure(proc_build_car_man, ...)`:
        *   Prints: "ACTION: Executing procedure 'proc_build_car_man'".
        *   Conceptually creates `ObjectAtom("car_man_1")` in AtomSpace.
        *   Returns `{'status': 'success', 'created_item_id': 'car_man_1'}`.
    *   `AtomSpace.record_action_outcome` strengthens the (newly created) `CognitiveSchematicLink` for (Context: "build novel", Proc: "proc_build_car_man", Goal: "teacher_pleased_by_novelty").
    *   `GoalSystem.mark_goal_achieved(goal_build_novel_blocks)`.
5.  **Internal Housekeeping:**
    *   `ECAN` updates STIs. `proc_build_car_man` and `car_man_1` might get higher LTI over time if reinforced.

**Cycle 2 (Teacher's Feedback):**
`agent.perception.add_command_to_queue("It's beautiful. What is it?")` (Assume this is directed at `car_man_1`)

1.  **Perception:** Command processed. `FeedbackAtom("fb_beautiful")` linked to `car_man_1`. New `QuestionAtom("q_what_is_it")` linked to `car_man_1`.
2.  **Goal Processing:**
    *   `GoalSystem` derives `GoalAtom("goal_answer_q_what_is_it")`. This becomes active.
    *   The positive feedback "beautiful" (via `FeedbackAtom`) would also trigger PLN to further increase the TV of the `CognitiveSchematicLink` related to `proc_build_car_man`. (Reinforcement learning aspect, CogPrime paper Sec 10.1)
3.  **Procedure Selection:**
    *   `GoalSystem` looks for a procedure for `goal_answer_q_what_is_it`.
    *   `MOSES` might find/generate `ProcedureAtom("proc_describe_car_man")` which involves stating the conceptual origin (car + man).
4.  **Action:**
    *   `Action.execute_procedure(proc_describe_car_man, ...)`:
        *   Prints: "ACTION: Executing procedure 'proc_describe_car_man'".
        *   Outputs: "It's a car man."
    *   Goal is marked achieved.

## 6. How Components Work Together (Cognitive Synergy)

*   **Perception to Goal:** Teacher's command (percept) is translated into an internal goal by the `GoalSystem`, guided by `ECAN`'s attention on the percept.
*   **Goal to Action (via MOSES & PLN):**
    *   The `GoalSystem` needs a `Procedure` for the active `Goal`.
    *   `MOSES` is tasked with finding or learning this procedure.
    *   `PLN` assists `MOSES` by evaluating candidate procedures (e.g., checking for "novelty" against the teacher's known experiences stored in `AtomSpace`).
    *   This interaction avoids MOSES blindly searching; PLN provides semantic guidance. (Synergy, CogPrime paper Sec 8.8.1, 8.8.2)
*   **ECAN Guiding Focus:** `ECAN` ensures that Atoms related to the current goal, recent percepts, and promising procedures have high STI, focusing the "computational resources" of `PLN` and `MOSES`.
*   **AtomSpace as Shared Memory:** All components read from and write to `AtomSpace`. `PLN` reasons over knowledge, `MOSES` stores learned procedures, `ECAN` manages attention values on Atoms, `GoalSystem` tracks goal states. This shared representation is crucial for integration.
*   **Learning from Feedback:** The teacher's positive feedback ("It's beautiful") reinforces the `CognitiveSchematicLink` in `AtomSpace` associated with the successful "build_car_man" procedure, making the agent more likely to use similar creative strategies in the future. This involves `PLN` updating TruthValues.

This simplified example hints at the "cognitive synergy" described in CogPrime: components don't just run in sequence but actively inform and leverage each other.

## 7. Next Steps for Extending the Example

*   **More Sophisticated PLN:** Implement actual probabilistic inference rules and TruthValue calculations (strength, confidence).
*   **Real MOSES:** Implement an evolutionary algorithm for program learning, with a defined program representation and genetic operators.
*   **Detailed ECAN:** Use a currency-based system for STI/LTI, with explicit HebbianLinks influencing attention spread.
*   **Episodic Memory:** Allow the agent to store and recall sequences of (percept, context, goal, procedure, outcome) as episodes in AtomSpace. These episodes can then inform future PLN reasoning and MOSES learning.
*   **Internal Simulation:** Before `Action.execute_procedure`, the agent could use a `SimulationEngine` to internally "run" the procedure and predict its outcome using PLN, especially for physical tasks.
*   **Map Formation:** Implement a process that periodically scans AtomSpace for recurring patterns (e.g., frequently co-activated Atoms, successful procedure patterns) and creates new, higher-level `ConceptAtoms` or `SchemaAtoms` representing these patterns (the "Cognitive Equation" from CogPrime paper Sec 8.6).
*   **Richer Environment:** Interface with a simple graphical block-building simulator instead of just text.
*   **Natural Language Understanding (NLU):** Integrate a more robust NLU component (like OpenCog's Link Grammar, conceptually) to parse commands instead of simple keyword spotting.

This tutorial provides a glimpse into the complex yet elegant design of CogPrime. Building a full AGI is a monumental task, but understanding these foundational interactions is a key first step.
