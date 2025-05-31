"""
Cognitive Pattern Generator

Generates sophisticated cognitive patterns for testing emergent behaviors
and distributed cognition synergy scenarios.

TODO: Future recursive expansions:
- Multi-agent cognitive pattern interactions
- Temporal cognitive pattern evolution
- Cross-modal pattern integration
- Hierarchical pattern emergence
"""

import random
import math
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta


class CognitivePatternGenerator:
    """
    Generates complex cognitive patterns for advanced testing scenarios.
    
    Implements emergent pattern generation following principles of:
    - Neural-symbolic integration
    - Distributed cognition
    - Adaptive attention mechanisms
    - Recursive implementation pathways
    """
    
    def __init__(self, pattern_complexity: str = "medium"):
        """
        Initialize cognitive pattern generator.
        
        Args:
            pattern_complexity: "simple", "medium", "complex", "emergent"
        """
        self.pattern_complexity = pattern_complexity
        self.pattern_history = []
        
    def generate_attention_pattern(self, time_steps: int = 100) -> Dict[str, Any]:
        """Generate temporal attention allocation patterns."""
        pattern = {
            "time_steps": time_steps,
            "attention_waves": [],
            "focus_transitions": [],
            "attention_entropy": []
        }
        
        # Generate attention waves with different frequencies
        for i in range(time_steps):
            # Multiple overlapping attention waves
            wave1 = 0.5 + 0.3 * math.sin(i * 0.1)  # Slow wave
            wave2 = 0.5 + 0.2 * math.sin(i * 0.3)  # Medium wave  
            wave3 = 0.5 + 0.1 * math.sin(i * 0.7)  # Fast wave
            
            combined_attention = (wave1 + wave2 + wave3) / 3
            pattern["attention_waves"].append(combined_attention)
            
            # Calculate attention entropy (measure of focus distribution)
            entropy = -combined_attention * math.log2(combined_attention + 1e-10)
            pattern["attention_entropy"].append(entropy)
            
        # Generate focus transitions
        current_focus = random.choice(["perception", "memory", "reasoning", "planning"])
        for i in range(0, time_steps, random.randint(5, 15)):
            new_focus = random.choice(["perception", "memory", "reasoning", "planning", "execution"])
            pattern["focus_transitions"].append({
                "time": i,
                "from_focus": current_focus,
                "to_focus": new_focus,
                "transition_strength": random.uniform(0.3, 1.0)
            })
            current_focus = new_focus
            
        return pattern
        
    def generate_learning_pattern(self, epochs: int = 50) -> Dict[str, Any]:
        """Generate learning curve patterns with different phases."""
        pattern = {
            "epochs": epochs,
            "learning_curve": [],
            "learning_phases": [],
            "knowledge_acquisition": [],
            "forgetting_curve": []
        }
        
        # Generate learning curve with realistic phases
        knowledge = 0.0
        for epoch in range(epochs):
            # Learning rate changes over time
            if epoch < 10:  # Initial rapid learning
                learning_rate = 0.1
            elif epoch < 30:  # Slower learning
                learning_rate = 0.05
            else:  # Plateau phase
                learning_rate = 0.01
                
            # Add noise and occasional breakthroughs
            noise = random.uniform(-0.02, 0.02)
            breakthrough = 0.1 if random.random() < 0.05 else 0.0
            
            knowledge += learning_rate + noise + breakthrough
            knowledge = min(knowledge, 1.0)  # Cap at 100%
            
            pattern["learning_curve"].append(knowledge)
            
            # Forgetting curve (knowledge decay)
            forgetting_rate = 0.001 * epoch  # Forgetting increases over time
            retained_knowledge = knowledge * (1 - forgetting_rate)
            pattern["forgetting_curve"].append(retained_knowledge)
            
        # Identify learning phases
        pattern["learning_phases"] = self._identify_learning_phases(pattern["learning_curve"])
        
        return pattern
        
    def _identify_learning_phases(self, learning_curve: List[float]) -> List[Dict[str, Any]]:
        """Identify distinct phases in learning curve."""
        phases = []
        window_size = 5
        
        for i in range(window_size, len(learning_curve) - window_size):
            # Calculate slope in window
            recent_slope = (learning_curve[i] - learning_curve[i-window_size]) / window_size
            
            if recent_slope > 0.05:
                phase_type = "rapid_learning"
            elif recent_slope > 0.01:
                phase_type = "steady_learning"
            elif recent_slope > -0.01:
                phase_type = "plateau"
            else:
                phase_type = "forgetting"
                
            phases.append({
                "epoch": i,
                "phase_type": phase_type,
                "slope": recent_slope,
                "knowledge_level": learning_curve[i]
            })
            
        return phases
        
    def generate_reasoning_pattern(self, depth: int = 5) -> Dict[str, Any]:
        """Generate reasoning chain patterns with different strategies."""
        pattern = {
            "reasoning_depth": depth,
            "reasoning_chains": [],
            "inference_strategies": [],
            "logical_consistency": [],
            "creative_leaps": []
        }
        
        # Generate reasoning chains
        for chain_id in range(random.randint(3, 8)):
            chain = {
                "chain_id": chain_id,
                "steps": [],
                "strategy": random.choice(["deductive", "inductive", "abductive", "analogical"]),
                "confidence_progression": []
            }
            
            confidence = random.uniform(0.7, 0.95)
            
            for step in range(depth):
                # Confidence typically decreases with reasoning depth
                confidence *= random.uniform(0.85, 0.98)
                
                # Occasional creative leaps boost confidence
                if random.random() < 0.1:
                    confidence = min(confidence + 0.2, 1.0)
                    pattern["creative_leaps"].append({
                        "chain_id": chain_id,
                        "step": step,
                        "confidence_boost": 0.2
                    })
                
                chain["steps"].append({
                    "step": step,
                    "reasoning_type": random.choice(["premise", "inference", "conclusion"]),
                    "confidence": confidence,
                    "complexity": random.uniform(0.2, 0.9)
                })
                
                chain["confidence_progression"].append(confidence)
                
            pattern["reasoning_chains"].append(chain)
            
        return pattern
        
    def generate_memory_pattern(self, memory_size: int = 1000) -> Dict[str, Any]:
        """Generate memory access and consolidation patterns."""
        pattern = {
            "memory_size": memory_size,
            "access_patterns": [],
            "consolidation_events": [],
            "interference_patterns": [],
            "memory_hierarchy": []
        }
        
        # Generate memory access patterns (following power law)
        for i in range(memory_size):
            # Power law distribution for memory access frequency
            access_frequency = 1.0 / ((i + 1) ** 0.8)
            recency_boost = math.exp(-i / 100.0)  # Recent memories accessed more
            
            total_access_prob = access_frequency * recency_boost
            
            pattern["access_patterns"].append({
                "memory_id": i,
                "access_frequency": access_frequency,
                "recency": recency_boost,
                "total_access_prob": total_access_prob,
                "memory_type": random.choice(["episodic", "semantic", "procedural", "working"])
            })
            
        # Generate consolidation events
        for event_id in range(random.randint(10, 30)):
            pattern["consolidation_events"].append({
                "event_id": event_id,
                "memory_ids": random.sample(range(memory_size), random.randint(5, 20)),
                "consolidation_strength": random.uniform(0.5, 1.0),
                "consolidation_type": random.choice(["systems", "synaptic", "cellular"])
            })
            
        return pattern
        
    def generate_distributed_pattern(self, agent_count: int = 5) -> Dict[str, Any]:
        """Generate distributed cognition synchronization patterns."""
        pattern = {
            "agent_count": agent_count,
            "synchronization_events": [],
            "communication_patterns": [],
            "emergent_behaviors": [],
            "consensus_formation": []
        }
        
        # Generate inter-agent communication patterns
        for time_step in range(100):
            # Communication probability based on cognitive load
            for agent_a in range(agent_count):
                for agent_b in range(agent_a + 1, agent_count):
                    if random.random() < 0.1:  # 10% chance of communication
                        pattern["communication_patterns"].append({
                            "time": time_step,
                            "from_agent": agent_a,
                            "to_agent": agent_b,
                            "information_content": random.uniform(0.1, 1.0),
                            "communication_type": random.choice(["query", "inform", "request", "share"])
                        })
                        
        # Generate synchronization events
        for sync_event in range(random.randint(5, 15)):
            participating_agents = random.sample(range(agent_count), random.randint(2, agent_count))
            pattern["synchronization_events"].append({
                "event_id": sync_event,
                "time": random.randint(0, 100),
                "participating_agents": participating_agents,
                "synchronization_strength": random.uniform(0.3, 1.0),
                "sync_type": random.choice(["attention", "memory", "goal", "action"])
            })
            
        # Generate emergent behaviors
        for behavior in range(random.randint(3, 8)):
            pattern["emergent_behaviors"].append({
                "behavior_id": behavior,
                "emergence_time": random.randint(20, 80),
                "complexity_level": random.uniform(0.3, 0.9),
                "stability": random.uniform(0.5, 0.95),
                "participating_agents": random.sample(range(agent_count), random.randint(2, agent_count)),
                "behavior_type": random.choice(["coordination", "competition", "cooperation", "adaptation"])
            })
            
        return pattern
        
    def generate_edge_case_patterns(self) -> Dict[str, Any]:
        """Generate edge case patterns for robust testing."""
        patterns = {
            "extreme_conditions": [],
            "failure_modes": [],
            "recovery_patterns": [],
            "boundary_behaviors": []
        }
        
        # Extreme cognitive load conditions
        patterns["extreme_conditions"].extend([
            {
                "condition": "cognitive_overload",
                "parameters": {"attention_demand": 1.5, "memory_pressure": 0.95},
                "expected_behavior": "graceful_degradation"
            },
            {
                "condition": "attention_deficit",
                "parameters": {"attention_availability": 0.1, "task_complexity": 0.8},
                "expected_behavior": "task_switching_increase"
            },
            {
                "condition": "memory_saturation", 
                "parameters": {"memory_usage": 0.99, "new_information_rate": 0.8},
                "expected_behavior": "selective_forgetting"
            }
        ])
        
        # Failure mode patterns
        patterns["failure_modes"].extend([
            {
                "failure_type": "attention_fragmentation",
                "trigger_conditions": {"concurrent_goals": 10, "interruption_rate": 0.9},
                "failure_probability": 0.8
            },
            {
                "failure_type": "reasoning_loop",
                "trigger_conditions": {"circular_dependencies": True, "depth_limit": None},
                "failure_probability": 0.6
            },
            {
                "failure_type": "memory_interference",
                "trigger_conditions": {"similar_patterns": 0.95, "consolidation_incomplete": True},
                "failure_probability": 0.7
            }
        ])
        
        # Recovery patterns
        patterns["recovery_patterns"].extend([
            {
                "recovery_mechanism": "attention_refocus",
                "trigger_threshold": 0.3,
                "recovery_time": random.uniform(2.0, 8.0),
                "success_rate": 0.85
            },
            {
                "recovery_mechanism": "memory_reorganization",
                "trigger_threshold": 0.8,
                "recovery_time": random.uniform(5.0, 20.0),
                "success_rate": 0.7
            },
            {
                "recovery_mechanism": "goal_prioritization",
                "trigger_threshold": 0.5,
                "recovery_time": random.uniform(1.0, 5.0),
                "success_rate": 0.9
            }
        ])
        
        return patterns