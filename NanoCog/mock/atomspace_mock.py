"""
AtomSpace Mock Data Generator

Generates realistic mock cognitive state data for testing distributed cognition systems.
Implements principles of emergent cognitive patterns and hypergraph encoding.
"""

import random
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class AtomSpaceMockGenerator:
    """
    Generates mock AtomSpace cognitive state data with realistic patterns.
    
    Features emergent cognitive behaviors and distributed robustness patterns.
    """
    
    def __init__(self, 
                 seed: Optional[int] = None,
                 complexity_level: str = "medium",
                 cognitive_focus: str = "general"):
        """
        Initialize the mock generator with cognitive parameters.
        
        Args:
            seed: Random seed for reproducible patterns
            complexity_level: "simple", "medium", "complex" - affects data richness
            cognitive_focus: "attention", "learning", "reasoning", "general"
        """
        if seed is not None:
            random.seed(seed)
        
        self.complexity_level = complexity_level
        self.cognitive_focus = cognitive_focus
        
        # Configure complexity-based parameters
        self._configure_complexity()
        
    def _configure_complexity(self):
        """Configure generation parameters based on complexity level."""
        complexity_configs = {
            "simple": {
                "atom_count_range": (500, 2000),
                "goal_count_range": (2, 5),
                "atom_types_count": 8,
                "schematic_count_range": (20, 50)
            },
            "medium": {
                "atom_count_range": (1000, 10000),
                "goal_count_range": (3, 8),
                "atom_types_count": 12,
                "schematic_count_range": (50, 200)
            },
            "complex": {
                "atom_count_range": (5000, 50000),
                "goal_count_range": (5, 15),
                "atom_types_count": 16,
                "schematic_count_range": (100, 500)
            }
        }
        
        config = complexity_configs.get(self.complexity_level, complexity_configs["medium"])
        self.config = config
        
    def generate_cognitive_state(self) -> Dict[str, Any]:
        """
        Generate comprehensive mock cognitive state data.
        
        Returns:
            Dictionary containing all cognitive state components
        """
        # Generate core cognitive components
        atom_count = random.randint(*self.config["atom_count_range"])
        active_goals = self._generate_active_goals()
        attention_distribution = self._generate_attention_distribution()
        atom_distribution = self._generate_atom_distribution()
        cognitive_schematics = self._generate_cognitive_schematics()
        bottlenecks = self._generate_bottlenecks()
        recommendations = self._generate_recommendations()
        
        # Generate summary metrics
        summary = self._generate_summary_metrics(
            atom_count, active_goals, attention_distribution
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "generator_version": "1.0.0",
            "complexity_level": self.complexity_level,
            "cognitive_focus": self.cognitive_focus,
            "atom_count": atom_count,
            "active_goals": active_goals,
            "attention_distribution": attention_distribution,
            "atom_distribution": atom_distribution,
            "cognitive_schematics": cognitive_schematics,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "summary": summary,
            "metadata": {
                "generation_time": time.time(),
                "adaptive_patterns": self._generate_adaptive_patterns(),
                "emergent_properties": self._generate_emergent_properties()
            }
        }
        
    def _generate_active_goals(self) -> List[Dict[str, Any]]:
        """Generate realistic active goal patterns."""
        goal_count = random.randint(*self.config["goal_count_range"])
        goals = []
        
        goal_templates = [
            "CognitiveLearning", "PatternRecognition", "MemoryConsolidation",
            "AttentionFocus", "ReasoningChain", "ProblemSolving",
            "LanguageProcessing", "SensorIntegration", "MotorPlanning",
            "EmotionalRegulation", "SocialInteraction", "MetaCognition"
        ]
        
        for i in range(goal_count):
            goal_name = random.choice(goal_templates) + f"_G{i+1}"
            goals.append({
                "handle": f"0x{random.randint(1000, 9999):x}",
                "name": goal_name,
                "type": "ConceptNode",
                "sti": random.uniform(0.5, 0.95),
                "lti": random.uniform(0.1, 0.5),
                "tv": [random.uniform(0.7, 0.99), random.uniform(0.6, 0.9)],
                "priority": random.choice(["high", "medium", "low"]),
                "progress": random.uniform(0.1, 0.9),
                "sub_goals": random.randint(0, 3)
            })
            
        return sorted(goals, key=lambda x: x["sti"], reverse=True)
        
    def _generate_attention_distribution(self) -> Dict[str, Any]:
        """Generate realistic attention allocation patterns."""
        return {
            "attention_stats": {
                "avg_sti": random.uniform(0.1, 0.3),
                "max_sti": random.uniform(0.8, 0.99),
                "min_sti": random.uniform(0.01, 0.1),
                "std_dev_sti": random.uniform(0.1, 0.3),
                "avg_lti": random.uniform(0.1, 0.3),
                "max_lti": random.uniform(0.7, 0.9),
                "min_lti": random.uniform(0.01, 0.1),
                "attention_focus_ratio": random.uniform(0.05, 0.15)
            },
            "high_sti_count": random.randint(50, 200),
            "high_sti_by_type": {
                "ConceptNode": random.randint(20, 100),
                "PredicateNode": random.randint(10, 50),
                "ListLink": random.randint(5, 30),
                "EvaluationLink": random.randint(10, 40),
                "HebbianLink": random.randint(0, 20)
            },
            "sti_distribution": {
                "very_high": random.randint(10, 50),
                "high": random.randint(50, 150),
                "medium": random.randint(200, 500),
                "low": random.randint(500, 1000),
                "very_low": random.randint(1000, 5000)
            },
            "attention_waves": self._generate_attention_waves()
        }
        
    def _generate_attention_waves(self) -> List[Dict[str, Any]]:
        """Generate attention wave patterns for temporal cognitive dynamics."""
        waves = []
        for i in range(random.randint(2, 5)):
            waves.append({
                "wave_id": f"wave_{i+1}",
                "frequency": random.uniform(0.1, 2.0),
                "amplitude": random.uniform(0.3, 0.8),
                "phase": random.uniform(0, 6.28),
                "cognitive_domain": random.choice([
                    "perception", "memory", "reasoning", "planning", "execution"
                ])
            })
        return waves
        
    def _generate_atom_distribution(self) -> Dict[str, Any]:
        """Generate realistic atom type distribution patterns."""
        base_atom_types = [
            "ConceptNode", "PredicateNode", "ListLink", "EvaluationLink", 
            "HebbianLink", "InheritanceLink", "SchemaNode", "VariableNode"
        ]
        
        extended_atom_types = base_atom_types + [
            "GroundedSchemaNode", "GroundedPredicateNode", "AndLink", "OrLink",
            "NotLink", "ImplicationLink", "EquivalenceLink", "SimilarityLink"
        ]
        
        atom_types = extended_atom_types[:self.config["atom_types_count"]]
        
        type_counts = {}
        for atom_type in atom_types:
            # Generate realistic distributions based on cognitive patterns
            if "Node" in atom_type:
                base_count = random.randint(100, 2000)
            else:  # Links
                base_count = random.randint(50, 1000)
                
            type_counts[atom_type] = base_count
        
        total_atoms = sum(type_counts.values())
        type_percentages = {
            atom_type: (count / total_atoms * 100)
            for atom_type, count in type_counts.items()
        }
        
        sorted_types = sorted(
            type_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            "total_atoms": total_atoms,
            "type_counts": type_counts,
            "type_percentages": type_percentages,
            "sorted_types": sorted_types,
            "diversity_index": self._calculate_diversity_index(type_percentages),
            "connectivity_patterns": self._generate_connectivity_patterns()
        }
        
    def _calculate_diversity_index(self, percentages: Dict[str, float]) -> float:
        """Calculate Shannon diversity index for atom types."""
        import math
        total = sum(percentages.values())
        if total == 0:
            return 0.0
            
        diversity = 0.0
        for percentage in percentages.values():
            if percentage > 0:
                p = percentage / total
                diversity -= p * math.log2(p)
                
        return diversity
        
    def _generate_connectivity_patterns(self) -> Dict[str, Any]:
        """Generate hypergraph connectivity patterns."""
        return {
            "avg_connectivity": random.uniform(2.5, 8.0),
            "max_connectivity": random.randint(20, 100),
            "clustering_coefficient": random.uniform(0.3, 0.8),
            "small_world_index": random.uniform(1.2, 4.0),
            "hub_nodes": random.randint(5, 25),
            "isolated_components": random.randint(0, 5)
        }
        
    def _generate_cognitive_schematics(self) -> Dict[str, Any]:
        """Generate cognitive schema execution patterns."""
        total_schematics = random.randint(*self.config["schematic_count_range"])
        
        return {
            "total_schematics": total_schematics,
            "success_counts": {
                "successful": random.randint(int(total_schematics * 0.6), int(total_schematics * 0.9)),
                "failed": random.randint(int(total_schematics * 0.05), int(total_schematics * 0.3)),
                "unknown": random.randint(0, int(total_schematics * 0.1))
            },
            "schematics_by_goal": {
                f"Goal{i}": random.randint(5, 30)
                for i in range(1, random.randint(5, 10))
            },
            "execution_patterns": self._generate_execution_patterns(),
            "learning_rates": {
                "fast_learning": random.uniform(0.1, 0.3),
                "medium_learning": random.uniform(0.05, 0.15),
                "slow_learning": random.uniform(0.01, 0.08)
            }
        }
        
    def _generate_execution_patterns(self) -> Dict[str, Any]:
        """Generate schema execution timing patterns."""
        return {
            "avg_execution_time": random.uniform(0.1, 2.0),
            "success_rate_trend": random.choice(["improving", "stable", "declining"]),
            "parallel_executions": random.randint(1, 8),
            "recursive_depth": random.randint(2, 6)
        }
        
    def _generate_bottlenecks(self) -> List[Dict[str, Any]]:
        """Generate realistic cognitive bottleneck patterns."""
        bottleneck_types = [
            ("attention_fragmentation", "high", "Attention spread too thin across multiple goals"),
            ("memory_interference", "medium", "Conflicting memory patterns causing retrieval delays"),
            ("reasoning_loops", "high", "Circular reasoning chains detected in inference"),
            ("resource_contention", "medium", "Multiple processes competing for cognitive resources"),
            ("pattern_overload", "low", "Too many simultaneous pattern matching operations"),
            ("goal_conflicts", "high", "Contradictory goals creating decision paralysis"),
            ("schema_thrashing", "medium", "Excessive schema switching without completion")
        ]
        
        bottlenecks = []
        num_bottlenecks = random.randint(2, 5)
        
        for i in range(num_bottlenecks):
            bottleneck_type, severity, description = random.choice(bottleneck_types)
            bottlenecks.append({
                "type": bottleneck_type,
                "severity": severity,
                "description": description,
                "impact_score": random.uniform(0.3, 0.9),
                "affected_goals": random.randint(1, 4),
                "duration": random.uniform(0.5, 10.0),
                "suggested_action": self._get_bottleneck_action(bottleneck_type)
            })
            
        return bottlenecks
        
    def _get_bottleneck_action(self, bottleneck_type: str) -> str:
        """Get suggested action for bottleneck type."""
        actions = {
            "attention_fragmentation": "Implement attention focusing mechanisms",
            "memory_interference": "Clear conflicting memory patterns",
            "reasoning_loops": "Add loop detection and breaking mechanisms",
            "resource_contention": "Implement resource scheduling",
            "pattern_overload": "Reduce pattern matching complexity",
            "goal_conflicts": "Resolve goal hierarchy conflicts",
            "schema_thrashing": "Implement schema persistence mechanisms"
        }
        return actions.get(bottleneck_type, "Monitor and analyze further")
        
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate adaptive cognitive optimization recommendations."""
        recommendation_types = [
            ("increase_attention_focus", "Focus attention on highest priority goals"),
            ("optimize_memory_consolidation", "Consolidate frequently accessed memory patterns"),
            ("enhance_reasoning_efficiency", "Implement more efficient reasoning chains"),
            ("balance_exploration_exploitation", "Adjust exploration vs exploitation balance"),
            ("improve_goal_coordination", "Better coordinate between competing goals"),
            ("upgrade_learning_algorithms", "Implement more advanced learning mechanisms"),
            ("enhance_pattern_recognition", "Improve pattern recognition accuracy")
        ]
        
        recommendations = []
        num_recommendations = random.randint(3, 6)
        
        for i in range(num_recommendations):
            rec_type, description = random.choice(recommendation_types)
            recommendations.append({
                "type": rec_type,
                "description": description,
                "priority": random.choice(["high", "medium", "low"]),
                "confidence": random.uniform(0.6, 0.95),
                "estimated_benefit": random.uniform(0.2, 0.8),
                "implementation_complexity": random.choice(["simple", "moderate", "complex"]),
                "prerequisites": random.randint(0, 3)
            })
            
        return sorted(recommendations, key=lambda x: (x["priority"] == "high", x["confidence"]), reverse=True)
        
    def _generate_summary_metrics(self, atom_count: int, active_goals: List, 
                                 attention_dist: Dict) -> Dict[str, Any]:
        """Generate summary cognitive metrics."""
        return {
            "cognitive_load": random.uniform(0.3, 0.9),
            "attention_efficiency": random.uniform(0.5, 0.95),
            "goal_completion_rate": random.uniform(0.4, 0.85),
            "learning_velocity": random.uniform(0.1, 0.7),
            "pattern_recognition_accuracy": random.uniform(0.7, 0.96),
            "memory_utilization": random.uniform(0.6, 0.9),
            "reasoning_depth": random.randint(3, 8),
            "adaptive_capacity": random.uniform(0.4, 0.9),
            "emergent_behavior_index": random.uniform(0.2, 0.8)
        }
        
    def _generate_adaptive_patterns(self) -> Dict[str, Any]:
        """Generate adaptive cognitive pattern metadata."""
        return {
            "adaptation_rate": random.uniform(0.05, 0.3),
            "pattern_stability": random.uniform(0.6, 0.9),
            "novelty_detection": random.uniform(0.3, 0.8),
            "meta_learning_active": random.choice([True, False]),
            "cognitive_flexibility": random.uniform(0.4, 0.9)
        }
        
    def _generate_emergent_properties(self) -> Dict[str, Any]:
        """Generate emergent cognitive property indicators."""
        return {
            "self_organization_level": random.uniform(0.3, 0.8),
            "creativity_index": random.uniform(0.2, 0.7),
            "consciousness_indicators": {
                "global_workspace_activity": random.uniform(0.4, 0.9),
                "attention_coherence": random.uniform(0.5, 0.95),
                "temporal_binding": random.uniform(0.3, 0.8)
            },
            "distributed_cognition_sync": random.uniform(0.6, 0.95)
        }