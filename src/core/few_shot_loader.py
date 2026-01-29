"""
Few-shot example loader and formatter for UML generation prompts.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from src.core.logger import Logger


class FewShotLoader:
    """Loads and formats few-shot examples for different agents."""
    
    def __init__(self, examples_path: Path, diagrams_path: Path):
        """
        Initialize the few-shot loader.
        
        Args:
            examples_path: Path to the few_shot_examples.json file
            diagrams_path: Path to the diagrams.json file (for diagrams)
        """
        self.examples_path = examples_path
        self.diagrams_path = diagrams_path
        self._examples = None
        self._diagrams = None
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load examples from JSON file."""
        if self._examples is None:
            try:
                with open(self.examples_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both array format and object with "examples" key
                    if isinstance(data, list):
                        self._examples = data
                    elif isinstance(data, dict) and "examples" in data:
                        self._examples = data["examples"]
                    else:
                        Logger.log_error(f"Unexpected JSON structure in {self.examples_path}")
                        self._examples = []
                Logger.log_info(f"Loaded {len(self._examples)} few-shot examples")
            except Exception as e:
                Logger.log_error(f"Failed to load few-shot examples: {e}")
                self._examples = []
        return self._examples
    
    def _load_diagrams(self) -> List[Dict[str, Any]]:
        """Load complete diagrams from JSON file."""
        if self._diagrams is None:
            try:
                with open(self.diagrams_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._diagrams = data if isinstance(data, list) else []
            except Exception as e:
                Logger.log_warning(f"Failed to load complete diagrams: {e}")
                self._diagrams = []
        return self._diagrams
    
    def _get_diagram_for_example(self, example_id: int) -> str:
        """
        Get the PlantUML diagram for a specific example ID.
        
        Args:
            example_id: The ID of the example
            
        Returns:
            The PlantUML diagram string or empty string if not found
        """
        diagrams = self._load_diagrams()
        for shot in diagrams:
            if shot.get("id") == example_id:
                return shot.get("diagram", "")
        return ""
    
    def format_for_class_extraction(self, num_examples: int = 3) -> str:
        """
        Format examples for the class extraction prompt.
        
        Args:
            num_examples: Number of examples to include
            
        Returns:
            Formatted string of examples with escaped curly braces
        """
        examples = self._load_examples()[:num_examples]
        
        if not examples:
            return "No examples available."
        
        formatted_parts = []
        
        for idx, example in enumerate(examples, 1):
            class_ext = example.get("class_extraction", {})
            requirements = class_ext.get("requirements", "")
            thought = class_ext.get("thought", "")
            classes = class_ext.get("classes", [])
            
            if not requirements or not classes:
                continue
            
            # Format the example
            example_text = f"""
            ## Example {idx}: {example.get('title', 'Untitled')}

            ### Requirements:
            {requirements}

            ### Analysis and Reasoning:
            {thought}

            ### Extracted Classes:
            """
            for cls in classes:
                attrs = cls.get("attributes", [])
                if attrs:
                    attrs_str = ", ".join([f"{a['name']}: {a['type']}" for a in attrs])
                    example_text += f"- {cls['name']} ({attrs_str})\n"
                else:
                    example_text += f"- {cls['name']} (no attributes)\n"
            
            formatted_parts.append(example_text)
        
        return "\n".join(formatted_parts)
    
    def format_for_relationship_extraction(self, num_examples: int = 3) -> str:
        """
        Format examples for the relationship extraction prompt.
        
        Args:
            num_examples: Number of examples to include
            
        Returns:
            Formatted string of examples with escaped curly braces
        """
        examples = self._load_examples()[:num_examples]
        
        if not examples:
            return "No examples available."
        
        formatted_parts = []
        
        for idx, example in enumerate(examples, 1):
            rel_ext = example.get("relationship_extraction", {})
            requirements = rel_ext.get("requirements", "")
            classes = rel_ext.get("classes", [])
            thought = rel_ext.get("thought", "")
            relationships = rel_ext.get("relationships", [])
            
            if not requirements or not relationships:
                continue
            
            # Format the example
            example_text = f"""
            ## Example {idx}: {example.get('title', 'Untitled')}

            ### Requirements:
            {requirements}

            ### Available Classes:
            {', '.join(classes)}

            ### Analysis and Reasoning:
            {thought}

            ### Extracted Relationships:
            """
            for rel in relationships:
                example_text += f"- {rel['source']} → {rel['target']} ({rel['type']})\n"
            
            formatted_parts.append(example_text)
        
        return "\n".join(formatted_parts)
    
    def format_for_generator(self, num_examples: int = 3) -> str:
        """
        Format examples for the PlantUML generator prompt.
        
        Args:
            num_examples: Number of examples to include
            
        Returns:
            Formatted string of examples with escaped curly braces
        """
        examples = self._load_examples()[:num_examples]
        
        if not examples:
            return "No examples available."
        
        formatted_parts = []
        
        for idx, example in enumerate(examples, 1):
            example_id = example.get("id")
            
            # Use class and relationship extraction data to show the plan
            class_ext = example.get("class_extraction", {})
            rel_ext = example.get("relationship_extraction", {})
            
            classes = class_ext.get("classes", [])
            relationships = rel_ext.get("relationships", [])
            
            # Get the diagram from complete_shots
            diagram = self._get_diagram_for_example(example_id)
            
            if not classes or not diagram:
                continue
            
            # Format the design plan
            plan_text = "## STRUCTURAL DECOMPOSITION\n\n### Classes:\n"
            for cls in classes:
                attrs = cls.get("attributes", [])
                if attrs:
                    attrs_str = ", ".join([f"{a['name']}: {a['type']}" for a in attrs])
                    plan_text += f"- {cls['name']} ({attrs_str})\n"
                else:
                    plan_text += f"- {cls['name']}\n"
            
            plan_text += "\n### Relationships:\n"
            for rel in relationships:
                plan_text += f"- {rel['source']} → {rel['target']} ({rel['type']})\n"
            
            # Format the example
            example_text = f"""
            ## Example {idx}: {example.get('title', 'Untitled')}

            ### Design Plan:
            {plan_text}

            ### Generated PlantUML:
            ```plantuml
            {diagram}
            ```
            """
            formatted_parts.append(example_text)
        
        return "\n".join(formatted_parts)
