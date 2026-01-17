"""
PlantUML tools for validation, rendering, and parsing diagrams.
"""

import re
import zlib
import base64
import logging
from typing import Dict, List, Any, Optional

import requests

try:
    from .models import PlantUMLResult
except ImportError:
    from models import PlantUMLResult

logger = logging.getLogger(__name__)


class PlantUMLTool:
    """
    Tool for validating and rendering PlantUML diagrams.
    
    This class interfaces with a PlantUML server to check syntax
    and generate diagram URLs.
    """
    
    def __init__(self, host: str = "http://localhost:8080"):
        """
        Initialize PlantUML tool.
        
        Args:
            host: PlantUML server host URL
        """
        self.host = host
        logger.info(f"PlantUML tool initialized with host: {host}")

    def extract_plantuml(self, text: str) -> str:
        """
        Extract PlantUML code from markdown blocks or raw text.
        
        Args:
            text: Text containing PlantUML code
            
        Returns:
            Extracted PlantUML code or empty string
        """
        if not text:
            return ""
        
        # Try to extract from ```plantuml ... ```
        fence_match = re.search(
            r"```\s*plantuml\s*(.*?)```",
            text,
            re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            return fence_match.group(1).strip()
        
        # Try to extract from @startuml ... @enduml
        tag_match = re.search(
            r"@startuml.*?@enduml",
            text,
            re.DOTALL | re.IGNORECASE
        )
        if tag_match:
            return tag_match.group(0).strip()
        
        return text.strip()

    def _encode_plantuml(self, plantuml_code: str) -> str:
        """
        Encode PlantUML code for URL.
        
        Args:
            plantuml_code: Raw PlantUML code
            
        Returns:
            URL-safe encoded string
        """
        code = plantuml_code.strip()
        
        if not code.startswith("@startuml"):
            code = f"@startuml\n{code}"
        if not code.endswith("@enduml"):
            code = f"{code}\n@enduml"
        
        compressed = zlib.compress(code.encode('utf-8'))[2:-4]
        encoded = base64.b64encode(compressed).translate(
            bytes.maketrans(
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
                b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
            )
        ).decode('utf-8')
        
        return encoded

    def get_diagram_url(self, plantuml_code: str, format: str = "png") -> str:
        """
        Generate a viewable URL for the PlantUML diagram.
        
        Args:
            plantuml_code: PlantUML diagram code
            format: Output format (png, svg, etc.)
            
        Returns:
            URL to view the diagram
        """
        diagram_code = self.extract_plantuml(plantuml_code)
        encoded = self._encode_plantuml(diagram_code)
        return f"{self.host}/{format}/{encoded}"
        
    def check_syntax(self, plantuml_code: str, timeout: int = 5) -> PlantUMLResult:
        """
        Validate PlantUML syntax with detailed error extraction.
        
        Args:
            plantuml_code: PlantUML code to validate
            timeout: Request timeout in seconds
            
        Returns:
            PlantUMLResult with validation status and detailed error if applicable.
        """
        logger.info("Validating PlantUML syntax")
        
        try:
            diagram_code = self.extract_plantuml(plantuml_code)
            encoded = self._encode_plantuml(diagram_code)
            
            url_png = f"{self.host}/png/{encoded}"
            response = requests.get(url_png, timeout=timeout)
            
            if response.status_code == 200 and response.content[:4] == b'\x89PNG':
                logger.info("Syntax validation passed (PNG rendered)")
                return PlantUMLResult(
                    is_valid=True,
                    url=url_png,
                    svg_url=f"{self.host}/svg/{encoded}"
                )
            
            logger.warning("PNG rendering failed. Fetching detailed syntax error...")
            url_txt = f"{self.host}/txt/{encoded}"
            error_response = requests.get(url_txt, timeout=timeout)
            
            detailed_error = error_response.text.strip()
            
            error_msg = f"PlantUML Syntax Error:\n{detailed_error}"
            logger.error(f"Syntax error detected: {error_msg}")
            
            return PlantUMLResult(
                is_valid=False,
                error=error_msg
            )
            
        except requests.exceptions.RequestException as e:
            error_msg = f"PlantUML Server Connection Error: {str(e)}"
            logger.error(error_msg)
            return PlantUMLResult(is_valid=False, error=error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error during syntax check: {str(e)}"
            logger.error(error_msg)
            return PlantUMLResult(is_valid=False, error=error_msg)


class PlantUMLParser:
    """
    Parser for extracting structured information from PlantUML diagrams.
    
    Extracts classes, attributes, and relationships from PlantUML code
    for evaluation purposes.
    """
    
    def __init__(self, plantuml_code: str):
        """
        Initialize parser with PlantUML code.
        
        Args:
            plantuml_code: PlantUML diagram code
        """
        self.plantuml_code = plantuml_code
        self.classes: Dict[str, Dict[str, List[str]]] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.parse()
    
    def parse(self) -> None:
        """Parse the PlantUML code."""
        try:
            self._extract_classes()
            self._extract_relationships()
            logger.debug(
                f"Parsed {len(self.classes)} classes and "
                f"{len(self.relationships)} relationships"
            )
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
    
    def _extract_classes(self) -> None:
        """Extract class definitions and their attributes."""
        class_pattern = r'class\s+(\w+)\s*\{([^}]*)\}'
        matches = re.finditer(
            class_pattern,
            self.plantuml_code,
            re.MULTILINE | re.DOTALL
        )
        
        for match in matches:
            class_name = match.group(1)
            class_body = match.group(2)
            
            attributes = []
            for line in class_body.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('--'):
                    attributes.append(line)
            
            self.classes[class_name] = {'attributes': attributes}
    
    def _extract_relationships(self) -> None:
        """Extract relationships between classes with cardinalities."""
        patterns = [
            # Generalization (either direction)
            (
                r'(\w+)\s*(?:"([^"]*)")?\s*<\|--\s*(?:"([^"]*)")?\s*(\w+)',
                'generalization'
            ),
            (
                r'(\w+)\s*(?:"([^"]*)")?\s*--|>\s*(?:"([^"]*)")?\s*(\w+)',
                'generalization'
            ),
            # Composition (either direction)
            (
                r'(\w+)\s*(?:"([^"]*)")?\s*\*--\s*(?:"([^"]*)")?\s*(\w+)',
                'composition'
            ),
            (
                r'(\w+)\s*(?:"([^"]*)")?\s*--\*\s*(?:"([^"]*)")?\s*(\w+)',
                'composition'
            ),
            # Aggregation (either direction)
            (
                r'(\w+)\s*(?:"([^"]*)")?\s*o--\s*(?:"([^"]*)")?\s*(\w+)',
                'aggregation'
            ),
            (
                r'(\w+)\s*(?:"([^"]*)")?\s*--o\s*(?:"([^"]*)")?\s*(\w+)',
                'aggregation'
            ),
            # Directed Association (either direction)
            (
                r'(\w+)\s*(?:"([^"]*)")?\s*-->\s*(?:"([^"]*)")?\s*(\w+)',
                'association'
            ),
            (
                r'(\w+)\s*(?:"([^"]*)")?\s*<--\s*(?:"([^"]*)")?\s*(\w+)',
                'association'
            ),
            # Simple Association (no arrow)
            (
                r'(\w+)\s*(?:"([^"]*)")?\s*--\s*(?:"([^"]*)")?\s*(\w+)',
                'association'
            ),
        ]
        
        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, self.plantuml_code):
                source = match.group(1)
                target = match.group(4)
                
                # Skip if source or target is None or empty
                if not source or not target:
                    continue
                
                self.relationships.append({
                    'type': rel_type,
                    'source': source,
                    'target': target,
                    'cardinality_source': (
                        match.group(2) if match.lastindex >= 2 else None
                    ),
                    'cardinality_target': (
                        match.group(3) if match.lastindex >= 3 else None
                    )
                })
