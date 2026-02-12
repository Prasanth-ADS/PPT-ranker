"""
Base agent class for the multi-agent evaluation system.

Includes:
- Deterministic inference (temperature=0)
- Retry with correction prompt on JSON parse failure
- Score validation guard (integer enforcement, negative clamping)
- Judge confidence tracking
"""
import ollama
from typing import Dict, Any, Optional, Tuple
import json
import re
from ..config import OLLAMA_HOST

# Strict JSON enforcement contract — appended to every judge prompt
STRICT_JSON_CONTRACT = """

CRITICAL: You are inside a production scoring system. Your response will be parsed using json.loads(). If you output anything other than valid JSON, the evaluation will fail and your scores will be discarded.

RULES:
- Return ONLY valid JSON.
- Do NOT include markdown formatting.
- Do NOT include headings or explanations outside JSON.
- Don't wrap JSON in backticks or code blocks.
- The response MUST start with {{ and end with }}.
- All scores MUST be integers (no decimals, no floating point).
- No negative scores.
- No null values.
- No additional fields beyond the schema.
- No missing fields from the schema."""

# Correction prompt for retry
CORRECTION_PROMPT = """Your previous response was not valid JSON and could not be parsed. 

Return ONLY valid JSON matching the exact schema requested. No extra text, no markdown, no explanations. Start with {{ and end with }}."""


class BaseAgent:
    """Base class for all evaluation agents"""
    
    def __init__(self, model_name: str, agent_name: str):
        """
        Initialize the agent.
        
        Args:
            model_name: Ollama model to use
            agent_name: Human-readable agent name
        """
        self.model_name = model_name
        self.agent_name = agent_name
        self.prompt_template = ""
    
    def evaluate(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate content and return results.
        
        Args:
            content: Dictionary with evaluation content
            
        Returns:
            Dictionary with evaluation results
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def _call_model(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.0) -> Optional[str]:
        """
        Call Ollama model with deterministic inference parameters.
        
        Args:
            prompt: Prompt to send to model
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation (default 0 for deterministic)
            
        Returns:
            Model response text or None if error
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': temperature,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1,
                    'num_predict': max_tokens,
                }
            )
            
            if response and 'message' in response:
                return response['message']['content']
            return None
            
        except Exception as e:
            print(f"  ❌ {self.agent_name} error: {e}")
            return None
    
    def _call_model_with_retry(self, prompt: str, max_tokens: int = 1200) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Call model and parse JSON with one retry on failure.
        
        Returns:
            Tuple of (parsed_result_or_None, confidence_level)
            confidence_level: "high" if first attempt, "medium" if retry, "low" if both fail
        """
        # Attempt 1: Primary call
        response = self._call_model(prompt, max_tokens=max_tokens)
        
        if response:
            result = self._parse_json_response(response)
            if result:
                return self._validate_all_scores(result), "high"
        
        # Attempt 2: Retry with correction prompt
        print(f"  🔄 {self.agent_name}: JSON parse failed, retrying with correction prompt...")
        
        retry_prompt = prompt + "\n\n" + CORRECTION_PROMPT
        response2 = self._call_model(retry_prompt, max_tokens=max_tokens)
        
        if response2:
            result2 = self._parse_json_response(response2)
            if result2:
                print(f"  ✅ {self.agent_name}: Retry succeeded")
                return self._validate_all_scores(result2), "medium"
        
        # Both attempts failed
        print(f"  ❌ {self.agent_name}: Both JSON parse attempts failed")
        return None, "low"
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from model response with robust error handling.
        
        Recovery chain:
        1. Direct json.loads()
        2. Strip markdown code blocks → json.loads()
        3. Regex extract first JSON object (dotall) → json.loads()
        4. Fix trailing commas → json.loads()
        """
        if not response:
            return None
        
        text = response.strip()
            
        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Strip markdown code blocks
        if '```' in text:
            if '```json' in text:
                start = text.find('```json') + 7
            else:
                start = text.find('```') + 3
            end = text.find('```', start)
            if end != -1:
                stripped = text[start:end].strip()
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass
        
        # Strategy 3: Extract JSON by brace matching (dotall-safe)
        if '{' in text:
            start = text.find('{')
            depth = 0
            end = start
            for i, c in enumerate(text[start:]):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = start + i + 1
                        break
            
            json_candidate = text[start:end]
            
            # 3a: Direct parse of extracted block
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass
            
            # 3b: Fix trailing commas before closing braces/brackets
            fixed = re.sub(r',\s*([}\]])', r'\1', json_candidate)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
            
            # 3c: Remove control characters and retry
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', fixed)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        
        print(f"  ❌ {self.agent_name} JSON parse exhausted all strategies")
        print(f"  Response preview: {text[:200]}...")
        return None
    
    def _validate_all_scores(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Universal score validation guard.
        
        - Ensures all numeric fields are integers
        - Clamps negatives to 0
        - Does NOT recompute totals (each judge does that in _validate_scores)
        """
        return self._deep_fix_numbers(result)
    
    def _deep_fix_numbers(self, obj: Any) -> Any:
        """Recursively fix numeric values in a dict/list structure."""
        if isinstance(obj, dict):
            return {k: self._deep_fix_numbers(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_fix_numbers(item) for item in obj]
        elif isinstance(obj, float):
            # Convert float scores to int, clamp negatives
            return max(0, int(round(obj)))
        elif isinstance(obj, int):
            # Clamp negatives
            return max(0, obj) if obj < 0 else obj
        return obj
    
    def get_info(self) -> Dict[str, str]:
        """Get agent information"""
        return {
            'agent_name': self.agent_name,
            'model': self.model_name,
            'type': self.__class__.__name__
        }
