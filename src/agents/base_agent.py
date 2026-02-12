"""
Base agent class for the multi-agent evaluation system.
"""
import ollama
from typing import Dict, Any, Optional
import json
import re
from ..config import OLLAMA_HOST

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
    
    def _call_model(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> Optional[str]:
        """
        Call Ollama model with the given prompt.
        
        Args:
            prompt: Prompt to send to model
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Model response text or None if error
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )
            
            if response and 'message' in response:
                return response['message']['content']
            return None
            
        except Exception as e:
            print(f"  ❌ {self.agent_name} error: {e}")
            return None
    
    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from model response with robust error handling.
        
        Args:
            response: Model response string
            
        Returns:
            Parsed JSON dict or None if parse error
        """
        if not response:
            return None
            
        try:
            # Strip markdown code blocks
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                response = response[start:end].strip()
            
            # Try to find JSON object boundaries
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                response = response[start:end]
            
            # Attempt to parse JSON
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            # Try to fix common issues
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r',\s*([}\]])', r'\1', response)
                return json.loads(fixed)
            except:
                pass
            
            print(f"  ❌ {self.agent_name} JSON parse error: {e}")
            print(f"  Response: {response[:200]}...")
            return None
        except Exception as e:
            print(f"  ❌ {self.agent_name} unexpected error: {e}")
            return None
    
    def get_info(self) -> Dict[str, str]:
        """Get agent information"""
        return {
            'agent_name': self.agent_name,
            'model': self.model_name,
            'type': self.__class__.__name__
        }
