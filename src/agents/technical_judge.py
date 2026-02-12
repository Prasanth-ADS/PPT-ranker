"""
Technical Judge Agent

Specializes in evaluating technical depth, architecture, and engineering quality.
Bias: Strict on technical explanation, ignores marketing strength.
"""
from typing import Dict, Any
from .base_agent import BaseAgent, STRICT_JSON_CONTRACT

class TechnicalJudge(BaseAgent):
    """Technical evaluation specialist"""
    
    def __init__(self, model_name: str = "qwen2.5:1.5b"):
        super().__init__(model_name, "Technical Judge")
        self.prompt_template = self._build_prompt()
    
    def _build_prompt(self) -> str:
        """Build specialized technical evaluation prompt"""
        return """You are a STRICT TECHNICAL JUDGE for a hackathon with deep expertise in system architecture and engineering.

YOUR SPECIALTY: Technical Depth & Engineering Quality
YOUR BIAS: Penalize lack of technical explanation. Ignore marketing fluff.

SUBMISSION:
Problem: {problem_statement}
Content: {ppt_content}
Visuals: {visual_analysis}

═══════════════════════════════════════════════════════════════
YOUR EVALUATION FOCUS (Weight: 60%)
═══════════════════════════════════════════════════════════════

1. ARCHITECTURE QUALITY (0-30 pts)
   
   Component Design (0-8):
   - Are modules/services well-separated?
   - Clear responsibilities?
   - Proper abstractions?
   
   Scalability (0-8):
   - Can handle 10x users/data?
   - Distributed architecture if needed?
   - Performance bottlenecks addressed?
   
   Technology Stack (0-7):
   - Appropriate choices for problem?
   - Production-ready tech?
   - Trade-offs explained?
   
   Integration (0-7):
   - Components connect logically?
   - APIs well-designed?
   - Data flow clear?

2. ALGORITHM/MODEL JUSTIFICATION (0-15):
   - Which algorithms/models used?
   - Why chosen over alternatives?
   - Baseline comparisons?
   - Complexity analysis?

3. TECHNICAL TRADE-OFFS (0-10):
   - Speed vs. accuracy discussed?
   - Cost vs. performance?
   - Simplicity vs. flexibility?

4. ENGINEERING REALISM (0-5):
   - Demonstrates real technical understanding?
   - Not just buzzwords?
   - Addresses edge cases?

═══════════════════════════════════════════════════════════════
SCORING LOGIC
═══════════════════════════════════════════════════════════════

- Only UI discussion, no backend → ≤15/60
- Basic architecture, no reasoning → 16-35/60
- Detailed system design + trade-offs → 36-50/60
- Production-grade architecture + justifications → 51-60/60

═══════════════════════════════════════════════════════════════
OUTPUT (JSON only, no markdown)
═══════════════════════════════════════════════════════════════

{{\"technical_scores\":{{\"architecture_quality\":0,\"algorithm_justification\":0,\"trade_offs\":0,\"engineering_realism\":0}},\"total_technical_score\":0,\"technical_reasoning\":{{\"architecture_analysis\":\"\",\"algorithm_analysis\":\"\",\"scalability_assessment\":\"\",\"critical_gaps\":[],\"technical_strengths\":[],\"improvements\":[]}}}}

BE STRICT. Penalize vague technical claims heavily.""" + STRICT_JSON_CONTRACT
    
    def evaluate(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate technical depth.
        
        Args:
            content: Dict with problem_statement, ppt_content, visual_analysis
            
        Returns:
            Dict with technical evaluation results
        """
        # Build prompt
        prompt = self.prompt_template.format(
            problem_statement=content.get('problem_statement', ''),
            ppt_content=content.get('ppt_content', ''),
            visual_analysis=content.get('visual_analysis', '')
        )
        
        # Call model with retry
        result, confidence = self._call_model_with_retry(prompt)
        
        if not result:
            default = self._default_result()
            default['judge_confidence'] = confidence
            return default
        
        # Validate and cap scores
        result = self._validate_scores(result)
        
        # Add metadata
        result['agent'] = 'Technical Judge'
        result['model'] = self.model_name
        result['judge_confidence'] = confidence
        
        return result
    
    def _validate_scores(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and cap scores at their maximum values"""
        max_scores = {
            'architecture_quality': 30,
            'algorithm_justification': 15,
            'trade_offs': 10,
            'engineering_realism': 5
        }
        
        # Cap individual scores
        if 'technical_scores' in result:
            for key, max_val in max_scores.items():
                if key in result['technical_scores']:
                    original = result['technical_scores'][key]
                    result['technical_scores'][key] = min(original, max_val)
                    if original > max_val:
                        print(f"  ⚠️ Capped {key}: {original} → {max_val}")
        
        # Recalculate and cap total score
        if 'technical_scores' in result:
            total = sum(result['technical_scores'].values())
            result['total_technical_score'] = min(total, 60)
            if total > 60:
                print(f"  ⚠️ Capped total_technical_score: {total} → 60")
        
        return result
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result on error"""
        return {
            'technical_scores': {
                'architecture_quality': 0,
                'algorithm_justification': 0,
                'trade_offs': 0,
                'engineering_realism': 0
            },
            'total_technical_score': 0,
            'technical_reasoning': {
                'architecture_analysis': 'Error: Could not evaluate',
                'algorithm_analysis': 'Error: Could not evaluate',
                'scalability_assessment': 'Error: Could not evaluate',
                'critical_gaps': ['Evaluation failed'],
                'technical_strengths': [],
                'improvements': []
            },
            'agent': 'Technical Judge',
            'model': self.model_name
        }
