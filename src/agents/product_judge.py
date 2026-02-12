"""
Product & Innovation Judge Agent

Specializes in evaluating problem framing, market clarity, and differentiation.
Bias: Business-oriented, penalizes vague impact claims, ignores deep algorithm details.
"""
from typing import Dict, Any
from .base_agent import BaseAgent, STRICT_JSON_CONTRACT

class ProductJudge(BaseAgent):
    """Product and innovation evaluation specialist"""
    
    def __init__(self, model_name: str = "qwen2.5:1.5b"):
        super().__init__(model_name, "Product & Innovation Judge")
        self.prompt_template = self._build_prompt()
    
    def _build_prompt(self) -> str:
        """Build specialized product evaluation prompt"""
        return """You are a PRODUCT & MARKET JUDGE for a hackathon with expertise in business strategy and innovation.

YOUR SPECIALTY: Problem Framing, Market Clarity, Strategic Positioning
YOUR BIAS: Penalize vague impact claims. Ignore deep technical details unless they create differentiation.

SUBMISSION:
Problem: {problem_statement}
Content: {ppt_content}
Visuals: {visual_analysis}

═══════════════════════════════════════════════════════════════
YOUR EVALUATION FOCUS (Weight: 25%)
═══════════════════════════════════════════════════════════════

1. PROBLEM DEFINITION (0-20 pts)
   
   Clarity & Specificity (0-8):
   - Is the problem clearly defined?
   - Specific enough to be actionable?
   - Backed by data/statistics?
   
   Target Users (0-6):
   - Who exactly benefits?
   - User personas clear?
   - Market size mentioned?
   
   Current Gaps (0-6):
   - What exists today?
   - Why is it insufficient?
   - Evidence of gap?

2. SOLUTION INNOVATION (0-25 pts)
   
   Problem-Solution Mapping (0-10):
   - Does solution address stated problem?
   - All problem aspects covered?
   - Clear causal connection?
   
   Differentiation (0-10):
   - What makes this unique?
   - How is it better than alternatives?
   - Competitive advantage clear?
   
   Innovation Level (0-5):
   - Novel approach or incremental?
   - Technical innovation OR business model innovation?
   - Defensible uniqueness?

3. MARKET POTENTIAL (0-15 pts)
   
   Impact Scale (0-6):
   - How many people affected?
   - Economic or social significance?
   - Measurable outcomes?
   
   Business Viability (0-5):
   - Revenue model mentioned?
   - Cost structure realistic?
   - Sustainable long-term?
   
   Go-to-Market (0-4):
   - Target market strategy?
   - Distribution channels?
   - Adoption barriers addressed?

═══════════════════════════════════════════════════════════════
SCORING LOGIC
═══════════════════════════════════════════════════════════════

- Generic problem, no data → ≤15/60
- Some clarity, weak differentiation → 16-35/60
- Clear problem + strong differentiation → 36-50/60
- Exceptional market insight + defensible moat → 51-60/60

═══════════════════════════════════════════════════════════════
OUTPUT (JSON only, no markdown)
═══════════════════════════════════════════════════════════════

{{\"product_scores\":{{\"problem_definition\":0,\"solution_innovation\":0,\"market_potential\":0}},\"total_product_score\":0,\"product_reasoning\":{{\"problem_assessment\":\"\",\"innovation_analysis\":\"\",\"market_analysis\":\"\",\"differentiation_summary\":\"\",\"product_strengths\":[],\"product_gaps\":[],\"improvements\":[]}}}}

BE CRITICAL OF VAGUE CLAIMS. Reward specific, defensible positioning.""" + STRICT_JSON_CONTRACT
    
    def evaluate(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate product and innovation aspects.
        
        Args:
            content: Dict with problem_statement, ppt_content, visual_analysis
            
        Returns:
            Dict with product evaluation results
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
        result['agent'] = 'Product & Innovation Judge'
        result['model'] = self.model_name
        result['judge_confidence'] = confidence
        
        return result
    
    def _validate_scores(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and cap scores at their maximum values"""
        max_scores = {
            'problem_definition': 20,
            'solution_innovation': 25,
            'market_potential': 15
        }
        
        # Cap individual scores
        if 'product_scores' in result:
            for key, max_val in max_scores.items():
                if key in result['product_scores']:
                    original = result['product_scores'][key]
                    result['product_scores'][key] = min(original, max_val)
                    if original > max_val:
                        print(f"  ⚠️ Capped {key}: {original} → {max_val}")
        
        # Recalculate and cap total score
        if 'product_scores' in result:
            total = sum(result['product_scores'].values())
            result['total_product_score'] = min(total, 60)
            if total > 60:
                print(f"  ⚠️ Capped total_product_score: {total} → 60")
        
        return result
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result on error"""
        return {
            'product_scores': {
                'problem_definition': 0,
                'solution_innovation': 0,
                'market_potential': 0
            },
            'total_product_score': 0,
            'product_reasoning': {
                'problem_assessment': 'Error: Could not evaluate',
                'innovation_analysis': 'Error: Could not evaluate',
                'market_analysis': 'Error: Could not evaluate',
                'differentiation_summary': 'Error: Could not evaluate',
                'product_strengths': [],
                'product_gaps': ['Evaluation failed'],
                'improvements': []
            },
            'agent': 'Product & Innovation Judge',
            'model': self.model_name
        }
