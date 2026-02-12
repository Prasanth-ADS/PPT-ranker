"""
Execution & Feasibility Judge Agent

Specializes in evaluating deployment realism, MVP completeness, and demo quality.
Bias: Practical, penalizes theoretical-only systems, rewards working prototypes.
"""
from typing import Dict, Any
from .base_agent import BaseAgent, STRICT_JSON_CONTRACT

class ExecutionJudge(BaseAgent):
    """Execution and feasibility evaluation specialist"""
    
    def __init__(self, model_name: str = "qwen2.5:1.5b"):
        super().__init__(model_name, "Execution & Feasibility Judge")
        self.prompt_template = self._build_prompt()
    
    def _build_prompt(self) -> str:
        """Build specialized execution evaluation prompt"""
        return """You are an EXECUTION & FEASIBILITY JUDGE for a hackathon with expertise in deployment and product delivery.

YOUR SPECIALTY: Deployment Realism, MVP Completeness, Demo Quality
YOUR BIAS: Penalize theoretical-only systems. Reward working prototypes over perfect plans.

SUBMISSION:
Problem: {problem_statement}
Content: {ppt_content}
Visuals: {visual_analysis}

═══════════════════════════════════════════════════════════════
YOUR EVALUATION FOCUS (Weight: 15%)
═══════════════════════════════════════════════════════════════

1. IMPLEMENTATION FEASIBILITY (0-15 pts)
   
   Deployment Strategy (0-6):
   - Clear deployment plan?
   - Hosting/infrastructure mentioned?
   - Environment (dev/staging/prod)?
   - CI/CD considerations?
   
   Tech Stack Realism (0-4):
   - Stack coherent and practical?
   - Not over-engineered?
   - Team can actually build this?
   
   Cost & Resource Awareness (0-3):
   - Infrastructure costs considered?
   - API costs mentioned?
   - Scalability vs. cost trade-offs?
   
   MVP Scope (0-2):
   - Is this buildable in hackathon timeframe?
   - Core features prioritized?
   - Realistic scope?

2. DEMO QUALITY (0-20 pts)
   
   Functionality (0-8):
   - Does demo actually work?
   - Core features demonstrated?
   - Not just mockups/slides?
   
   Workflow Clarity (0-6):
   - User flow clear and logical?
   - End-to-end process shown?
   - No confusing gaps?
   
   Technical Demonstration (0-6):
   - Shows technical capability?
   - Proves concept works?
   - Addresses key technical challenges?

3. EXECUTION RISKS (0-5 pts)
   
   - Identified integration challenges?
   - Edge cases considered?
   - Fallback strategies?
   - Testing mentioned?

═══════════════════════════════════════════════════════════════
SCORING LOGIC
═══════════════════════════════════════════════════════════════

- Pure theory, no demo → ≤10/40
- Mockups only, vague deployment → 11-20/40
- Working prototype, partial deployment plan → 21-30/40
- Complete demo + realistic deployment strategy → 31-40/40

═══════════════════════════════════════════════════════════════
OUTPUT (JSON only, no markdown)
═══════════════════════════════════════════════════════════════

{{\"execution_scores\":{{\"implementation_feasibility\":0,\"demo_quality\":0,\"execution_risks\":0}},\"total_execution_score\":0,\"execution_reasoning\":{{\"deployment_assessment\":\"\",\"demo_analysis\":\"\",\"feasibility_summary\":\"\",\"risk_factors\":[],\"execution_strengths\":[],\"practical_concerns\":[],\"improvements\":[]}}}}

REWARD WORKING DEMOS HEAVILY. Penalize theoretical hand-waving.""" + STRICT_JSON_CONTRACT
    
    def evaluate(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate execution and feasibility.
        
        Args:
            content: Dict with problem_statement, ppt_content, visual_analysis
            
        Returns:
            Dict with execution evaluation results
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
        result['agent'] = 'Execution & Feasibility Judge'
        result['model'] = self.model_name
        result['judge_confidence'] = confidence
        
        return result
    
    def _validate_scores(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and cap scores at their maximum values"""
        max_scores = {
            'implementation_feasibility': 15,
            'demo_quality': 20,
            'execution_risks': 5
        }
        
        # Cap individual scores
        if 'execution_scores' in result:
            for key, max_val in max_scores.items():
                if key in result['execution_scores']:
                    original = result['execution_scores'][key]
                    result['execution_scores'][key] = min(original, max_val)
                    if original > max_val:
                        print(f"  ⚠️ Capped {key}: {original} → {max_val}")
        
        # Recalculate and cap total score
        if 'execution_scores' in result:
            total = sum(result['execution_scores'].values())
            result['total_execution_score'] = min(total, 40)
            if total > 40:
                print(f"  ⚠️ Capped total_execution_score: {total} → 40")
        
        return result
    
    def _default_result(self) -> Dict[str, Any]:
        """Return default result on error"""
        return {
            'execution_scores': {
                'implementation_feasibility': 0,
                'demo_quality': 0,
                'execution_risks': 0
            },
            'total_execution_score': 0,
            'execution_reasoning': {
                'deployment_assessment': 'Error: Could not evaluate',
                'demo_analysis': 'Error: Could not evaluate',
                'feasibility_summary': 'Error: Could not evaluate',
                'risk_factors': ['Evaluation failed'],
                'execution_strengths': [],
                'practical_concerns': [],
                'improvements': []
            },
            'agent': 'Execution & Feasibility Judge',
            'model': self.model_name
        }
