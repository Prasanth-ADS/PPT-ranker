"""
Multi-Agent Evaluation Orchestrator

Coordinates the execution of all judge agents and aggregates results.
"""
from typing import Dict, Any
from .agents import TechnicalJudge, ProductJudge, ExecutionJudge, Aggregator

class MultiAgentEvaluator:
    """Orchestrates multi-agent evaluation"""
    
    def __init__(self, model_name: str = "qwen2.5:1.5b"):
        """
        Initialize multi-agent evaluator.
        
        Args:
            model_name: Ollama model to use for judges
        """
        self.model_name = model_name
        
        # Initialize judges
        self.technical_judge = TechnicalJudge(model_name)
        self.product_judge = ProductJudge(model_name)
        self.execution_judge = ExecutionJudge(model_name)
        
        # Initialize aggregator with tech-focused weights
        self.aggregator = Aggregator({
            'technical': 0.60,
            'product': 0.25,
            'execution': 0.15
        })
    
    def evaluate(self, problem_statement: str, ppt_content: str, visual_analysis: str = "") -> Dict[str, Any]:
        """
        Execute multi-agent evaluation.
        
        Args:
            problem_statement: Problem being addressed
            ppt_content: Extracted PPT text content
            visual_analysis: Visual assessment results
            
        Returns:
            Aggregated evaluation results from all judges
        """
        print("\n" + "="*70)
        print("🤖 MULTI-AGENT EVALUATION SYSTEM")
        print("="*70)
        
        # Prepare content for judges
        content = {
            'problem_statement': problem_statement,
            'ppt_content': ppt_content,
            'visual_analysis': visual_analysis
        }
        
        # Execute judges sequentially
        print("\n1️⃣ Technical Judge evaluating...")
        technical_result = self.technical_judge.evaluate(content)
        print(f"   ✅ Technical Score: {technical_result.get('total_technical_score', 0)}/60")
        
        print("\n2️⃣ Product & Innovation Judge evaluating...")
        product_result = self.product_judge.evaluate(content)
        print(f"   ✅ Product Score: {product_result.get('total_product_score', 0)}/60")
        
        print("\n3️⃣ Execution & Feasibility Judge evaluating...")
        execution_result = self.execution_judge.evaluate(content)
        print(f"   ✅ Execution Score: {execution_result.get('total_execution_score', 0)}/40")
        
        # Aggregate results
        print("\n4️⃣ Aggregating results...")
        judge_results = {
            'technical': technical_result,
            'product': product_result,
            'execution': execution_result
        }
        
        final_result = self.aggregator.aggregate(judge_results)
        
        print(f"\n{'='*70}")
        print(f"📊 FINAL SCORE: {final_result['final_score']}/140")
        print(f"📈 Category: {final_result['score_category']}")
        print(f"🎯 Confidence: {final_result['confidence']['level']} ({final_result['confidence']['percentage']}%)")

        
        if final_result['variance_analysis']['high_disagreement']:
            print(f"⚠️  High disagreement detected between judges")
            print(f"   {final_result['variance_analysis']['disagreement_interpretation']}")
        
        print(f"{'='*70}\n")
        
        return final_result
    
    def get_judge_info(self) -> Dict[str, Any]:
        """Get information about all judges"""
        return {
            'technical_judge': self.technical_judge.get_info(),
            'product_judge': self.product_judge.get_info(),
            'execution_judge': self.execution_judge.get_info(),
            'aggregator_weights': self.aggregator.weights
        }
