"""
Aggregator & Confidence Engine

Collects all judge outputs, normalizes scores, detects variance, and estimates confidence.
"""
from typing import Dict, Any, List
import statistics

class Aggregator:
    """Aggregates multiple judge evaluations into final score"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize aggregator.
        
        Args:
            weights: Dict mapping judge names to weights (must sum to 1.0)
        """
        if weights is None:
            # Default tech-focused weights
            self.weights = {
                'technical': 0.60,
                'product': 0.25,
                'execution': 0.15
            }
        else:
            self.weights = weights
        
        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def aggregate(self, judge_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate judge results into final evaluation.
        
        Judges with judge_confidence='low' are excluded from the weighted average.
        Remaining weights are rebalanced proportionally.
        
        Args:
            judge_results: Dict mapping judge type to their results
        
        Returns:
            Aggregated results with variance analysis and confidence
        """
        # Determine which judges are valid (not low confidence)
        valid_judges = {}
        excluded_judges = []
        
        for judge_type in ['technical', 'product', 'execution']:
            result = judge_results.get(judge_type, {})
            confidence = result.get('judge_confidence', 'high')
            
            if confidence == 'low':
                excluded_judges.append(judge_type)
                print(f"  ⚠️ Excluding {judge_type} judge (low confidence / parse failure)")
            else:
                valid_judges[judge_type] = result
        
        # Extract raw scores from valid judges only
        scores = {}
        for judge_type in ['technical', 'product', 'execution']:
            if judge_type in valid_judges:
                result = valid_judges[judge_type]
                if judge_type == 'technical':
                    scores[judge_type] = result.get('total_technical_score', 0)
                elif judge_type == 'product':
                    scores[judge_type] = result.get('total_product_score', 0)
                elif judge_type == 'execution':
                    scores[judge_type] = result.get('total_execution_score', 0)
            else:
                scores[judge_type] = 0  # Placeholder, won't be used in weighting
        
        # Rebalance weights for valid judges only
        if valid_judges:
            valid_weight_sum = sum(self.weights[j] for j in valid_judges)
            rebalanced_weights = {
                j: self.weights[j] / valid_weight_sum for j in valid_judges
            }
        else:
            # All judges failed — return zero score
            rebalanced_weights = {}
        
        # Normalize to 140-point scale
        max_raw = {'technical': 60, 'product': 60, 'execution': 40}
        max_contribution = {'technical': 84, 'product': 35, 'execution': 21}
        
        normalized_scores = {}
        for judge_type in ['technical', 'product', 'execution']:
            raw = scores[judge_type]
            normalized_scores[judge_type] = (raw / max_raw[judge_type]) * max_contribution[judge_type]
        
        # Calculate weighted final score (only from valid judges)
        if rebalanced_weights:
            final_score = sum(
                normalized_scores[j] * rebalanced_weights[j]
                for j in rebalanced_weights
            )
        else:
            final_score = 0.0
        
        # Variance analysis
        variance_analysis = self._analyze_variance(normalized_scores)
        
        # Confidence estimation (accounts for excluded judges)
        confidence = self._estimate_confidence(variance_analysis, scores)
        if excluded_judges:
            confidence['level'] = 'Low' if len(excluded_judges) >= 2 else 'Medium'
            confidence['percentage'] = max(20, confidence['percentage'] - 20 * len(excluded_judges))
            confidence['excluded_judges'] = excluded_judges
        
        # Determine category
        category = self._determine_category(final_score)
        
        # Compile all reasoning
        all_strengths, all_weaknesses, all_improvements = self._compile_insights(judge_results)
        
        return {
            'final_score': round(final_score, 1),
            'max_score': 140,
            'score_category': category,
            'judge_scores': {
                'technical': round(normalized_scores['technical'], 1),
                'product': round(normalized_scores['product'], 1),
                'execution': round(normalized_scores['execution'], 1)
            },
            'raw_judge_scores': scores,
            'weights_used': rebalanced_weights if rebalanced_weights else self.weights,
            'excluded_judges': excluded_judges,
            'variance_analysis': variance_analysis,
            'confidence': confidence,
            'aggregated_insights': {
                'top_strengths': all_strengths[:5],
                'critical_weaknesses': all_weaknesses[:5],
                'priority_improvements': all_improvements[:3]
            },
            'judge_details': judge_results
        }
    
    def _analyze_variance(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze score variance across judges"""
        score_list = list(scores.values())
        
        if len(score_list) < 2:
            return {'high_disagreement': False, 'variance': 0, 'std_dev': 0}
        
        variance = statistics.variance(score_list)
        std_dev = statistics.stdev(score_list)
        max_diff = max(score_list) - min(score_list)
        
        # Flag high disagreement if std dev > 15% of mean
        mean_score = statistics.mean(score_list)
        high_disagreement = std_dev > (mean_score * 0.15)
        
        return {
            'high_disagreement': high_disagreement,
            'variance': round(variance, 2),
            'std_dev': round(std_dev, 2),
            'max_difference': round(max_diff, 2),
            'disagreement_interpretation': self._interpret_disagreement(scores)
        }
    
    def _interpret_disagreement(self, scores: Dict[str, float]) -> str:
        """Interpret what disagreement pattern means"""
        tech = scores['technical']
        prod = scores['product']
        exec_score = scores['execution']
        
        interpretations = []
        
        if tech > prod + 10 and tech > exec_score + 10:
            interpretations.append("Strong engineering but weak business positioning")
        elif prod > tech + 10:
            interpretations.append("Marketing-heavy but technically shallow")
        elif exec_score > tech + 5 and exec_score > prod + 5:
            interpretations.append("Built something workable but concept may be weak")
        elif tech < 20 and prod < 15 and exec_score < 10:
            interpretations.append("Consensus: Weak across all dimensions")
        elif tech > 60 and prod > 25 and exec_score > 15:
            interpretations.append("Strong agreement: High quality submission")
        else:
            interpretations.append("Balanced evaluation across judges")
        
        return " | ".join(interpretations)
    
    def _estimate_confidence(self, variance_analysis: Dict[str, Any], scores: Dict[str, float]) -> Dict[str, Any]:
        """Estimate confidence in the evaluation"""
        std_dev = variance_analysis['std_dev']
        mean_score = statistics.mean(list(scores.values()))
        
        # Confidence based on variance
        if std_dev < mean_score * 0.1:
            level = "High"
            percentage = 90
        elif std_dev < mean_score * 0.15:
            level = "Medium-High"
            percentage = 75
        elif std_dev < mean_score * 0.25:
            level = "Medium"
            percentage = 60
        else:
            level = "Low"
            percentage = 40
        
        return {
            'level': level,
            'percentage': percentage,
            'reasoning': f"Judge agreement: {100 - int(std_dev/mean_score*100) if mean_score > 0 else 0}%"
        }
    
    def _determine_category(self, score: float) -> str:
        """Determine competitive category based on score"""
        if score >= 120:
            return "Top-tier / Likely Winner"
        elif score >= 100:
            return "Strong Contender"
        elif score >= 75:
            return "Average / Needs Refinement"
        elif score >= 50:
            return "Weak / Major Gaps"
        else:
            return "Non-competitive"
    
    def _compile_insights(self, judge_results: Dict[str, Dict[str, Any]]) -> tuple:
        """Compile strengths, weaknesses, and improvements from all judges"""
        all_strengths = []
        all_weaknesses = []
        all_improvements = []
        
        for judge_type, result in judge_results.items():
            reasoning_key = f"{judge_type}_reasoning"
            reasoning = result.get(reasoning_key, {})
            
            # Defensive handling: ensure reasoning is a dict
            if not isinstance(reasoning, dict):
                print(f"  ⚠️ Warning: {reasoning_key} is not a dict, skipping insights for {judge_type}")
                continue
            
            # Extract strengths
            if judge_type == 'technical':
                all_strengths.extend(reasoning.get('technical_strengths', []))
                all_weaknesses.extend(reasoning.get('critical_gaps', []))
            elif judge_type == 'product':
                all_strengths.extend(reasoning.get('product_strengths', []))
                all_weaknesses.extend(reasoning.get('product_gaps', []))
            elif judge_type == 'execution':
                all_strengths.extend(reasoning.get('execution_strengths', []))
                all_weaknesses.extend(reasoning.get('practical_concerns', []))
            
            # Extract improvements
            all_improvements.extend(reasoning.get('improvements', []))
        
        return all_strengths, all_weaknesses, all_improvements
