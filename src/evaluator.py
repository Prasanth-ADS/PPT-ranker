import json
import time
import hashlib
import os
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama
from .config import CACHE_DIR, ENABLE_CACHE, OLLAMA_MODEL

# Advanced Hackathon Evaluation Framework - Diagnostic & Improvement-Oriented
EVALUATION_PROMPT = """You are a senior hackathon judge evaluating a structured hackathon presentation.

You must:
- Score strictly
- Provide reasoning for every score
- Identify missing elements
- Suggest concrete improvements
- Estimate competitiveness

Be analytical. Be precise. Do NOT use motivational tone.

SUBMISSION TO EVALUATE:
Problem Statement: {problem_statement}
Presentation Content: {ppt_content}
Visual Assessment: {visual_analysis}

═══════════════════════════════════════════════════════════════
SCORING RUBRIC (140 points total)
═══════════════════════════════════════════════════════════════

1. PROBLEM DEFINITION & RELEVANCE (0-20)
   Evaluate: Specificity, Quantification, Clear target users, Theme alignment
   
   Scoring Logic:
   - Generic problem → ≤ 8
   - Some clarity but no data → 9-14
   - Clear, quantified, scoped → 15-20

2. SOLUTION & INNOVATION (0-25)
   Evaluate: Logical mapping to problem, Clear differentiation, Technical novelty, Avoidance of buzzwords
   
   Scoring Logic:
   - Generic AI usage → ≤ 10
   - Moderate improvement → 11-18
   - Strong differentiation + defensible idea → 19-25

3. TECHNICAL DEPTH (0-30)
   Evaluate: Architecture clarity, Algorithm/model justification, Trade-off discussion, Scalability thinking, Engineering realism
   
   Scoring Logic:
   - Only UI discussion → ≤ 10
   - Basic architecture without reasoning → 11-20
   - Detailed system design + trade-offs → 21-30

4. IMPLEMENTATION FEASIBILITY (0-15)
   Evaluate: Deployment realism, Stack coherence, Cost awareness, MVP completeness
   
   Scoring Logic:
   - Theoretical only → ≤ 6
   - Partial feasibility → 7-11
   - Clear deployment + constraints → 12-15

5. IMPACT & MARKET POTENTIAL (0-20)
   Evaluate: Defined user segment, Scale of impact, Sustainability, Revenue or value model
   
   Scoring Logic:
   - Vague impact → ≤ 8
   - Moderate impact reasoning → 9-14
   - Clear scalable impact → 15-20

6. DEMO QUALITY (0-20)
   Evaluate: Functional clarity, Logical workflow, Depth of demonstration

7. PRESENTATION & DEFENSIBILITY (0-10)
   Evaluate: Logical flow, Technical confidence, Q&A readiness

═══════════════════════════════════════════════════════════════
CATEGORY DEFINITIONS
═══════════════════════════════════════════════════════════════

120-140 → "Top-tier / Likely Winner"
100-119 → "Strong Contender"
75-99 → "Average / Needs Refinement"
50-74 → "Weak / Major Gaps"
Below 50 → "Non-competitive"

═══════════════════════════════════════════════════════════════
STRICT RULES
═══════════════════════════════════════════════════════════════

- Total must equal sum of section scores
- Missing sections → penalize strongly
- No text outside JSON
- Deterministic scoring
- Avoid generic praise

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON ONLY, no markdown)
═══════════════════════════════════════════════════════════════

{{"scores":{{"problem_definition":0,"solution_innovation":0,"technical_depth":0,"implementation_feasibility":0,"impact_market":0,"demo_quality":0,"presentation_defensibility":0}},"section_reasoning":{{"problem_definition":{{"why_this_score":"","key_gaps_detected":[],"improvement_actions":[]}},"solution_innovation":{{"why_this_score":"","key_gaps_detected":[],"improvement_actions":[]}},"technical_depth":{{"why_this_score":"","key_gaps_detected":[],"improvement_actions":[]}},"implementation_feasibility":{{"why_this_score":"","key_gaps_detected":[],"improvement_actions":[]}},"impact_market":{{"why_this_score":"","key_gaps_detected":[],"improvement_actions":[]}},"demo_quality":{{"why_this_score":"","key_gaps_detected":[],"improvement_actions":[]}},"presentation_defensibility":{{"why_this_score":"","key_gaps_detected":[],"improvement_actions":[]}}}},"overall_analysis":{{"total_score_out_of_140":0,"score_category":"","top_strengths":[],"critical_weaknesses":[],"top_3_priority_improvements":[],"winning_probability_estimate_percent":0,"confidence_level":""}},"rank_reason":"Concise 2-3 sentence final verdict"}}

Be critical and analytical. Begin evaluation now."""


def _get_cache_key(ppt_text, problem_statement, visual_context):
    """Generate cache key from content hash."""
    content = f"{ppt_text[:3000]}|{problem_statement}|{visual_context[:500]}"
    return hashlib.md5(content.encode()).hexdigest()


def _get_cached_result(cache_key):
    """Get cached result if available and valid."""
    if not ENABLE_CACHE:
        return None
    cache_path = os.path.join(CACHE_DIR, f"eval_{cache_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("total_score", 0) > 0:
                    return data
        except:
            pass
    return None


def _save_to_cache(cache_key, result):
    """Cache successful result."""
    if not ENABLE_CACHE or result.get("total_score", 0) == 0:
        return
    cache_path = os.path.join(CACHE_DIR, f"eval_{cache_key}.json")
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f)
    except:
        pass


def _parse_response(text):
    """Fast JSON extraction from model response."""
    text = text.strip()
    
    # Try direct parse
    try:
        result = json.loads(text)
        if "total_score" in result or "scores" in result:
            return _normalize(result)
    except:
        pass
    
    # Extract from code blocks
    if "```" in text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return _normalize(json.loads(match.group(1)))
            except:
                pass
    
    # Find JSON by braces
    start = text.find('{')
    if start != -1:
        depth, end = 0, start
        for i, c in enumerate(text[start:]):
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = start + i + 1
                    break
        try:
            return _normalize(json.loads(text[start:end]))
        except:
            # Clean and retry
            cleaned = re.sub(r'\s+', ' ', text[start:end])
            try:
                return _normalize(json.loads(cleaned))
            except:
                pass
    
    # Regex fallback
    result = {"scores": {}, "total_score": 0, "reasoning": {"pros": [], "cons": [], "technical_depth": ""}, "rank_reason": "Parsed"}
    
    for name in ["problem_alignment", "innovation", "technical_implementation", "execution_clarity", "impact", "aesthetics"]:
        m = re.search(rf'"{name}"\s*:\s*(\d+)', text)
        if m:
            result["scores"][name] = min(int(m.group(1)), 25)
    
    m = re.search(r'"total_score"\s*:\s*(\d+)', text)
    if m:
        result["total_score"] = min(int(m.group(1)), 100)
    elif result["scores"]:
        result["total_score"] = sum(result["scores"].values())
    
    m = re.search(r'"rank_reason"\s*:\s*"([^"]+)"', text)
    if m:
        result["rank_reason"] = m.group(1)[:150]
    
    return result


def _normalize(result):
    """Ensure result has required fields."""
    if "total_score" not in result:
        result["total_score"] = sum(result.get("scores", {}).values())
    if "reasoning" not in result:
        result["reasoning"] = {"pros": [], "cons": [], "technical_depth": "N/A"}
    if "rank_reason" not in result:
        result["rank_reason"] = "Evaluated"
    if "scores" not in result:
        result["scores"] = {}
    return result


def _call_model(prompt, max_new_tokens=800):
    """Call Ollama model."""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.1,
                'num_predict': max_new_tokens,
            }
        )
        
        # Extract the response content
        if response and 'message' in response:
            return response['message']['content']
        return None
        
    except Exception as e:
        print(f"  ❌ Ollama error: {e}")
        print(f"  💡 Make sure Ollama is running. Check with: ollama list")
        raise


def evaluate_submission(ppt_text, problem_statement, visual_context="", model_name="ollama"):
    """Evaluate a single submission using Ollama model."""
    cache_key = _get_cache_key(ppt_text, problem_statement, visual_context)
    cached = _get_cached_result(cache_key)
    if cached:
        return cached

    try:
        # Truncate for speed (less content = faster response)
        prompt = EVALUATION_PROMPT.format(
            problem_statement=problem_statement[:500],
            ppt_content=ppt_text[:10000],  # Reduced for local model
            visual_analysis=visual_context[:300]
        )
        
        response_text = _call_model(prompt)
        if not response_text:
            return {"total_score": 0, "rank_reason": "No response", "scores": {}, "reasoning": {}}
        
        result = _parse_response(response_text)
        
        if result.get("total_score", 0) > 0:
            _save_to_cache(cache_key, result)
        
        return result

    except Exception as e:
        return {
            "total_score": 0,
            "rank_reason": f"Error: {str(e)[:100]}",
            "scores": {},
            "reasoning": {"error": str(e)[:200]}
        }


def batch_evaluate(submissions, problem_statement, model_name="ollama", max_concurrent=1):
    """
    Evaluate submissions with optimized timing.
    Pre-checks cache for all submissions first.
    """
    results = {}
    uncached = []
    
    # PHASE 1: Check all caches first (instant, no model calls)
    print("  📋 Checking cache...")
    for sub in submissions:
        team_name = sub['team_name']
        cache_key = _get_cache_key(
            sub['ppt_text'],
            problem_statement,
            sub.get('visual_context', '')
        )
        cached = _get_cached_result(cache_key)
        if cached:
            print(f"  ✓ {team_name}: {cached.get('total_score', 0)} pts (cached)")
            results[team_name] = cached
        else:
            uncached.append(sub)
    
    if not uncached:
        print("  ✅ All results from cache!")
        return results
    
    # PHASE 2: Evaluate uncached submissions
    print(f"  🤖 Evaluating {len(uncached)} teams with {model_name}...")
    
    # No need to pre-load with Ollama - service handles model lifecycle
    
    last_call_time = 0
    
    for i, sub in enumerate(uncached):
        team_name = sub['team_name']
        
        # Small delay between requests to prevent memory issues
        elapsed = time.time() - last_call_time
        if elapsed < MIN_REQUEST_INTERVAL and i > 0:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        
        print(f"  [{i+1}/{len(uncached)}] {team_name}...")
        
        last_call_time = time.time()
        result = evaluate_submission(
            sub['ppt_text'],
            problem_statement,
            sub.get('visual_context', ''),
            model_name
        )
        results[team_name] = result
        
        score = result.get('total_score', 0)
        if score > 0:
            print(f"  ✓ {team_name}: {score} pts")
        else:
            print(f"  ✗ {team_name}: {result.get('rank_reason', 'Failed')[:50]}")
    
    return results


# ============= PARALLEL PROCESSING UTILITIES =============

def prepare_submissions_parallel(team_data, download_map, process_file_func, visual_scorer, detect_type_func, max_workers=4):
    """
    Prepare all submissions in parallel (extraction + visual scoring).
    This runs BEFORE any model calls.
    
    Returns: List of prepared submission dicts
    """
    submissions = []
    
    def process_team(row):
        team_name = row['Team Name']
        problem = row.get('Problem Statement', 'General Project')
        local_path = download_map.get(team_name)
        
        result = {
            "team_name": team_name,
            "problem_statement": problem,
            "ppt_text": "",
            "visual_context": "",
            "visual_data": {},
            "local_path": local_path
        }
        
        if local_path and os.path.exists(local_path):
            # Extract content
            content = process_file_func(local_path)
            result["ppt_text"] = content if content else ""
            
            # Visual scoring
            ftype = detect_type_func(local_path)
            visual_data = visual_scorer.evaluate(local_path, ftype)
            result["visual_data"] = visual_data
            result["visual_context"] = f"Visual:{visual_data.get('visual_score',0)}/15 {visual_data.get('feedback','')}"
        
        return result
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_team, row): row['Team Name'] for row in team_data}
        for future in as_completed(futures):
            try:
                sub = future.result()
                if sub["ppt_text"]:  # Only add if we got content
                    submissions.append(sub)
            except Exception as e:
                print(f"  Error processing {futures[future]}: {e}")
    
    return submissions


def evaluate_architecture(ppt_text, project_name, model_name="ollama"):
    """
    Perform a detailed architectural evaluation of the submission.
    Returns: String containing the detailed report.
    """
    # Create specific cache key for this mode
    content_hash = hashlib.md5(f"ARCH_EVAL|{project_name}|{ppt_text[:5000]}".encode()).hexdigest()
    
    if ENABLE_CACHE:
        cache_path = os.path.join(CACHE_DIR, f"arch_eval_{content_hash}.txt")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    print(f"  [CACHE HIT] Detail evaluation for {project_name}")
                    return f.read()
            except:
                pass

    try:
        print(f"  🧠 Generating Detailed Architecture Report for {project_name}...")
        
        # Format the prompt
        prompt = DETAILED_EVALUATION_PROMPT.format(
            project_name=project_name,
            ppt_content=ppt_text[:12000]  # Allow more context for detailed review
        )
        
        response_text = _call_model(prompt, max_new_tokens=1500) # Need more tokens for detailed report
        
        if not response_text:
            return "Error: No response from model."
            
        # Cache text result
        if ENABLE_CACHE and len(response_text) > 100:
             cache_path = os.path.join(CACHE_DIR, f"arch_eval_{content_hash}.txt")
             try:
                 with open(cache_path, 'w', encoding='utf-8') as f:
                     f.write(response_text)
             except:
                 pass
                 
        return response_text

    except Exception as e:
        return f"Error generating report: {str(e)}"
