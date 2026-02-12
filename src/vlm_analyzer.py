"""
VLM Analyzer Module — MiniCPM-V 2.6

Performs structured visual understanding on presentation slide images
using MiniCPM-V 2.6 via Ollama's multimodal API.

Outputs structured JSON with components, connections, technologies,
and pipeline summary extracted from workflow/architecture diagrams.
"""
import json
import re
import ollama
from typing import Dict, Any, Optional, List
from .config import VLM_MODEL, VLM_MAX_TOKENS, ENABLE_VLM


# ============= VLM PROMPT TEMPLATE =============

VLM_PROMPT = """You are a vision-language model performing structured workflow extraction.

From the given image:

1. Identify all components in the workflow.
2. Identify directed relationships between components.
3. Detect any mentioned technologies (frameworks, tools, ML models, APIs, databases).
4. Infer the high-level purpose of the workflow.
5. DO NOT evaluate quality.
6. DO NOT provide suggestions.
7. DO NOT generate long explanations.
8. Output strictly valid JSON.

Return only JSON in this exact format:
{"components": [{"name": "", "type": "", "description": ""}], "connections": [{"from": "", "to": "", "relation": ""}], "technologies_detected": [], "pipeline_summary": ""}"""


# ============= EMPTY/DEFAULT RESULT =============

def _empty_result() -> Dict[str, Any]:
    """Return an empty VLM result structure."""
    return {
        "components": [],
        "connections": [],
        "technologies_detected": [],
        "pipeline_summary": ""
    }


# ============= JSON PARSING =============

def _parse_vlm_json(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from VLM response with robust error handling.
    Handles markdown code blocks, partial JSON, and common formatting issues.
    """
    if not response_text:
        return None

    text = response_text.strip()

    # Strip markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()

    # Try direct parse
    try:
        data = json.loads(text)
        return _validate_vlm_json(data)
    except json.JSONDecodeError:
        pass

    # Find JSON by braces
    start = text.find("{")
    if start != -1:
        depth, end = 0, start
        for i, c in enumerate(text[start:]):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = start + i + 1
                    break
        try:
            data = json.loads(text[start:end])
            return _validate_vlm_json(data)
        except json.JSONDecodeError:
            # Try fixing trailing commas
            cleaned = re.sub(r",\s*([}\]])", r"\1", text[start:end])
            try:
                data = json.loads(cleaned)
                return _validate_vlm_json(data)
            except json.JSONDecodeError:
                pass

    print(f"  ⚠️ VLM JSON parse failed. Response: {text[:200]}...")
    return None


def _validate_vlm_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize VLM output against the expected schema."""
    result = _empty_result()

    # Components
    if "components" in data and isinstance(data["components"], list):
        for comp in data["components"]:
            if isinstance(comp, dict) and "name" in comp:
                result["components"].append({
                    "name": str(comp.get("name", "")),
                    "type": str(comp.get("type", "")),
                    "description": str(comp.get("description", ""))
                })

    # Connections
    if "connections" in data and isinstance(data["connections"], list):
        for conn in data["connections"]:
            if isinstance(conn, dict) and "from" in conn and "to" in conn:
                result["connections"].append({
                    "from": str(conn.get("from", "")),
                    "to": str(conn.get("to", "")),
                    "relation": str(conn.get("relation", ""))
                })

    # Technologies
    if "technologies_detected" in data and isinstance(data["technologies_detected"], list):
        result["technologies_detected"] = [
            str(t) for t in data["technologies_detected"] if t
        ]

    # Summary
    if "pipeline_summary" in data:
        result["pipeline_summary"] = str(data["pipeline_summary"])[:500]

    return result


# ============= VLM INFERENCE =============

def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Run MiniCPM-V 2.6 on a single image and return structured JSON.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with components, connections, technologies_detected, pipeline_summary
    """
    if not ENABLE_VLM:
        return _empty_result()

    try:
        response = ollama.chat(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": VLM_PROMPT,
                    "images": [image_path]
                }
            ],
            options={
                "temperature": 0,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": VLM_MAX_TOKENS,
            }
        )

        if response and "message" in response:
            content = response["message"]["content"]
            parsed = _parse_vlm_json(content)
            if parsed:
                return parsed

        print("  ⚠️ VLM returned empty or unparseable response")
        return _empty_result()

    except Exception as e:
        print(f"  ❌ VLM inference error: {e}")
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            print("  💡 CUDA OOM: Try reducing VLM_IMAGE_MAX_DIM in config.py")
        return _empty_result()


def _merge_vlm_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge VLM results from multiple slides into a single consolidated output.
    De-duplicates components and technologies across slides.
    """
    merged = _empty_result()

    seen_components = set()
    seen_connections = set()
    all_technologies = set()
    summaries = []

    for result in results:
        # Merge components (de-duplicate by name)
        for comp in result.get("components", []):
            comp_key = comp.get("name", "").lower().strip()
            if comp_key and comp_key not in seen_components:
                seen_components.add(comp_key)
                merged["components"].append(comp)

        # Merge connections (de-duplicate by from+to)
        for conn in result.get("connections", []):
            conn_key = f"{conn.get('from', '').lower()}->{conn.get('to', '').lower()}"
            if conn_key not in seen_connections:
                seen_connections.add(conn_key)
                merged["connections"].append(conn)

        # Merge technologies
        for tech in result.get("technologies_detected", []):
            all_technologies.add(tech)

        # Collect summaries
        summary = result.get("pipeline_summary", "")
        if summary:
            summaries.append(summary)

    merged["technologies_detected"] = sorted(list(all_technologies))

    # Combine summaries into one
    if summaries:
        if len(summaries) == 1:
            merged["pipeline_summary"] = summaries[0]
        else:
            merged["pipeline_summary"] = " | ".join(summaries)[:500]

    return merged


def analyze_presentation(image_paths: List[str]) -> Dict[str, Any]:
    """
    Run VLM analysis on multiple slide images and return consolidated JSON.

    Args:
        image_paths: List of paths to slide images

    Returns:
        Consolidated dictionary with merged components, connections, technologies
    """
    if not ENABLE_VLM:
        print("  ⏭️ VLM analysis skipped (ENABLE_VLM=false)")
        return _empty_result()

    if not image_paths:
        print("  ⚠️ No slide images provided for VLM analysis")
        return _empty_result()

    print(f"  🔬 Running MiniCPM-V 2.6 on {len(image_paths)} slide(s)...")
    results = []

    for i, img_path in enumerate(image_paths):
        print(f"    Slide {i+1}/{len(image_paths)}...", end=" ")
        result = analyze_image(img_path)

        num_components = len(result.get("components", []))
        if num_components > 0:
            print(f"✓ ({num_components} components detected)")
        else:
            print("○ (no components)")

        results.append(result)

    merged = _merge_vlm_results(results)

    total_components = len(merged["components"])
    total_connections = len(merged["connections"])
    total_tech = len(merged["technologies_detected"])
    print(f"  📊 VLM Summary: {total_components} components, {total_connections} connections, {total_tech} technologies")

    return merged


def vlm_result_to_context(vlm_data: Dict[str, Any]) -> str:
    """
    Serialize VLM result into a compact text context for the Llama prompt.
    
    Args:
        vlm_data: VLM analysis result dictionary
        
    Returns:
        Formatted string for inclusion in the evaluation prompt
    """
    if not vlm_data or not vlm_data.get("components"):
        return ""

    try:
        return json.dumps(vlm_data, indent=2)
    except Exception:
        return ""
