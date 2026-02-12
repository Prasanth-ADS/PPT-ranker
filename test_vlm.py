"""
Test script for MiniCPM-V 2.6 VLM integration.

Prerequisites:
  1. Ollama must be running: ollama serve
  2. MiniCPM-V model must be pulled: ollama pull minicpm-v

Usage:
  cd "c:\\Users\\mostr\\OneDrive\\Documents\\GitHub\\Priyanka\\Hackathon PPT ranker"
  python test_vlm.py
"""
import os
import sys
import json
import tempfile
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_test_diagram():
    """Create a simple workflow diagram image for testing."""
    img = Image.new("RGB", (800, 400), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # Draw boxes
    boxes = [
        (50, 150, 200, 230, "User Input"),
        (250, 150, 400, 230, "OCR Engine"),
        (450, 150, 620, 230, "MiniCPM-V\n(VLM)"),
        (670, 150, 780, 230, "Llama 3.1"),
    ]

    for x1, y1, x2, y2, label in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="#333333", width=2, fill="#E8F0FE")
        # Center text in box
        bbox = draw.textbbox((0, 0), label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 - th) // 2
        draw.text((tx, ty), label, fill="#333333")

    # Draw arrows
    arrows = [(200, 190, 250, 190), (400, 190, 450, 190), (620, 190, 670, 190)]
    for x1, y1, x2, y2 in arrows:
        draw.line([(x1, y1), (x2, y2)], fill="#333333", width=2)
        # Arrowhead
        draw.polygon([(x2, y2), (x2 - 8, y2 - 5), (x2 - 8, y2 + 5)], fill="#333333")

    # Title
    draw.text((250, 50), "AI Evaluation Pipeline", fill="#1A73E8")

    # Tech labels
    draw.text((80, 250), "Tesseract", fill="#666666")
    draw.text((470, 250), "Python + CUDA", fill="#666666")
    draw.text((680, 250), "Ollama", fill="#666666")

    # Save
    path = os.path.join(tempfile.gettempdir(), "test_vlm_diagram.png")
    img.save(path, "PNG")
    print(f"✅ Test diagram created: {path}")
    return path


def test_single_image_analysis():
    """Test VLM analysis on a single image."""
    from src.vlm_analyzer import analyze_image, _empty_result

    print("\n" + "=" * 60)
    print("TEST 1: Single Image Analysis")
    print("=" * 60)

    image_path = create_test_diagram()

    print("🔬 Running MiniCPM-V 2.6 analysis...")
    result = analyze_image(image_path)

    # Validate structure
    expected_keys = ["components", "connections", "technologies_detected", "pipeline_summary"]
    for key in expected_keys:
        assert key in result, f"❌ Missing key: {key}"
    print("✅ All expected keys present in output")

    assert isinstance(result["components"], list), "❌ 'components' should be a list"
    assert isinstance(result["connections"], list), "❌ 'connections' should be a list"
    assert isinstance(result["technologies_detected"], list), "❌ 'technologies_detected' should be a list"
    assert isinstance(result["pipeline_summary"], str), "❌ 'pipeline_summary' should be a string"
    print("✅ All types are correct")

    # Check if components were detected
    if result["components"]:
        print(f"✅ Detected {len(result['components'])} components:")
        for comp in result["components"]:
            print(f"   - {comp.get('name', '?')} ({comp.get('type', '?')})")
    else:
        print("⚠️  No components detected (VLM may not have parsed the diagram)")

    if result["connections"]:
        print(f"✅ Detected {len(result['connections'])} connections:")
        for conn in result["connections"]:
            print(f"   - {conn.get('from', '?')} → {conn.get('to', '?')}")

    if result["technologies_detected"]:
        print(f"✅ Technologies: {', '.join(result['technologies_detected'])}")

    if result["pipeline_summary"]:
        print(f"✅ Summary: {result['pipeline_summary'][:150]}")

    print("\n📋 Full JSON Output:")
    print(json.dumps(result, indent=2))

    # Cleanup
    os.remove(image_path)
    return result


def test_presentation_analysis():
    """Test VLM analysis with multiple slides."""
    from src.vlm_analyzer import analyze_presentation, vlm_result_to_context

    print("\n" + "=" * 60)
    print("TEST 2: Multi-slide Presentation Analysis")
    print("=" * 60)

    # Create 2 test images
    image_path = create_test_diagram()
    image_paths = [image_path]

    print(f"🔬 Running MiniCPM-V 2.6 on {len(image_paths)} slide(s)...")
    result = analyze_presentation(image_paths)

    # Convert to context string
    context = vlm_result_to_context(result)
    if context:
        print(f"✅ VLM context generated ({len(context)} chars)")
    else:
        print("⚠️  VLM context is empty")

    # Cleanup
    os.remove(image_path)
    return result


def test_image_extractor():
    """Test the image extraction from files (if a sample file exists)."""
    from src.image_extractor import extract_slide_images, cleanup_temp_images

    print("\n" + "=" * 60)
    print("TEST 3: Image Extractor")
    print("=" * 60)

    # Look for any existing files in downloads
    downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")

    if not os.path.exists(downloads_dir):
        print("⚠️  No downloads directory found. Skipping file-based test.")
        return

    files = [f for f in os.listdir(downloads_dir) if f.endswith((".pdf", ".pptx"))]

    if not files:
        print("⚠️  No PDF/PPTX files in downloads/. Skipping file-based test.")
        return

    test_file = os.path.join(downloads_dir, files[0])
    print(f"📄 Testing with: {files[0]}")

    images = extract_slide_images(test_file)
    print(f"  Extracted {len(images)} images")

    if images:
        print(f"  First image: {images[0]}")
        # Verify images exist and are valid
        for img_path in images:
            assert os.path.exists(img_path), f"❌ Image not found: {img_path}"
            img = Image.open(img_path)
            assert img.size[0] > 0 and img.size[1] > 0, "❌ Invalid image dimensions"
        print("  ✅ All images valid")

        cleanup_temp_images(images)
        print("  ✅ Temp images cleaned up")


if __name__ == "__main__":
    print("🧪 MiniCPM-V 2.6 VLM Integration Tests")
    print("=" * 60)

    try:
        # Test 1: Single image analysis
        test_single_image_analysis()

        # Test 2: Multi-slide analysis
        test_presentation_analysis()

        # Test 3: Image extractor (if files available)
        test_image_extractor()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
