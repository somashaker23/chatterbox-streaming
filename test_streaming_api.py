#!/usr/bin/env python3
"""
Test script to validate the streaming API implementation.

This script tests the API signatures and basic functionality of the
streaming methods without requiring the full model to be loaded.
"""

import sys
import inspect
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all required components can be imported"""
    print("Testing imports...")
    try:
        from chatterbox.mtl_tts import (
            ChatterboxMultilingualTTS,
            StreamingMetrics,
            SUPPORTED_LANGUAGES,
            punc_norm
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_api_signatures():
    """Test that streaming methods have correct signatures"""
    print("\nTesting API signatures...")
    
    # Import without loading dependencies
    import ast
    with open('src/chatterbox/mtl_tts.py', 'r') as f:
        tree = ast.parse(f.read())
    
    # Find ChatterboxMultilingualTTS class
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ChatterboxMultilingualTTS':
            methods = {m.name: m for m in node.body if isinstance(m, ast.FunctionDef)}
            
            # Check inference_stream
            if 'inference_stream' in methods:
                m = methods['inference_stream']
                args = [arg.arg for arg in m.args.args]
                print(f"✓ inference_stream found with {len(args)} parameters")
                # Check for Generator return type
                if m.returns and 'Generator' in ast.unparse(m.returns):
                    print("  ✓ Returns Generator type")
            else:
                print("✗ inference_stream not found")
                return False
            
            # Check _process_token_buffer
            if '_process_token_buffer' in methods:
                m = methods['_process_token_buffer']
                args = [arg.arg for arg in m.args.args]
                print(f"✓ _process_token_buffer found with {len(args)} parameters")
            else:
                print("✗ _process_token_buffer not found")
                return False
            
            # Check generate_stream
            if 'generate_stream' in methods:
                m = methods['generate_stream']
                args = [arg.arg for arg in m.args.args]
                print(f"✓ generate_stream found with {len(args)} parameters")
                # Verify required parameters
                required_params = ['text', 'language_id']
                for param in required_params:
                    if param in args:
                        print(f"  ✓ Required parameter '{param}' present")
                    else:
                        print(f"  ✗ Required parameter '{param}' missing")
                        return False
            else:
                print("✗ generate_stream not found")
                return False
    
    return True


def test_streaming_metrics():
    """Test StreamingMetrics dataclass"""
    print("\nTesting StreamingMetrics...")
    
    import ast
    with open('src/chatterbox/mtl_tts.py', 'r') as f:
        content = f.read()
        tree = ast.parse(content)
    
    # Find StreamingMetrics
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'StreamingMetrics':
            fields = [n.target.id for n in node.body if isinstance(n, ast.AnnAssign)]
            print(f"✓ StreamingMetrics found with {len(fields)} fields")
            
            required_fields = [
                'latency_to_first_chunk',
                'rtf',
                'total_generation_time',
                'total_audio_duration',
                'chunk_count'
            ]
            
            for field in required_fields:
                if field in fields:
                    print(f"  ✓ Field '{field}' present")
                else:
                    print(f"  ✗ Field '{field}' missing")
                    return False
            
            return True
    
    print("✗ StreamingMetrics not found")
    return False


def test_language_support():
    """Test that all 23 languages are supported"""
    print("\nTesting language support...")
    
    with open('src/chatterbox/mtl_tts.py', 'r') as f:
        content = f.read()
    
    # Extract SUPPORTED_LANGUAGES
    import re
    match = re.search(r'SUPPORTED_LANGUAGES = \{([^}]+)\}', content, re.DOTALL)
    if match:
        langs_text = match.group(1)
        lang_codes = re.findall(r'"(\w+)":', langs_text)
        print(f"✓ Found {len(lang_codes)} supported languages")
        
        if len(lang_codes) == 23:
            print("  ✓ Correct number of languages (23)")
        else:
            print(f"  ✗ Expected 23 languages, found {len(lang_codes)}")
            return False
        
        # Check for key languages
        key_langs = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ar', 'hi']
        for lang in key_langs:
            if lang in lang_codes:
                print(f"  ✓ Language '{lang}' supported")
            else:
                print(f"  ✗ Language '{lang}' missing")
                return False
        
        return True
    
    print("✗ SUPPORTED_LANGUAGES not found")
    return False


def test_documentation():
    """Test that methods have proper documentation"""
    print("\nTesting documentation...")
    
    with open('src/chatterbox/mtl_tts.py', 'r') as f:
        content = f.read()
    
    methods = ['inference_stream', '_process_token_buffer', 'generate_stream']
    for method in methods:
        # Find method and check for docstring
        pattern = rf'def {method}\([^)]*\).*?"""([^"]*)"""'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            docstring = match.group(1).strip()
            if len(docstring) > 50:
                print(f"✓ {method} has documentation ({len(docstring)} chars)")
            else:
                print(f"⚠ {method} has short documentation")
        else:
            print(f"✗ {method} missing docstring")
            return False
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Streaming TTS API Validation")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("API Signatures", test_api_signatures),
        ("StreamingMetrics", test_streaming_metrics),
        ("Language Support", test_language_support),
        ("Documentation", test_documentation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ All validation tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    import re
    sys.exit(main())
