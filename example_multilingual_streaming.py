"""
Example demonstrating streaming TTS with ChatterboxMultilingualTTS.

This example shows how to use the new streaming methods to generate
multilingual audio incrementally for real-time playback.
"""

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def example_streaming_synthesis():
    """
    Example of streaming multilingual TTS synthesis.
    """
    print("=" * 60)
    print("Multilingual Streaming TTS Example")
    print("=" * 60)
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nInitializing model on {device}...")
    
    # Note: This example assumes you have the model downloaded.
    # For testing without downloading, you would use:
    # model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    # Example texts in different languages
    examples = [
        ("Hello, this is a test of the streaming synthesis system.", "en"),
        ("Hola, ¿cómo estás hoy?", "es"),
        ("Bonjour, comment allez-vous?", "fr"),
        ("你好，今天天气真不错。", "zh"),
        ("こんにちは、お元気ですか？", "ja"),
    ]
    
    print("\nThis example demonstrates the streaming API usage:")
    print("\nCode example:")
    print("-" * 60)
    print("""
    # Stream audio chunks as they are generated
    all_chunks = []
    for audio_chunk, metrics in model.generate_stream(
        text=text,
        language_id=lang,
        audio_prompt_path="reference.wav",  # Optional
        chunk_size=25,
        context_window=50,
        fade_duration=0.02,
        print_metrics=True
    ):
        # Process each chunk in real-time
        # e.g., play audio, stream to network, etc.
        all_chunks.append(audio_chunk)
        
        # Access metrics
        print(f"Chunk {metrics.chunk_count} - "
              f"Latency: {metrics.latency_to_first_chunk:.3f}s")
    
    # Combine all chunks for final audio
    full_audio = torch.cat(all_chunks, dim=-1)
    ta.save("output.wav", full_audio, model.sr)
    
    # Final metrics
    print(f"RTF: {metrics.rtf:.3f}")
    print(f"Total audio: {metrics.total_audio_duration:.3f}s")
    """)
    print("-" * 60)
    
    print("\nSupported languages:")
    supported_langs = ChatterboxMultilingualTTS.get_supported_languages()
    for code, name in sorted(supported_langs.items()):
        print(f"  {code}: {name}")
    
    print("\n" + "=" * 60)
    print("Key features of streaming synthesis:")
    print("=" * 60)
    print("✓ Real-time audio generation with low latency")
    print("✓ Incremental token-by-token synthesis")
    print("✓ Context window overlap for smooth audio transitions")
    print("✓ Fade-in smoothing to remove audio pops")
    print("✓ Detailed metrics (RTF, latency, chunk count)")
    print("✓ Support for all 23 languages")
    print("✓ CFG and repetition penalty for quality")
    print("✓ Watermarked output for responsible AI")
    
    print("\n" + "=" * 60)
    print("Example texts to test:")
    print("=" * 60)
    for text, lang in examples:
        print(f"\n[{lang}] {text}")


if __name__ == "__main__":
    example_streaming_synthesis()
