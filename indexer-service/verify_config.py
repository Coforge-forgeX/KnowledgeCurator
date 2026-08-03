"""
Configuration Verification Script
Run this to check if your Azure OpenAI embedding settings are properly loaded.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.core.config import settings

    print("=" * 60)
    print("Configuration Verification")
    print("=" * 60)
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Script location: {os.path.dirname(__file__)}")

    print("\n" + "=" * 60)
    print("Azure OpenAI Embedding Configuration")
    print("=" * 60)

    # Check API Base
    api_base = settings.lightrag.AZURE_OPENAI_EMBEDDING_API_BASE
    print(f"\n✓ API Base: {'✓ SET' if api_base else '✗ NOT SET'}")
    if api_base:
        print(f"  Value: {api_base}")
    else:
        print(f"  ✗ AZURE_OPENAI_EMBEDDING_MODEL_API_BASE is not configured")

    # Check API Key
    api_key = settings.lightrag.AZURE_OPENAI_EMBEDDING_API_KEY
    print(f"\n✓ API Key: {'✓ SET' if api_key else '✗ NOT SET'}")
    if api_key:
        print(f"  Value: {api_key[:10]}...{api_key[-4:]} ({len(api_key)} chars)")
    else:
        print(f"  ✗ AZURE_OPENAI_EMBEDDING_MODEL_API_KEY is not configured")

    # Check Deployment
    deployment = settings.lightrag.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    print(f"\n✓ Deployment: {'✓ SET' if deployment else '✗ NOT SET'}")
    if deployment:
        print(f"  Value: {deployment}")
    else:
        print(f"  ✗ AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL is not configured")

    # Check API Version
    api_version = settings.lightrag.AZURE_OPENAI_EMBEDDING_API_VERSION
    print(f"\n✓ API Version: {api_version}")

    # Check if all required settings are present
    print("\n" + "=" * 60)
    if api_base and api_key and deployment:
        print("✓ All required settings are configured!")
        print("\nConstructed endpoint URL:")
        endpoint = f"{api_base}openai/deployments/{deployment}/embeddings?api-version={api_version}"
        print(f"  {endpoint}")
    else:
        print("✗ Missing required settings!")
        print("\nPlease ensure these environment variables are set in your .env file:")
        if not api_base:
            print("  - AZURE_OPENAI_EMBEDDING_MODEL_API_BASE")
        if not api_key:
            print("  - AZURE_OPENAI_EMBEDDING_MODEL_API_KEY")
        if not deployment:
            print("  - AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL")

    print("=" * 60)

except Exception as e:
    print(f"✗ Error loading configuration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
