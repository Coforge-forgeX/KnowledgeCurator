"""
Global storage configuration for the KB curator agent.
Imported by main.py to initialize the storage client, and by tools to access it.
"""

import os
from typing import Optional
from common_adapters.storage import StorageSettings, StorageFactory, StorageClient

storage_client: Optional[StorageClient] = None
container_name: str = ""


def initialize_storage() -> tuple[Optional[StorageClient], str]:
    """
    Initialize the global storage client and container name.
    Called once at application startup (in main.py).
    """
    global storage_client, container_name

    try:
        provider = os.getenv('STORAGE_PROVIDER', '').strip().lower()
        if not provider:
            if os.getenv('AZURE_STORAGE_CONNECTION_STRING') or os.getenv('AZURE_BLOB_STORAGE_CONNECTION_STRING'):
                provider = 'azure'
            elif os.getenv('AWS_ACCESS_KEY_ID') or os.getenv('AWS_REGION'):
                provider = 'aws'
            else:
                raise ValueError("Storage provider not configured. Set STORAGE_PROVIDER or provide credentials.")
            os.environ['STORAGE_PROVIDER'] = provider

        # Support legacy Azure connection string variable
        if provider == 'azure' and not os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
            legacy_conn = os.getenv('AZURE_BLOB_STORAGE_CONNECTION_STRING')
            if legacy_conn:
                os.environ['AZURE_STORAGE_CONNECTION_STRING'] = legacy_conn

        storage_settings = StorageSettings.from_env()
        storage_client = StorageFactory.from_settings(storage_settings)
        print(f"✓ Storage client initialized: {provider}")

    except Exception as e:
        print(f"✗ Failed to initialize storage client: {e}")
        storage_client = None

    container_name = (
        os.getenv('STORAGE_CONTAINER_NAME') or
        os.getenv('STORAGE_BUCKET_NAME') or
        os.getenv('AZURE_BLOB_STORAGE_CONTAINER_NAME') or
        ''
    )

    return storage_client, container_name


def get_storage_client() -> Optional[StorageClient]:
    return storage_client


def get_container_name() -> str:
    return container_name
