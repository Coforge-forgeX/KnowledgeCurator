from kbcurator.server.server import mcp
from azure.storage.blob import generate_blob_sas, BlobSasPermissions, BlobServiceClient
import requests
from dotenv import load_dotenv
import os
import base64
from datetime import datetime, timedelta
import traceback

load_dotenv(os.path.abspath(os.path.join(os.getcwd(),'.env')))

@mcp.tool()
def fetch_blob_structure():
    """
    Fetches the directory structure from Azure Blob Storage and returns a dictionary:
    {domain1: [kb1, kb2, ...], domain2: [kb1, ...], ...}

    Returns:
        dict: {domain: [kb, ...], ...}
    """
    connection_string = os.getenv('AZURE_BLOB_STORAGE_CONNECTION_STRING')
    container_name = os.getenv('AZURE_BLOB_STORAGE_CONTAINER_NAME')

    if not connection_string or not container_name:
        return {}

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        kbs_dict = {}

        for blob in container_client.list_blobs():
            # Expecting blob.name like "domain/kb/filename"
            parts = blob.name.split('/')
            if len(parts) >= 2:
                domain, kb = parts[0], parts[1]
                if domain not in kbs_dict:
                    kbs_dict[domain] = set()
                kbs_dict[domain].add(kb)

        # Convert sets to lists for JSON serializability
        kbs_dict = {d: list(kbs) for d, kbs in kbs_dict.items()}

        return kbs_dict
    except Exception as e:
        return {"error": str(e)}

#@mcp.tool()
def upload_files_and_get_urls(container_name: str, file_path: str, file_names: list, file_contents: list, expiry_years: int = 10):
    """
    Uploads files to Azure Blob Storage and returns their long-lived SAS download URLs.

    Args:
        container_name (str): Name of the container where file will be uploaded.
        file_path (str): Path of the uploaded file.
        file_names (list): List of file names to upload.
        file_contents (list): List of file contents (bytes or str).
        expiry_years (int): Years until the SAS token expires (default: 10).

    Returns:
        dict: {file_name: download_url or error}
    """
    connection_string = os.getenv('AZURE_BLOB_STORAGE_CONNECTION_STRING')

    if not connection_string or not container_name:
        return {"error": "Azure Blob Storage configuration is missing."}

    if not (len(file_names) == len(file_contents)):
        return {"error": "file_names and file_contents must have the same length."}

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        result = {}
        account_name = str(blob_service_client.account_name) if blob_service_client.account_name else ""
        if not account_name:
            return {"error": "Could not determine Azure Storage account name."}
        for fname, fcontent in zip(file_names, file_contents):
            if isinstance(fcontent, str):
                try:
                    fcontent = base64.b64decode(fcontent)
                except Exception:
                    pass
            blob_path = file_path + f"/{fname}"
            blob_client = container_client.get_blob_client(blob_path)
            try:
                blob_client.upload_blob(fcontent, overwrite=True)
                expiry = datetime.now() + timedelta(days=365 * expiry_years)
                sas_token = generate_blob_sas(
                    account_name=account_name,
                    container_name=container_name,
                    blob_name=blob_path,
                    account_key=blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=expiry
                )
                download_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_path}?{sas_token}"
                result[fname] = download_url
            except Exception as e:
                result[fname] = f"Error: {str(e)}, Fcontent: {fcontent}"
        return result
    except Exception as e:
        return {"error": str(e)}

# @mcp.tool()
def upload_files_to_blob(domain: str, kbs: list, file_names: list, file_contents: list, index_flag: bool = True):
    """
    Uploads files to Azure Blob Storage under the specified domain and KB directories.

    Args:
        domain (str): The domain name (top-level directory).
        kbs (list): List of KB names (subdirectories).
        file_names (list): List of file names to upload.
        file_contents (list): List of file contents (bytes or str).

    Returns:
        str: Summary of uploaded files or error message.
    """
    if index_flag:
        connection_string = os.getenv('AZURE_BLOB_STORAGE_CONNECTION_STRING')
        container_name = os.getenv('AZURE_BLOB_STORAGE_CONTAINER_NAME')

        if not connection_string or not container_name:
            return "Azure Blob Storage configuration is missing."

        if not (len(file_names) == len(file_contents)):
            return "file_names and file_contents must have the same length."


        try:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service_client.get_container_client(container_name)
            uploaded_files = []

            # Accepts files as input: file_names and file_contents must be lists of same length
            # Each file will be uploaded to every specified KB under the domain
            for kb in kbs:
                for fname, fcontent in zip(file_names, file_contents):
                    # Decode base64 string to bytes
                    if isinstance(fcontent, str):
                        try:
                            fcontent = base64.b64decode(fcontent)
                        except Exception as decode_err:
                            return {"error": f"Base64 decode error for {fname}: {decode_err}"}
                    elif isinstance(fcontent, bytes):
                        pass  # already bytes
                    else:
                        return {"error": f"Unsupported data type: {type(fcontent)}"}
                    blob_path = f"{domain}/{kb}/{fname}"
                    blob_client = container_client.get_blob_client(blob_path)
                    blob_client.upload_blob(fcontent, overwrite=True)
                    uploaded_files.append(blob_path)

            return {"uploaded_files": uploaded_files}
        except Exception as e:
            return {"error": str(e)}
    
@mcp.tool()
def delete_files_from_blob(domain: str, kbs: list, file_names: list):
    """
    Deletes files from Azure Blob Storage under the specified domain and KB directories.

    Args:
        domain (str): The domain name (top-level directory).
        kbs (list): List of KB names (subdirectories).
        file_names (list): List of file names to delete.

    Returns:
        str: Summary of deleted files or error message.
    """
    connection_string = os.getenv('AZURE_BLOB_STORAGE_CONNECTION_STRING')
    container_name = os.getenv('AZURE_BLOB_STORAGE_CONTAINER_NAME')

    if not connection_string or not container_name:
        return "Azure Blob Storage configuration is missing."

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        deleted_files = []

        for kb in kbs:
            for fname in file_names:
                blob_path = f"{domain}/{kb}/{fname}"
                blob_client = container_client.get_blob_client(blob_path)
                try:
                    blob_client.delete_blob()
                    deleted_files.append(blob_path)
                except Exception as e:
                    # File might not exist, skip or log as needed
                    pass

        return f"Deleted files: {deleted_files}"
    except Exception as e:
        return f"Error deleting files: {e}"

# New MCP tool: fetch file and return downloadable URL
# @mcp.tool()
def get_blob_download_url(domain: str, kb: str, file_name: str, expiry_minutes: int = 5):
    """
    Generates a downloadable URL for a file in Azure Blob Storage using a SAS token.

    Args:
        domain (str): The domain name (top-level directory).
        kb (str): The KB name (subdirectory).
        file_name (str): The file name to fetch.
        expiry_minutes (int): Minutes until the SAS token expires (default: 30).

    Returns:
        dict: {"download_url": url} or {"error": message}
    """
    connection_string = os.getenv('AZURE_BLOB_STORAGE_CONNECTION_STRING')
    container_name = os.getenv('AZURE_BLOB_STORAGE_CONTAINER_NAME')

    if not connection_string or not container_name:
        return {"error": "Azure Blob Storage configuration is missing."}

    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_path = f"{domain}/{kb}/{file_name}"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)

        # Check if blob exists
        if not blob_client.exists():
            return {"error": f"File not found: {blob_path}"}

        print("Connected to Blob!")
        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_path,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now() + timedelta(minutes=expiry_minutes)
        )
        print("SAS token generated successfully!")
        download_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_path}?{sas_token}"
        print("Download URL generated successfully!")
        return {"download_url": download_url}
    except Exception as e:
        return {"error": str(e)}