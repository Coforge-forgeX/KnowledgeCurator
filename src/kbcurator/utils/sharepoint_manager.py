from typing import List, Optional, Dict, BinaryIO
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
import os
import logging
from datetime import datetime, timedelta
import msal
import requests
from pathlib import Path
# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class SharePointManager:
    """
    SharePoint manager that follows the same schema as BlobManager
    for seamless integration with existing vector_store
    """
    
    def __init__(self, client_id: str, client_secret: str, tenant_id: str, 
                 site_hostname: str, site_path: str):
        """
        Initialize SharePoint manager
        
        Args:
            client_id: Azure AD app client ID
            client_secret: Azure AD app client secret
            tenant_id: Azure AD tenant ID
            site_hostname: SharePoint site hostname (e.g., 'ntlgnoida.sharepoint.com')
            site_path: Site path (e.g., '/sites/DeliverySuccessStories')
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.site_hostname = site_hostname
        self.site_path = site_path
        self.access_token = None
        self.site_id = None
        
        # Initialize MSAL app
        self.authority = f"https://login.microsoftonline.com/{tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]
        self.app = msal.ConfidentialClientApplication(
            client_id,
            authority=self.authority,
            client_credential=client_secret
        )
        
    def authenticate(self) -> bool:
        """Authenticate and get access token"""
        try:
            result = self.app.acquire_token_for_client(scopes=self.scope)
            if "access_token" in result:
                self.access_token = result["access_token"]
                logger.info("Successfully authenticated with Microsoft Graph API")
                return True
            else:
                logger.error(f"Authentication failed: {result.get('error_description')}")
                return False
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def _ensure_authenticated(self):
        """Ensure we have a valid access token"""
        if not self.access_token:
            self.authenticate()
    
    def get_site_id(self) -> Optional[str]:
        """Get SharePoint site ID"""
        self._ensure_authenticated()
        
        try:
            site_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_hostname}:{self.site_path}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.get(site_url, headers=headers)
            response.raise_for_status()
            
            site_data = response.json()
            self.site_id = site_data.get('id')
            logger.info(f"Retrieved site ID: {self.site_id}")
            return self.site_id
        except Exception as e:
            logger.error(f"Error getting site ID: {str(e)}")
            return None
    
    def _get_all_files_recursive(self, folder_path: str = "") -> List[Dict]:
        """Recursively get all files from folder and subfolders"""
        all_files = []
        
        if not self.site_id:
            self.get_site_id()

        try:
            # Build URL
            if folder_path:
                folder_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/root:{folder_path}:/children"
            else:
                folder_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/root/children"
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = requests.get(folder_url, headers=headers)
            response.raise_for_status()
            
            items = response.json().get('value', [])
            
            for item in items:
                # If it's a folder, recurse
                if 'folder' in item:
                    subfolder_path = f"{folder_path}/{item['name']}" if folder_path else item['name']
                    all_files.extend(self._get_all_files_recursive(subfolder_path))
                
                # If it's a file, check extension filter
                elif 'file' in item:
                    all_files.append(item)
        except Exception as e:
            logger.error(f"Error listing folder contents: {str(e)}")
        
        return all_files
    
    def download_file(self, file_id: str) -> Optional[bytes]:
        """
        Download file content (matching BlobManager.download_blob signature)
        
        Args:
            file_id: SharePoint item ID
            
        Returns:
            File content as bytes
        """
        self._ensure_authenticated()
        
        try:
            # Get file metadata to get download URL
            file_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/items/{file_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.get(file_url, headers=headers)
            response.raise_for_status()
            
            file_data = response.json()
            download_url = file_data.get('@microsoft.graph.downloadUrl')
            
            if not download_url:
                logger.error(f"No download URL for file ID: {file_id}")
                return None
            
            # Download the file
            response = requests.get(download_url)
            response.raise_for_status()
            
            logger.info(f"Downloaded file ID {file_id} ({len(response.content)} bytes)")
            return response.content
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {str(e)}")
            return None
    
    def get_file_metadata(self, file_id: str) -> Optional[Dict]:
        """
        Get file metadata
        
        Args:
            file_id: SharePoint item ID
            
        Returns:
            File metadata dictionary
        """
        self._ensure_authenticated()
        
        try:
            file_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drive/items/{file_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.get(file_url, headers=headers)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error getting file metadata: {str(e)}")
            return None
    
    def get_data_from_file(self,content):
        
        try:
            endpoint = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')
            api_key = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY')
            if not endpoint or not api_key:
                print({"error": "Azure Document Intelligence endpoint or key not set in environment variables."})
            doc_client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(api_key))

            # Pass the stream directly to Document Intelligence
            poller = doc_client.begin_analyze_document(
                "prebuilt-read",
                body=AnalyzeDocumentRequest(bytes_source=content),
                locale="en-US"
            )
            result = poller.result()

            file_content = result.content
            return file_content
        
        except Exception as e:
            print({"error": f"Failed to process PDF with Document Intelligence: {e}"})
            return None


    def extract_data(self, folder_path: str = "") -> List[Dict]:
        """
        Get all documents with metadata in the exact schema used by vector_store
        This matches the format expected by your existing pipeline
        
        Args:
            folder_path: Folder path to extract from
            file_extensions: File types to include
            
        Returns:
            List of document dictionaries with schema:
            {
                'id': str,              # Unique identifier
                'name': str,            # File name
                'content': bytes,       # File content
                'text': str,            # Extracted text
                'metadata': {
                    'source': str,      # Source identifier
                    'file_type': str,   # File extension
                    'size': int,        # File size in bytes
                    'created_at': str,  # Creation timestamp
                    'modified_at': str, # Modification timestamp
                    'url': str          # Web URL
                }
            }
        """
        if not self.site_id:
            self.get_site_id()
        
        documents = []
        files = self._get_all_files_recursive(folder_path)
        
        logger.info(f"Found {len(files)} files to process")
        
        for file_item in files:
            try:
                file_id = file_item.get('id')
                file_name = file_item.get('name')
                
                # Download content
                content = self.download_file(file_id)
                if not content:
                    logger.warning(f"Could not download {file_name}")
                    continue
                
                # Extract text
                extension = Path(file_name).suffix.lower()
                text = None
                
                try:
                    endpoint = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')
                    api_key = os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY')
                    if not endpoint or not api_key:
                        print({"error": "Azure Document Intelligence endpoint or key not set in environment variables."})
                    doc_client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(api_key))

                    # Pass the stream directly to Document Intelligence
                    poller = doc_client.begin_analyze_document(
                        "prebuilt-read",
                        body=AnalyzeDocumentRequest(bytes_source=content),
                        locale="en-US"
                    )
                    result = poller.result()

                    file_content = result.content
                except Exception as e:
                    print({"error": f"Failed to process PDF with Document Intelligence: {e}"})
                
                # Build document in exact schema format
                doc = {
                    'id': file_id,
                    'name': file_name,
                    'content': file_content,
                    'text': text,
                    'metadata': {
                        'source': f"sharepoint://{self.site_hostname}{self.site_path}/{file_name}",
                        'file_type': extension,
                        'size': file_item.get('size', 0),
                        'created_at': file_item.get('createdDateTime', ''),
                        'modified_at': file_item.get('lastModifiedDateTime', ''),
                        'url': file_item.get('webUrl', ''),
                        'sharepoint_id': file_id,
                        'mime_type': file_item.get('file', {}).get('mimeType', ''),
                        'graph_url': file_item.get('@microsoft.graph.downloadUrl', '')
                    }
                }
                
                documents.append(doc)
                logger.info(f"Processed: {file_name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_item.get('name')}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(documents)} documents")
        return documents