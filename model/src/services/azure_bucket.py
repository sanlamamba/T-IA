import os

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


class AzureBucket:
    load_dotenv()
    CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')

    def __init__(self, container_name: str = "models-saves"):
        self.blob_service_client = BlobServiceClient.from_connection_string(self.CONNECTION_STRING)
        self.container_name = container_name
        self.container_saves = self.blob_service_client.get_container_client(container_name)

    def list_blobs(self):
        list_of_blobs = self.container_saves.list_blobs()
        for blob in list_of_blobs:
            print(blob.name)

    def get_blob(self, blob_name: str):
        return self.blob_service_client.get_blob_client(self.container_name, blob_name)
    
    def download_locally(
        self, 
        blob_name: str, 
        local_download_path: str = f"{os.getcwd()}/model/processed"
    ) -> None:
        print(f"Downloading model...")
        os.makedirs(local_download_path, exist_ok=True)
        download_file_path = os.path.join(local_download_path, os.path.basename(blob_name))

        with open(download_file_path, "wb") as download_file:
            download_file.write(self.container_saves.download_blob(blob_name).readall())

        print(f"Model downloaded to {download_file_path}")
            