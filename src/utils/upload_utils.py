import os
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from pathlib import Path

load_dotenv()


CLIENT_SECRET_FILE = os.getenv("CLIENT_SECRET_FILE")
FOLDER_ID = os.getenv("FOLDER_ID")
SCOPES = os.getenv("SCOPES").split(",")  


def authenticate_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)

def upload_image(local_path, folder_id=FOLDER_ID):
    service = authenticate_drive()

    local_path = Path(local_path)
    file_metadata = {
        'name': local_path.name,
        'parents': [folder_id] if folder_id else []  
    }
    media = MediaFileUpload(str(local_path), mimetype='image/jpeg')  

    try:
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        file_url = f"https://drive.google.com/file/d/{file['id']}/view"
        print(f"Uploaded {local_path.name} -> {file_url}")
        return file_url
    except Exception as e:
        print(f"Upload failed: {e}")
        return None
    
def upload_batch(metadata_list, folder_id=FOLDER_ID):
    for meta in metadata_list:
        if meta.get('thumbnail_path'):
            url = upload_image(meta['thumbnail_path'], folder_id)
            meta['thumbnail_url'] = url
        if meta.get('webp_path'):
            url = upload_image(meta['webp_path'], folder_id)
            meta['webp_url'] = url
    return metadata_list