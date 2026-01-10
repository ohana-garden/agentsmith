from python.helpers import dotenv

async def get_root_password():
    """Get root password directly from environment - no RFC needed"""
    return _get_root_password()
    
def _provide_root_password(public_key_pem: str):
    """Legacy function - not used in containerized deployment"""
    return _get_root_password()

def _get_root_password():
    return dotenv.get_dotenv_value(dotenv.KEY_ROOT_PASSWORD) or ""

