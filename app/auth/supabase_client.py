import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "https://wulciqolussrmlerycsf.supabase.co"
)

SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "sb_publishable_RMgnIPlmgQKRp0vN7Txt3g_vzt8IxVa"
)

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)