from src.config.settings import settings

def test_pipeline_config():
    print("--- Testing Config Ingestion ---")
    print(f"Environment: {settings.APP_ENV}")
    print(f"LLM Provider: {settings.LLM_PROVIDER}")
    print(f"Model Name: {settings.LLM_MODEL}")
    print(f"Vector DB URL: {settings.VECTOR_DB_URL}")
    print(f"Page Boost Weight: {settings.PAGE_BOOST_WEIGHT}")
    
    print(f"Obfuscated Key: {settings.LLM_API_KEY}")
    if settings.LLM_API_KEY.get_secret_value():
        print("API Key parsed and accessible successfully.")
    else:
        print("Warning: API Key is empty.")

if __name__ == "__main__":
    test_pipeline_config()