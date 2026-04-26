from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # DB & cache
    database_url: str = "postgresql://cw_user:cw_pass@localhost:5432/citywhisper"
    redis_url: str = "redis://localhost:6379"

    # LLM (switched from OpenAI to Groq)
    # openai_api_key: str = "sk-proj-..."
    groq_api_key: str = os.getenv("GROQ_API_KEY")
    llm_model: str = "llama-3.3-70b-versatile"
    llm_max_tokens: int = 300

    # POI pipeline tuning
    confidence_threshold: float = 0.45
    lookahead_radius_m: int = 3000
    poi_search_radius_m: int = 500
    target_word_min: int = 55
    target_word_max: int = 80

    # Tracking & logs
    mlflow_tracking_uri: str = "./mlruns"
    log_level: str = "INFO"


settings = Settings()
