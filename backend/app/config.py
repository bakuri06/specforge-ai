from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 600.0
    vision_model: str = "qwen2.5vl:7b"
    reasoning_model: str = "deepseek-r1:7b"
    formatter_model: str = "qwen2.5:7b"
    cors_origins: str = "http://localhost:5173"
    storage_dir: str = "./storage"


settings = Settings()
