"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Trading stays disabled by default."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    okx_api_key: str = ""
    okx_secret_key: str = ""
    okx_passphrase: str = ""
    okx_base_url: str = "https://www.okx.com"
    okx_timeout_seconds: float = Field(default=10.0, gt=0)

    # Hard safety defaults — Phase 1 must not trade.
    trading_enabled: bool = False
    live_trading_enabled: bool = False

    log_level: str = "INFO"

    def assert_trading_disabled(self) -> None:
        """Refuse to proceed if live trading was accidentally enabled."""
        if self.live_trading_enabled:
            raise RuntimeError(
                "LIVE_TRADING_ENABLED is true, but this Phase 1 build "
                "does not support live trading. Set LIVE_TRADING_ENABLED=false."
            )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
