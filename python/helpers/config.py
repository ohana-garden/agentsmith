"""
AgentSmith Configuration Loader
Loads from TOML with environment variable overrides
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from TOML file with environment overrides.
    
    Priority (highest to lowest):
    1. Environment variables
    2. TOML config file
    3. Default values
    """
    # Find config file
    if config_path is None:
        config_path = os.getenv("AGENTSMITH_CONFIG", "/a0/config/agentsmith.toml")
    
    config = {}
    
    # Load from TOML if exists
    if Path(config_path).exists():
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
    
    # Apply environment overrides
    config = _apply_env_overrides(config)
    
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config"""
    
    # Server settings
    if "server" not in config:
        config["server"] = {}
    config["server"]["port"] = int(os.getenv("PORT", os.getenv("WEB_UI_PORT", config.get("server", {}).get("port", 80))))
    config["server"]["host"] = os.getenv("HOST", config.get("server", {}).get("host", "0.0.0.0"))
    
    # LLM settings
    if "llm" not in config:
        config["llm"] = {}
    
    # Provider detection from API keys
    if os.getenv("API_KEY_OPENROUTER"):
        config["llm"]["provider"] = "openrouter"
        config["llm"]["api_key"] = os.getenv("API_KEY_OPENROUTER")
    elif os.getenv("OPENAI_API_KEY"):
        config["llm"]["provider"] = "openai"
        config["llm"]["api_key"] = os.getenv("OPENAI_API_KEY")
    elif os.getenv("ANTHROPIC_API_KEY"):
        config["llm"]["provider"] = "anthropic"
        config["llm"]["api_key"] = os.getenv("ANTHROPIC_API_KEY")
    
    # Explicit overrides
    if os.getenv("CHAT_MODEL_PROVIDER"):
        config["llm"]["provider"] = os.getenv("CHAT_MODEL_PROVIDER")
    if os.getenv("CHAT_MODEL_NAME"):
        config["llm"]["model"] = os.getenv("CHAT_MODEL_NAME")
    
    # Graph settings
    if "graph" not in config:
        config["graph"] = {}
    config["graph"]["host"] = os.getenv("FALKORDB_HOST", config.get("graph", {}).get("host", "localhost"))
    config["graph"]["port"] = int(os.getenv("FALKORDB_PORT", config.get("graph", {}).get("port", 16379)))
    config["graph"]["password"] = os.getenv("FALKORDB_PASSWORD", config.get("graph", {}).get("password", "CH@NG3M3N()W!"))
    config["graph"]["graph_name"] = os.getenv("GRAPH_NAME", config.get("graph", {}).get("graph_name", "agentsmith"))
    config["graph"]["enabled"] = os.getenv("GRAPH_ENABLED", "true").lower() == "true"
    
    # MCP settings
    if "mcp" not in config:
        config["mcp"] = {}
    if os.getenv("MCP_SERVERS_JSON"):
        import json
        try:
            config["mcp"]["servers"] = json.loads(os.getenv("MCP_SERVERS_JSON"))
        except:
            pass
    
    return config


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 80


@dataclass  
class LLMConfig:
    provider: str = "openrouter"
    model: str = "anthropic/claude-sonnet-4-20250514"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class GraphConfig:
    enabled: bool = True
    host: str = "localhost"
    port: int = 16379
    password: str = ""
    graph_name: str = "agentsmith"


@dataclass
class AgentSmithConfig:
    """Main configuration object"""
    server: ServerConfig = field(default_factory=ServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "AgentSmithConfig":
        """Load configuration from file and environment"""
        raw = load_config(config_path)
        
        return cls(
            server=ServerConfig(**raw.get("server", {})),
            llm=LLMConfig(**{k: v for k, v in raw.get("llm", {}).items() if k in LLMConfig.__dataclass_fields__}),
            graph=GraphConfig(**{k: v for k, v in raw.get("graph", {}).items() if k in GraphConfig.__dataclass_fields__})
        )


# Global config instance
_config: Optional[AgentSmithConfig] = None


def get_config() -> AgentSmithConfig:
    """Get the global configuration"""
    global _config
    if _config is None:
        _config = AgentSmithConfig.load()
    return _config


def reload_config(config_path: Optional[str] = None) -> AgentSmithConfig:
    """Reload configuration"""
    global _config
    _config = AgentSmithConfig.load(config_path)
    return _config
