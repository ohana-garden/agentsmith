from python.helpers import dotenv
dotenv.save_dotenv_value("ANONYMIZED_TELEMETRY", "false")

# Make browser_use import optional
try:
    import browser_use
    import browser_use.utils
    BROWSER_USE_AVAILABLE = True
except ImportError:
    browser_use = None
    BROWSER_USE_AVAILABLE = False
