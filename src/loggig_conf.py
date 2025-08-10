import logging, os
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()]
)
