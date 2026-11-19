import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./emp_selection.db")
SECRET_KEY = os.getenv("SECRET_KEY", "")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
