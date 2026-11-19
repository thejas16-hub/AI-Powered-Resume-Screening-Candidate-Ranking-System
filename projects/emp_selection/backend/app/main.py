from fastapi import FastAPI

from app.routers import auth, jobs, reports, resumes

app = FastAPI(title="Employee Selection API")

app.include_router(auth.router)
app.include_router(jobs.router)
app.include_router(resumes.router)
app.include_router(reports.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
