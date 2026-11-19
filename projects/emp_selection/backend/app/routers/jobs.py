from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.auth import get_current_active_user, get_current_admin_user
from app.database import get_db
from app.models import (
    User,
    Job as JobModel,
    Resume as ResumeModel,
    ResumeBatch as ResumeBatchModel,
    CandidateScore,
)
from app.schemas import (
    JobCreate,
    JobUpdate,
    Job as JobSchema,
    ResumeBatch as ResumeBatchSchema,
    CandidateScoreWithResume,
)


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/", response_model=JobSchema, status_code=status.HTTP_201_CREATED)
def create_job(
    job_data: JobCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new job requirement definition.
    """
    # Check if user has permission to create jobs
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can create jobs"
        )

    # Create new job
    db_job = JobModel(
        **job_data.model_dump(),
        creator_id=current_user.id
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


@router.get("/", response_model=List[JobSchema])
def get_jobs(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    active_only: bool = Query(True, description="Return only active jobs"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get list of jobs with pagination.
    """
    query = db.query(JobModel)

    if active_only:
        query = query.filter(JobModel.is_active == True)

    # Admin users can see all jobs, regular users only see active ones
    if not current_user.is_admin:
        query = query.filter(JobModel.is_active == True)

    jobs = query.offset(skip).limit(limit).all()
    return jobs


@router.get("/{job_id}", response_model=JobSchema)
def get_job(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific job by ID.
    """
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Check if user can view this job
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    return job


@router.put("/{job_id}", response_model=JobSchema)
def update_job(
    job_id: int,
    job_data: JobUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update a job requirement definition.
    """
    # Check if user has permission to update jobs
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can update jobs"
        )

    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Update job fields
    update_data = job_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(job, field, value)

    db.commit()
    db.refresh(job)
    return job


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(
    job_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Delete a job (soft delete - sets is_active to False).
    """
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Soft delete - set is_active to False
    job.is_active = False
    db.commit()


@router.post("/{job_id}/activate", response_model=JobSchema)
def activate_job(
    job_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Activate a deactivated job.
    """
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    job.is_active = True
    db.commit()
    db.refresh(job)
    return job


@router.get("/{job_id}/batches", response_model=List[ResumeBatchSchema])
def get_job_batches(
    job_id: int,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get resume batches for a specific job.
    """
    # Check if job exists
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Check if user can view this job
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    batches = (
        db.query(ResumeBatchModel)
        .filter(ResumeBatchModel.job_id == job_id)
        .order_by(ResumeBatchModel.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return batches


@router.get("/{job_id}/candidates", response_model=List[CandidateScoreWithResume])
def get_job_candidates(
    job_id: int,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    min_score: Optional[float] = Query(None, ge=0, le=1, description="Minimum overall score"),
    shortlisted_only: bool = Query(False, description="Return only shortlisted candidates"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get ranked candidates for a specific job with filtering options.
    """
    # Check if job exists
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Check if user can view this job
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    # Build query
    query = (
        db.query(CandidateScore)
        .filter(CandidateScore.job_id == job_id)
        .join(CandidateScore.resume)
    )

    # Apply filters
    if min_score is not None:
        query = query.filter(CandidateScore.overall_score >= min_score)

    if shortlisted_only:
        query = query.filter(CandidateScore.is_shortlisted == True)

    # Apply sorting and pagination
    candidate_scores = (
        query.order_by(CandidateScore.rank.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Convert to response format with resume data
    results = []
    for score in candidate_scores:
        result = CandidateScoreWithResume(
            **score.__dict__,
            resume=score.resume
        )
        results.append(result)

    return results


@router.post("/{job_id}/candidates/{candidate_id}/shortlist")
def shortlist_candidate(
    job_id: int,
    candidate_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Shortlist a candidate for a job.
    """
    # Check if user has permission
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can shortlist candidates"
        )

    # Get candidate score
    candidate_score = (
        db.query(CandidateScore)
        .filter(
            CandidateScore.job_id == job_id,
            CandidateScore.id == candidate_id
        )
        .first()
    )

    if not candidate_score:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate score not found"
        )

    # Toggle shortlist status
    candidate_score.is_shortlisted = not candidate_score.is_shortlisted
    db.commit()

    return {
        "message": f"Candidate {'shortlisted' if candidate_score.is_shortlisted else 'removed from shortlist'}",
        "is_shortlisted": candidate_score.is_shortlisted
    }


@router.get("/{job_id}/stats")
def get_job_statistics(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get statistics for a job.
    """
    # Check if job exists
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Check if user can view this job
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    # Get candidate count
    candidate_count = (
        db.query(CandidateScore)
        .filter(CandidateScore.job_id == job_id)
        .count()
    )

    # Get batch count
    batch_count = (
        db.query(ResumeBatchModel)
        .filter(ResumeBatchModel.job_id == job_id)
        .count()
    )

    # Get average score
    avg_score_result = (
        db.query(db.func.avg(CandidateScore.overall_score))
        .filter(CandidateScore.job_id == job_id)
        .scalar()
    )
    avg_score = float(avg_score_result) if avg_score_result else 0.0

    # Get shortlisted count
    shortlisted_count = (
        db.query(CandidateScore)
        .filter(
            CandidateScore.job_id == job_id,
            CandidateScore.is_shortlisted == True
        )
        .count()
    )

    return {
        "job_id": job_id,
        "total_candidates": candidate_count,
        "total_batches": batch_count,
        "average_score": avg_score,
        "shortlisted_candidates": shortlisted_count,
        "shortlist_rate": shortlisted_count / max(candidate_count, 1)
    }


@router.post("/{job_id}/recalculate-scores")
def recalculate_scores(
    job_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Recalculate scores for all candidates in a job.
    """
    from app.services.scorer import scorer

    # Check if job exists
    job = db.query(JobModel).filter(JobModel.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    # Get all resumes for this job through batches
    batches = db.query(ResumeBatchModel).filter(ResumeBatchModel.job_id == job_id).all()
    resume_ids = []
    for batch in batches:
        resumes = db.query(ResumeModel).filter(ResumeModel.batch_id == batch.id).all()
        resume_ids.extend([resume.id for resume in resumes])

    # Get all candidate scores for this job
    candidate_scores = (
        db.query(CandidateScore)
        .filter(CandidateScore.job_id == job_id)
        .all()
    )

    # Get resumes for scoring
    resumes = db.query(ResumeModel).filter(ResumeModel.id.in_(resume_ids)).all()

    # Recalculate scores
    ranked_results = scorer.rank_candidates(job, resumes, candidate_scores)

    # Update candidate scores in database
    for result in ranked_results:
        candidate_score = next(
            (cs for cs in candidate_scores if cs.resume_id == result['resume_id']),
            None
        )

        if candidate_score:
            # Update existing score
            candidate_score.overall_score = result['overall_score']
            candidate_score.skills_score = result['score_breakdown'].skills_score
            candidate_score.experience_score = result['score_breakdown'].experience_score
            candidate_score.education_score = result['score_breakdown'].education_score
            candidate_score.certification_score = result['score_breakdown'].certification_score
            candidate_score.cgpa_score = result['score_breakdown'].cgpa_score
            candidate_score.semantic_similarity = result['score_breakdown'].semantic_similarity
            candidate_score.job_fit_confidence = result['score_breakdown'].job_fit_confidence
            candidate_score.missing_requirements = result['missing_requirements']
            candidate_score.matched_requirements = result['matched_requirements']
            candidate_score.strengths = result['strengths']
            candidate_score.weaknesses = result['weaknesses']
            candidate_score.rank = result['rank']
        else:
            # Create new score
            new_score = CandidateScore(
                job_id=job_id,
                resume_id=result['resume_id'],
                overall_score=result['overall_score'],
                skills_score=result['score_breakdown'].skills_score,
                experience_score=result['score_breakdown'].experience_score,
                education_score=result['score_breakdown'].education_score,
                certification_score=result['score_breakdown'].certification_score,
                cgpa_score=result['score_breakdown'].cgpa_score,
                semantic_similarity=result['score_breakdown'].semantic_similarity,
                job_fit_confidence=result['score_breakdown'].job_fit_confidence,
                missing_requirements=result['missing_requirements'],
                matched_requirements=result['matched_requirements'],
                strengths=result['strengths'],
                weaknesses=result['weaknesses'],
                rank=result['rank']
            )
            db.add(new_score)

    db.commit()

    return {
        "message": f"Scores recalculated for {len(ranked_results)} candidates",
        "total_candidates": len(ranked_results)
    }
