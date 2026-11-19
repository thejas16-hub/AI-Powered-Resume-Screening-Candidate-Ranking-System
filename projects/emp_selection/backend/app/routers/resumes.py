import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import logging

from app.auth import get_current_active_user, get_current_admin_user
from app.database import get_db
from app.models import User, Job, ResumeBatch, Resume, CandidateScore
from app.schemas import (
    ResumeBatchCreate, ResumeBatch, Resume, CandidateScore,
    BatchProcessingResponse, BatchUploadResponse, FileUploadResponse
)
from app.services.parsing import parser as resume_parser
from app.services.anonymizer import anonymizer as resume_anonymizer
from app.services.scorer import scorer as candidate_scorer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resumes", tags=["resumes"])


# Configure upload directory
UPLOAD_DIR = Path("./uploads/resumes")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/batch", response_model=ResumeBatch, status_code=status.HTTP_201_CREATED)
def create_resume_batch(
    batch_data: ResumeBatchCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new resume batch for processing.
    """
    # Check if job exists and is active
    job = db.query(Job).filter(Job.id == batch_data.job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )

    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    # Create batch directory
    batch_id = str(uuid.uuid4())
    batch_dir = UPLOAD_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Create resume batch record
    db_batch = ResumeBatch(
        name=batch_data.name,
        job_id=batch_data.job_id,
        upload_path=str(batch_dir),
        status="pending",
        file_count=0
    )
    db.add(db_batch)
    db.commit()
    db.refresh(db_batch)

    return db_batch


@router.post("/batch/{batch_id}/upload", response_model=BatchUploadResponse)
async def upload_resumes_to_batch(
    batch_id: int,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload resumes to a batch (supports PDF and DOCX).
    """
    # Check batch exists and is in pending state
    batch = db.query(ResumeBatch).filter(ResumeBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )

    if batch.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot upload to batch with status: {batch.status}"
        )

    # Check if user has permission
    job = db.query(Job).filter(Job.id == batch.job_id).first()
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    uploaded_files = []
    errors = []

    for file in files:
        try:
            # Validate file type
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ['.pdf', '.docx']:
                errors.append({
                    "filename": file.filename,
                    "error": f"Unsupported file type: {file_ext}. Only PDF and DOCX are supported."
                })
                continue

            # Generate unique filename
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            file_path = Path(batch.upload_path) / unique_filename

            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Create resume record
            db_resume = Resume(
                batch_id=batch_id,
                filename=file.filename,
                original_path=str(file_path),
                status="uploaded"
            )
            db.add(db_resume)

            uploaded_files.append(
                FileUploadResponse(
                    filename=file.filename,
                    size=len(content),
                    content_type=file.content_type,
                    upload_id=unique_filename
                )
            )

        except Exception as e:
            logger.error(f"Failed to upload file {file.filename}: {e}")
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })

    # Update batch file count
    batch.file_count += len(uploaded_files)
    db.commit()

    return BatchUploadResponse(
        batch_id=batch_id,
        total_files=len(files),
        uploaded_files=uploaded_files,
        errors=errors
    )


@router.post("/batch/{batch_id}/process", response_model=BatchProcessingResponse)
def process_resume_batch(
    batch_id: int,
    background_tasks: BackgroundTasks,
    anonymize: bool = True,
    score_immediately: bool = True,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Start processing a resume batch (parsing, anonymization, and scoring).
    """
    # Check batch exists
    batch = db.query(ResumeBatch).filter(ResumeBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )

    if batch.status not in ["pending", "uploaded"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot process batch with status: {batch.status}"
        )

    # Update batch status
    batch.status = "processing"
    db.commit()

    # Start background processing
    background_tasks.add_task(
        process_batch_background,
        batch_id,
        anonymize,
        score_immediately,
        current_user.id
    )

    return BatchProcessingResponse(
        batch_id=batch_id,
        total_files=batch.file_count,
        processed_files=0,
        status="processing",
        estimated_completion_time="5-10 minutes"
    )


def process_batch_background(
    batch_id: int,
    anonymize: bool,
    score_immediately: bool,
    user_id: int
):
    """
    Background task for processing resume batch.
    """
    from app.database import SessionLocal
    db = SessionLocal()

    try:
        # Get batch and job
        batch = db.query(ResumeBatch).filter(ResumeBatch.id == batch_id).first()
        if not batch:
            logger.error(f"Batch {batch_id} not found")
            return

        job = db.query(Job).filter(Job.id == batch.job_id).first()
        if not job:
            logger.error(f"Job {batch.job_id} not found for batch {batch_id}")
            return

        # Get resumes for this batch
        resumes = db.query(Resume).filter(Resume.batch_id == batch_id).all()
        if not resumes:
            logger.warning(f"No resumes found in batch {batch_id}")
            batch.status = "completed"
            db.commit()
            return

        processed_count = 0
        total_count = len(resumes)

        for resume in resumes:
            try:
                # Parse resume
                logger.info(f"Parsing resume: {resume.filename}")
                parsed_data = resume_parser.parse_resume(resume.original_path)

                # Update resume with parsed data
                resume.skills = parsed_data['skills']
                resume.languages = parsed_data['languages']
                resume.certifications = parsed_data['certifications']
                resume.education = parsed_data['education']
                resume.experience_years = parsed_data['total_experience_years']
                resume.cgpa = parsed_data['cgpa']
                resume.parsed_at = datetime.utcnow()

                # Anonymize if requested
                if anonymize:
                    logger.info(f"Anonymizing resume: {resume.filename}")
                    anonymized_path, removed_identifiers = resume_anonymizer.anonymize_file(
                        resume.original_path
                    )
                    resume.anonymized_path = anonymized_path
                    resume.is_anonymized = True
                    resume.anonymized_at = datetime.utcnow()

                    # Store anonymization report
                    anonymization_report = resume_anonymizer.generate_anonymization_report(
                        removed_identifiers
                    )
                    # Note: In a real implementation, store this report

                db.commit()
                processed_count += 1

                # Update batch progress periodically
                if processed_count % 10 == 0 or processed_count == total_count:
                    progress = processed_count / total_count * 100
                    logger.info(f"Batch {batch_id}: {processed_count}/{total_count} processed ({progress:.1f}%)")

            except Exception as e:
                logger.error(f"Failed to process resume {resume.filename}: {e}")
                resume.error_message = str(e)
                db.commit()

        # Score candidates if requested
        if score_immediately and processed_count > 0:
            logger.info(f"Scoring candidates for batch {batch_id}")
            score_batch_candidates(batch_id, job.id, db)

        # Update batch status
        batch.status = "completed"
        batch.processed_at = datetime.utcnow()
        db.commit()

        logger.info(f"Batch {batch_id} processing completed successfully")

    except Exception as e:
        logger.error(f"Failed to process batch {batch_id}: {e}")
        batch = db.query(ResumeBatch).filter(ResumeBatch.id == batch_id).first()
        if batch:
            batch.status = "failed"
            batch.error_message = str(e)
            db.commit()

    finally:
        db.close()


def score_batch_candidates(batch_id: int, job_id: int, db: Session):
    """
    Score all candidates in a batch.
    """
    # Get job
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        logger.error(f"Job {job_id} not found")
        return

    # Get resumes for this batch
    resumes = db.query(Resume).filter(Resume.batch_id == batch_id).all()
    if not resumes:
        logger.warning(f"No resumes found in batch {batch_id}")
        return

    # Score candidates
    ranked_results = candidate_scorer.rank_candidates(job, resumes)

    # Save scores to database
    for result in ranked_results:
        candidate_score = CandidateScore(
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
        db.add(candidate_score)

    db.commit()
    logger.info(f"Scored {len(ranked_results)} candidates for batch {batch_id}")


@router.get("/batch/{batch_id}", response_model=ResumeBatch)
def get_resume_batch(
    batch_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get resume batch details.
    """
    batch = db.query(ResumeBatch).filter(ResumeBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )

    # Check job access
    job = db.query(Job).filter(Job.id == batch.job_id).first()
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    return batch


@router.get("/batch/{batch_id}/resumes", response_model=List[Resume])
def get_batch_resumes(
    batch_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get resumes in a batch.
    """
    # Check batch exists
    batch = db.query(ResumeBatch).filter(ResumeBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )

    # Check job access
    job = db.query(Job).filter(Job.id == batch.job_id).first()
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    resumes = (
        db.query(Resume)
        .filter(Resume.batch_id == batch_id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return resumes


@router.get("/batch/{batch_id}/status", response_model=BatchProcessingResponse)
def get_batch_status(
    batch_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get processing status of a batch.
    """
    batch = db.query(ResumeBatch).filter(ResumeBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )

    # Check job access
    job = db.query(Job).filter(Job.id == batch.job_id).first()
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    # Get processed count
    processed_count = (
        db.query(Resume)
        .filter(
            Resume.batch_id == batch_id,
            Resume.parsed_at.isnot(None)
        )
        .count()
    )

    return BatchProcessingResponse(
        batch_id=batch_id,
        total_files=batch.file_count,
        processed_files=processed_count,
        status=batch.status,
        estimated_completion_time=None  # Could calculate based on processing rate
    )


@router.delete("/batch/{batch_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_resume_batch(
    batch_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Delete a resume batch and all associated files.
    """
    batch = db.query(ResumeBatch).filter(ResumeBatch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )

    # Delete associated resumes and scores first
    db.query(CandidateScore).filter(
        CandidateScore.resume_id.in_(
            db.query(Resume.id).filter(Resume.batch_id == batch_id)
        )
    ).delete(synchronize_session=False)

    db.query(Resume).filter(Resume.batch_id == batch_id).delete(synchronize_session=False)

    # Delete batch
    db.delete(batch)
    db.commit()

    # Delete uploaded files
    try:
        if batch.upload_path and os.path.exists(batch.upload_path):
            shutil.rmtree(batch.upload_path)
    except Exception as e:
        logger.warning(f"Failed to delete batch files: {e}")


@router.get("/{resume_id}", response_model=Resume)
def get_resume(
    resume_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific resume by ID.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )

    # Check batch and job access
    batch = db.query(ResumeBatch).filter(ResumeBatch.id == resume.batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )

    job = db.query(Job).filter(Job.id == batch.job_id).first()
    if not job.is_active and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Job is not active"
        )

    return resume


@router.post("/{resume_id}/reparse")
def reparse_resume(
    resume_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Reparse a specific resume.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )

    try:
        # Reparse resume
        parsed_data = resume_parser.parse_resume(resume.original_path)

        # Update resume with parsed data
        resume.skills = parsed_data['skills']
        resume.languages = parsed_data['languages']
        resume.certifications = parsed_data['certifications']
        resume.education = parsed_data['education']
        resume.experience_years = parsed_data['total_experience_years']
        resume.cgpa = parsed_data['cgpa']
        resume.parsed_at = datetime.utcnow()
        resume.error_message = None

        db.commit()

        return {
            "message": "Resume reparsed successfully",
            "resume_id": resume_id
        }

    except Exception as e:
        logger.error(f"Failed to reparse resume {resume_id}: {e}")
        resume.error_message = str(e)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reparse resume: {e}"
        )


@router.post("/{resume_id}/reanonymize")
def reanonymize_resume(
    resume_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Reanonymize a specific resume.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )

    try:
        # Anonymize resume
        anonymized_path, removed_identifiers = resume_anonymizer.anonymize_file(
            resume.original_path
        )

        # Update resume
        resume.anonymized_path = anonymized_path
        resume.is_anonymized = True
        resume.anonymized_at = datetime.utcnow()

        db.commit()

        # Generate anonymization report
        report = resume_anonymizer.generate_anonymization_report(removed_identifiers)

        return {
            "message": "Resume reanonymized successfully",
            "resume_id": resume_id,
            "anonymization_score": report['anonymization_score'],
            "identifiers_removed": report['summary']['total_identifiers_removed']
        }

    except Exception as e:
        logger.error(f"Failed to reanonymize resume {resume_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reanonymize resume: {e}"
        )


# Import datetime at the end to avoid circular imports
from datetime import datetime
