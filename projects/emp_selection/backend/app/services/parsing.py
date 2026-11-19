"""Resume and document parsing."""
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile

import fitz  # PyMuPDF
import docx
import spacy
from spacy.language import Language

from app.models import Resume


class ResumeParser:
    """Parser for extracting information from resume files (PDF/DOCX)."""

    def __init__(self):
        """Initialize the parser with spaCy NLP model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, download it
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Regular expressions for common resume patterns
        self.email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_regex = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        self.cgpa_regex = r'(?:CGPA|GPA|Grade)[:\s]*([0-9]\.[0-9]{1,2})(?:/[0-9]\.?[0-9]?)?'
        self.year_regex = r'(?:20\d{2}|19\d{2})'

        # Common skills and languages to look for
        self.common_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'kotlin', 'swift'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra'],
            'frameworks': ['django', 'flask', 'fastapi', 'react', 'angular', 'vue', 'spring', 'node.js'],
            'tools': ['docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'jenkins', 'ansible']
        }

        self.common_certifications = [
            'aws certified', 'google cloud', 'azure certified', 'pmp', 'scrum master',
            'cisco certified', 'comptia', 'oracle certified', 'red hat certified'
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using PyMuPDF."""
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")
        return text

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file using python-docx."""
        text = ""
        try:
            doc = docx.Document(docx_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {e}")
        return text

    def extract_text(self, file_path: str) -> str:
        """Extract text from file based on extension."""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def extract_contact_info(self, text: str) -> Dict[str, List[str]]:
        """Extract contact information from text."""
        # Find emails
        emails = re.findall(self.email_regex, text, re.IGNORECASE)

        # Find phone numbers
        phones = re.findall(self.phone_regex, text)

        # Remove duplicates and clean up
        emails = list(set([email.lower() for email in emails]))
        phones = list(set(phones))

        return {
            'emails': emails,
            'phones': phones
        }

    def extract_education(self, text: str) -> List[Dict[str, Any]]:
        """Extract education information from text."""
        education_entries = []

        # Look for education section patterns
        education_patterns = [
            r'(?:Education|Academic Background|Qualifications)[:\s]*(.*?)(?:\n\n|\n\s*\n|\Z)',
            r'(?:Bachelor|Master|PhD|Diploma|Degree).*?(?:20\d{2}|19\d{2})',
        ]

        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]

                # Extract degree information
                degree_match = re.search(r'(Bachelor|Master|PhD|Diploma|Degree)\s*(?:of|in)?\s*([A-Za-z\s&]+)', match, re.IGNORECASE)
                if degree_match:
                    degree_type = degree_match.group(1)
                    field = degree_match.group(2).strip()

                    # Extract university
                    uni_match = re.search(r'(?:at|from|,)\s*([A-Za-z\s&]+(?:University|College|Institute|School))', match, re.IGNORECASE)
                    university = uni_match.group(1).strip() if uni_match else ""

                    # Extract years
                    year_matches = re.findall(self.year_regex, match)
                    years = list(set(year_matches))

                    # Extract CGPA
                    cgpa_match = re.search(self.cgpa_regex, match, re.IGNORECASE)
                    cgpa = float(cgpa_match.group(1)) if cgpa_match else None

                    education_entries.append({
                        'degree_type': degree_type,
                        'field': field,
                        'university': university,
                        'years': years,
                        'cgpa': cgpa
                    })

        return education_entries

    def extract_experience(self, text: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Extract work experience information and calculate total years."""
        experience_entries = []
        total_years = 0.0

        # Look for experience section patterns
        experience_patterns = [
            r'(?:Experience|Work History|Employment)[:\s]*(.*?)(?:\n\n|\n\s*\n|\Z)',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}\s*[-–to]+\s*(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4})',
        ]

        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]

                # Extract company and position
                position_match = re.search(r'([A-Za-z\s&]+)\s*(?:at|@|-|–)\s*([A-Za-z\s&]+)', match)
                if position_match:
                    position = position_match.group(1).strip()
                    company = position_match.group(2).strip()

                    # Extract duration
                    duration_match = re.search(r'(\d+)\s*(?:years?|months?|yrs?)', match, re.IGNORECASE)
                    duration = float(duration_match.group(1)) if duration_match else 0

                    if 'month' in (duration_match.group(0).lower() if duration_match else ''):
                        duration = duration / 12  # Convert months to years

                    experience_entries.append({
                        'position': position,
                        'company': company,
                        'duration_years': duration
                    })

                    total_years += duration

        return total_years, experience_entries

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using NLP and pattern matching."""
        skills = set()
        text_lower = text.lower()

        # Look for skills section
        skills_section_match = re.search(r'(?:Skills|Technical Skills|Competencies)[:\s]*(.*?)(?:\n\n|\n\s*\n|\Z)', text, re.IGNORECASE | re.DOTALL)

        if skills_section_match:
            skills_text = skills_section_match.group(1)
            # Split by commas, bullets, or newlines
            potential_skills = re.split(r'[,•\-\n]\s*', skills_text)
            for skill in potential_skills:
                skill = skill.strip()
                if skill and len(skill) > 1:
                    skills.add(skill)

        # Look for common skills in the entire text
        for category, skill_list in self.common_skills.items():
            for skill in skill_list:
                if skill in text_lower:
                    skills.add(skill)

        # Use spaCy to extract noun chunks that might be skills
        doc = self.nlp(text[:10000])  # Process first 10k chars for efficiency
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if len(chunk_text) > 2 and len(chunk_text) < 50:
                # Check if it looks like a skill (contains common tech terms)
                tech_indicators = ['script', 'sql', 'db', 'cloud', 'api', 'dev', 'ops', 'ml', 'ai']
                if any(indicator in chunk_text for indicator in tech_indicators):
                    skills.add(chunk_text)

        return list(skills)

    def extract_languages(self, text: str) -> List[str]:
        """Extract programming languages from text."""
        languages = set()
        text_lower = text.lower()

        # Common programming languages
        programming_languages = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'kotlin', 'swift', 'ruby', 'php', 'scala', 'haskell', 'perl', 'r',
            'matlab', 'sql', 'html', 'css', 'bash', 'shell', 'powershell'
        ]

        for lang in programming_languages:
            if lang in text_lower:
                languages.add(lang)

        # Look for languages section
        lang_section_match = re.search(r'(?:Languages|Programming Languages)[:\s]*(.*?)(?:\n\n|\n\s*\n|\Z)', text, re.IGNORECASE | re.DOTALL)

        if lang_section_match:
            lang_text = lang_section_match.group(1)
            potential_langs = re.split(r'[,•\-\n]\s*', lang_text)
            for lang in potential_langs:
                lang = lang.strip().lower()
                if lang and len(lang) > 1:
                    languages.add(lang)

        return list(languages)

    def extract_certifications(self, text: str) -> List[str]:
        """Extract certifications from text."""
        certifications = set()
        text_lower = text.lower()

        # Look for common certification patterns
        for cert in self.common_certifications:
            if cert in text_lower:
                certifications.add(cert.title())

        # Look for certifications section
        cert_section_match = re.search(r'(?:Certifications|Certificates|Qualifications)[:\s]*(.*?)(?:\n\n|\n\s*\n|\Z)', text, re.IGNORECASE | re.DOTALL)

        if cert_section_match:
            cert_text = cert_section_match.group(1)
            potential_certs = re.split(r'[,•\-\n]\s*', cert_text)
            for cert in potential_certs:
                cert = cert.strip()
                if cert and len(cert) > 2:
                    certifications.add(cert)

        return list(certifications)

    def extract_cgpa(self, text: str) -> Optional[float]:
        """Extract CGPA from text."""
        cgpa_matches = re.findall(self.cgpa_regex, text, re.IGNORECASE)
        if cgpa_matches:
            # Take the highest CGPA found
            cgpa_values = [float(match) for match in cgpa_matches]
            return max(cgpa_values)
        return None

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Parse a resume file and extract all relevant information."""
        try:
            # Extract text from file
            text = self.extract_text(file_path)

            # Extract various information
            contact_info = self.extract_contact_info(text)
            education = self.extract_education(text)
            total_experience, experience_details = self.extract_experience(text)
            skills = self.extract_skills(text)
            languages = self.extract_languages(text)
            certifications = self.extract_certifications(text)
            cgpa = self.extract_cgpa(text)

            return {
                'raw_text': text[:10000],  # Store first 10k chars for reference
                'contact_info': contact_info,
                'education': education,
                'total_experience_years': total_experience,
                'experience_details': experience_details,
                'skills': skills,
                'languages': languages,
                'certifications': certifications,
                'cgpa': cgpa,
                'file_info': {
                    'path': file_path,
                    'filename': Path(file_path).name,
                    'size': os.path.getsize(file_path)
                }
            }

        except Exception as e:
            raise ValueError(f"Failed to parse resume: {e}")

    def create_resume_model(self, file_path: str, batch_id: int) -> Resume:
        """Parse resume and create a Resume model instance."""
        parsed_data = self.parse_resume(file_path)

        # Get CGPA from education entries if not directly extracted
        cgpa = parsed_data['cgpa']
        if not cgpa and parsed_data['education']:
            for edu in parsed_data['education']:
                if edu.get('cgpa'):
                    cgpa = edu['cgpa']
                    break

        resume = Resume(
            batch_id=batch_id,
            filename=Path(file_path).name,
            original_path=file_path,
            skills=parsed_data['skills'],
            languages=parsed_data['languages'],
            certifications=parsed_data['certifications'],
            education=parsed_data['education'],
            experience_years=parsed_data['total_experience_years'],
            cgpa=cgpa,
            parsed_at=None  # Will be set when saved to DB
        )

        return resume


# Global parser instance
parser = ResumeParser()