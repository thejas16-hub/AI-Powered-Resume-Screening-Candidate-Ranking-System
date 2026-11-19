"""Anonymizer service to remove personal identifiers from resumes."""
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

import fitz  # PyMuPDF
import docx
from docx.shared import RGBColor
import spacy
from spacy.tokens import Doc
from app.services.parsing import ResumeParser


class ResumeAnonymizer:
    """Anonymizer for removing personal identifiers from resumes."""

    def __init__(self):
        """Initialize the anonymizer with spaCy NLP model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Patterns for personal identifiers
        self.personal_patterns = {
            'name': [
                r'^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$',  # Capitalized names at start of lines
                r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Titles with names
            ],
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]',
            'address': [
                r'\b\d+\s+[A-Za-z\s]+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Lane|Ln\.|Drive|Dr\.)\b',
                r'\b(?:P\.?O\.?\s+Box|PO\s+Box)\s+\d+\b',
            ],
            'social_media': [
                r'linkedin\.com/[A-Za-z0-9-]+',
                r'twitter\.com/@[A-Za-z0-9_]+',
                r'github\.com/[A-Za-z0-9-]+',
                r'facebook\.com/[A-Za-z0-9.]+',
            ],
            'id_numbers': [
                r'\b[A-Z]{2,3}\d{6,}\b',  # Passport/ID numbers
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN-like patterns
                r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b',  # Credit card-like
            ],
            'date_of_birth': [
                r'\b(?:DOB|Date of Birth|Born)[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b',
                r'\b(?:Age|Aged)[:\s]*(\d{1,3})\s*(?:years?|yrs?)\b',
            ],
        }

        # Common names to watch for (to reduce false positives)
        self.common_first_names = {
            'john', 'jane', 'david', 'sarah', 'michael', 'emily', 'robert', 'lisa',
            'william', 'mary', 'james', 'patricia', 'richard', 'jennifer', 'charles',
            'linda', 'thomas', 'barbara', 'christopher', 'susan'
        }

    def anonymize_text(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Anonymize personal identifiers in text.

        Returns:
            Tuple of (anonymized_text, removed_identifiers)
        """
        original_text = text
        removed_identifiers = {
            'names': [],
            'emails': [],
            'phones': [],
            'addresses': [],
            'social_media': [],
            'id_numbers': [],
            'dates_of_birth': [],
        }

        # Step 1: Remove emails
        emails = re.findall(self.personal_patterns['email'], text, re.IGNORECASE)
        for email in emails:
            text = text.replace(email, '[EMAIL_REMOVED]')
            removed_identifiers['emails'].append(email)

        # Step 2: Remove phone numbers
        phones = re.findall(self.personal_patterns['phone'], text)
        for phone in phones:
            text = text.replace(phone, '[PHONE_REMOVED]')
            removed_identifiers['phones'].append(phone)

        # Step 3: Remove addresses
        for address_pattern in self.personal_patterns['address']:
            addresses = re.findall(address_pattern, text, re.IGNORECASE)
            for address in addresses:
                text = text.replace(address, '[ADDRESS_REMOVED]')
                removed_identifiers['addresses'].append(address)

        # Step 4: Remove social media links
        for social_pattern in self.personal_patterns['social_media']:
            social_links = re.findall(social_pattern, text, re.IGNORECASE)
            for link in social_links:
                text = text.replace(link, '[SOCIAL_MEDIA_REMOVED]')
                removed_identifiers['social_media'].append(link)

        # Step 5: Remove ID numbers
        for id_pattern in self.personal_patterns['id_numbers']:
            id_numbers = re.findall(id_pattern, text)
            for id_num in id_numbers:
                text = text.replace(id_num, '[ID_NUMBER_REMOVED]')
                removed_identifiers['id_numbers'].append(id_num)

        # Step 6: Remove dates of birth and age
        for dob_pattern in self.personal_patterns['date_of_birth']:
            dob_matches = re.findall(dob_pattern, text, re.IGNORECASE)
            for dob in dob_matches:
                text = re.sub(dob_pattern.replace(r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})', dob),
                             '[DOB_REMOVED]', text, flags=re.IGNORECASE)
                removed_identifiers['dates_of_birth'].append(dob)

        # Step 7: Use spaCy to identify and remove names with more context
        doc = self.nlp(text[:10000])  # Process first 10k chars for efficiency

        # Track names found by spaCy
        spacy_names = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'PER']:
                name = ent.text.strip()
                # Check if it's likely a name (not a common word)
                if (len(name.split()) >= 2 or
                    (len(name.split()) == 1 and name.lower() not in self.common_first_names)):
                    spacy_names.append(name)

        # Remove names found by spaCy
        for name in spacy_names:
            # Replace whole name
            text = text.replace(name, '[NAME_REMOVED]')
            removed_identifiers['names'].append(name)

            # Also replace variations (like "John's" -> "[NAME_REMOVED]'s")
            name_parts = name.split()
            if len(name_parts) > 1:
                # Replace first name only if followed by last name context
                first_name = name_parts[0]
                if first_name.lower() not in self.common_first_names:
                    # Be more careful with single names
                    pattern = r'\b' + re.escape(first_name) + r'\b'
                    text = re.sub(pattern, '[NAME_REMOVED]', text)

        # Step 8: Remove names using pattern matching (fallback)
        for name_pattern in self.personal_patterns['name']:
            names = re.findall(name_pattern, text, re.MULTILINE)
            for name in names:
                if name not in removed_identifiers['names']:  # Avoid duplicates
                    text = text.replace(name, '[NAME_REMOVED]')
                    removed_identifiers['names'].append(name)

        # Clean up: Remove any standalone "removed" markers that might create weird spacing
        text = re.sub(r'\[\w+_REMOVED\]\s+\[\w+_REMOVED\]', '[PERSONAL_INFO_REMOVED]', text)
        text = re.sub(r'\[\w+_REMOVED\]{2,}', '[PERSONAL_INFO_REMOVED]', text)

        return text, removed_identifiers

    def anonymize_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> Tuple[str, Dict[str, List[str]]]:
        """
        Anonymize a PDF file by extracting text, anonymizing, and creating a new PDF.

        Returns:
            Tuple of (output_path, removed_identifiers)
        """
        if output_path is None:
            output_dir = tempfile.mkdtemp(prefix="anonymized_")
            output_path = os.path.join(output_dir, f"anonymized_{Path(pdf_path).name}")

        # Extract text from PDF
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")

        # Anonymize the text
        anonymized_text, removed_identifiers = self.anonymize_text(text)

        # Create a new anonymized PDF
        try:
            with fitz.open() as new_doc:
                # Create a page with anonymized text
                page = new_doc.new_page()

                # Add anonymized text with a note
                note = "ANONYMIZED RESUME - PERSONAL IDENTIFIERS HAVE BEEN REMOVED\n\n"
                full_text = note + anonymized_text

                # Insert text (simplified - real implementation might need better formatting)
                rect = page.rect + (36, 36, -36, -36)  # Add margins
                page.insert_textbox(rect, full_text, fontsize=10)

                new_doc.save(output_path)
        except Exception as e:
            raise ValueError(f"Failed to create anonymized PDF: {e}")

        return output_path, removed_identifiers

    def anonymize_docx(self, docx_path: str, output_path: Optional[str] = None) -> Tuple[str, Dict[str, List[str]]]:
        """
        Anonymize a DOCX file.

        Returns:
            Tuple of (output_path, removed_identifiers)
        """
        if output_path is None:
            output_dir = tempfile.mkdtemp(prefix="anonymized_")
            output_path = os.path.join(output_dir, f"anonymized_{Path(docx_path).name}")

        try:
            # Load the document
            doc = docx.Document(docx_path)

            # Extract all text for anonymization
            all_text = []
            for paragraph in doc.paragraphs:
                all_text.append(paragraph.text)

            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        all_text.append(cell.text)

            full_text = "\n".join(all_text)

            # Anonymize the text
            anonymized_text, removed_identifiers = self.anonymize_text(full_text)

            # Create a new document with anonymized content
            new_doc = docx.Document()

            # Add a header note
            header_para = new_doc.add_paragraph()
            header_para.add_run("ANONYMIZED RESUME - PERSONAL IDENTIFIERS HAVE BEEN REMOVED").bold = True
            new_doc.add_paragraph()  # Add empty line

            # Add anonymized text (simplified - real implementation would preserve formatting)
            lines = anonymized_text.split('\n')
            for line in lines:
                if line.strip():  # Skip empty lines
                    new_doc.add_paragraph(line)

            new_doc.save(output_path)

        except Exception as e:
            raise ValueError(f"Failed to anonymize DOCX: {e}")

        return output_path, removed_identifiers

    def anonymize_file(self, file_path: str, output_path: Optional[str] = None) -> Tuple[str, Dict[str, List[str]]]:
        """
        Anonymize a file based on its extension.

        Returns:
            Tuple of (output_path, removed_identifiers)
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return self.anonymize_pdf(file_path, output_path)
        elif file_ext == '.docx':
            return self.anonymize_docx(file_path, output_path)
        else:
            raise ValueError(f"Unsupported file format for anonymization: {file_ext}")

    def calculate_anonymization_score(self, removed_identifiers: Dict[str, List[str]]) -> float:
        """
        Calculate a score indicating how much personal information was removed.
        Higher score = more personal information was removed.
        """
        weights = {
            'names': 2.0,
            'emails': 1.5,
            'phones': 1.5,
            'addresses': 2.0,
            'social_media': 1.0,
            'id_numbers': 2.5,
            'dates_of_birth': 2.0,
        }

        total_score = 0.0
        max_possible_score = 0.0

        for category, items in removed_identifiers.items():
            weight = weights.get(category, 1.0)
            item_count = len(items)

            # Score for this category
            category_score = weight * item_count
            total_score += category_score

            # Assume max 3 items per category for normalization
            max_possible_score += weight * 3

        # Normalize score to 0-1 range
        if max_possible_score > 0:
            normalized_score = total_score / max_possible_score
        else:
            normalized_score = 0.0

        return min(normalized_score, 1.0)  # Cap at 1.0

    def generate_anonymization_report(self, removed_identifiers: Dict[str, List[str]]) -> Dict[str, any]:
        """
        Generate a detailed report of what was anonymized.
        """
        anonymization_score = self.calculate_anonymization_score(removed_identifiers)

        report = {
            'anonymization_score': anonymization_score,
            'summary': {
                'total_identifiers_removed': sum(len(items) for items in removed_identifiers.values()),
                'categories_affected': [cat for cat, items in removed_identifiers.items() if items],
            },
            'details': removed_identifiers,
            'interpretation': self._interpret_anonymization_score(anonymization_score),
        }

        return report

    def _interpret_anonymization_score(self, score: float) -> str:
        """Interpret the anonymization score."""
        if score >= 0.7:
            return "Excellent anonymization - significant personal information was removed"
        elif score >= 0.4:
            return "Good anonymization - moderate personal information was removed"
        elif score >= 0.1:
            return "Limited anonymization - some personal information was removed"
        else:
            return "Minimal anonymization - little to no personal information was found"

    def anonymize_resume_data(self, parsed_data: Dict[str, any]) -> Dict[str, any]:
        """
        Anonymize parsed resume data (for use after parsing).
        """
        anonymized_data = parsed_data.copy()

        # Remove contact info from parsed data
        if 'contact_info' in anonymized_data:
            anonymized_data['contact_info'] = {
                'emails': [],
                'phones': [],
            }

        # Anonymize education entries (remove personal names from university/degree)
        if 'education' in anonymized_data:
            for edu_entry in anonymized_data['education']:
                # Remove any personal names from university field
                if 'university' in edu_entry:
                    # Simple pattern to remove names (in reality would need more sophisticated approach)
                    university = edu_entry['university']
                    # This is a simplified version - real implementation would use NLP
                    edu_entry['university'] = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                                                    '[NAME_REMOVED]',
                                                    university)

        # Anonymize experience details
        if 'experience_details' in anonymized_data:
            for exp_entry in anonymized_data['experience_details']:
                # Remove company names that might contain personal names
                if 'company' in exp_entry:
                    company = exp_entry['company']
                    # Remove common personal name patterns from company names
                    exp_entry['company'] = re.sub(r'\b(?:[A-Z][a-z]+){1,2}\s+(?:Inc|Corp|LLC|Ltd)\b',
                                                 '[COMPANY_ANONYMIZED]',
                                                 company)

        return anonymized_data


# Global anonymizer instance
anonymizer = ResumeAnonymizer()
