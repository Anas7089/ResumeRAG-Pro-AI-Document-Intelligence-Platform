# app/schemas.py
from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
# from pydantic import BaseModel
# from app.schemas import ResumeData

class Experience(BaseModel):
    company: Optional[str] =Field(None, description = 'Company name')
    role: Optional[str] = Field(None, description = 'Jon title or role.')
    duration: Optional[str] = Field(None, description= 'Start/End dates or total duration.')
    description: Optional[str] =Field(None, description='Summary of responsibilities or achievements.')

class Education(BaseModel):
    degree: Optional[str] = Field(None, description="Degree or certification name.")
    institution: Optional[str] = Field(None, description="School, university, or institution name.")
    year: Optional[str] = Field(None, description="Graduation year or completion date.")

class ResumeData(BaseModel):
    document_title: str = Field(..., description="The title of the document.")
    candidate_name: str = Field(..., description="The full name of the candidate.")
    email: Optional[EmailStr] = Field(None, description="Candidate's email.")
    phone: Optional[str] = Field(None, description="Candidate's phone number.")
    professional_summary: Optional[str] = Field(None, description="The candidate's profile summary.")
    key_skills: List[str] = Field(default_factory=list, description="A list of key skills.")
    work_experience: List[Experience] = Field(default_factory=list, description="A list of work history entries.")
    education: List[Education] = Field(default_factory=list, description="A list of education entries.")

class ExtractionResponse(BaseModel):
    status: str
    data: ResumeData
    file_id: Optional[str] = Field(None, description="Unique ID for RAG index lookup.")