from pydantic import BaseModel, EmailStr
from typing import Optional, List, Any, Dict
from datetime import datetime

# --- USER SCHEMAS ---
class UserBase(BaseModel):
    full_name: str
    email: EmailStr
    gender: Optional[str] = None
    age: Optional[int] = None
    phone: Optional[str] = None
    profile_pic: Optional[str] = None

class UserCreate(UserBase):
    password: str
    role: str = "User"

class UserOut(UserBase):
    id: int
    points: int
    role: str

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    phone: Optional[str] = None

class UserPasswordChange(BaseModel):
    current_password: str
    new_password: str

# --- COMPANY SCHEMAS ---
class CompanyBase(BaseModel):
    company_name: str
    email: EmailStr
    logo_url: Optional[str] = None
    industry_category: Optional[str] = None

class CompanyCreate(CompanyBase):
    password: str

class CompanyOut(CompanyBase):
    id: int

    class Config:
        from_attributes = True

# --- CAMPAIGN SCHEMAS ---
class CampaignBase(BaseModel):
    video_url: Optional[str] = None
    description: Optional[str] = None
    product_name: Optional[str] = None
    product_price: Optional[str] = None
    product_description: Optional[str] = None
    product_ingredients: Optional[str] = None
    product_photo_url: Optional[str] = None
    questions_list: Optional[List[Dict[str, Any]]] = None
    promo_code: Optional[str] = None
    discount_value: Optional[str] = None

class CampaignUpdate(BaseModel):
    video_url: Optional[str] = None
    product_name: Optional[str] = None
    product_price: Optional[str] = None
    product_description: Optional[str] = None
    product_ingredients: Optional[str] = None
    product_photo_url: Optional[str] = None
    promo_code: Optional[str] = None
    discount_value: Optional[str] = None
    questions_list: Optional[List[Dict[str, Any]]] = None

class CampaignCreate(CampaignBase):
    company_id: int

class CampaignOut(CampaignBase):
    id: int
    company_id: int

    class Config:
        from_attributes = True

# --- RESULT SCHEMAS ---
class ResultBase(BaseModel):
    survey_data: Dict[str, Any]
    neural_score: Optional[float] = None

class ResultCreate(ResultBase):
    user_id: int
    campaign_id: int

class ResultOut(ResultBase):
    id: int
    user_id: int
    campaign_id: int

    class Config:
        from_attributes = True

# --- HISTORY SCHEMAS ---
class HistoryBase(BaseModel):
    status: Optional[str] = 'In Processing'
    earned_points: int = 0
    earned_promo_code: Optional[str] = None

class HistoryCreate(HistoryBase):
    user_id: int
    campaign_id: int

class HistoryOut(HistoryBase):
    id: int
    user_id: int
    campaign_id: int
    completion_date: datetime
    status: Optional[str] = 'In Processing'
    company_name: Optional[str] = None
    industry_category: Optional[str] = None
    company_logo_url: Optional[str] = None

    class Config:
        from_attributes = True

# --- TOKEN SCHEMAS ---
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
