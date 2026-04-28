from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import List, Optional
import os
import shutil
import json
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
import uuid


# Adjust import based on the library used. Here we use 'jose'
# Install via: pip install python-jose[cryptography] passlib[bcrypt]
try:
    from jose import JWTError, jwt
except ImportError:
    import jwt  # Fallback to PyJWT if jose is not installed
    JWTError = jwt.InvalidTokenError


import models, schemas
from database import engine, get_db

# Create the database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Neuromarketing Backend API")

os.makedirs("static/logos", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the uploads directory to serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Authentication Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "Rawan_Brain_Decoder_Super_Secret_HS256")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# --- AUTHENTICATION UTILITIES ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# --- DEPENDENCIES ---
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = schemas.TokenData(email=email, role=payload.get("role"))
    except JWTError:
        raise credentials_exception
    
    # Try finding the user in the Users table
    user = db.query(models.User).filter(models.User.email == token_data.email).first()
    if user:
        return user
    
    # Try finding the user in the Companies table
    company = db.query(models.Company).filter(models.Company.email == token_data.email).first()
    if company:
        # Assign a pseudo-role attribute for the RBAC dependency
        company.role = "Company"
        return company

    raise credentials_exception

def require_role(allowed_roles: List[str]):
    def role_checker(current_user = Depends(get_current_user)):
        if getattr(current_user, "role", "User") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted"
            )
        return current_user
    return role_checker


# --- ROUTES ---

@app.post("/signup", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if email exists in Users or Companies
    if db.query(models.User).filter(models.User.email == user.email).first() or \
       db.query(models.Company).filter(models.Company.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        full_name=user.full_name,
        email=user.email,
        password=hashed_password,
        gender=user.gender,
        age=user.age,
        phone=user.phone,
        profile_pic=user.profile_pic,
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Check Users table
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if user and verify_password(form_data.password, user.password):
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email, "role": user.role}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer", "role": user.role}
    
    # Check Companies table
    company = db.query(models.Company).filter(models.Company.email == form_data.username).first()
    if company and verify_password(form_data.password, company.password):
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": company.email, "role": "Company"}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer", "role": "Company"}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.get("/users/me", response_model=schemas.UserOut)
def read_users_me(current_user: models.User = Depends(require_role(["Admin", "User", "Company"]))):
    return current_user

@app.put("/users/me", response_model=schemas.UserOut)
def update_user_me(user_update: schemas.UserUpdate, db: Session = Depends(get_db), current_user: models.User = Depends(require_role(["Admin", "User", "Company"]))):
    # Depending on the application, we might want this to apply to companies as well, but the instruction is for users
    if hasattr(current_user, 'full_name'):
        if user_update.full_name is not None:
            current_user.full_name = user_update.full_name
        if user_update.gender is not None:
            current_user.gender = user_update.gender
        if user_update.age is not None:
            current_user.age = user_update.age
        if user_update.phone is not None:
            current_user.phone = user_update.phone
        db.commit()
        db.refresh(current_user)
        return current_user
    else:
        raise HTTPException(status_code=400, detail="Only standard users can update their profile via this endpoint")

@app.post("/users/me/profile-pic", response_model=schemas.UserOut)
async def upload_profile_pic(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role(["Admin", "User", "Company"]))
):
    if not hasattr(current_user, 'profile_pic'):
        raise HTTPException(status_code=400, detail="Current user type does not support profile pictures")
    
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    # Generate unique filename
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join("uploads", filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Remove old profile pic if it's a local file
    if current_user.profile_pic and current_user.profile_pic.startswith("/uploads/"):
        old_file = current_user.profile_pic.lstrip("/")
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
            except Exception:
                pass

    # Save the new URL in the database
    # In a real production setup, this would be the server's public URL, but relative is fine if frontend prepends the base URL
    current_user.profile_pic = f"/uploads/{filename}"
    db.commit()
    db.refresh(current_user)
    return current_user

@app.delete("/users/me/profile-pic", response_model=schemas.UserOut)
async def remove_profile_pic(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_role(["Admin", "User", "Company"]))
):
    if not hasattr(current_user, 'profile_pic'):
        raise HTTPException(status_code=400, detail="Current user type does not support profile pictures")

    # Remove old profile pic file if it exists locally
    if current_user.profile_pic and current_user.profile_pic.startswith("/uploads/"):
        old_file = current_user.profile_pic.lstrip("/")
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
            except Exception:
                pass

    current_user.profile_pic = None
    db.commit()
    db.refresh(current_user)
    return current_user

@app.post("/users/change-password")
def change_password(password_data: schemas.UserPasswordChange, db: Session = Depends(get_db), current_user: models.User = Depends(require_role(["Admin", "User", "Company"]))):
    if not verify_password(password_data.current_password, current_user.password):
        raise HTTPException(status_code=400, detail="Incorrect current password")
    
    current_user.password = get_password_hash(password_data.new_password)
    db.commit()
    return {"message": "Password updated successfully"}

@app.get("/admin/users", response_model=List[schemas.UserOut])
def get_all_users(db: Session = Depends(get_db), current_user: models.User = Depends(require_role(["Admin"]))):
    users = db.query(models.User).filter(models.User.role == "User").all()
    return users

@app.delete("/admin/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(require_role(["Admin"]))):
    user_to_delete = db.query(models.User).filter(models.User.id == user_id).first()
    if not user_to_delete:
        raise HTTPException(status_code=404, detail="User not found")
    
    # We shouldn't delete other Admins using this endpoint, though they're already filtered out in the get request
    if user_to_delete.role == "Admin":
         raise HTTPException(status_code=403, detail="Cannot delete an admin account")

    db.delete(user_to_delete)
    db.commit()
    return {"message": "User deleted successfully"}

# Example RBAC endpoint for Admin only
@app.get("/admin/dashboard")
def get_admin_data(current_user: models.User = Depends(require_role(["Admin"]))):
    return {"message": "Welcome to the Admin Dashboard", "user": current_user.email}

# Example RBAC endpoint for Company only
@app.get("/company/data")
def get_company_data(current_user: models.Company = Depends(require_role(["Company"]))):
    return {"message": "Welcome to the Company Portal", "company_name": current_user.company_name}

@app.post("/admin/add-company", response_model=schemas.CompanyOut, status_code=status.HTTP_201_CREATED)
def add_company(
    company_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    industry_category: str = Form(...),
    logo: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if db.query(models.User).filter(models.User.email == email).first() or \
       db.query(models.Company).filter(models.Company.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    logo_filename = f"{int(datetime.utcnow().timestamp())}_{logo.filename}"
    logo_path = os.path.join("static", "logos", logo_filename)
    with open(logo_path, "wb") as buffer:
        shutil.copyfileobj(logo.file, buffer)
    
    logo_url = f"/{logo_path}".replace("\\", "/")
    
    hashed_password = get_password_hash(password)
    db_company = models.Company(
        company_name=company_name,
        email=email,
        password=hashed_password,
        logo_url=logo_url,
        industry_category=industry_category
    )
    db.add(db_company)
    db.commit()
    db.refresh(db_company)
    return db_company

@app.get("/admin/companies", response_model=List[schemas.CompanyOut])
def get_companies(industry_category: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(models.Company)
    if industry_category:
        query = query.filter(models.Company.industry_category == industry_category)
    return query.all()

@app.get("/admin/companies/{category}", response_model=List[schemas.CompanyOut])
def get_companies_by_category(category: str, db: Session = Depends(get_db)):
    return db.query(models.Company).filter(models.Company.industry_category == category).all()

@app.post("/company/update-campaign", response_model=schemas.CampaignOut)
def update_campaign(
    video_url: Optional[str] = Form(None),
    product_name: Optional[str] = Form(None),
    product_price: Optional[str] = Form(None),
    product_description: Optional[str] = Form(None),
    product_ingredients: Optional[str] = Form(None),
    promo_code: Optional[str] = Form(None),
    discount_value: Optional[str] = Form(None),
    questions_list: Optional[str] = Form(None),
    product_photo: Optional[UploadFile] = File(None),
    current_user: models.Company = Depends(require_role(["Company"])),
    db: Session = Depends(get_db)
):
    # Check if campaign exists
    db_campaign = db.query(models.Campaign).filter(models.Campaign.company_id == current_user.id).first()
    
    update_data = {}
    if video_url is not None: update_data["video_url"] = video_url
    if product_name is not None: update_data["product_name"] = product_name
    if product_price is not None: update_data["product_price"] = product_price
    if product_description is not None: update_data["product_description"] = product_description
    if product_ingredients is not None: update_data["product_ingredients"] = product_ingredients
    if promo_code is not None: update_data["promo_code"] = promo_code
    if discount_value is not None: update_data["discount_value"] = discount_value
    
    if questions_list is not None:
        try:
            update_data["questions_list"] = json.loads(questions_list)
        except json.JSONDecodeError:
            pass
            
    if product_photo is not None and product_photo.filename:
        photo_filename = f"{int(datetime.utcnow().timestamp())}_{product_photo.filename}"
        os.makedirs("static/products", exist_ok=True)
        photo_path = os.path.join("static", "products", photo_filename)
        with open(photo_path, "wb") as buffer:
            shutil.copyfileobj(product_photo.file, buffer)
        update_data["product_photo_url"] = f"/{photo_path}".replace("\\", "/")
    
    if db_campaign:
        # Update existing
        for key, value in update_data.items():
            setattr(db_campaign, key, value)
    else:
        # Create new
        db_campaign = models.Campaign(
            company_id=current_user.id,
            **update_data
        )
        db.add(db_campaign)
    
    db.commit()
    db.refresh(db_campaign)
    return db_campaign

@app.get("/company/my-campaign", response_model=schemas.CampaignOut)
def get_my_campaign(
    current_user: models.Company = Depends(require_role(["Company"])),
    db: Session = Depends(get_db)
):
    campaign = db.query(models.Campaign).filter(models.Campaign.company_id == current_user.id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign

@app.get("/campaign/{company_name}", response_model=schemas.CampaignOut)
def get_campaign_by_company(company_name: str, db: Session = Depends(get_db)):
    # company_name might be formatted like "globaltech", so we search using ilike and removing spaces
    # Alternatively, the frontend encodes the exact DB name or we just do a direct match or case-insensitive match
    # A simple ilike will handle basic case differences, but replacing spaces in SQL might be needed if frontend strips them
    # Since sqlite doesn't easily support REPLACE without extensions, let's just do a direct match and assume the frontend can handle encoding spaces or we fetch all and match.
    # Actually, the user's frontend is using: brandId = encodeURIComponent(company.company_name.toLowerCase().replace(/\s+/g, ''))
    # This means the backend receives "globaltechinc". We need to match this.
    companies = db.query(models.Company).all()
    target_company = None
    for c in companies:
        if c.company_name.lower().replace(" ", "") == company_name.lower():
            target_company = c
            break
            
    if not target_company:
        raise HTTPException(status_code=404, detail="Company not found")
        
    campaign = db.query(models.Campaign).filter(models.Campaign.company_id == target_company.id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found for this company")
        
    return campaign

from pydantic import BaseModel
class UserResultSubmit(BaseModel):
    campaign_id: int
    survey_data: dict

@app.post("/user/submit-result")
def submit_result(
    result_data: UserResultSubmit,
    current_user: models.User = Depends(require_role(["User"])),
    db: Session = Depends(get_db)
):
    new_result = models.Result(
        user_id=current_user.id,
        campaign_id=result_data.campaign_id,
        survey_data=result_data.survey_data
    )
    db.add(new_result)
    db.commit()
    db.refresh(new_result)
    return {"status": "success", "result_id": new_result.id}
