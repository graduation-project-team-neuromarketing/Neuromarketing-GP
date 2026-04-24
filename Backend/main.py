from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import List, Optional
import os

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

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        return {"access_token": access_token, "token_type": "bearer"}
    
    # Check Companies table
    company = db.query(models.Company).filter(models.Company.email == form_data.username).first()
    if company and verify_password(form_data.password, company.password):
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": company.email, "role": "Company"}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

@app.get("/users/me", response_model=schemas.UserOut)
def read_users_me(current_user: models.User = Depends(require_role(["Admin", "User", "Company"]))):
    return current_user

# Example RBAC endpoint for Admin only
@app.get("/admin/dashboard")
def get_admin_data(current_user: models.User = Depends(require_role(["Admin"]))):
    return {"message": "Welcome to the Admin Dashboard", "user": current_user.email}

# Example RBAC endpoint for Company only
@app.get("/company/data")
def get_company_data(current_user: models.Company = Depends(require_role(["Company"]))):
    return {"message": "Welcome to the Company Portal", "company_name": current_user.company_name}
