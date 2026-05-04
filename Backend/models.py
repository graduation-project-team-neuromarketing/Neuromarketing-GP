from sqlalchemy import Column, Integer, String, ForeignKey, JSON, DateTime, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    full_name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)  # hashed password
    gender = Column(String)
    age = Column(Integer)
    phone = Column(String)
    profile_pic = Column(String, nullable=True)
    points = Column(Integer, default=0)
    role = Column(String, default="User")  # Admin / Company / User

    results = relationship("Result", back_populates="user")
    histories = relationship("History", back_populates="user")

class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    icon_url = Column(String, nullable=True)

    companies = relationship("Company", back_populates="category")

class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    company_name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)  # hashed password
    logo_url = Column(String, nullable=True)
    industry_category = Column(String) # To be deleted later
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)

    category = relationship("Category", back_populates="companies")
    campaigns = relationship("Campaign", back_populates="company")

class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("companies.id"))
    video_url = Column(String, nullable=True)
    description = Column(String, nullable=True)
    product_name = Column(String, nullable=True)
    product_price = Column(String, nullable=True)
    product_description = Column(String, nullable=True)
    product_ingredients = Column(String, nullable=True)
    product_photo_url = Column(String, nullable=True)
    questions_list = Column(JSON, nullable=True)  # JSON format to allow flexibility
    promo_code = Column(String, nullable=True)
    discount_value = Column(String, nullable=True)

    company = relationship("Company", back_populates="campaigns")
    results = relationship("Result", back_populates="campaign")
    histories = relationship("History", back_populates="campaign")

class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    survey_data = Column(JSON)  # JSON format
    neural_score = Column(Float, nullable=True)# the prediction of the model for the campaign

    user = relationship("User", back_populates="results")
    campaign = relationship("Campaign", back_populates="results")

class History(Base):
    __tablename__ = "history"

    # SQLite needs a primary key even for association/history tables
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    completion_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="In Processing")
    earned_points = Column(Integer, default=0)
    earned_promo_code = Column(String, nullable=True)

    user = relationship("User", back_populates="histories")
    campaign = relationship("Campaign", back_populates="histories")
