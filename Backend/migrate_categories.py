import asyncio
from sqlalchemy.orm import Session
from database import SessionLocal
import models

def migrate_categories():
    db = SessionLocal()
    try:
        # Get all unique industry categories from existing companies
        companies = db.query(models.Company).all()
        unique_categories = set(company.industry_category for company in companies if company.industry_category)

        icon_mapping = {
            "FMCG": "shopping_cart",
            "Beauty & Fragrances": "eco",
            "Sportswear & Athletics": "sports_kabaddi",
            "Consumer Electronics": "devices"
        }

        # Create Category objects if they don't exist
        category_map = {}
        for cat_name in unique_categories:
            default_icon = icon_mapping.get(cat_name, "category")
            category = db.query(models.Category).filter(models.Category.name == cat_name).first()
            if not category:
                # Provide a generic default description and icon
                category = models.Category(
                    name=cat_name,
                    description=f"{cat_name} Category",
                    icon_url=default_icon
                )
                db.add(category)
                db.commit()
                db.refresh(category)
            else:
                # Fix existing broken categories
                if category.icon_url == "/static/categories/default_icon.png":
                    category.icon_url = default_icon
                    db.commit()
                    db.refresh(category)
            category_map[cat_name] = category.id

        # Update companies with the new category_id
        for company in companies:
            if company.industry_category and company.industry_category in category_map:
                company.category_id = category_map[company.industry_category]
        
        db.commit()
        print(f"Successfully migrated {len(companies)} companies into {len(unique_categories)} categories.")
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    migrate_categories()
