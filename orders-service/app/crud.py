from sqlalchemy import select
from sqlalchemy.orm import Session
from . import models, schemas

def create_order(db: Session, data: schemas.OrderCreate, price: float | None) -> models.Order:
    order = models.Order(**data.model_dump(), predicted_price=price)
    db.add(order)
    db.commit()
    db.refresh(order)
    return order

def get_order(db: Session, oid: int) -> models.Order | None:
    return db.get(models.Order, oid)

def list_orders(db: Session, limit: int = 20, offset: int = 0) -> list[models.Order]:
    return list(db.scalars(select(models.Order).order_by(models.Order.id.desc()).limit(limit).offset(offset)))

def update_order(db: Session, order: models.Order, changes: dict) -> models.Order:
    for key, value in changes.items():
        setattr(order, key, value)
    db.commit()
    db.refresh(order)
    return order

def delete_order(db: Session, order: models.Order) -> None:
    db.delete(order)
    db.commit()
