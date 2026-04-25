from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Text, ARRAY
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
from app.db.database import Base


class POITable(Base):
    __tablename__ = "pois"

    poi_id           = Column(String,  primary_key=True)
    name             = Column(String,  nullable=False, index=True)
    lat              = Column(Float,   nullable=False)
    lon              = Column(Float,   nullable=False)
    geom             = Column(Geometry("POINT", srid=4326))   # PostGIS spatial column
    category         = Column(String,  nullable=False, index=True)
    description      = Column(Text)
    source_url       = Column(String)
    address          = Column(String)
    opening_hours    = Column(String)
    source_count     = Column(Integer, default=1)
    confidence_score = Column(Float,   default=0.0, index=True)
    raw_tags         = Column(JSON,    default=dict)
    created_at       = Column(DateTime, server_default=func.now())
    updated_at       = Column(DateTime, server_default=func.now(), onupdate=func.now())


class UserPreferenceTable(Base):
    __tablename__ = "user_preferences"

    user_id    = Column(String,  primary_key=True)
    weights    = Column(ARRAY(Float), nullable=False)
    tone       = Column(String,  default="informative")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
