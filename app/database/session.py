#!/usr/bin/env python3
"""
SQLAlchemy engine factory and FastAPI dependency.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        # SQLite fallback for local dev / tests
        url = "sqlite:///./nfl_dev.db"
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


engine = get_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
