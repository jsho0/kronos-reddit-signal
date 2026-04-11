"""
Database engine setup.

- WAL mode enabled via connection event hook (SQLite only, no-op on other DBs)
- busy_timeout=10000ms prevents "database is locked" errors under concurrent access
- get_session() is a context manager that commits on success, rolls back on error
"""
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from storage.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionFactory = None


def get_engine(database_url: str = None):
    """
    Return the module-level engine, creating it if needed.
    Passing database_url on first call sets the URL for the process lifetime.
    Subsequent calls ignore database_url (cached engine is returned).
    """
    global _engine, _SessionFactory

    if _engine is not None:
        return _engine

    if database_url is None:
        from config import DATABASE_URL
        database_url = DATABASE_URL

    connect_args = {}
    engine_kwargs = {"echo": False}

    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if database_url == "sqlite://":
            # In-memory SQLite: use StaticPool so every get_session() call
            # shares the same connection (and therefore the same database).
            # Without this, each new connection gets a fresh empty database.
            from sqlalchemy.pool import StaticPool
            engine_kwargs["poolclass"] = StaticPool

    _engine = create_engine(
        database_url,
        connect_args=connect_args,
        **engine_kwargs,
    )

    # WAL mode + busy timeout — applied to every new SQLite connection
    if database_url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=10000")
            cursor.close()

    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
    logger.debug("Database engine created: %s", database_url)
    return _engine


def init_db(database_url: str = None):
    """
    Create all tables if they don't exist. Safe to call repeatedly (CREATE IF NOT EXISTS).
    Call this once at application startup.
    """
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    logger.info("Database tables initialized")


@contextmanager
def get_session(database_url: str = None) -> Session:
    """
    Context manager that yields a SQLAlchemy Session.
    Commits on clean exit, rolls back on exception.

    Usage:
        with get_session() as session:
            session.add(some_model)
    """
    if _SessionFactory is None:
        get_engine(database_url)

    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
