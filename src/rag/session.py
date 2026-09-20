import os
import time
import tempfile
import uuid

def create_session():
    session_id = str(uuid.uuid4())

    session_dir = os.path.join(
        tempfile.gettempdir(),
        "rag_sessions",
        session_id
    )

    os.makedirs(session_dir, exist_ok=True)

    return session_id, session_dir


def cleanup_session(session_dir: str):
    if session_dir and os.path.exists(session_dir):
        import shutil
        shutil.rmtree(session_dir, ignore_errors=True)


def cleanup_old_sessions(max_age_hours=24):
    sessions_root = os.path.join(
        tempfile.gettempdir(),
        "rag_sessions"
    )

    if not os.path.exists(sessions_root):
        return

    current_time = time.time()
    max_age_seconds = max_age_hours * 60 * 60

    for session_id in os.listdir(sessions_root):
        session_dir = os.path.join(sessions_root, session_id)

        if not os.path.isdir(session_dir):
            continue

        age = current_time - os.path.getmtime(session_dir)

        if age > max_age_seconds:
            cleanup_session(session_dir)


def initialize_session():
    cleanup_old_sessions(max_age_hours=24)

    _, session_dir = create_session()

    return session_dir