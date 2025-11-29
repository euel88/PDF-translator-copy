"""
백엔드 서버 모듈 - PDFMathTranslate 구조 기반
Flask REST API 및 비동기 작업 처리
"""
import os
import uuid
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    from flask import Flask, request, jsonify, send_file
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from pdf2zh.high_level import translate
from pdf2zh.config import config


@dataclass
class Task:
    """번역 작업"""
    id: str
    input_path: str
    output_path: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    progress: int = 0
    total_pages: int = 0
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "total_pages": self.total_pages,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TaskManager:
    """작업 관리자"""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()

    def create_task(self, input_path: str) -> Task:
        """새 작업 생성"""
        task_id = str(uuid.uuid4())
        task = Task(id=task_id, input_path=input_path)

        with self._lock:
            self._tasks[task_id] = task

        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """작업 조회"""
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs):
        """작업 업데이트"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)

    def delete_task(self, task_id: str):
        """작업 삭제"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]

    def list_tasks(self) -> list:
        """모든 작업 목록"""
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]


# 전역 작업 관리자
task_manager = TaskManager()


def create_app() -> Optional[Any]:
    """Flask 앱 생성"""
    if not FLASK_AVAILABLE:
        return None

    app = Flask(__name__)

    # 업로드 폴더
    upload_folder = Path(tempfile.gettempdir()) / "pdf2zh_uploads"
    upload_folder.mkdir(exist_ok=True)
    app.config["UPLOAD_FOLDER"] = str(upload_folder)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

    @app.route("/api/health", methods=["GET"])
    def health():
        """헬스 체크"""
        return jsonify({"status": "ok"})

    @app.route("/api/translate", methods=["POST"])
    def translate_pdf():
        """PDF 번역 요청"""
        # 파일 확인
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # 파일 저장
        filename = f"{uuid.uuid4()}_{file.filename}"
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)

        # 옵션
        source_lang = request.form.get("source_lang", "English")
        target_lang = request.form.get("target_lang", "Korean")
        service = request.form.get("service", "openai")

        # 작업 생성
        task = task_manager.create_task(input_path)

        # 백그라운드 처리
        thread = threading.Thread(
            target=_process_task,
            args=(task.id, input_path, source_lang, target_lang, service)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            "task_id": task.id,
            "status": "pending",
            "message": "Translation started"
        })

    @app.route("/api/task/<task_id>", methods=["GET"])
    def get_task_status(task_id: str):
        """작업 상태 조회"""
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404

        return jsonify(task.to_dict())

    @app.route("/api/task/<task_id>/download", methods=["GET"])
    def download_result(task_id: str):
        """결과 파일 다운로드"""
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404

        if task.status != "completed":
            return jsonify({"error": "Task not completed"}), 400

        if not task.output_path or not os.path.exists(task.output_path):
            return jsonify({"error": "Output file not found"}), 404

        return send_file(
            task.output_path,
            as_attachment=True,
            download_name=os.path.basename(task.output_path)
        )

    @app.route("/api/tasks", methods=["GET"])
    def list_tasks():
        """모든 작업 목록"""
        return jsonify({"tasks": task_manager.list_tasks()})

    return app


def _process_task(
    task_id: str,
    input_path: str,
    source_lang: str,
    target_lang: str,
    service: str
):
    """작업 처리"""
    try:
        task_manager.update_task(task_id, status="processing")

        # 출력 경로
        output_path = input_path.replace(".pdf", "_translated.pdf")

        def callback(msg: str):
            # 진행률 파싱
            if "페이지" in msg and "/" in msg:
                try:
                    parts = msg.split("/")
                    current = int(parts[0].split()[-1])
                    total = int(parts[1].split()[0])
                    task_manager.update_task(
                        task_id,
                        progress=current,
                        total_pages=total
                    )
                except:
                    pass

        result = translate(
            input_path=input_path,
            output_path=output_path,
            source_lang=source_lang,
            target_lang=target_lang,
            service=service,
            callback=callback,
        )

        if result.success:
            task_manager.update_task(
                task_id,
                status="completed",
                output_path=output_path,
                completed_at=datetime.now()
            )
        else:
            task_manager.update_task(
                task_id,
                status="failed",
                error=result.error,
                completed_at=datetime.now()
            )

    except Exception as e:
        task_manager.update_task(
            task_id,
            status="failed",
            error=str(e),
            completed_at=datetime.now()
        )


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """서버 실행"""
    app = create_app()
    if app is None:
        print("Flask가 설치되지 않았습니다. pip install flask 로 설치하세요.")
        return

    print(f"서버 시작: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


# Celery 지원 (선택적)
try:
    from celery import Celery

    celery_app = Celery(
        "pdf2zh",
        broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    )

    @celery_app.task(bind=True)
    def translate_task(
        self,
        input_path: str,
        output_path: str,
        source_lang: str,
        target_lang: str,
        service: str,
        **kwargs
    ):
        """Celery 번역 작업"""
        def callback(msg: str):
            if "페이지" in msg and "/" in msg:
                try:
                    parts = msg.split("/")
                    current = int(parts[0].split()[-1])
                    total = int(parts[1].split()[0])
                    self.update_state(
                        state="PROGRESS",
                        meta={"current": current, "total": total}
                    )
                except:
                    pass

        result = translate(
            input_path=input_path,
            output_path=output_path,
            source_lang=source_lang,
            target_lang=target_lang,
            service=service,
            callback=callback,
            **kwargs
        )

        return {
            "success": result.success,
            "output_path": result.output_path,
            "page_count": result.page_count,
            "error": result.error
        }

    CELERY_AVAILABLE = True

except ImportError:
    celery_app = None
    CELERY_AVAILABLE = False


if __name__ == "__main__":
    run_server(debug=True)
