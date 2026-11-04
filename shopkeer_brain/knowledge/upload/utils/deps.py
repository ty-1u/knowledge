from functools import lru_cache

from shopkeer_brain.knowledge.upload.service.import_file_service import ImportFileService
from shopkeer_brain.knowledge.upload.service.task_service import TaskService

@lru_cache
def get_task_service() -> TaskService:
    return TaskService()

@lru_cache
def get_import_file_service() -> ImportFileService:
    return ImportFileService(get_task_service())
