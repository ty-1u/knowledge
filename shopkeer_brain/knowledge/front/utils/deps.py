from functools import lru_cache


@lru_cache
def get_task_service():
    from shopkeer_brain.knowledge.front.service.task_service import TaskService
    return TaskService()

@lru_cache
def get_import_file_service():
    from shopkeer_brain.knowledge.front.service.import_file_service import ImportFileService
    return ImportFileService(get_task_service())

@lru_cache
def get_query_service():
    from shopkeer_brain.knowledge.front.service.query_service import QueryService
    return QueryService()
