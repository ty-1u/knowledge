"""Front import-service compatibility wrapper.

The canonical implementation lives in the upload module. Keeping this wrapper
prevents the query and legacy front entrypoints from loading a second,
out-of-date import pipeline.
"""

from shopkeer_brain.knowledge.upload.service.import_file_service import ImportFileService

__all__ = ["ImportFileService"]
