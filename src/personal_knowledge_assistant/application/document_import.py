from __future__ import annotations

from asyncio import to_thread
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from personal_knowledge_assistant.domain import (
    ImportResult,
    ImportStatus,
    StoredDocument,
)
from personal_knowledge_assistant.indexing import (
    DocumentIndexingService,
    IndexingStatus,
)
from personal_knowledge_assistant.ingestion import (
    DocumentIngestionService,
)
from personal_knowledge_assistant.providers.base import (
    DocumentStoreProvider,
)


class DocumentUploadError(Exception):
    """文档上传错误基类。"""


class InvalidUploadFileNameError(DocumentUploadError):
    """上传文件名为空或包含不安全路径。"""


class UnsupportedUploadTypeError(DocumentUploadError):
    """上传文件类型不受支持。"""


class UploadTooLargeError(DocumentUploadError):
    """上传文件超过大小限制。"""


class DocumentStorageError(DocumentUploadError):
    """上传文件无法安全保存。"""


@dataclass(frozen=True, slots=True)
class DocumentImportOutcome:
    """上传、导入和索引完成后的统一结果。"""

    import_status: ImportStatus
    indexing_status: IndexingStatus
    document_id: str
    content_hash: str
    file_name: str
    chunk_count: int
    deactivated_chunk_count: int


class DocumentImportService:
    """协调上传文件保存、解析和索引。"""

    _SUPPORTED_EXTENSIONS = frozenset(
        {
            ".txt",
            ".md",
            ".markdown",
            ".pdf",
        }
    )

    _WINDOWS_RESERVED_NAMES = frozenset(
        {
            "con",
            "prn",
            "aux",
            "nul",
            "com1",
            "com2",
            "com3",
            "com4",
            "com5",
            "com6",
            "com7",
            "com8",
            "com9",
            "lpt1",
            "lpt2",
            "lpt3",
            "lpt4",
            "lpt5",
            "lpt6",
            "lpt7",
            "lpt8",
            "lpt9",
        }
    )

    def __init__(
        self,
        *,
        ingestion_service: DocumentIngestionService,
        indexing_service: DocumentIndexingService,
        upload_directory: Path,
        max_upload_bytes: int,
        document_store: DocumentStoreProvider | None = None,
    ) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be greater than zero.")

        self._ingestion_service = ingestion_service
        self._indexing_service = indexing_service
        self._upload_directory = upload_directory
        self._max_upload_bytes = max_upload_bytes

        # pgvector 模式下使用 PostgreSQL 中的持久化文档状态。
        # memory 模式和普通单元测试可以不提供。
        self._document_store = document_store

    @property
    def max_upload_bytes(self) -> int:
        """返回允许的最大上传字节数。"""

        return self._max_upload_bytes

    async def import_upload(
        self,
        *,
        file_name: str,
        content: bytes,
        force_reindex: bool = False,
    ) -> DocumentImportOutcome:
        """保存上传文件，并完成导入和索引。"""

        normalized_file_name = self._validate_file_name(file_name)

        if len(content) > self._max_upload_bytes:
            raise UploadTooLargeError("Uploaded document exceeds the size limit.")

        stored_path = await to_thread(
            self._store_upload,
            normalized_file_name,
            content,
        )

        # Loader 和本地内存 Catalog 都是同步实现，
        # 放到工作线程中执行，避免阻塞 FastAPI 事件循环。
        import_result = await to_thread(
            self._ingestion_service.import_document,
            stored_path,
        )

        # 如果配置了 PostgreSQL 文档存储，
        # 使用数据库中的活动版本重新判断导入状态。
        if self._document_store is not None:
            stored_document = await self._document_store.get_document_by_id(
                import_result.document.document_id
            )

            import_result = self._apply_persistent_import_status(
                import_result=import_result,
                stored_document=stored_document,
            )

        indexing_result = await self._indexing_service.index(
            import_result,
            force=force_reindex,
        )

        return DocumentImportOutcome(
            import_status=import_result.status,
            indexing_status=indexing_result.status,
            document_id=indexing_result.document_id,
            content_hash=indexing_result.content_hash,
            file_name=(import_result.document.metadata.file_name),
            chunk_count=indexing_result.chunk_count,
            deactivated_chunk_count=(indexing_result.deactivated_chunk_count),
        )

    def _apply_persistent_import_status(
        self,
        *,
        import_result: ImportResult,
        stored_document: StoredDocument | None,
    ) -> ImportResult:
        """根据PostgreSQL活动版本重新判断导入状态。

        InMemoryDocumentCatalog 会在应用重启后清空，
        因此生产环境必须以 PostgreSQL 中的活动版本为准。
        """

        document = import_result.document

        # 数据库中完全不存在，或者文档已删除且不存在活动版本。
        if stored_document is None or stored_document.active_content_hash is None:
            return ImportResult(
                status=ImportStatus.CREATED,
                document=document,
                previous_content_hash=None,
            )

        previous_content_hash = stored_document.active_content_hash

        # 文件内容与数据库活动版本相同，不重新生成向量。
        if previous_content_hash == document.content_hash:
            return ImportResult(
                status=ImportStatus.UNCHANGED,
                document=document,
                previous_content_hash=(previous_content_hash),
            )

        # document_id 相同但内容哈希不同，属于文档更新。
        return ImportResult(
            status=ImportStatus.UPDATED,
            document=document,
            previous_content_hash=(previous_content_hash),
        )

    def _validate_file_name(
        self,
        file_name: str,
    ) -> str:
        """拒绝空文件名、路径穿越和不支持的类型。"""

        normalized = file_name.strip()

        if not normalized:
            raise InvalidUploadFileNameError("Upload file name cannot be empty.")

        if len(normalized) > 255:
            raise InvalidUploadFileNameError("Upload file name is too long.")

        if "/" in normalized or "\\" in normalized or normalized in {".", ".."}:
            raise InvalidUploadFileNameError("Upload file name must not contain a path.")

        path = Path(normalized)
        suffix = path.suffix.casefold()

        if suffix not in self._SUPPORTED_EXTENSIONS:
            raise UnsupportedUploadTypeError(f"Unsupported upload type: {suffix or '<none>'}")

        # Windows 会把 CON.txt、CON.backup.txt 等
        # 文件名都识别为保留设备名。
        reserved_name_candidate = path.name.split(
            ".",
            maxsplit=1,
        )[0].casefold()

        if reserved_name_candidate in self._WINDOWS_RESERVED_NAMES:
            raise InvalidUploadFileNameError("Upload file name is reserved by Windows.")

        return normalized

    def _store_upload(
        self,
        file_name: str,
        content: bytes,
    ) -> Path:
        """使用临时文件和原子替换保存上传内容。"""

        try:
            upload_root = self._upload_directory.expanduser().resolve()

            upload_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination = (upload_root / file_name).resolve()

            if destination.parent != upload_root:
                raise InvalidUploadFileNameError("Upload path escapes the storage directory.")

            temporary_path: Path | None = None

            try:
                with NamedTemporaryFile(
                    mode="wb",
                    dir=upload_root,
                    prefix=".upload-",
                    suffix=destination.suffix,
                    delete=False,
                ) as temporary_file:
                    temporary_file.write(content)
                    temporary_path = Path(temporary_file.name)

                temporary_path.replace(destination)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

            return destination
        except DocumentUploadError:
            raise
        except OSError as exc:
            raise DocumentStorageError("Failed to store uploaded document.") from exc
