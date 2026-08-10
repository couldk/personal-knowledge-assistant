from personal_knowledge_assistant.ingestion.catalog import (
    DocumentCatalog,
    InMemoryDocumentCatalog,
)
from personal_knowledge_assistant.ingestion.markdown_loader import (
    MarkdownLoader,
)
from personal_knowledge_assistant.ingestion.pdf_loader import PdfLoader
from personal_knowledge_assistant.ingestion.registry import LoaderRegistry
from personal_knowledge_assistant.ingestion.service import (
    DocumentIngestionService,
)
from personal_knowledge_assistant.ingestion.text_loader import TextLoader


def create_default_loader_registry() -> LoaderRegistry:
    """创建包含所有内置Loader的默认注册表。"""

    return LoaderRegistry(
        [
            TextLoader(),
            MarkdownLoader(),
            PdfLoader(),
        ]
    )


def create_default_ingestion_service(
    catalog: DocumentCatalog | None = None,
) -> DocumentIngestionService:
    """创建可以直接使用的默认文档导入服务。"""

    document_catalog: DocumentCatalog = (
        catalog if catalog is not None else InMemoryDocumentCatalog()
    )

    return DocumentIngestionService(
        registry=create_default_loader_registry(),
        catalog=document_catalog,
    )
