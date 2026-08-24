from llama_index.core.node_parser import SentenceSplitter

from personal_knowledge_assistant.chunking.exceptions import (
    EmptyChunkingResultError,
)
from personal_knowledge_assistant.domain import (
    ChunkingConfig,
    ChunkMetadata,
    DocumentChunk,
    LoadedDocument,
    create_chunk_id,
)


class SentenceChunker:
    """使用LlamaIndex SentenceSplitter按文档结构切块。"""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
    ) -> None:
        self._config = config or ChunkingConfig()

        self._splitter = SentenceSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )

    @property
    def config(self) -> ChunkingConfig:
        """返回当前切块配置。"""

        return self._config

    def split(
        self,
        document: LoadedDocument,
    ) -> list[DocumentChunk]:
        """逐个DocumentPart切块，并保留来源信息。"""

        chunks: list[DocumentChunk] = []

        for part in document.parts:
            # 不把只有空白字符的Part交给分词器。
            if not part.text.strip():
                continue

            fragments = self._splitter.split_text(part.text)
            chunk_index = 0

            for fragment in fragments:
                text = fragment.strip()

                if not text:
                    continue

                chunk_id = create_chunk_id(
                    document_id=document.document_id,
                    content_hash=document.content_hash,
                    part_index=part.part_index,
                    chunk_index=chunk_index,
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        text=text,
                        metadata=ChunkMetadata(
                            content_hash=document.content_hash,
                            source_path=document.metadata.source_path,
                            file_name=document.metadata.file_name,
                            document_type=document.metadata.document_type,
                            modified_at=document.metadata.modified_at,
                            part_index=part.part_index,
                            chunk_index=chunk_index,
                            page_number=part.page_number,
                            section_path=part.section_path,
                        ),
                    )
                )

                chunk_index += 1

        if not chunks:
            raise EmptyChunkingResultError(document.document_id)

        return chunks
