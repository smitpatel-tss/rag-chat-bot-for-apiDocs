from pathlib import Path
from typing import List
from src.ingestion.nodes import SmartChunk
from src.ingestion.parser import MarkdownASTParser
from src.ingestion.enricher import HierarchyEnricher
from src.ingestion.sanitizer import GenericDocumentSanitizer
from src.ingestion.classifier import DeterministicClassifier
from src.ingestion.llm_enricher import LLMMetadataEnricher
from src.ingestion.chunk_builder import ChunkBuilder

class IngestionPipeline:
    def __init__(self, llm_provider):
        self.parser = MarkdownASTParser()
        self.enricher = HierarchyEnricher(llm_provider=llm_provider)
        self.sanitizer = GenericDocumentSanitizer()
        self.classifier = DeterministicClassifier()
        self.llm_enricher = LLMMetadataEnricher(llm_provider=llm_provider)
        self.chunk_builder = ChunkBuilder()

    def process_file(self, file_path: str, source_url: str, doc_id: str, fast_test: bool = False) -> List[SmartChunk]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file missing at: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        doc_identity = self.enricher.bootstrap_document_identity(text)
        global_metadata = self.enricher.extract_global_metadata(text)
        nodes = self.parser.parse(text)
        enriched_nodes = self.enricher.enrich(nodes, global_metadata, doc_identity)
        clean_nodes = self.sanitizer.sanitize(enriched_nodes)
        final_nodes = self.classifier.classify_and_merge(clean_nodes)
        
        if fast_test:
            structural_nodes = [n for n in final_nodes if n.type in ["code", "table"]]
            other_nodes = [n for n in final_nodes if n.type not in ["code", "table"]]
            test_structural = self.llm_enricher.enrich_nodes_with_llm(structural_nodes[:2])
            final_nodes = test_structural + structural_nodes[2:] + other_nodes
        else:
            final_nodes = self.llm_enricher.enrich_nodes_with_llm(final_nodes)
        
        smart_chunks = self.chunk_builder.assemble_chunks(
            final_nodes, 
            source_url=source_url, 
            doc_id=doc_id
        )
        
        return smart_chunks