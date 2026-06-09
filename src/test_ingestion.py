from src.ingestion.pipeline import IngestionPipeline
from src.llm.factory import LLMFactory  

def test_pipeline():
    pipeline = IngestionPipeline(llm_provider=LLMFactory)
    
    local_path = r"C:\Users\smit.patel\Desktop\CodeVilla\RND\api_documentation.md"
    frontend_url = "https://docs.trackwizz.com/api/quality-check" 
    canonical_id = "trackwizz_api_v1.1"
    
    chunks = pipeline.process_file(
        file_path=local_path, 
        source_url=frontend_url, 
        doc_id=canonical_id,
        fast_test=True
    )
    
    print(f"Total Vector Chunks Created: {len(chunks)}\n")
    
    if chunks:
        c = chunks[0]
        print("============= CHUNK 1 GENERATED PAYLOAD SCHEMA =============")
        print(f"Doc ID      : {c.doc_id}")
        print(f"Chunk ID    : {c.chunk_id}")
        print(f"Prev ID     : {c.prev_chunk_id}")
        print(f"Next ID     : {c.next_chunk_id}")
        print(f"Token Count : {c.token_count}")
        print(f"Source URL  : {c.source_url}")
        print(f"Anchor Link : {c.anchor}")
        print(f"Title       : {c.title}")
        print(f"Parent Hdr  : {c.parent_header}")
        print(f"Attributes  : {c.attributes}\n")

        c = chunks[1]
        print("============= CHUNK 2 GENERATED PAYLOAD SCHEMA =============")
        print(f"Doc ID      : {c.doc_id}")
        print(f"Chunk ID    : {c.chunk_id}")
        print(f"Prev ID     : {c.prev_chunk_id}")
        print(f"Next ID     : {c.next_chunk_id}")
        print(f"Token Count : {c.token_count}")
        print(f"Source URL  : {c.source_url}")
        print(f"Anchor Link : {c.anchor}")
        print(f"Title       : {c.title}")
        print(f"Parent Hdr  : {c.parent_header}")
        print(f"Attributes  : {c.attributes}\n")

if __name__ == "__main__":
    test_pipeline()