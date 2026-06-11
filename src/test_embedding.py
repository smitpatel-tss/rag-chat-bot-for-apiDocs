from src.pipeline.run_ingestion import run_pipeline

def test_embedding_pipeline():
    path = r"C:\Users\smit.patel\Desktop\CodeVilla\RND\api_doc_test_md.md"
    frontend_url = "https://docs.trackwizz.com/api/quality-check"
    document_id = "trackwizz_api_v1.1"
    result = run_pipeline(mdfile_path=path, frontend_url=frontend_url, document_id=document_id)

    for chunk, embedding in result:
        display_chunk(chunk)
        print(f"Embedding: \n {embedding.embedding[:100]}" )


def display_chunk(c):
        print("\n\n------------------------------------------------------------------")
        print(f"Doc ID      : {c.doc_id}")
        print(f"Chunk ID    : {c.chunk_id}")
        print(f"Prev ID     : {c.prev_chunk_id}")
        print(f"Next ID     : {c.next_chunk_id}")
        print(f"Token Count : {c.token_count}")
        print(f"Source URL  : {c.source_url}")
        print(f"Anchor Link : {c.anchor}")
        print(f"Title       : {c.title}")
        print(f"doc version : {c.document_version}")
        print(f"Parent Hdr  : {c.parent_header}")
        print(f"section path: {c.section_path}")
        print(f"chunk type  : {c.chunk_type}")
        print(f"Attributes  : {c.attributes}\n")
        print(f"--> Content :\n{c.content}")

if __name__== "__main__":
     test_embedding_pipeline()