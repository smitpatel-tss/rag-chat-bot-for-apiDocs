from src.pipeline.run_ingestion import run_pipeline


def test_embedding_pipeline():
    try:
        configs = [
        {
                "path": r"C:\Users\smit.patel\Desktop\CodeVilla\TSS\MD-html-convertions\md_files\TrackWizz KYC Photo Extraction API A116 Document V1.md",
                "frontend_url": "TrackWizz KYC Photo Extraction API A116 Document V1.html",
                "document_id": "A116",
        },
        {
                "path": r"C:\Users\smit.patel\Desktop\CodeVilla\TSS\MD-html-convertions\md_files\TrackWizz Photo Match API A109 v1.md",
                "frontend_url": "TrackWizz Photo Match API A109 v1.html",
                "document_id": "A109",
        },
        {
                "path": r"C:\Users\smit.patel\Desktop\CodeVilla\TSS\MD-html-convertions\md_files\TrackWizz Quality Check API A108X v1.1.md",
                "frontend_url": "TrackWizz Quality Check API A108X v1.1.html",
                "document_id": "A108X",
        },
        ]

        for config in configs:
                print(f"Ingesting {config['document_id']}...")

                run_pipeline(
                        mdfile_path=config["path"],
                        frontend_url=config["frontend_url"],
                        document_id=config["document_id"],
                )

                print(f" {config['document_id']} complete...")

        print("Ingetion COMPLETE!")

    except Exception as e:
        print("FAILED due to:")
        print(e)


#     for chunk, embedding in result:
#         display_chunk(chunk)
#         print(f"Embedding: \n {embedding.embedding[:100]}" )


# def display_chunk(c):
#         print("\n\n------------------------------------------------------------------")
#         print(f"Doc ID      : {c.doc_id}")
#         print(f"Chunk ID    : {c.chunk_id}")
#         print(f"Prev ID     : {c.prev_chunk_id}")
#         print(f"Next ID     : {c.next_chunk_id}")
#         print(f"Token Count : {c.token_count}")
#         print(f"Source URL  : {c.source_url}")
#         print(f"Anchor Link : {c.anchor}")
#         print(f"Title       : {c.title}")
#         print(f"doc version : {c.document_version}")
#         print(f"Parent Hdr  : {c.parent_header}")
#         print(f"section path: {c.section_path}")
#         print(f"chunk type  : {c.chunk_type}")
#         print(f"Attributes  : {c.attributes}\n")
#         print(f"--> Content :\n{c.content}")


if __name__ == "__main__":
    test_embedding_pipeline()