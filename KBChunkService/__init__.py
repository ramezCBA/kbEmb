import logging
import azure.functions as func
import uuid
import json
import KBChunkService.blobService as BlobService
import KBChunkService.FormattingService as FormattingService
import KBChunkService.indexes as indexes
import pandas as pd
from bs4 import BeautifulSoup

def main(BlobTrigger: func.InputStream):
    try:
        logging.info(f"Python blob trigger function processed blob \n"
                     f"Name: {BlobTrigger.name}\n"
                     f"Blob Size: {BlobTrigger.length} bytes")

        filename = BlobTrigger.name.split('/')[-1]  # Get File name when the blob is added
        blob_path = filename
        logging.info(f"Processing file: {filename}")

        stream = BlobService.read_stream_from_blob(blob_path)
        html_content = stream.read().decode('utf-8')  # Read and decode HTML content
        logging.info("Successfully read and decoded HTML content")

        soup = BeautifulSoup(html_content, 'html.parser')
        page_content = soup.get_text(separator=' ')  # Extract text content from HTML
        logging.info("Extracted text content from HTML")

        final_df = pd.DataFrame(columns=["Filename", "Content", "Embeddings"])
        docs = FormattingService.text_to_docs(page_content) # Format the content and create chunks based on token size
        logging.info("Formatted content and created document chunks")

        page_contents = [doc.page_content.replace("\n", "").replace("\t", "").replace("\r", "").replace("\u00a0", "").replace("\u201c", "").replace("\u201d", "") for doc in docs]
        
        df_html = FormattingService.start_process(content=page_contents, f_name=filename)  # Generate Embeddings and store in dataframe
        logging.info("Generated embeddings and stored them in a dataframe")

        final_df = pd.concat([final_df, df_html], ignore_index=True)
        
        data = {
            "value": final_df.apply(lambda row: {
                "id": str(uuid.uuid4()),
                "title": row['Filename'],
                "content": row['Content'],
                "category": filename.replace(".html", ""),
                "contentVector": row['Embeddings'],
                "@search.action": "upload"
            }, axis=1).tolist()
        }
        
        # Write JSON to file
        output_json_filename = f'{filename.replace(".html", "")}.json'
        outputjson = json.dumps(data["value"])
        logging.info(f"Generated JSON for output: {output_json_filename}")

        # Write json file to Blob storage
        BlobService.write_to_blob(outputjson, output_json_filename)
        logging.info(f"Written JSON to blob storage: {output_json_filename}")

        indexes.Create_Search_Index()
        logging.info("Created search index")

        indexes.Load_Doc_to_Index(output_json_filename)
        logging.info("Loaded document to search index")

    except Exception as e:
        logging.error("An error occurred during processing.")
        logging.error(f"Error: {str(e)}")