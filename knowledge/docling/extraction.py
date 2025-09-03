from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls
import os

converter = DocumentConverter()

# --------------------------------------------------------------
# Document extraction function that handles both URLs and file paths
# --------------------------------------------------------------

def extract_document(source):
    """
    Extract content from a document using Docling
    
    Parameters:
    source (str): URL, file path, or file object to convert
    
    Returns:
    tuple: (document, markdown_output, json_output)
    """
    result = converter.convert(source)
    document = result.document
    markdown_output = document.export_to_markdown() if document else ""
    json_output = document.export_to_dict() if document else {}
    
    return document, markdown_output, json_output

# Example usage for PDF (commented out as it will be handled by Streamlit)
# document, markdown_output, json_output = extract_document("https://arxiv.org/pdf/2408.09869")
# print(markdown_output)

# --------------------------------------------------------------
# Basic HTML extraction
# --------------------------------------------------------------

result = converter.convert("https://ds4sd.github.io/docling/")

document = result.document
markdown_output = document.export_to_markdown()
print(markdown_output)

# --------------------------------------------------------------
# Scrape multiple pages using the sitemap
# --------------------------------------------------------------


# Only enable this when there is a need for parsing HTML files
# sitemap_urls = get_sitemap_urls("https://ds4sd.github.io/docling/")
# conv_results_iter = converter.convert_all(sitemap_urls)

# docs = []
# for result in conv_results_iter:
#     if result.document:
#         document = result.document
#         docs.append(document)
