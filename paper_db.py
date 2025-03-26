import re
import requests
import logging
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def follow_redirects(url):
    """
    Follow redirects for a given URL.

    Args:
        url: The URL to follow redirects for.

    Returns:
        The final URL after following redirects and the response text.
    """
    response = requests.head(url, timeout=5, allow_redirects=True)
    return response.url, response.text


def clean_text(text):
    """
    Clean the given text by removing unnecessary spaces, tabs, and newlines.

    Args:
        text: The text to clean.

    Returns:
        The cleaned text.
    """
    text = text.replace('\xa0', ' ')  # Replace non-breaking spaces with regular spaces
    text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces and tabs with a single space
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    return text.strip()  # Strip leading and trailing whitespace


class paperDB:
    """A class that handles the scraping of websites and PDFs and hold langchain documents for processing."""
    def __init__(self, text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)):
        """
        Initialize the paperDB instance.

        Args:
            text_splitter: The text splitter to use for splitting documents into chunks.
        """
        self.text_splitter = text_splitter
        self.documents = []
        self.abstracts = []
        self.global_doc_count = 0
        self.vectorstore = None

    def is_pdf(self, url: str):
        """
        Check if the URL points to a PDF file.

        Args:
            url: The URL to check.

        Returns:
            True if the URL points to a PDF file, False otherwise.
        """
        parsed_url = urlparse(url)
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '').lower()
        return 'pdf' in parsed_url.path.lower() or content_type == 'application/pdf'

    def scrape_pdf(self, pdf_link: str):
        """
        Scrape the PDF for text.

        Args:
            pdf_link: The link to the PDF file.

        Returns:
            A list of cleaned pages from the PDF.
            If the PDF is not reachable, it returns None.

        Raises:
            Exception: If there is an error loading the PDF.
        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'} 
            pdf_loader = PyPDFLoader(pdf_link, headers=headers)
            pages = pdf_loader.load_and_split(text_splitter=self.text_splitter)

            # Clean the text in each page
            for page in pages:
                if page.page_content:
                    page.page_content = clean_text(page.page_content)

            return pages

        except Exception as e:
            logging.error(f"Error loading {pdf_link}: {e}")
            return None

    def scrape_website(self, web_link: str):
        """
        Scrape the website for text.

        Args:
            web_link: The link to the website.

        Returns:
            A list of cleaned pages from the website.
            If the website is not reachable, it returns None.

        Raises:
            Exception: If there is an error loading the website.
        """
        try:
            final_url, _ = follow_redirects(web_link)
            loader = WebBaseLoader(final_url)
            pages = loader.load_and_split(text_splitter=self.text_splitter)

            # Clean the text in each page
            for page in pages:
                if page.page_content:
                    page.page_content = clean_text(page.page_content)

            return pages

        except Exception as e:
            logging.error(f"Error loading {web_link}: {e}")
            return None

    def process_url(self, url: str):
        """
        Process the URL to extract text, chunk it, and store it in the vectorstore.
        This method checks if the URL is a PDF or a website and processes it accordingly.

        Args:
            url: The URL to process.
        """
        try:
            if self.is_pdf(url):
                docs = self.scrape_pdf(url)
            else:
                docs = self.scrape_website(url)

            if docs:
                # add unique doc_id to each document
                for doc in docs:
                    doc.metadata['doc_id'] = self.global_doc_count
                    self.global_doc_count += 1

                self.documents.extend(docs)
                if not self.vectorstore:
                    self.vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(model="text-embedding-3-small"))
                    self.retriever = self.vectorstore.as_retriever(search_kwargs={'k': 20})
                else:
                    self.vectorstore.add_documents(docs)
                logging.info(f"Processed {len(docs)} documents from {url}")
            else:
                logging.info(f"No documents found for {url}")
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")

    def process_urls_parallel(self, urls: list, max_workers=5):
        """
        Process multiple URLs in parallel to extract text, chunk it, and store it in the vectorstore.

        Args:
            urls: A list of URLs to process.
            max_workers: The maximum number of worker threads to use.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_url, urls)
