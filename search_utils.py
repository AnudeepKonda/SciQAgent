import requests
from xml.etree import ElementTree
from bs4 import BeautifulSoup as bs


class ArxivSearch:
    """Class to search arXiv for articles matching a query."""

    BASE_URL = "http://export.arxiv.org/api/query"

    @staticmethod
    def search(query: str, max_results: int = 5) -> list[dict]:
        """
        Search arXiv for articles matching a query.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.

        Returns:
            A list of dictionaries containing article details.
        """
        params = {
            "search_query": f"all:{query}",  # Properly format query
            "start": 0,
            "max_results": max_results
        }

        response = requests.get(ArxivSearch.BASE_URL, params=params)
        if response.status_code != 200:
            return f"Error: Unable to fetch results from arXiv. Status Code: {response.status_code}"

        root = ElementTree.fromstring(response.content)
        articles = []

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
            link = entry.find("{http://www.w3.org/2005/Atom}id").text

            articles.append({
                "Title": title.strip(),
                "Abstract": summary.strip(),
                "Link": link.strip().replace('/abs/', '/pdf/')  # Convert to PDF link
            })

        return articles


class PubMedSearch:
    """Class to search PubMed for articles matching a query."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    @staticmethod
    def search(query: str, max_results: int = 5) -> list[dict]:
        """
        Search PubMed for articles matching a query.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.

        Returns:
            A list of dictionaries containing article details.
        """
        search_url = f"{PubMedSearch.BASE_URL}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": max_results
        }

        response = requests.get(search_url, params=params)
        if response.status_code != 200:
            return f"Error: Unable to fetch results from PubMed. Status Code: {response.status_code}"

        root = ElementTree.fromstring(response.content)
        pmids = [pmid.text for pmid in root.findall(".//Id")]

        return PubMedSearch.fetch_articles(pmids)

    @staticmethod
    def fetch_articles(pmids: list[str]) -> list[dict]:
        """
        Fetch article details for given PubMed IDs.

        Args:
            pmids: A list of PubMed IDs.

        Returns:
            A list of dictionaries containing article details.
        """
        if not pmids:
            return "No articles found."

        fetch_url = f"{PubMedSearch.BASE_URL}efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }

        response = requests.get(fetch_url, params=params)
        if response.status_code != 200:
            return f"Error: Unable to fetch articles. Status Code: {response.status_code}"

        root = ElementTree.fromstring(response.content)
        articles = []

        for article in root.findall(".//PubmedArticle"):
            # Safely extract DOI, checking for its existence
            doi = None
            doi_elem = article.find(".//ArticleId[@IdType='doi']")
            if doi_elem is not None:
                doi = doi_elem.text

            pmid = article.find(".//PMID").text
            title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else "No Title"
            abstract = article.find(".//AbstractText").text if article.find(".//AbstractText") is not None else "No Abstract"
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            articles.append({
                "Title": title,
                "Abstract": abstract,
                "PubMedLink": link,
                "Link": "https://doi.org/" + doi if doi else "No DOI"
            })

        return articles


class BioRxivSearch:
    """Class to search bioRxiv for preprints matching a query."""

    BASE_URL = "https://www.biorxiv.org/search/"

    @staticmethod
    def search(query: str, max_results: int = 5) -> list[dict]:
        """
        Search bioRxiv for preprints matching a query.

        Args:
            query: The search query.
            max_results: The maximum number of results to return.

        Returns:
            A list of dictionaries containing article details.
        """

        formatted_query = "+".join(query.split())  # Convert spaces to '+'
        search_url = f"{BioRxivSearch.BASE_URL}{formatted_query}%20numresults%3A{max_results}%20format_result%3Acondensed"

        response = requests.get(search_url)
        if response.status_code != 200:
            return f"Error: Unable to fetch results from bioRxiv. Status Code: {response.status_code}"

        html = bs(response.text, "html.parser")
        articles = html.find_all("li", class_="search-result")[:max_results]  # Get top results

        results = []
        for article in articles:
            # Extract title
            title_tag = article.find("span", class_="highwire-cite-title")
            title = title_tag.text.strip() if title_tag else "No title available"

            # Extract URL
            link_tag = article.find("a", class_="highwire-cite-linked-title")
            url = "https://www.biorxiv.org" + link_tag["href"] if link_tag else "No link available"

            # Fetch abstract separately (optional, but slows it down)
            abstract = "No abstract available"
            abstract_page = requests.get(url)
            if abstract_page.status_code == 200:
                abstract_html = bs(abstract_page.text, "html.parser")
                abstract_section = abstract_html.find("div", class_="section abstract")
                if abstract_section:
                    abstract = abstract_section.text.replace("Abstract", "").strip()

            results.append({
                "Title": title,
                "Abstract": abstract,
                "Link": url + '.full'
            })

        return results
