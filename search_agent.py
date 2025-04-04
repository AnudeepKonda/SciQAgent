import logging
import os
from typing import Dict, List

import dspy
import time

from dspy_signatures import QueryExpansionSignature, RelevanceSignature, SourceSelectionSignature
from search_utils import ArxivSearch, BioRxivSearch, PubMedSearch

logger = logging.getLogger('SciQAgent')

MAX_PAPERS = 5  # Maximum number of papers to return for the search


def expand_query(conversation: str, model: str = "openai/gpt-4o-mini", temperature: float = 0.7) -> List[str]:
    """
    Expands a scientific query into three versions for searching in arXiv and PubMed.

    Args:
        conversation: The original scientific query to expand and the subsequent conversation history.
        model: The OpenAI model to use for expansion.
        temperature: Temperature setting for the model's creativity.

    Returns:
        A dictionary containing:
            - expanded_queries: List of three independent search terms.
            - updated_query: An updated query based on the conversation history.
    """
    # Set up the LLM
    lm = dspy.LM(model, api_key=os.getenv("OPENAI_API_KEY"), temperature=temperature)
    dspy.configure(lm=lm)

    # Create and use the predictor
    expander = dspy.ChainOfThought(QueryExpansionSignature)
    response = expander(chat_history=conversation)
    logger.info(f"expand_query COT: {response}")

    return response


def rank_papers_with_llm(papers: List[Dict[str, str]], query: str, model: str = "gpt-4o-mini") -> List[Dict[str, str]]:
    """
    Use LLM to rank papers based on relevance to the query.

    Args:
        papers: List of paper dictionaries, each containing 'title' and 'abstract'.
        query: The search query.
        model: The OpenAI model to use for ranking.

    Returns:
        List of papers sorted by relevance (most relevant first)
    """
    # Set up the LLM with temperature 0 for consistent scoring
    lm = dspy.LM(model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.)
    dspy.configure(lm=lm)

    # Create the predictor
    relevance_predictor = dspy.Predict(RelevanceSignature)

    ranked_papers = []
    for paper in papers:
        # Evaluate the paper's relevance using DSPy
        result = relevance_predictor(
            paper_title=paper['Title'],
            paper_abstract=paper['Abstract'],
            query=query
        )

        # Add paper and score to ranked list
        ranked_papers.append((paper, result.relevance_score))

    # Sort papers by relevance score in descending order
    ranked_papers.sort(key=lambda x: x[1], reverse=True)

    # Return papers without scores
    return [paper[0] for paper in ranked_papers]


class SearchAgent:
    """
    A search agent that uses a combination of query expansion, arXiv and PubMed searches to find relevant research papers.
    It ranks the results using an LLM to determine relevance to the original query.
    """
    @staticmethod
    def search(query: str, conversation: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a search workflow to retrieve and rank research papers based on a given query.

        Args:
            query (str): The original search query provided by the user.
            conversation (str): Conversation history that lead to the user question.
            max_results (int, optional): The maximum number of results to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing the top-ranked research papers. 
                                  Each dictionary includes metadata such as title, authors, and abstract.
            str: The updated query after query expansion, if applicable.

        Raises:
            Exception: Logs and handles exceptions that occur during query expansion or search processes.
        """
        # Step 1: Expand the original query using the QueryExpansionTool
        try:
            response = expand_query(conversation)
            expanded_queries = response['expanded_queries']
            updated_query = response['updated_query']
            if updated_query:
                query = updated_query

        except Exception as e:
            logging.error(f"Error during query expansion: {e}")
            return []

        logger.info(f"""Expanded queries: {chr(10).join(expanded_queries)}""")

        # setup dspy lm and create the source selection prediction
        lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"), temperature=0.)
        dspy.configure(lm=lm)
        source_selector = dspy.Predict(SourceSelectionSignature)

        all_results = []
        for expanded_query in expanded_queries:
            logger.info(f"\nSearching for papers with expanded query: {expanded_query}")

            source = source_selector(query=expanded_query)['source']
            results = None

            # Step 2: Use arXiv or PubMed or BioRxiv search tools to get results
            if source.lower() == 'arxiv':
                results = ArxivSearch.search(expanded_query, max_results=max_results)
            elif source.lower() == 'pubmed':
                results = PubMedSearch.search(expanded_query, max_results=max_results)
            elif source.lower == 'biorxiv':
                results = BioRxivSearch.search(expanded_query, max_results=max_results)

            # Combine results from both tools
            if results and isinstance(results, list):
                all_results.extend(results)
            time.sleep(.5)

        logger.info(f"\nTotal results: {len(all_results), source}")

        # Step 3: Rank the results using the LLM
        # Use the LLM to rank the papers based on relevance to the original query
        all_results = rank_papers_with_llm(all_results, query)

        # Return top 5 overall results based on LLM relevance scoring
        return all_results[:MAX_PAPERS], updated_query
