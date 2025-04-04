from typing import List, Literal
import dspy


class QueryExpansionSignature(dspy.Signature):
    """Signature for expanding a scientific query."""
    chat_history: str = dspy.InputField(desc="The conversation history with the user. The last message should be the user's query")
    expanded_queries: List[str] = dspy.OutputField(desc="Ten independent queries. Should be keyword style queries, not full sentences, for searching on pubmed and arxiv, etc. The topics should follow the current conversation provided in chat_history")
    updated_query: str = dspy.OutputField(desc="The updated query based on the conversation history. This should be a full sentence, not a keyword style query, to be used for similarity search in the vectorstore, if no update is needed, return empty string")

class SourceSelectionSignature(dspy.Signature):
    """Signature for selecting the source to search for papers."""
    query: str = dspy.InputField(desc="The expanded scientific query to determine the source for")
    source: Literal['arxiv', 'pubmed', 'biorxiv'] = dspy.OutputField(desc="The relevant website to search for papers given the domain of the query")


class RelevanceSignature(dspy.Signature):
    """Signature for ranking papers based on relevance to a query."""
    paper_title: str = dspy.InputField(desc="The title of the research paper")
    paper_abstract: str = dspy.InputField(desc="The abstract of the research paper")
    query: str = dspy.InputField(desc="The search query")
    relevance_score: int = dspy.OutputField(desc="Numerical assessment of the paper's relevance to the user's search query on a scale of 0-10, \
                                                where 10 indicates perfect alignment with the query (highly relevant) and 0 indicates no topical connection (irrelevant)")


class QueryRouterSignature(dspy.Signature):
    """Signature for routing a query to either the vectorstore or the search agent."""
    query: str = dspy.InputField(desc="The user's query")
    abstracts: str = dspy.InputField(desc="Abstracts of all the research papers currently in the custom database, can be empty")
    output: Literal['vectorstore', 'search'] = dspy.OutputField(desc="Do the provided abstracts contain information relevant to the user's query? if yes, output vectorstore, else output search")


class AnswerGenerationSignature(dspy.Signature):
    """Signature for generating an answer based on the retrieved context."""
    query: str = dspy.InputField(desc="The user's query")
    context: str = dspy.InputField(desc="The context retrieved from the vectorstore or search agent")
    answer: str = dspy.OutputField(desc="The generated answer in markdown format based on the retrieved context and feedback. \
                                        Answer should have bullet points for key information with appropriate citation for every bullet, \
                                        strictly use provided SOURCE_ID of each CONTENT as the citation label with the LINK as the hyperlink, eg: [source_id] \
                                        It should look visually good and be easy to read.")


class AnswerRefinerSignature(dspy.Signature):
    """Signature for refining the generated answer."""
    query: str = dspy.InputField(desc="The user's query")
    context: str = dspy.InputField(desc="The context retrieved from the vectorstore or search agent")
    generated_answer: str = dspy.InputField(desc="The generated answer from the previous step")
    feedback: str = dspy.InputField(desc="Feedback for refining the current answer")
    refined_answer: str = dspy.OutputField(desc="The refined answer based on the feedback given the context and previous answer. You should only output markdown format. Remeber to give a bullet point style answer with appropriate citation for every bullet - strictly use provided SOURCE_ID for each CONTENT as the citation label, the LINK as the hyperlink")


class AnswerAssessorSignature(dspy.Signature):
    """Signature for assessing the generated answer.
    Check for hallucinations and inaccuracies"""
    query: str = dspy.InputField(desc="The user's query")
    context: str = dspy.InputField(desc="The context retrieved from the vectorstore or search agent")
    generated_answer: str = dspy.InputField(desc="The generated answer from the previous step")
    is_hallucination: str = dspy.OutputField(desc="Whether the generated answer contains hallucinations or not, if yes, output the hallucinated content")
    is_inaccurate: str = dspy.OutputField(desc="Whether the generated answer contains inaccuracies or not, if yes, output the inaccurate content")
