import os
import logging
from typing import List
from typing_extensions import TypedDict
from typing import Annotated

import dspy
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages


from paper_db import paperDB
from search_agent import SearchAgent
from dspy_signatures import AnswerGenerationSignature, QueryRouterSignature, AnswerRefinerSignature, AnswerAssessorSignature


logger = logging.getLogger('rag_agent')


class RAGAgentState(TypedDict):
    """State dictionary for the RAG Agent, holding conversation context and relevant information."""
    query: str  # User's query in the current conversation
    retrieved_context: str  # The context retrieved from documents relevant to the query
    generated_answer: str  # The answer generated based on the retrieved context
    feedback: str  # Feedback for refining the current answer (e.g., for hallucinations, inaccuracies)
    messages: Annotated[List, add_messages]  # History of conversation messages with added context


class RAGAgent:
    """Agent for multi-turn scientific conversations, utilizing retrieval-augmented generation (RAG) for answering queries."""

    def __init__(self):
        """Initialize the RAG Agent by setting up the paper database and workflow graph."""
        # Initialize the (research) paper database
        self.db = paperDB()

        # Create and initialize the workflow graph
        self.graph = self.create_graph()

    def create_graph(self):
        """
        Creates and configures the LangGraph workflow for managing the RAG agent's tasks.

        This graph defines the steps for retrieving relevant documents, generating answers,
        assessing feedback, and refining the answer if necessary.
        """
        def search(state: RAGAgentState):
            """
            Search for relevant scientific documents using the search agent.

            Args:
                state (RAGAgentState): The current state of the RAG agent, containing the query and conversation history.
            """
            logger.info("\n\n***SEARCH***")
            full_conversation = "\n".join([msg.content for msg in state['messages']])
            paper_list = SearchAgent.search(state['query'], full_conversation)

            # Process URLs to extract images and chunked text and add embeddings to vectorstore
            self.db.process_urls_parallel([paper['Link'] for paper in paper_list if paper['Link']])
            self.db.abstracts.extend([paper['Abstract'] for paper in paper_list if paper['Abstract']])

            logger.info(f"Processed {len(paper_list)} documents from search agent")
            logger.info("***END_SEARCH***\n\n")

        def route_query(state: RAGAgentState):
            """
            Routes the query to either the vectorstore or the search agent based on the current paperDB contents.

            Args:
                state (RAGAgentState): The current state of the RAG agent

            Returns:
                str: The result of routing the query, determining the next step.
            """
            logger.info("\n\n***ROUTE_QUERY***")
            abstract_text = "\n******\n".join(self.db.abstracts)
            print(f"Abstracts: {abstract_text}")

            lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"), temperature=0.)
            dspy.configure(lm=lm)

            query_router = dspy.ChainOfThought(QueryRouterSignature)
            output = query_router(query=state['query'], abstracts=abstract_text)
            logger.info(f"Query routing result: {output}")
            logger.info("***END_ROUTE_QUERY***\n\n")
            return output['output']

        def generate_feedback(state: RAGAgentState):
            """
            Generates feedback on the answer based on the context and answer's quality.

            Args:
                state (RAGAgentState): The current state of the RAG agent, including the query, retrieved context, and generated answer.

            Returns:
                dict: Feedback about the answer (e.g., inaccuracies or hallucinations).
            """
            logger.info("\n\n***GENERATE_FEEDBACK***")
            lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"), temperature=0.)
            dspy.configure(lm=lm)
            answer_assessor = dspy.Predict(AnswerAssessorSignature)
            assessment = answer_assessor(query=state['query'], context=state['retrieved_context'], generated_answer=state['generated_answer'])

            feedback = ""
            if assessment['is_hallucination']:
                feedback += "The generated answer contains hallucinations. "
            if assessment['is_inaccurate']:
                feedback += "The generated answer contains inaccuracies. "

            if not feedback:
                feedback = "The generated answer is accurate and does not contain hallucinations."

            logger.info(f"Feedback: {feedback}")
            logger.info("END_GENERATE_FEEDBACK")

            return {'feedback': feedback}

        def assess_feedback(state: RAGAgentState):
            """
            Assesses the generated feedback and decides whether the answer should be refined or if the process is complete.

            Args:
                state (RAGAgentState): The current state of the RAG agent, containing feedback information.

            Returns:
                str: "refine" to trigger answer refinement or "end" to end the process.
            """
            logger.info("\n\n***ASSESS_FEEDBACK***")
            if "The generated answer contains hallucinations" in state['feedback'] or \
                "The generated answer contains inaccuracies" in state['feedback']:
                logger.info("Refine the answer")
                logger.info("***END_ASSESS_FEEDBACK***\n\n")
                return "refine"
            else:
                logger.info("END")
                logger.info("***END_ASSESS_FEEDBACK***\n\n")
                return "end"

        def refine_answer(state: RAGAgentState):
            """
            Refines the generated answer using the feedback to improve its quality.

            Args:
                state (RAGAgentState): The current state of the RAG agent, including the query, context, generated answer, and feedback.

            Returns:
                dict: The refined answer.
            """
            logger.info("\n\n***REFINE_ANSWER***")
            lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)
            dspy.configure(lm=lm)
            answer_refiner = dspy.Predict(AnswerRefinerSignature)
            answer = answer_refiner(query=state['query'],
                                    context=state['retrieved_context'],
                                    generated_answer=state['generated_answer'],
                                    feedback=state['feedback'])['refined_answer']
            logger.info(f"Refined answer: {answer}")
            logger.info("***END_REFINE_ANSWER***\n\n")

            return {'generated_answer': answer}

        def retrieve_documents(state: RAGAgentState):
            """
            Retrieve relevant documents from the database based on the user's query.

            Args:
                state (RAGAgentState): The current state of the RAG agent, including the user's query.

            Returns:
                dict: The context of retrieved documents.
            """
            logger.info("\n\n***RETRIEVE_DOCUMENTS***")
            retrieved_docs = self.db.retriever.invoke(state['query'])

            context = "\n******\n".join([f"""SOURCE_ID: {doc.metadata['doc_id']},{doc.metadata['page'] if 'page' in doc.metadata else ""}\nLINK: {doc.metadata['source']}\nCONTENT: {doc.page_content}""" for doc in retrieved_docs])
            logger.info("***END_RETRIEVE_DOCUMENTS******\n\n")

            return {'retrieved_context': context}

        def generate_answer(state: RAGAgentState):
            """
            Generates an answer based on the retrieved context and previous conversation history.

            Args:
                state (RAGAgentState): The current state of the RAG agent, including the user's query and conversation history.

            Returns:
                dict: The generated answer and updated message history.
            """
            logger.info("\n\n***GENERATE_ANSWER***")
            full_conversation = "\n".join([msg.content for msg in state['messages']])

            # Use DSPy to generate an answer based on the updated context
            lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
            dspy.configure(lm=lm)

            answer_generator = dspy.Predict(AnswerGenerationSignature)
            answer = answer_generator(
                query=state['query'],
                context=full_conversation + "\n******\n" + state['retrieved_context']
            )['answer']

            logger.info("***END_GENERATE_ANSWER***\n\n")

            return {'generated_answer': answer, "messages": [{"role": "assistant", "content": answer}]}

        print("Creating RAG Agent graph...")
        # Define graph workflow
        workflow = StateGraph(RAGAgentState)
        workflow.add_node("search", search)
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("generate_feedback", generate_feedback)
        workflow.add_node("refine_answer", refine_answer)

        workflow.add_conditional_edges(
            START,
            route_query,
            {
                "search": "search",
                "vectorstore": "retrieve",
            },
        )
        # search to retrieve
        workflow.add_edge("search", "retrieve")
        # retrieve to generate_answer
        workflow.add_edge("retrieve", "generate_answer")
        # generate_answer to generate_feedback
        workflow.add_edge("generate_answer", "generate_feedback")

        # conditional edges for assess_answer to refine_answer or end
        workflow.add_conditional_edges(
            "generate_feedback",
            assess_feedback,
            {
                "refine": "refine_answer",
                "end": END,
            },
        )
        # refine_answer to assess_answer
        workflow.add_edge("refine_answer", "generate_feedback")

        print("RAG Agent graph created successfully.")

        return workflow.compile()

    def invoke(self, state: RAGAgentState) -> RAGAgentState:
        """
        Invoke the RAG agent for multi-turn conversations.

        Args:
            state (RAGAgentState): The current state of the agent, which is updated after each turn.

        Returns:
            RAGAgentState: The updated state after processing the current query.
        """
        # Initialize or maintain the state across turns
        return self.graph.invoke(state)
