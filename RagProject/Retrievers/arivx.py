from langchain_community.retrievers import ArxivRetriever

#this thing is used beacuse the arxiv is not running in modern python version
from typing import List
import arxiv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class ArxivRetriever(BaseRetriever):
    load_max_docs: int = Field(default=2)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=self.load_max_docs,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        documents = []
        for result in client.results(search):
            doc = Document(
                page_content=result.summary,
                metadata={
                    "Title": result.title,
                    "Authors": ", ".join(a.name for a in result.authors),
                    "Published": str(result.published),
                    "Entry_ID": result.entry_id,
                    "PDF_URL": result.pdf_url,
                },
            )
            documents.append(doc)

        return documents




# create the retriever
retriever = ArxivRetriever(
    load_max_docs=2,      # number of papers to retrieve
    load_all_available_meta=True
)

# query arxiv
docs = retriever.invoke("large language models")

# print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Summary:", doc.page_content[:500])  # print first 500 characters