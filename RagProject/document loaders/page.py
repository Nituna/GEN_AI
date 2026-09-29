#This magic mock is used beacuse in this python verion webloader is not working python 3.11 or 3.12 is a great choice for this
import sys
from unittest.mock import MagicMock

# Block the heavy dependencies from loading and hanging
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

from langchain_community.document_loaders import WebBaseLoader
url = "https://www.apple.com/in/macbook-pro/"
data=WebBaseLoader(url)
docs=data.load()
print(len(docs))  #Len is 1 becuase it is one website and it is loaded as one document. If we want to load multiple documents we can use the list of urls in the WebBaseLoader
print(docs[0].page_content)  #This will print the content of the website loaded as a document