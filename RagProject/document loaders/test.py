#document loader
from langchain_community.document_loaders import TextLoader
#text based splitting
from langchain_text_splitters import CharacterTextSplitter
splitter = CharacterTextSplitter(
    separator=" ", 
    chunk_size=10, 
    chunk_overlap=1)
data=TextLoader("document loaders/notes.txt",encoding="utf8") #encoding is used because the this python version is not suppporting the default encoding of the text file. So we have to specify the encoding of the text file to be utf8
docs=data.load()
chunks=splitter.split_documents(docs)
#print(len(docs))
#print(chunks[0].page_content) #you can only ssee one chunk at a time because the text is splitted into multiple chunks and each chunk is loaded as a document. If we want to see all the chunks we can use the for loop to iterate through the chunks and print the content of each chunk.
for i in chunks:
    print(i.page_content)