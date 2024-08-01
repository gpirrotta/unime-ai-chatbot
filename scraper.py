from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd


PREFIX = 'https://www.unime.it'
dataset = []

## SCRAPE PRIMA TIPOLOGIA PAGINA FAQ (DOMANDA/RISPOSTA)
url = 'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/faq-immatricolazioni'



page = requests.get(url, verify=False)
print(url)
soup = BeautifulSoup(page.text)
title = soup.find('h2', {'class' : 'title-page__title'}).text.strip()
for item in soup.find_all('div', {'class' : 'accordion-item'}):
    question =  item.find('button', {'class' : 'accordion-button'}).text.strip()
    answer = item.find('div', {'class':'field--name-field-testo-paragrafo'}).text.strip()
    body =  question + '\n' + answer
    attachments = item.find('div', {'class':'paragraph__allegati'})
    if attachments:
        body = body + '\nLink relativi:\n' 
        for a in attachments.find_all('a'):
            body = body + a.text + f" ({PREFIX}{a.get('href')})\n"
    dataset.append({'url':url, 'title': title, 'body':body})

time.sleep(3)    

## SCRAPE SECONDA TIPOLOGIA PAGINA FAQ (PAGINA INTERA)

page_links = ['https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/corsi-liberi-aa-202324-modalita-e-termini',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/contemporanea-iscrizione-due-corsi-di-istruzione-superiore',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/corsi-numero-chiuso',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/corsi-singoli',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/i-documenti-necessari',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/i-vari-tipi-di-corsi-di-laurea',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/il-sistema-pago-par',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/immatricolazioni-personale-pa',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/info-point',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/live-chat-studenti',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/ricongiunzione-eo-ripresa-carriera-ex-studenti',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/sedi-didattiche-distaccate',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/tasse-ed-esenzioni',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/tempo-parziale',
              'https://www.unime.it/didattica/immatricolazioni-e-iscrizioni/trasferimenti'
              ]

for url in page_links:
    page = requests.get(url)

    soup = BeautifulSoup(page.text)
    title = soup.find('h2', {'class' : 'title-page__title'}).text.strip()
    print(f"Parsing page: {title}")
    body = soup.find('div', {'class' : 'block-field-blocknodepagefield-contenuto'}).get_text(separator = '\n', strip = True)
    for attachments in soup.find_all('div', {'class':'paragraph__allegati'}):
        if attachments:
            body = body + '\nLink relativi:\n' 
            for a in attachments.find_all('a'):
                body = body + a.text + f" ({PREFIX}{a.get('href')})\n"
    dataset.append({'url':url, 'title': title, 'body':body})
    time.sleep(3)
    
    
df = pd.DataFrame(dataset)
df['body'] = df['body'].str.replace('Document', ' ')
df['body'] = df['body'].str.replace("\n",' ').str.replace('  ',' ').str.strip()

df.to_csv('./data/dataset.csv', index=False)


chunks = DataFrameLoader(
        df, page_content_column="body"
    ).load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, length_function=len
        )
    )
    
for doc in chunks:
    title = doc.metadata["title"]
    content = doc.page_content
    url = doc.metadata["url"]
    final_content = f"TITLE: {title}\BODY: {content}\nURL: {url}"
    doc.page_content = final_content
    
  
load_dotenv()
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002') 

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("./db")