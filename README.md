# UNIME-AI-CHATBOT

Quando l'AI entra in Ateneo, maggiori informazioni [qui](https://medium.com/@gpirrotta/di-intelligenze-artificiali-atenei-e-servizi-digitali-0e9b679dc65a)

### Installazione

**Installare** le librerie con il comando: 

```
pip install -r requirements.txt
```

**Rinominare** il file `.env.sample` in `.env` e inserire la **OPENAI_API_KEY**


**Effetuare** lo scraping (solo la prima volta)
```
python scraper.py
```

**Avviare** l'applicazione con il comando
```
streamlit run start.py
```

**Aprire** il browser all'URL 
```
http://localhost:8501/
```
