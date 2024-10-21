# MLR - MRV.LLM.RAG  
**Chatbot Inteligente para Help Desk da MRV**

Este projeto implementa o **MLR (MRV.LLM.RAG)**, um chatbot desenvolvido para o sistema de produção da MRV. Ele utiliza a **API do Google Gemini** para interpretar dúvidas e uma **base de conhecimento estruturada em SQL Server** para fornecer respostas precisas. Com essa abordagem, o chatbot garante que nenhuma resposta seja inventada, respondendo apenas com base nas informações previamente cadastradas. Isso otimiza a eficiência do help desk, liberando a equipe para focar em tarefas mais críticas. 

---

## 📋 Funcionalidades
- **Interpretação de linguagem natural (NLP)** com Google Gemini.
- **Busca eficiente** por respostas em uma base de conhecimento.
- **Controle total das respostas**, evitando respostas inventadas.
- **Fallback seguro** para casos onde não há resposta disponível.
- **Escalabilidade**: Suporte a novas entradas na base de dados de forma simples.

---

## 🛠️ Tecnologias Utilizadas
- **Python** – Linguagem principal para lógica e integração.
- **Google Gemini API** – Para interpretar as perguntas.
- **SQL Server** – Banco de dados para perguntas e respostas frequentes.
- **FAISS / Pinecone** (opcional) – Busca rápida por similaridade.
- **Google Generative AI SDK** – Conexão com a API do Gemini.
