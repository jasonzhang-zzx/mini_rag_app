from extract_from_pdf import extract_text_from_pdf, split_text
from llm_api import get_completion, build_prompt, prompt_template
from vectorDB import MyVectorDBConnector
from embeddings import get_embedding

class RAG_bot:
    def __init__(self, vector_db, llm_api, n_results=30):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        # 1. search the vector db to get the top n results
        search_results = self.vector_db.search(user_query, top_k=self.n_results)
        # 2. format the search results into a prompt for the llm_api
        prompt = build_prompt(
            prompt_template=prompt_template,
            context=search_results["documents"][0],
            question=user_query,
        )
        # 3. call the llm_api to get the response
        response = self.llm_api(prompt)
        return response

if __name__ == "__main__":
    print("You are chatting with a RAG chatbot! Input 'quit' or 'exit' to end the chat. \n")
    vec_db = MyVectorDBConnector("defualt", embedding_function=get_embedding)
    bot = RAG_bot(vector_db=vec_db, llm_api=get_completion)

    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Bye! ")
            break
        response = bot.chat(user_query)
        print(f"Bot: {response}\n")