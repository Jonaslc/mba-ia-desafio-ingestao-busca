from search import get_vector_store, retrieve_context

query = "qual faturamento da Âmbar Games"
store = get_vector_store()
try:
    results = store.similarity_search_with_score(query, k=10)
    print("RESULTS (content, distance):")
    for item in results:
        print(item)
except Exception as e:
    print("Erro na busca:", e)

print("\nCONTEXTO RECUPERADO:\n")
print(retrieve_context(query))