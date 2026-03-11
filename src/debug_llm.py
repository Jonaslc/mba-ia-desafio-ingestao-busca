from search import retrieve_context, get_llm

query = "qual faturamento da Âmbar Games"
context = retrieve_context(query)
print("PROMPT ENVIADO:\n")
from search import PROMPT_TEMPLATE
prompt = PROMPT_TEMPLATE.format(contexto=context, pergunta=query)
print(prompt[:2000])
print("\n=== LLM RESPONSE ===\n")
llm = get_llm()
print(llm(prompt))