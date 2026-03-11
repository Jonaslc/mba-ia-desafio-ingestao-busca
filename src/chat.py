from search import answer_question


def main():
    print("Faça sua pergunta:")
    try:
        while True:
            question = input().strip()
            if not question:
                continue
            if question.lower() in ("sair", "exit", "quit"):
                break
            print(f"PERGUNTA: {question}")
            try:
                response = answer_question(question)
            except Exception:
                response = "Não foi possível processar a pergunta."
            print(f"RESPOSTA: {response}\n")
    except KeyboardInterrupt:
        print("\nEncerrando chat.")


if __name__ == "__main__":
    main()