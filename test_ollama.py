
import ollama

while True:
    question = input("Ask your question: ")

    if question.lower() == "exit":
        break

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    print(response.message.content)
