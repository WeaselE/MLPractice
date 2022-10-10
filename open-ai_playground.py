import os
import openai

openai.api_key = os.environ.get('OPEN_AI_KEY')

prompt = input()
while True:
    response = openai.Completion.create(
        model="text-curie-001",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response = response['choices'][0]['text']
    print(response)
    prompt = input('\n')
    prompt = response + '\n' + prompt
