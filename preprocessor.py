import json

# Open and read the JSON file
data = json.load(open('questions.json', 'r'))

question = ""
x = 0
while x < int(len(data)/2):
    question = data[x*2]
    for answer in data[x*2+1]:
        for i in range(128):
            print(f"")
        print(f"question: \"{question}\"")
        print(f"answer: \"{answer}\"")
    x += 1
