import json

# Open and read the JSON files
data1 = json.load(open('QuestionsNewFormat.json', 'r'))
data2 = json.load(open('questions.json', 'r'))

question = ""
x = 0
while x < len(data1):
    if type(data1[x]) == str:
        question = data1[x]
    else:
        for i in range(128):
            print(f"")
        print(f"question: \"{question}\"")
        print(f"answer: \"{data1[x][2]}\"")
    x += 1

x = 0
while x < int(len(data2)/4):
    for i in range(128):
        print(f"")
    print(f"question: \"{data2[x*4+1]}\"")
    print(f"answer: \"{data2[x*4+3]}\"")
    x += 1
