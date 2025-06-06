import json

# Open and read the JSON file
data = json.load(open('questions.json', 'r'))

for x in range(int(len(data)/4)):
    print(f"")
    print(f"question: \"{data[x*4+1]}\"")
    print(f"answer: \"{data[x*4+3]}\"")
