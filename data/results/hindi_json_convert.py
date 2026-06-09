import json
input_file = "data/results/perspective_data/xJCWF0z6C7k.json"
output_file = "data/results/perspective_data/xJCWF0z6C7k_hindi.json"

with open(input_file,"r",encoding="utf-8") as f:
    data = json.load(f)
with open(output_file,"w",encoding="utf-8") as f:
    json.dump(data,f,ensure_ascii=False,indent=2)

print("Check hindi text json file")