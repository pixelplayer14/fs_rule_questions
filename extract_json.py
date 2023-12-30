from pypdf import PdfReader
import re
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def pdfToTxt():
    reader = PdfReader("rules.pdf")
    number_of_pages = len(reader.pages)
    pages = reader.pages[79:98]
    with open("EV.txt",'w',encoding='utf-8') as f:
        for page in pages:
            f.write(page.extract_text())

def txtToJson():
    '''
    Converts the txt document generated in pdfToTxt() to a json file where each (sub)section is of the following format:
    {id:"section nr", content:"rule",subsections:} 
    The regex assumes that the rule definition starts with a subsection identifier after a newline. This is not only the case for rule definitions. These cases are altered manually in EV.txt for now.
    '''
    allSections = {"id":"root","content":None,"subsections":[]}

    with open("EV.txt","r",encoding='utf-8') as f:
        text = f.read()
        all_rules = re.findall(r'\nEV(?:.|\n)*?(?=\nEV)',text)
        for rule in all_rules:
            id = re.match(r'\nEV(?:\s|\.|\d)*\d',rule).group(0)
            id = id.replace("\n","").replace(" ","")
            sections = id.replace("EV","").split(".")
            rule = rule.replace("\n","").replace(id,"")
            iSections = list(map(int,sections))
            currentLevel = allSections
            doNotAppend = False
            for i in range(len(iSections)):
                if len(currentLevel["subsections"])<iSections[i]:
                    currentLevel["subsections"].append({"id":id,"content":rule,"subsections":[]})
                    break
                
                #print(currentLevel[-1])
                currentLevel = currentLevel["subsections"][-1]
            print(id)
    with open("EV.json","w") as jf:
        json.dump(allSections,jf,indent=4,sort_keys=True)

def stringToQuestion_allenai():

    from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

    model_name = "allenai/t5-small-squad2-question-generation"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def run_model(input_string, **generator_args):
        input_ids = tokenizer.encode(input_string, return_tensors="pt")
        res = model.generate(input_ids, **generator_args)
        output = tokenizer.batch_decode(res, skip_special_tokens=True)
        print(output)
        return output


    run_model("All TS wiring that runs outside of TS enclosures must:Be enclosed in separate orange non-conductive conduit or use an orange shieldedcable. The conduit must be securely anchored to the vehicle, but not to the wire, atleast at each end.Be securely anchored at least at each end so that it can withstand a force of 200 Nwithout straining the cable end crimp.Bodywork is not sufficient to meet this enclosure requirement.")


tokenizer = None
model = None
def initPipeline():
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
    model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")

def stringToQuestion_potsawee(context):
    context = context.replace('\n', ' ')

    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
    return question_answer

def getRuleLeafs(rules):
    leafs=[]
    if len(rules["subsections"])==0:
       return [(rules["content"],rules["id"])]
    else:
        for subsection in rules["subsections"]:
            leafs = leafs + getRuleLeafs(subsection)
        return leafs

initPipeline()

with open("EV.json") as ruleFile:
    rulesJson = json.load(ruleFile)

rules =getRuleLeafs(rulesJson)
for rule in rules:
    qa = stringToQuestion_potsawee(rule[0])
    q = qa.split("<sep>")[0]
    print(q+f" ({rule[1]})")
    with open("questions.txt","+a") as qf:
        qf.write(q+f"({rule[1]})"+"\n")
    
