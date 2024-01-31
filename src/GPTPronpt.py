import os
import json
from openai import OpenAI, RateLimitError
import time

client = OpenAI(
    api_key="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",
)

def removeDone(ToDo, Done):
    ToDo = [file.split(".")[0] for file in ToDo]
    Done = [file.split(".")[0] for file in Done]
    returnList = list(set(ToDo) - set(Done))
    returnList = [f"{file}.json" for file in returnList]
    return returnList

def makeIt(files, inputPath, outputPath):
    errorCount = 0
    rateLimitCount = 0
    done = 0
    
    for i, file in enumerate(files):
        while True:
            try: 
                print(f"Start File: {file} | Count: {done}/{len(files)} | From: {inputPath} | Erros: {errorCount} | RateLimitErros: {rateLimitCount}")
                
                jsonCarregado = json.load(open(f"{inputPath}/{file}"))
                jsonGPT = {}
                
                retorno = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Based on this text, return the sentiment: 0 for negative, 1 for postive \n Use only one token for response \n {jsonCarregado['text']}",
                        }
                    ],
                    model="gpt-3.5-turbo",
                )
                
                jsonGPT["labelGPT"] = int(retorno.choices[0].message.content)
                jsonGPT["label"] = jsonCarregado["text"]
                jsonGPT["text"] = jsonCarregado["text"]
                jsonGPT["file"] = file
                
                json.dump(jsonGPT, open(f"{outputPath}/{file}", "w"), indent=4)
                done = done + 1
                break
                
            except RateLimitError as e:
                rateLimitCount = rateLimitCount + 1
                print(f"RateLimitError File: {file} | Count: {done}/{len(files)} | Errors: {errorCount} | RateLimitErros: {rateLimitCount}")
                time.sleep(20)
            except Exception as e:
                errorCount = errorCount + 1
                print(f"Error File: {file} | Count: {done}/{len(files)} | Errors: {errorCount} | RateLimitErros: {rateLimitCount} | Error: {e}")
                    
            
        
inputPath = "dataset/outputs"
outputPath = "dataset/outputsGPT"

DoneTestPosFiles  = os.listdir(f"{outputPath}/test/pos")
DoneTestNegFiles  = os.listdir(f"{outputPath}/test/neg")
DoneTrainPosFiles = os.listdir(f"{outputPath}/train/pos")
DoneTrainNegFiles = os.listdir(f"{outputPath}/train/neg")

ToDoneTestPosFiles  = os.listdir(f"{inputPath}/test/pos")
ToDoneTestNegFiles  = os.listdir(f"{inputPath}/test/neg")
ToDoneTrainPosFiles = os.listdir(f"{inputPath}/train/pos")
ToDoneTrainNegFiles = os.listdir(f"{inputPath}/train/neg")

ToDoneTestPosFiles  = removeDone(ToDoneTestPosFiles,  DoneTestPosFiles)
ToDoneTestNegFiles  = removeDone(ToDoneTestNegFiles,  DoneTestNegFiles)
ToDoneTrainPosFiles = removeDone(ToDoneTrainPosFiles, DoneTrainPosFiles)
ToDoneTrainNegFiles = removeDone(ToDoneTrainNegFiles, DoneTrainNegFiles)

makeIt(ToDoneTestPosFiles,  f"{inputPath}/test/pos", f"{outputPath}/test/pos")
makeIt(ToDoneTestNegFiles,  f"{inputPath}/test/neg", f"{outputPath}/test/neg")
makeIt(ToDoneTrainPosFiles, f"{inputPath}/train/pos", f"{outputPath}/train/pos")
makeIt(ToDoneTrainNegFiles, f"{inputPath}/train/neg", f"{outputPath}/train/neg")
