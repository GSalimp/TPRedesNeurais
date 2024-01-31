import os
from langchain.embeddings import LlamaCppEmbeddings
import json
import time 

def removeDone(ToDo, Done):
    ToDo = [file.split(".")[0] for file in ToDo]
    Done = [file.split(".")[0] for file in Done]
    returnList = list(set(ToDo) - set(Done))
    returnList = [f"{file}.txt" for file in returnList]
    return returnList

def makeIt(model, allFiles,  allOutputs, allInputs, allSize, allTitles, labels):
    
    countList      = [0, 0, 0, 0]
    countErrorList = [0, 0, 0, 0]
    
    for i in range(allSize):
        try: 
            fileString = allFiles[i%4][countList[i%4]]
            file = open(f"{allInputs[i%4]}/{fileString}")

            print("----------------------------------------------------------------------------------------------------------------")
            print(f"Start File: {fileString} | From: {allTitles[i%4]} | Count: {countList[i%4]}/{len(allFiles[i%4])} | Erros: {countErrorList[i%4]}")
            
            text = file.read()
            jsonSendoFeito = {}
            
            tempo_inicio = time.time()
            sentence_embedding = model.embed_query(text)
            tempo_fim = time.time()
            
            jsonSendoFeito["sentence_embedding"] = sentence_embedding
            jsonSendoFeito["text"] = text
            jsonSendoFeito["label"] = labels[i%4]
            jsonSendoFeito["file"] = fileString
            
            outputFile = open(f"{allOutputs[i%4]}/{fileString}", "w")
            json.dump(jsonSendoFeito, outputFile, indent=4)
            
            print(f"\nFineshed File: {fileString} | From: {allTitles[i%4]} | Time: {tempo_fim - tempo_inicio}")
            
            countList[i%4] = countList[i%4] + 1
        
        except Exception as e:
            print(f"Error File: {fileString} | From: {allTitles[i%4]} | Count: {countList[i%4]}/{len(allFiles[i%4])} | Erros: {countErrorList[i%4]}")
            countErrorList[i%4] += 1
        
inputPath  = "dataset/inputs"
outputPath = "dataset/outputs"

ToDoTestPosFiles  = os.listdir(f"{inputPath}/test/pos")
ToDoTestNegFiles  = os.listdir(f"{inputPath}/test/neg")
ToDoTrainPosFiles = os.listdir(f"{inputPath}/train/pos")
ToDoTrainNegFiles = os.listdir(f"{inputPath}/train/neg")

DoneTestPosFiles  = os.listdir(f"{outputPath}/test/pos")
DoneTestNegFiles  = os.listdir(f"{outputPath}/test/neg")
DoneTrainPosFiles = os.listdir(f"{outputPath}/train/pos")
DoneTrainNegFiles = os.listdir(f"{outputPath}/train/neg")


print(f"TestPos: {len(ToDoTestPosFiles)} | TestNeg: {len(ToDoTestNegFiles)} | TrainPos: {len(ToDoTrainPosFiles)} | TrainNeg: {len(ToDoTrainNegFiles)}")
ToDoTestPosFiles  = removeDone(ToDoTestPosFiles,  DoneTestPosFiles)
ToDoTestNegFiles  = removeDone(ToDoTestNegFiles,  DoneTestNegFiles)
ToDoTrainPosFiles = removeDone(ToDoTrainPosFiles, DoneTrainPosFiles)
ToDoTrainNegFiles = removeDone(ToDoTrainNegFiles, DoneTrainNegFiles)
print(f"TestPos: {len(ToDoTestPosFiles)} | TestNeg: {len(ToDoTestNegFiles)} | TrainPos: {len(ToDoTrainPosFiles)} | TrainNeg: {len(ToDoTrainNegFiles)}")

allFiles = []
allFiles.append(ToDoTestPosFiles)
allFiles.append(ToDoTestNegFiles)
allFiles.append(ToDoTrainPosFiles)
allFiles.append(ToDoTrainNegFiles)

allOutputs = [f"{outputPath}/test/pos", f"{outputPath}/test/neg", f"{outputPath}/train/pos", f"{outputPath}/train/neg"]
allInputs = [f"{inputPath}/test/pos", f"{inputPath}/test/neg", f"{inputPath}/train/pos", f"{inputPath}/train/neg"]

allSize = len(ToDoTestPosFiles) + len(ToDoTestNegFiles) + len(ToDoTrainPosFiles) + len(ToDoTrainNegFiles)
alltitles = ["test/pos", "test/neg", "train/pos", "train/neg"]    

labels = [1, 0, 1, 0]

print("------------------------------------------------------------")
model = LlamaCppEmbeddings(model_path="model/yarn-llama-2-7b-128k.Q4_K_M.gguf", use_mlock=True, n_ctx=2048)
print("------------------------------------------------------------")

makeIt(model, allFiles,  allOutputs, allInputs, allSize, alltitles, labels)
