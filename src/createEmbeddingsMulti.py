import os
from langchain.embeddings import LlamaCppEmbeddings
import json
import threading
import time 
import sys

def removeDone(ToDo, Done):
    ToDo = [file.split(".")[0] for file in ToDo]
    Done = [file.split(".")[0] for file in Done]
    returnList = list(set(ToDo) - set(Done))
    returnList = [f"{file}.txt" for file in returnList]
    return returnList

def divideAndRun(numberOfThereds):
                
    inputPath  = "./dataset/input"
    outputPath = "./dataset/outputs"

    ToDoTestPosFiles  = os.listdir(f"{inputPath}/test/pos")
    ToDoTestNegFiles  = os.listdir(f"{inputPath}/test/neg")
    ToDoTrainPosFiles = os.listdir(f"{inputPath}/train/pos")
    ToDoTrainNegFiles = os.listdir(f"{inputPath}/train/neg")

    DoneTestPosFiles  = os.listdir(f"{outputPath}/test/pos")
    DoneTestNegFiles  = os.listdir(f"{outputPath}/test/neg")
    DoneTrainPosFiles = os.listdir(f"{outputPath}/train/pos")
    DoneTrainNegFiles = os.listdir(f"{outputPath}/train/neg")

    ToDoTestPosFiles  = removeDone(ToDoTestPosFiles,  DoneTestPosFiles)
    ToDoTestNegFiles  = removeDone(ToDoTestNegFiles,  DoneTestNegFiles)
    ToDoTrainPosFiles = removeDone(ToDoTrainPosFiles, DoneTrainPosFiles)
    ToDoTrainNegFiles = removeDone(ToDoTrainNegFiles, DoneTrainNegFiles)

    allFiles = [ToDoTestPosFiles, ToDoTestNegFiles, ToDoTrainPosFiles, ToDoTrainNegFiles]
    allOutputs = [f"{outputPath}/test/pos", f"{outputPath}/test/neg", f"{outputPath}/train/pos", f"{outputPath}/train/neg"]
    allInpus = [f"{inputPath}/test/pos", f"{inputPath}/test/neg", f"{inputPath}/train/pos", f"{inputPath}/train/neg"]
    alltitles = ["test/pos", "test/neg", "train/pos", "train/neg"]
    labels = [1, 0, 1, 0]    

    theredCount = 0
    theredList = []
    for k, files in enumerate(allFiles):
        
        # if i != 1: continue 
            
        for i in range(numberOfThereds):
            original_stdout = sys.stdout
            sys.stdout = None
            model = LlamaCppEmbeddings(model_path="model/yarn-llama-2-7b-128k.Q4_K_M.gguf", use_mlock=True, n_ctx=2048)
            sys.stdout = original_stdout

            slide = int(len(files)/numberOfThereds)
            sliceStart = i*slide
            sliceEnd = (i+1)*slide-1 if i != numberOfThereds-1 else len(files)

            thered = threading.Thread(target=makeIt, args=(model, files[sliceStart:sliceEnd], allOutputs[k], allInpus[k], alltitles[k], theredCount, labels[k]))
            theredList.append(thered)
            theredCount += 1

    for i in range(len(theredList)):
        theredList[i].start()

    for i in range(len(theredList)):
        theredList[i].join()
    
        
def makeIt(model, files,  output, input, title, theredCount, label):
    
    countList = 0
    countErros = 0
    
    for fileString in files:
        try:
            file = open(f"{input}/{fileString}")

            print(f"\n-----Start File: {fileString} | From: {title} | Count: {countList}/{len(files)} | Thread: {theredCount} | Erros: {countErros}-----\n")
            
            text = file.read()
            jsonToDo = {}
            
            tempo_inicio = time.time()
            sentence_embedding = model.embed_query(text)
            tempo_fim = time.time()
            
            jsonToDo["text"] = text
            jsonToDo["sentence_embedding"] = sentence_embedding
            jsonToDo["label"] = label
            jsonToDo["file"] = f"{title}/{fileString}"
            outputFile = open(f"{output}/{fileString}", "w")
            json.dump(jsonToDo, outputFile, indent=4)
            
            print(f"\n-----Fineshed File: {fileString} | From: {title} | Time: {tempo_fim - tempo_inicio} | Thread: {theredCount}-----")
            
            countList = countList + 1
            
        except Exception as e:
            print(e)
            print(f"Error File: {fileString} | From: {title} | Thread: {theredCount}")
            countErros += 1


# numero de threads real = numero de threads * 4
divideAndRun(numberOfThereds=1)