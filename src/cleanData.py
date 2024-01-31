import os
import json

count = 0

def fixJson(changedFiles, originalFiles, currLabel, folder):
    global count
    
    print()
    print(f"folder: {folder}")
    
    for i in range(len(changedFiles)):
        outPath = f"{outputPath}/{folder}/{changedFiles[i]}"
        
        try:
            print(f"file: {outPath} | i: {i}/{len(changedFiles)}")
            
            fp = open(outPath)
            data = json.load(fp)
            os.remove(outPath)
            
            embeadings = []
            text = ""
            label = currLabel
            originalFile = changedFiles[i]
            
            if "sentence_embedding" in data:
                embeadings = data["sentence_embedding"]
                text = data["text"]
            
            elif "embedding" in data:
                embeadings = data["embedding"]
                
                for file in originalFiles:
                    if file["file"] == originalFile:
                        text = file["text"]
                        break
            
            data.clear()
            data["sentence_embedding"] = embeadings
            data["text"] = text
            data["label"] = label
            data["file"] = f"{folder}/{originalFile}"
            
            json.dump(data, open(outPath, "w+"), indent=4)
        
        except Exception as e:
            print(f"Error in file: {i}")
            os.remove(outPath)
            count += 1                
        
inputPath  = "/home/mateus/WSL/Neurais/TP/dataset/input"
outputPath = "/home/mateus/WSL/Neurais/TP/dataset/outputs"

TestPosIn  = os.listdir(f"{inputPath}/test/pos")
TestNegIn  = os.listdir(f"{inputPath}/test/neg")
TrainPosIn = os.listdir(f"{inputPath}/train/pos")
TrainNegIn = os.listdir(f"{inputPath}/train/neg")

TestPosOut  = os.listdir(f"{outputPath}/test/pos")
TestNegOut  = os.listdir(f"{outputPath}/test/neg")
TrainPosOut = os.listdir(f"{outputPath}/train/pos")
TrainNegOut = os.listdir(f"{outputPath}/train/neg")

fixJson(TestPosOut,  TestPosIn,  1, "test/pos")
fixJson(TestNegOut,  TestNegIn,  0, "test/neg")
fixJson(TrainPosOut, TrainPosIn, 1, "train/pos")
fixJson(TrainNegOut, TrainNegIn, 0, "train/neg")

print(f"Total of errors: {count}")

# JUNTAR OS ARQUIVOS DE TESTE E TREINO EM UMA UNICA PASTA
# RENOMEAR OS ARQUIVOS PARA QUE NÃO HAJA CONFLITO DE NOMES