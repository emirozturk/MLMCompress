from transformers import BertForMaskedLM, AutoTokenizer
import torch
import timeit
import os
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


def torchBatchEncode(inputArray,bertTokenizer, bertModel,predSize):
    ids = bertTokenizer(inputArray,padding=True, truncation=True,add_special_tokens=True, return_tensors='pt')

    mask_token_index = [torch.where(x==bertTokenizer.mask_token_id)[0] for x in ids["input_ids"]]

    ids = ids.to(device)
    predict = bertModel(**ids)[0]

    predictionList = []
    for i in range(len(predict)): 
        predictionList.append(torch.topk(predict[i, mask_token_index[i], :],predSize,dim=1)[1])

    resultList = []
    for pred in predictionList:
        result = ""
        if len(pred)>0:
            result = [x.replace("##","") for x in bertTokenizer.convert_ids_to_tokens(pred[0])]
        resultList.append(result)
    return resultList


def torchInit(bertModelName,checkPointName):    
    bert_tokenizer = AutoTokenizer.from_pretrained(bertModelName,model_max_length=50)
    bert_model = BertForMaskedLM.from_pretrained(bertModelName+checkPointName)
    bert_model.eval()
    bert_model.to(device)
    return bert_tokenizer, bert_model


def RLE(byteArray):
    result=bytes()
    i = 0
    while (i <= len(byteArray)-1):
        count = 1
        ch = byteArray[i]
        if ch == 0:
            j = i
            while (j < len(byteArray)-1): 
                if (byteArray[j] == byteArray[j + 1]): 
                    count = count + 1
                    j = j + 1
                else: 
                    break
            result +=  ch.to_bytes(1,'little') + count.to_bytes(1,'little')
            i = j + 1
        else:
            result += ch.to_bytes(1,'little')
            i+=1
    return result


def UnRLE(byteArray):
    result = bytes()
    i = 0
    while (i <= len(byteArray) - 1):
        ch = byteArray[i]
        if ch == 0:
            run_count = int(byteArray[i+1])
            for _ in range(run_count):
                result += ch.to_bytes(1,'little')
            i = i + 2
        else:
            result += ch.to_bytes(1,'little')
            i+=1
    return result

import huff
import re

def refreshWindow(w, newWord,windowSize):
    if len(w) < windowSize:
        w += [newWord]
    else:
        w = w[1:] + [newWord]
    return w


def addSpaces(text):
    return re.sub('([.:,\'\"!?()-])', r' \1 ', text)


def removeSpaces(text):
    return re.sub(' ([.:,\'\"!?()-]) ', r'\1', text)


def Decompress(bertTokenizer, bertModel, byteArray, windowSize, predictionSize,lang):
    codesLen = int.from_bytes(byteArray[0:4],byteorder='little')
    rawsLen = int.from_bytes(byteArray[4:8],byteorder='little')
    codes = byteArray[8:codesLen+8]
    raws = byteArray[8+codesLen:]

    raws = "".join(huff.huffman_decode(raws, lang)).split(" ")

    codes = list(codes)
    
    decompressed = raws[0]+" "
    window = [raws[0]]
    rawsCounter = 1
    codeCounter = 0
    while codeCounter < len(codes):
        if codes[codeCounter] == 0:
            decompressed += raws[rawsCounter]+" "
            window = refreshWindow(window,raws[rawsCounter],windowSize)
            rawsCounter+=1
        elif codes[codeCounter] == 1:
            decompressed += " "
        else:
            inputForBert = " ".join(window)
            result = torchBatchEncode([inputForBert + " " + bertTokenizer.mask_token], bertTokenizer, bertModel, predictionSize)[0]
            word = result[codes[codeCounter]-2]
            decompressed += word + " "
            window = refreshWindow(window,word,windowSize)
        codeCounter+=1
    
    return removeSpaces(decompressed)[:-1]


def Compress(bertTokenizer, bertModel, text, windowSize, predictionSize,lang):
    text = addSpaces(text)
    words = text.split(" ")
    window = [words[0]]
    codes = []
    raws = [words[0]]
    inputs = []
    for word in words[1:]:
        if word == "":
            continue
        inputs.append(" ".join(window+[bertTokenizer.mask_token])) 
        window = refreshWindow(window, word,windowSize) #1 for mask        

    predictions = torchBatchEncode(inputs,bertTokenizer,bertModel,predictionSize)

    predictionCounter = 0
    for word in words[1:]:
        if word == "":
            codes.append(1)
            continue
        else:
            result = predictions[predictionCounter]
            if word in result:
                codes.append(result.index(word)+2) #First 2 indexes are escapes. 2 will be subtracted from index value at decompression stage
            else:
                codes.append(0)
                raws.append(word)
        predictionCounter+=1

    codesBytes = bytes(bytearray(codes))
    rawsString = " ".join(raws)

    rawCodeLength = len(codesBytes) #FOR statistics, can be removed safely
    rawStringLength = len(rawsString) #FOR statistics, can be removed safely

    rawsBytes = huff.huffman_encode(rawsString, lang)

    compressedCodeLength = len(codesBytes) #FOR statistics, can be removed safely
    compressedStringLength = len(rawsBytes) #FOR statistics, can be removed safely
    
    codesLen = len(codesBytes).to_bytes(4,byteorder='little')
    rawsLen = len(rawsBytes).to_bytes(4,byteorder='little')
    compressed = codesLen+rawsLen+codesBytes+rawsBytes
    #print(codesLen,rawsLen,codesBytes,rawsBytes)
    return rawCodeLength,compressedCodeLength,rawStringLength,compressedStringLength,compressed #,codec


configs = [
        ("en1000","bert-base-cased"),
        ("en1000","dbmdz_bert-tiny-historic-multilingual-cased"),
        ("en1000","tzyLee_quant-tinybert"),

        ("Multilingual","tzyLee_quant-tinybert-2023-10-14"),        
        ("Multilingual","dbmdz_bert-tiny-historic-multilingual-cased"),
        ("Multilingual","bert-base-multilingual-cased"),
        ]

lineCounter = 0
lineLimit = 0
start = 0
encodingType = "utf8"
pathStart = os.path.dirname(__file__) + "/corpus/"
checkPointName = "" #"/checkpoint-890000"
langArray = ["de","en","es","fr","it","nl","tr"]
wps = [(5,64),(5,128),(5,254),(10,64),(10,128),(10,254),(15,64),(15,128),(15,254)]
for tup in tqdm(configs):
    fileName,modelName = tup
    start_time = timeit.default_timer()
    bertTokenizer, bertModel = torchInit(modelName,checkPointName)
    modelLoadTime = (timeit.default_timer() - start_time)
    with open(pathStart+fileName+".txt", encoding=encodingType) as file:
        
      lines = file.readlines()
      result = f"{fileName}-{modelName}-Load time-{modelLoadTime}\n" 

      for w,p in tqdm(wps[start:]):
          result +=f"w-{w}p-{p}\n"
          result += "rawcodelength;compressedcodelength;rawstringlength;compressedstringlength;ratio;comptime;decomptime\n"            
          lineCounter=0
          langCounter=0
          for line in tqdm(lines):
              if lineCounter<lineLimit:
                  lineCounter+=1
                  continue
              lineLimit=0
              start = 0
              if fileName=="Multilingual":
                  if lineCounter!=0 and lineCounter%10==0:
                      langCounter+=1
                      langCounter = langCounter%7
                      print(langArray[langCounter])
              else:
                  langCounter = 1
              start_time = timeit.default_timer()
              rawCodeLength,compressedCodeLength,rawStringLength,compressedStringLength,compressed = Compress(bertTokenizer,bertModel,line,w,p,langArray[langCounter])
              compTime = (timeit.default_timer() - start_time)
                    
              start_time = timeit.default_timer()
              decompressed = Decompress(bertTokenizer,bertModel,compressed,w,p,langArray[langCounter])
              decompTime = (timeit.default_timer() - start_time)

              ratio = str(len(compressed)*100/len(line)).replace(".",",")
              comptime = str(compTime).replace(".",",")
              decomptime = str(decompTime).replace(".",",")
              result += f"{rawCodeLength};{compressedCodeLength};{rawStringLength};{compressedStringLength};{ratio};{comptime};{decomptime}\n"

              with open(pathStart+f"{fileName}-{modelName.split('/')[-1]}-results.txt","a") as results:
                  results.write(result)
                  result = ""
              lineCounter+=1