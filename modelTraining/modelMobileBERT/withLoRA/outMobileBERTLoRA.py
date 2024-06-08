import cv2
import time
import torch
import pytesseract
from peft import PeftModel
from tkinter import filedialog
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer


print("\n\n\nSelect the imgFile Yo !")
inImg = filedialog.askopenfilename(filetypes=[("imgFile", "*.png")])


t1 = time.time()
# Windows
#pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
# Mac
pytesseract.pytesseract.tesseract_cmd = r"/usr/local/Cellar/tesseract/5.3.4_1/bin/tesseract"

idMap = {'Miscellaneous Category - Miscellaneous Bills': 3,
         'Travel Category - Flight Ticket': 7,
         'Electronics Category - Electronic Gadgets Bill/Receipt': 0,
         'Travel Category - Train Ticket': 8,
         'Travel Category - Cab Bill/Receipt': 6,
         'Room Stay Category - Stay Bill/Receipt': 4,
         'Travel Category - Bus Ticket': 5,
         'Food Category - Restaurant Bill/Receipt': 1,
         'Fuel Category - Fuel Bill/Receipt': 2}

labelMap = {3: 'Miscellaneous Category - Miscellaneous Bills',
            7: 'Travel Category - Flight Ticket',
            0: 'Electronics Category - Electronic Gadgets Bill/Receipt',
            8: 'Travel Category - Train Ticket',
            6: 'Travel Category - Cab Bill/Receipt',
            4: 'Room Stay Category - Stay Bill/Receipt',
            5: 'Travel Category - Bus Ticket',
            1: 'Food Category - Restaurant Bill/Receipt',
            2: 'Fuel Category - Fuel Bill/Receipt'}

img = cv2.imread(inImg, 0)
plt.imshow(img, cmap = "gray")
plt.show(block=False)

txtOut = pytesseract.image_to_string(img)
print("\n\nThe OCR Contents - ", txtOut)

model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=len(labelMap), id2label=idMap, label2id=labelMap)
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

model = PeftModel.from_pretrained(model, 'trainedMobileBERTLoRA')

loraModel = model.merge_and_unload()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
loraModel.to(device)

print("\n\n\n")
if len(txtOut)>50:
    encodedData = tokenizer(txtOut, padding="max_length", truncation=True, max_length=512)
    input_ids = torch.tensor([encodedData["input_ids"]], dtype=torch.uint8).to(device).long()
    attention_mask = torch.tensor([encodedData["attention_mask"]], dtype=torch.uint8).to(device)
    
    with torch.no_grad():
        outputs = loraModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predOut = torch.argmax(logits, dim=1)

    confVals = torch.nn.functional.softmax(logits, dim=1)
    confOut = float(confVals[0, predOut.item()])
    print("Confidence Yo: ", confOut)
    if confOut>0.7:
       print("outStuff: ", labelMap[predOut.item()])
    else:
        print("outStuff: Miscellaneous Category - Miscellaneous Bills")
else:
    print("outStuff: Miscellaneous Category - Miscellaneous Bills")

t2 = time.time()
print("ExecTime: ", str(t2-t1) + " Secs")