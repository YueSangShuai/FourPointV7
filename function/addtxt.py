import os

filePath="/media/yuesang/KESU/Robotmaster/dataset/data/images/train/"
filenames=os.listdir(filePath)
Note=open('train.txt',mode='w')
for filename in filenames:
    Note.write(os.path.join(filePath,filename))
    Note.write("\n")