import pandas as pd
import re
from copy import deepcopy


SIZE = "120"
ORI_MPI_PATH = f"./Ori-Mpi/mpi_{SIZE}.csv"
INVERSE_MPI_PATH = f"./Inverse-Mpi/mpi_{SIZE}.csv"


def inverse_text(text):
    if re.search(r"\bLove\b",text):
        return re.sub(r"\bLove\b","Hate",text)
    elif re.search(r"\bAre not\b",text):
        return re.sub(r"\bAre not\b","Are",text)
    elif re.search(r"\bAre\b",text):
        return re.sub(r"\bAre\b","Are not",text)
    elif re.search(r"\bDislike\b",text):
        return re.sub(r"\bDislike\b","Like",text)
    elif re.search(r"\bRarely\b",text):
        return re.sub(r"\bRarely\b","Often",text)
    elif re.search(r"\bPrefer to\b",text):
        return re.sub(r"\bPrefer to\b","Prefer not to",text)
    elif re.search(r"\bDo not \b",text):
        return re.sub(r"\bDo not \b","",text).capitalize()
    elif re.search(r"\bDon't \b",text):
        return re.sub(r"\bDon't \b","",text).capitalize()
    elif re.search(r"\bTry not to \b",text):
        return re.sub(r"\bTry not to \b","",text).capitalize()
    elif re.search(r"\bCan't\b",text):
        return re.sub(r"\bCan't\b","Can",text)
    elif re.search(r"\bCan\b",text):
        return re.sub(r"\bCan\b","Can't",text)
    elif re.search(r"\bNever ",text):
        return re.sub(r"\bNever ","",text).capitalize()
    elif re.search(r"\bSeldom\b",text):
        return re.sub(r"\bSeldom\b","Often",text)
    elif re.search(r"\bWill not\b",text):
        return re.sub(r"\bWill not\b","Will",text)
    elif re.search(r"\bWould never\b",text):
        return re.sub(r"\bWould never\b","Would",text)
    elif re.search(r"\bWould like to\b",text):
        return re.sub(r"\bWould like to\b","Wouldn't like to",text)
    elif re.search(r"\bWould\b",text):
        return re.sub(r"\bWould\b","Wouldn't",text)
    else:
        return "Don't "+text.lower()




if __name__ == "__main__":
    data = pd.read_csv(ORI_MPI_PATH)
    ori_data = deepcopy(data)
    for idx,item in data.iterrows():
        data.at[idx,"text"] = inverse_text(item["text"])
    data.to_csv(INVERSE_MPI_PATH,index=False)
    print((data!=ori_data).sum())
    
# Some original Typos:
'''
120: 51
1k: 17, 919-940 (38, 920, 929 have "")
'''