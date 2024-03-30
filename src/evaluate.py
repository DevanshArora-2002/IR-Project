from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import os
model = SentenceTransformer('all-MiniLM-L6-v2')
gen_path = os.path.join(os.path.dirname(os.getcwd()), 'Baseline_Results')
orig_path = gen_path+'/original.txt'
model_path = gen_path+"/model.txt"
with open(orig_path, 'r', encoding='utf-8') as file:
    orig_data = file.read()

with open(model_path, 'r', encoding='utf-8') as file:
    model_data = file.read()

original_data_lst = list(orig_data.strip('@_@'))
model_data_lst = list(model_data.strip('@_@'))
def get_scores(list1,list2,model):
    scores = []
    for i in range(len(list1)):
        sentences = [list1[i],list2[i]]
        embedded_list = model.encode(sentences)
        similarity = cos_sim(embedded_list[0],embedded_list[1])
        scores.append(similarity)
    return scores