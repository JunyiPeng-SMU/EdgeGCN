import pandas as pd
import numpy as np
import dgl
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
patient = pd.read_excel(r'/public/pengjunyi/Code/Node/OA/patient_labels.xls')
Cxy = np.zeros((600,600))
Bxy = np.zeros((600,600))
Ixy = np.zeros((600,600))
Kxy = np.zeros((600,600))
Wxy = np.zeros((600,600))
Jxy = np.zeros((600,600))

BIxy = np.zeros((600,600))
KWxy = np.zeros((600,600))
KJxy = np.zeros((600,600))
WJxy = np.zeros((600,600))
Gxy = np.zeros((600,600))

for i in range(600):
    for j in range(600):
        age_patient_x = patient['age'][i]
        age_patient_y = patient['age'][j]
        gender_patient_x = patient['gender'][i]
        gender_patient_y = patient['gender'][j]
        BMI_patient_x = patient['BMI'][i]
        BMI_patient_y = patient['BMI'][j]
        injure_patient_x = patient['受伤史'][i]
        injure_patient_y = patient['受伤史'][j]
        KL_patient_x = patient['KL分级'][i]
        KL_patient_y = patient['KL分级'][j]
        WOMAC_patient_x = patient['WOMAC'][i]
        WOMAC_patient_y = patient['WOMAC'][j]
        width_patient_x = patient['关节宽度'][i]
        width_patient_y = patient['关节宽度'][j]
        C_point = 0
        B_point = 0
        KW_point = 0
        KJ_point = 0
        WJ_point = 0
        if abs(age_patient_x-age_patient_y) <5:
            C_point+=1
        if gender_patient_x == gender_patient_y:
            C_point+=1
        Cxy[i][j] = C_point
        if abs(BMI_patient_x-BMI_patient_y) <3:
            Bxy[i][j] = 1
            B_point += 1
        if injure_patient_x == injure_patient_y:
            Ixy[i][j] = 1
            B_point += 1
        BIxy[i][j] = B_point
        if KL_patient_x == KL_patient_y:
            Kxy[i][j] = 1
            KW_point += 1
            KJ_point += 1
        if WOMAC_patient_x == WOMAC_patient_y:
            Wxy[i][j] = 1
            KW_point += 1
            WJ_point += 1
        if abs(width_patient_x-width_patient_y) <1:
            Jxy[i][j] = 1
            KJ_point += 1
            WJ_point += 1
        KWxy[i][j] = KW_point
        KJxy[i][j] = KJ_point
        WJxy[i][j] = WJ_point

def get_fig(matrix,name):
    matrix[matrix>0]=1
    matrix2 = matrix - np.diag(np.ones(600))
    a = list()
    b = list()
    for i1 in range(600):
        for j1 in range(600):
            if matrix2[i1][j1]>0:
                a.append(i1)
                b.append(j1)
    g1 = dgl.DGLGraph()
    g1.add_nodes(600)
    src = th.tensor(a)
    dst = th.tensor(b)
    g1.add_edges(src, dst)
    plt.figure()
    nx.draw(g1.to_networkx(), pos=nx.spring_layout(g1.to_networkx()), with_labels=False,node_size=5,arrows=False)
    plt.savefig(r'/public/pengjunyi/Code/Node/OA/%s.png'%name)

if __name=='__main__':
    matrix = Cxy*BIxy*KWxy*Jxy