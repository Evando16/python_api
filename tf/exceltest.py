import tensorflow as tf
from openpyxl import load_workbook

# read input data
wb = load_workbook(filename='dados.xlsx', read_only=True)
ws = wb['TREINAMENTO- RESPOSTA A CARGA']


# read xlxs
data = []
index = 0
for row in ws.rows:
    data.append([])

    for cell in row:
        data[index].append(cell.value)
    
    index += 1

#print(data[0])

# output
output = []
for i in range(0, len(data[0])):
    output.append([])
    output[i].append(data[0][i])

#print(output)

# input
input = []
for i in range(1, len(data)):
    for j in range(0, len(data[i])):
        if i == 1:
            input.append([])
            input[j].append(1)

        input[j].append(data[i][j])

#print(input)