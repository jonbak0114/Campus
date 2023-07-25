1.Excel to Json
import openpyxl
import json

excel_file = "C:/Users/서종민/Desktop/Excel to Json.xlsx"
json_file = "C:/Users/서종민/Desktop/Excel to Json(결과).json"

wb = openpyxl.load_workbook(excel_file, read_only=True)
sheet = wb.worksheets[0]

key_list = [cell.value for cell in sheet[1]]

data_dict = {}

key_index = 0  # Fix: Set the key_index to 0 instead of 1

for row in sheet.iter_rows(min_row=2, values_only=True):
    tmp_dict = {key_list[i]: row[i] for i in range(len(row))}
    data_dict[tmp_dict[key_list[key_index]]] = tmp_dict

wb.close()

with open(json_file, 'w', encoding='utf-8') as fp:
    json.dump(data_dict, fp, indent=4, ensure_ascii=False)
결과 
    "1": {
        "연번": 1,
        "출원번호": "CN2022-10840296",
        "출원연도": "2022",
        "출원인(정비)": "Xuzhou Qichuang Manufacturing Co.,Ltd.",
        "발명의명칭": "A 3d printer",
        "대분류": "장비·디바이스",
        "대분류코드": "C",
        "중분류": "생산현장",
        "중분류코드": "CD",
        "소분류": "3D 프린팅",
        "소분류코드": "CDC",
        "국가코드": "CN",
        "특허실용구분": "특허공개",
        "법적상태": "공개",
        "소멸이유": null,
        "출원인": "徐州齐创制造有限公司",
        "출원인원문": "徐州齐创制造有限公司",
        "피인용문헌수": null,
        "패밀리문헌수": 1,


2.Json개수 세기
import json

def count_string_with_common_values(json_data, target_values):
    count = 0
    if isinstance(json_data, dict):
        values = list(json_data.values())
        if all(value in values for value in target_values):
            count += 1
        for value in values:
            count += count_string_with_common_values(value, target_values)
    elif isinstance(json_data, list):
        for item in json_data:
            count += count_string_with_common_values(item, target_values)
    return count

# JSON 파일 경로
file_path = "C:/Users/서종민/Desktop/Json 개수 세기.json"

# JSON 파일 열기
with open(file_path, encoding='UTF-8') as file:
    # JSON 데이터 파싱
    json_data = json.load(file)

# 공통 값들
target_values = ["CN", "2022"]

# 공통 값을 가진 항목 개수 구하기
string_count = count_string_with_common_values(json_data, target_values)

print(f"The count of strings with common values {target_values}: {string_count}")
결과
정리한 표를 데이터로 보여주면 될 듯




3.대표청구항 Json to 고정벡터
import json
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Load sentences from JSON file
sentences = []
with open("C:/Users/서종민/Desktop/대표청구항.json", encoding="utf-8") as file:
    data = json.load(file)
    sentences = [item["대표청구항"] for item in data["Sheet1"]]

# Convert sentences to fixed-length vectors
sentence_embeddings = model.encode(sentences, show_progress_bar=True)

# Convert sentence embeddings to a list of lists
embedding_list = [embedding.tolist() for embedding in sentence_embeddings]

# Save the fixed-length vectors to a JSON file
output_file_path = " C:/Users/서종민/Desktop/ 대표청구항(결과).json"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(embedding_list, output_file)

4.요약 Json to 고정벡터
import json
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Load sentences from JSON file
sentences = []
with open("C:/Users/서종민/Desktop/요약.json", encoding="utf-8") as file:
    data = json.load(file)
    sentences = [item["요약"] for item in data["Sheet1"]]

# Convert sentences to fixed-length vectors
sentence_embeddings = model.encode(sentences, show_progress_bar=True)

# Convert sentence embeddings to a list of lists
embedding_list = [embedding.tolist() for embedding in sentence_embeddings]

# Save the fixed-length vectors to a JSON file
output_file_path = "C:/Users/서종민/Desktop/요약(결과).json"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(embedding_list, output_file)


5.대표청구항고정벡터 3차원 축소
import numpy as np
from sklearn.decomposition import PCA
import json

with open("C:/Users/서종민/Desktop/대표청구항고정벡터.json", 'r', encoding="utf-8") as file:
    data = json.load(file)

sentence_vectors = [item for item in data]

X = np.array(sentence_vectors)

pca = PCA(n_components=3)
sentence_vectors_3d = pca.fit_transform(X)

output_file = "C:/Users/서종민/Desktop/대표청구항고정벡터(축소).json"

output_data = sentence_vectors_3d.tolist()

with open(output_file, 'w', encoding="utf-8") as file:
    json.dump(output_data, file, indent=4, ensure_ascii=False)


6.요약고정벡터 3차원 축소
import numpy as np
from sklearn.decomposition import PCA
import json

with open("C:/Users/서종민/Desktop/요약고정벡터.json", 'r', encoding="utf-8") as file:
    data = json.load(file)

sentence_vectors = [item for item in data]

X = np.array(sentence_vectors)

pca = PCA(n_components=3)
sentence_vectors_3d = pca.fit_transform(X)

output_file = "C:/Users/서종민/Desktop/요약고정벡터(축소).json"

output_data = sentence_vectors_3d.tolist()

with open(output_file, 'w', encoding="utf-8") as file:
    json.dump(output_data, file, indent=4, ensure_ascii=False)

7.대표청구항.요약 고정벡터 클러스터링(3차원)
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

def dbscan_clustering_3d(file_path, subset_size, epsilon=0.1, min_samples=5):
    # 파일로부터 데이터 읽어오기
    with open(file_path, 'r') as file:
        data = json.load(file)

    vectors = np.array(data)

    # Choose a subset of the data
    subset_vectors = vectors[:subset_size]

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(subset_vectors)

    # 시각화를 위한 3D 산점도 플롯
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:  # 잡음 포인트는 검정색으로 표시
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xyz = subset_vectors[class_member_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=50, c=[col], marker='o')

    ax.set_title('DBSCAN Clustering (3D)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# 파일 경로와 클러스터링에 사용할 데이터 크기를 설정합니다.
file_path = "C:/Users/서종민/Desktop/ 대표청구항고정벡터(축소).json"  # 파일 경로를 적절히 수정해주세요
subset_size = 196132  # 클러스터링에 사용할 데이터 크기를 조정합니다.
epsilon = 0.05  # 적절한 반경 값을 설정합니다. (더 작은 값으로 변경)
min_samples = 5 # 적절한 최소 이웃 개수를 설정합니다.

# DBSCAN 클러스터링을 수행하고 결과를 3D 시각화합니다.
dbscan_clustering_3d(file_path, subset_size, epsilon=epsilon, min_samples=min_samples)


from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

def dbscan_clustering_3d(file_path, subset_size, epsilon=0.1, min_samples=5):
    # 파일로부터 데이터 읽어오기
    with open(file_path, 'r') as file:
        data = json.load(file)

    vectors = np.array(data)

    # Choose a subset of the data
    subset_vectors = vectors[:subset_size]

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(subset_vectors)

    # 시각화를 위한 3D 산점도 플롯
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:  # 잡음 포인트는 검정색으로 표시
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xyz = subset_vectors[class_member_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=50, c=[col], marker='o')

    ax.set_title('DBSCAN Clustering (3D)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# 파일 경로와 클러스터링에 사용할 데이터 크기를 설정합니다.
file_path = "C:/Users/서종민/Desktop/요약고정벡터(축소).json"  # 파일 경로를 적절히 수정해주세요
subset_size = 196132  # 클러스터링에 사용할 데이터 크기를 조정합니다.
epsilon = 0.05  # 적절한 반경 값을 설정합니다. (더 작은 값으로 변경)
min_samples = 5 # 적절한 최소 이웃 개수를 설정합니다.

# DBSCAN 클러스터링을 수행하고 결과를 3D 시각화합니다.
dbscan_clustering_3d(file_path, subset_size, epsilon=epsilon, min_samples=min_samples)


