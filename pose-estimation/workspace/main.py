import os, json, re
import pandas as pd
import matplotlib.pyplot as plt
from pose_estimation.config import PoseEstimationConfig
from pose_estimation.assess import PoseEstimationAssessor

def dict2json(path: str) -> dict:
    with open(path) as file: 
        return json.load(file) 

def flatten_keys(s1: str, s2: str) -> str:
    s1 = s1.replace(" ", "_")
    s2 = s2.replace(" ", "_")
    
    s = f"{s1}_{s2}"
    return re.sub('[^A-Za-z_]+', '', s)

DIRECTORY: str = PoseEstimationConfig.ARTIFACTS.value
stroke_datasets: list = ["malong", "harimoto", "zhang_jike"]
video_times_ms: list = [8000, 15000, 15000]

template_path: str = f"{DIRECTORY}template/livestream_template_flattened.json"
template_dict = dict2json(template_path)

pandas_datasets = {}
for i in range(len(stroke_datasets)):
    stroke_dataset_files = os.listdir(f"{DIRECTORY}{stroke_datasets[i]}")
    stroke_dataset_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    stroke_dataset_dict = dict2json(template_path)
    
    for j in range(len(stroke_dataset_files)):
        stroke_dataset_files[j] = f"{DIRECTORY}{stroke_datasets[i]}/{stroke_dataset_files[j]}"
        stroke_dataset_file_dict = dict2json(stroke_dataset_files[j])
        
        for key, values in stroke_dataset_file_dict.items():
            if key != "timestamp":
                for k,v in values.items():
                    flattened_key = flatten_keys(key, k)
                    stroke_dataset_dict[flattened_key].append(v)
            else:
                stroke_dataset_dict[key].append(values)
    df = pd.DataFrame.from_dict(stroke_dataset_dict)
    df["time"] = range(0, video_times_ms[i], int(video_times_ms[i]/len(df)))[:len(df)]
    pandas_datasets[stroke_datasets[i]] = df

x = "time"
y_values = ["right_elbow_x", "right_elbow_y", "right_elbow_z"]
for player in stroke_datasets:
    for y in y_values:
        pandas_datasets[player].plot(kind = 'line', x = x, y = y)
        plt.savefig(f"{DIRECTORY}/plots/{player}_{x}_{y}.png") 

players = ["zhang_jike", "harimoto"]

actual = pandas_datasets[players[0]][["right_elbow_x"]].to_numpy()
pred = pandas_datasets[players[1]][["right_elbow_x"]].to_numpy()

min_len = min(len(actual), len(pred))

pea = PoseEstimationAssessor()
metrics = pea.classify(actual=actual[:min_len], pred=pred[:min_len])
print(players)
print(metrics)
