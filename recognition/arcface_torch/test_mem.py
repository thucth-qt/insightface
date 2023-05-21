# %%
from psutil import *
import psutil
import time
import csv 

with open("/home/thucth/Biometrics/insightface/recognition/arcface_torch/TRACKING1.csv", "a") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Time", "MEM Percentage", "CPU Percentage"])
times = 0
while True:
    vm = psutil.virtual_memory()    
    cpu = psutil.cpu_percent()
    if vm.percent > 60:
        print("Exit because memory usage is high")
        break
    print(f"mem is good - using {vm.percent}%")
    
    
    with open("/home/thucth/Biometrics/insightface/recognition/arcface_torch/TRACKING1.csv", "a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), vm.percent, cpu])
    
    time.sleep(30)
    times+=1
    if times > 100000:
        break

