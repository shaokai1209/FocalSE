## How to train your model
Firstly, set the related path in start.sh file, then <br>
```bash
bash start.sh
```

## How to Inference
```bash
mkdir checkpoint
cd checkpoint
wget https://huggingface.co/Dongchao/AcademiCodec/resolve/main/HiFi-Codec-16k-320d
bash test.sh
```
python read_events.py /home/shaokai/AcademiCodec-master/egs/HiFi-Codec-16k-320d/logs_focalse/logs/events.out.tfevents.1781604581.old-eight.1077964.0 /home/shaokai/AcademiCodec-master/egs/HiFi-Codec-16k-320d/output.txt
