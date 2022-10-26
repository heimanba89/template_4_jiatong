from espnet2.diar.label_processor.dnc_aggregation import DNCAggregate
from espnet2.fileio.rttm import DNCRttmReader
import torch
import numpy as np


rttm_file = "three_class_dump/raw/org/test/dnc_rttm"
specific_meeting_id = "R8001_M8004"
loading = "refer_1.txt"

rttm_reader = DNCRttmReader(rttm_file)
label_processor = DNCAggregate(
    win_length=512,  # 32ms
    hop_length=256,  # 16ms
    spk_embed_window=1,
    spk_embed_shift=1,
    center=False,
    soft_label=False,
    separate_detection=False,
    add_osd=True,
)

label_processor2 = DNCAggregate(
    win_length=512,  # 32ms
    hop_length=256,  # 16ms
    spk_embed_window=5,
    spk_embed_shift=5,
    center=False,
    soft_label=False,
    separate_detection=False,
    add_osd=True,
)

with open(loading, "r", encoding="utf-8") as f:
    text = f.read().split("\n")[0]
    text = text.split(" ")[1:]
    text = np.array(list(map(int, text)))
    text[text > 1] = 2


for meeting_id in rttm_reader.keys():
    if meeting_id == specific_meeting_id:
        meeting_info = rttm_reader[meeting_id]
        meeting_info = torch.tensor(meeting_info).unsqueeze(0)
        embed_out, olens = label_processor(meeting_info)
        embed_out2, _ = label_processor2(meeting_info)

        embed_out = embed_out[0].int().cpu().numpy()
        embed_out2 = embed_out2[0].int().cpu().numpy()
        break

# text reference
# embed_out our own implementation
print(embed_out)
# print(embed_out2)
print(text)

print("--- overlap ---")
print(np.sum(embed_out == 1))
print(np.sum(text == 1))
print(np.sum(embed_out2 == 1))


print("--- same ---")
min_len = min(len(embed_out), len(text))
print(np.sum(embed_out[:min_len] == text[:min_len]))


print("--- silence ---")
print(np.sum(embed_out == 0))
print(np.sum(text == 0))
print(np.sum(embed_out2 == 0))


print("--- length ---")
print(len(embed_out))
print(len(text))
print(len(embed_out2))


print("--- details --")
for i in range(0, 3):
    for j in range(0, 3):
        print(i, j)
        print(np.sum(np.logical_and(embed_out == i, text == j)))


# size=80000
# hop=1000
# print(list(embed_out[size:size+hop]))
# print(list(text[size:size+hop]))
