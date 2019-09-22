folder_path = "/".join(__file__.split("/")[:-1])
image_size=512
label_ranks = { "epidural": 0, "intraparenchymal": 1, "intraventricular": 2, "subarachnoid": 3, "subdural": 4, "any": 5 }

labels = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]
label_ranks = {}
for i in range(len(labels)):
    label_ranks[labels[i]] = i
