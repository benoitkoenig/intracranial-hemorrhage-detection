folder_path = "/".join(__file__.split("/")[:-1])

labels = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]
label_ranks = {}
for i in range(len(labels)):
    label_ranks[labels[i]] = i

stage_1_training_count = 674262
stage_1_training_any_count = 97103
stage_1_studies_count = 34 # edit with the real study count
max_slices_per_study = 4 # edit when real value is known
