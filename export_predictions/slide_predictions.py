import pandas as pd
import sys

from intracranial_hemorrhage_detection.constants import folder_path

def slide_predictions(sliding_param):
    "For every row in results.csv, copies it in slided_rows.csv with the probability raised to the power sliding_param"
    assert sliding_param > 0, "A negative sliding_param will result in all probabilities being greater than 1"
    df = pd.read_csv("%s/outputs/results.csv" % folder_path)
    df["Label"] = df["Label"] ** sliding_param
    df.to_csv("%s/outputs/slided_results.csv" % folder_path, header=True, index=False)

# Read arguments from python command
sliding_param = None
for param in sys.argv:
    try:
        sliding_param = float(param)
    except:
        pass

if (sliding_param == None):
    print("Usage: python export_predictions.slide_predictions [sliding_param]")
    print("A sliding_param between 0 and 1 will move the probabilities upward, a sliding_param above 1 will move the probabilities downward")
    exit(-1)

slide_predictions(sliding_param)
