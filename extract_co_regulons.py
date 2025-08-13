import pandas as pd


def get_common_targets(df, tf_list):
    """
    Extract the Target jointly regulated by the specified TF list

    parameters:
    df: DataFrame containing two columns, column names should be ‘TF’ and 'Target'
    tf_list: A list of TF names that need to be checked

    return:
    co_target: List of co-regulated Targets
    """
    # Screening of each TF-regulated Target
    tf_target_dict = {}
    for tf in tf_list:
        targets = df[df['TF'] == tf]['Target'].unique().tolist()
        tf_target_dict[tf] = set(targets)
        print(f"{tf} :the number of targe it regulates: {len(targets)}")

    # Calculate the intersection
    if not tf_target_dict:
        return []

    co_target = set(tf_target_dict[tf_list[0]])  # Initialize the intersection as the Target of the first TF
    for tf in tf_list[1:]:
        co_target &= tf_target_dict[tf]  # 

    return tf_list + list(co_target)


# sample data
# data = {
#     'TF': ['NFIB', 'ZFP148', 'DLX2', 'BCL11B', 'HSF2', 'HSF2', 'TCF4', 'TCF4', 'PAX3', 'PAX3'],
#     'Target': ['ITGB7', 'ITGB7', 'ITGB7', 'ITGB7', 'ITGB7', 'Gene2', 'ITGB7', 'Gene3', 'ITGB7', 'Gene4']
# }
# df = pd.DataFrame(data)

df = pd.read_csv("./demo_data/aged_opc/ag_opc_prediction4_3000.csv")



# Specify the TF list to be checked
#tf_list = ['HSF2', 'TCF4', 'PAX3']

# Some of the identified transcription factors that exhibit synergistic regulation
tf_list = ['PAX6','ELF1','IRF1','NFKB2']
tf_list = ['NR2F1','ESR1','POU2F1']

# Extract the jointly regulated Targets
co_target = get_common_targets(df, tf_list)

# 
print(f"\n{', '.join(tf_list)} Target genes that are coordinately regulated in the late stage of aging:")
if co_target:
    print(co_target)
else:
    print("No common Target for regulation has been found")



