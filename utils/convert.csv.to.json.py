import pandas as pd
import json
import os
import numpy as np


temp = pd.read_csv("./papers.csv")
temp = temp.set_index("Alias")
paper_dict = temp.to_dict("index")


def csv_to_multiple_json_files(input_csv_file, output_root):
    def number_to_str(x):
        if isinstance(x, str):
            return x
        elif isinstance(x, float):
            return str(x).ljust(5,'0')

    # Read the CSV file
    df = pd.read_csv(input_csv_file)
    df['ID_Accuracy'] = df['ID_Accuracy'].apply(number_to_str)
    df['Near-OOD_AUROC'] = df['Near-OOD_AUROC'].apply(number_to_str)
    df['Far-OOD_AUROC'] = df['Far-OOD_AUROC'].apply(number_to_str)

    #dtype={
    #    'ID_Accuracy': 'string',
    #    'Near-OOD_AUROC': 'string',
    #    'Far-OOD_AUROC': 'string'       
    #}
    
    # Extract the first float value from the strings in the "NearOOD AUROC" and "FarOOD AUROC" columns
    df["ID_Accuracy_Mean"] = df["ID_Accuracy"].str.extract(r"(\d+\.\d+)", expand=False).apply(float).apply('{:.2f}'.format)
    df["ID_Accuracy_Mean"] = df["ID_Accuracy_Mean"].fillna('/')
    df["ID_Accuracy_Mean"] = df["ID_Accuracy_Mean"].replace('nan', '/')
    df["Near-OOD_AUROC_Mean"] = df["Near-OOD_AUROC"].str.extract(r"(\d+\.\d+)", expand=False).apply(float).apply('{:.2f}'.format)
    df["Far-OOD_AUROC_Mean"] = df["Far-OOD_AUROC"].str.extract(r"(\d+\.\d+)", expand=False).apply(float).apply('{:.2f}'.format)

    # Rank methods according to NearOOD AUROC by default
    df["Rank"] = df["Near-OOD_AUROC_Mean"].rank(ascending=False, method='first').astype(int)

    # Set up saving directory
    dataset = input_csv_file.split('/')[-1].split('.')[0]
    saving_dir = os.path.join(output_root, dataset)
    os.makedirs(saving_dir, exist_ok=True)

    for i in range(len(df)):
        training = df.iloc[i]["Training"]
        postproc = df.iloc[i]["Postprocessor"]
        extra_desc = df.iloc[i]["Additional_Description"]
        output_filename = f"{training}_{postproc}"
        if isinstance(extra_desc, str) and len(extra_desc) > 0:
            output_filename += f"_{extra_desc}"
        output_filename += ".json"
        output_filename = output_filename.replace(' ', '_')

        json_data = df.iloc[i:i+1].to_dict("records")

        training_with_link_list = []
        for temp in training.split(' + '):
            if temp in paper_dict:
                training_with_link_list.append(
                    f"<a href={paper_dict[temp]['Link']}>{temp}</a>"
                )
            else:
                training_with_link_list.append(
                    temp
                )
        training_with_link = ' + '.join(training_with_link_list)
        json_data[0]["Training"] = training_with_link

        pp_with_link_list = []
        for temp in postproc.split(' + '):
            pp_with_link_list.append(
                f"<a href={paper_dict[temp]['Link']}>{temp}</a>"
            )
        pp_with_link = ' + '.join(pp_with_link_list)
        json_data[0]["Postprocessor"] = pp_with_link

        #json_data[0]["Training_Link"] = paper_dict[training]["Link"] if training in paper_dict else ""
        #json_data[0]["PP_Link"] = paper_dict[postproc]["Link"]
        
        if not isinstance(extra_desc, str):
            json_data[0]["Additional_Description"] = ""

        # Write the JSON object to a file, one record per line
        with open(os.path.join(saving_dir, output_filename), "w") as file:
            for record in json_data:
                file.write(json.dumps(record, indent=2) + "\n")

        #print(f"Created {os.path.join(saving_dir, output_filename)}")


if __name__ == "__main__":
    for file_id in os.listdir("../results"):
        input_csv_file = f"../results/{file_id}"
        output_root = "../model_info"
        csv_to_multiple_json_files(input_csv_file, output_root)