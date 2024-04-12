import os
from pathlib import Path
import shutil
import re

def process_file(file_path, output_file):
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        replacements = [
            (r"configs/networks/resnet18_32x32\.yml", "configs/networks/convnext_base.yml"),
            (r"configs/datasets/cifar10/cifar10\.yml", "configs/datasets/tiny_inet/tiny_inat.yml"),
            (r"configs/pipelines/test/test_ood.yml", "configs/pipelines/test/test_hc_ood.yml"),
            (r"configs/datasets/cifar10/cifar10_ood\.yml", "configs/datasets/tiny_inet/tiny_inat_ood.yml"),
            (r"--num_workers \d", "--num_workers 1"),
            (r"--network.checkpoint '.*?'", "--network.checkpoint 'pretrained_models/checkpoint-best.pth'")
        ]

        for old, new in replacements:
            matches = re.findall(old, content)
            if len(matches) != 1:
                raise ValueError(f"Replacement for '{old}' did not occur exactly once.")
            content = re.sub(old, new, content)

        content = content.split("############################################")[0]

        with open(file_path, 'w') as file:
            file.write(content)

        with open(output_file, 'a') as file:
            file.write(f"echo 'Running {Path(file_path).parent.name}'\n")
            file.write(content + '\n')

        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main(directory):
    file_count = 0
    processed_count = 0
    output_file = "tiny_inat_all.sh"

    open(output_file, 'w').close()

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("cifar10_"):
                file_count += 1
                original_path = os.path.join(root, file)
                copy_path = os.path.join(root, file.replace("cifar10_", "tiny_inat_"))

                shutil.copyfile(original_path, copy_path)

                if process_file(copy_path, output_file):
                    processed_count += 1
                    print(f"{processed_count}/{file_count} - [{root}] done")
                else:
                    os.remove(copy_path)
                    print(f"{processed_count}/{file_count} - [{root}] failed")

    print(f"Processing completed. Total: {file_count}, Processed: {processed_count}")

if __name__ == "__main__":
    main("scripts/ood")
