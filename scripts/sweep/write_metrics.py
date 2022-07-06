import csv
import os

import numpy as np


def make_args_list(benchmarks, methods, metrics, benchmark_dict):
    args_list = []
    for metric in metrics:
        for benchmark in set(benchmarks) & set(benchmark_dict[metric]):
            for method in methods:
                args_list.append([benchmark, method, metric])
    return args_list


def write_metric(args, folder_list, save_line_dict, benchmark_dict):

    metric_list = [
        'fpr95', 'auroc', 'aupr_in', 'aupr_out', 'ccr_4', 'ccr_3', 'ccr_2',
        'ccr_1', 'acc'
    ]
    save_list = []
    for metric in args.metric2save:
        save_list.append(metric_list.index(metric) + 1)

    for metric in args.metrics:
        if metric == 'ood':
            for benchmark in set(args.benchmarks) & set(
                    benchmark_dict[metric]):
                args_list = make_args_list([benchmark], args.methods, ['ood'],
                                           benchmark_dict)
                sub_form_content = []
                for key_param in args_list:
                    for folder in folder_list:
                        key_folder = folder.split('_')
                        if all(key in key_folder for key in key_param):
                            target_folder = folder
                            break
                    else:
                        print("No respective folder path, something's wrong.")
                        raise FileNotFoundError
                        # quit()

                    with open(
                            os.path.join(args.output_dir, target_folder,
                                         'ood.csv'), 'r') as f:
                        lines = f.readlines()[save_line_dict[key_param[-1]]:]
                    sub_line_content = {}
                    sub_line_content['method/{}'.format(
                        args.metric2save)] = key_param[1]
                    for line in lines:
                        split = line.split(',')
                        content = ''
                        for metric in save_list:
                            content = content + '{:.2f}'.format(
                                float(split[metric])) + ' / '
                        else:
                            content = content[:-3]
                        # use method name as key
                        sub_line_content[split[0]] = content
                    sub_form_content.append(sub_line_content)
                csv_path = os.path.join(args.output_dir,
                                        '{}_ood.csv'.format(key_param[0]))
                with open(csv_path, 'w', newline='') as csvfile:
                    fieldnames = order_fieldnames(
                        list(sub_form_content[0].keys()), args)
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for sub_line_content in sub_form_content:
                        writer.writerow(sub_line_content)

        elif metric == 'osr':
            sub_form_content = []
            for method in args.methods:
                args_list = make_args_list(args.benchmarks, [method], ['osr'],
                                           benchmark_dict)
                sub_line_content = {}

                for key_param in args_list:
                    sub_line_content['method/{}'.format(
                        args.metric2save)] = key_param[1]
                    target_folder = []
                    seeds = ['seed1', 'seed2', 'seed3', 'seed4', 'seed5']
                    for seed in seeds:
                        key_param.append(seed)
                        for folder in folder_list:
                            key_folder = folder.split('_')
                            if all(key in key_folder for key in key_param):
                                target_folder.append(folder)
                                break
                        else:
                            print(
                                "No respective folder path, something's wrong."
                            )
                            raise FileNotFoundError
                            # quit()
                        key_param.pop(-1)

                    temp = np.ndarray(shape=(len(seeds), len(save_list)))
                    for i, folder in enumerate(target_folder):
                        with open(
                                os.path.join(args.output_dir, folder,
                                             'ood.csv'), 'r') as f:
                            lines = f.readlines(
                            )[save_line_dict[key_param[-1]]:]
                        for line in lines:
                            split = line.split(',')
                            for j, metric_index in enumerate(save_list):
                                temp[i][j] = split[metric_index]
                    content = ''
                    for item in np.mean(temp, axis=0):
                        content = content + '{:.2f}'.format(item) + ' / '
                    else:
                        content = content[:-3]

                    sub_line_content[key_param[0]] = content
                sub_form_content.append(sub_line_content)

            csv_path = os.path.join(args.output_dir, 'total_osr.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = order_fieldnames(list(sub_form_content[0].keys()),
                                              args)
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for sub_line_content in sub_form_content:
                    writer.writerow(sub_line_content)


def write_total(args, folder_list, save_line_dict, benchmark_dict,
                main_content_extract_dict):
    main_form_content = []
    for method in args.methods:
        main_line_content = {}
        for metric in args.metrics:
            args_list = make_args_list(args.benchmarks, [method], [metric],
                                       benchmark_dict)
            for key_param in args_list:
                main_line_content['method --> auroc'] = key_param[1]

                if metric == 'ood':
                    for folder in folder_list:
                        key_folder = folder.split('_')
                        if all(key in key_folder for key in key_param):
                            target_folder = folder
                            break
                    else:
                        print("No respective folder path, something's wrong.")
                        # quit()

                    with open(
                            os.path.join(args.output_dir, target_folder,
                                         'ood.csv'), 'r') as f:
                        lines = f.readlines()[save_line_dict[key_param[-1]]:]

                    content = ''
                    for line in lines:
                        if line.split(',')[0] in main_content_extract_dict[
                                key_param[-1]]:

                            # take auroc only
                            content = content + '{:.2f}'.format(
                                float(line.split(',')[2])) + ' / '
                    else:
                        content = content[:-3]
                    # use benchmark name as key
                    main_line_content[key_param[0]] = content

                if metric == 'osr':
                    target_folder = []
                    seeds = ['seed1', 'seed2', 'seed3', 'seed4', 'seed5']
                    for seed in seeds:
                        key_param.append(seed)
                        for folder in folder_list:
                            key_folder = folder.split('_')
                            if all(key in key_folder for key in key_param):
                                target_folder.append(folder)
                                break
                        else:
                            print(
                                "No respective folder path, something's wrong."
                            )
                            # quit()
                        key_param.pop(-1)

                    temp = np.ndarray(shape=(len(seeds), 1))
                    for i, folder in enumerate(target_folder):
                        with open(
                                os.path.join(args.output_dir, folder,
                                             'ood.csv'), 'r') as f:
                            lines = f.readlines(
                            )[save_line_dict[key_param[-1]]:]
                        for line in lines:
                            split = line.split(',')
                            temp[i] = split[2]
                    content = '{:.2f}'.format(np.mean(temp, axis=0).item())
                    main_line_content[key_param[0]] = content

        main_form_content.append(main_line_content)

    csv_path = os.path.join(args.output_dir, 'total_result.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = order_fieldnames(list(main_form_content[0].keys()), args)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for main_line_content in main_form_content:
            writer.writerow(main_line_content)


verify_dir = './results/total'
for folder in os.listdir(verify_dir):
    if os.path.isdir(os.path.join(verify_dir, folder)):
        if 'ood.csv' not in os.listdir(os.path.join(verify_dir, folder)):
            # if 'seed1' in folder.split('_'):
            print(folder)


def order_fieldnames(keys, args):

    ordered_keys = []
    ordered_keys.append(keys[0])
    for item in args.benchmarks:
        if item in keys:
            ordered_keys.append(item)

    return ordered_keys
