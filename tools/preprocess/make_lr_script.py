import os
import argparse
import pandas as pd

def make_lr_script(args):
    df = pd.read_csv(args.input_file)
    df = df[['dataset_name', 'model_name', 'val_loss', 'train_acc', 'lr', 'base_lr']]

    # dataset and model names
    dataset_list = df['dataset_name'].unique()
    model_list = df['model_name'].unique()

    # create file with specified file name
    output_file = os.path.join(args.results_dir, f'{args.output_file}.sh')
    f = open(output_file, "w")

    #for loop for each dataset
    for dataset in dataset_list:
        # write dataset name
        f.write(f'# {dataset}\n')

        # for loop for each model
        for model in model_list:

            # filter the subset of the dataset based on the dataset and the model
            df_subset = df[(df['model_name'] == model) & (df['dataset_name'] == dataset)]
            
            # sort subset based on val loss (lowest to highest)
            df_subset_sorted = df_subset.sort_values(by=['val_loss'], ascending=True)

            # get index and train acc corresponding to best val loss (0)
            best_idx = 0
            best_acc = df_subset_sorted['train_acc'].iloc[best_idx]

            while best_acc < 50 and best_idx < (len(df_subset) - 1):
                # increment index of next best val loss
                best_idx += 1
                # get val loss and train acc corresponding to next best
                best_acc = df_subset_sorted['train_acc'].iloc[best_idx]
                # print when best_acc < 50 (mostly cotton/soy/ultra fine-grained datasets)
                print(dataset, model, best_idx, best_acc)
            
            # if none of the accuracies surpass 50 then just return the highest accuracies
            if best_acc < 50:
                # sort subset based on train acc (highest to lowest)
                df_subset_sorted = df_subset.sort_values(by=['train_acc'], ascending=False) 
                print(df_subset_sorted)
                lr = df_subset_sorted[args.lr_var].iloc[0]

            # returns lr corresponding to best val loss
            else:
                lr = df_subset_sorted[args.lr_var].iloc[best_idx]
            
            # check the current subset and the selected LR
            # print(dataset, model, lr, df_subset.head())

            #write to file
            line = f'{args.prefix} --dataset_name {dataset} --lr {lr} --model_name "{model}{args.suffix}"\n'
            f.write(line)
        f.write('\n')

    f.close()

    return 0


def parse_args():

    parser = argparse.ArgumentParser()

    #parser arguments
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('data', 'stage1_cal_backbones.csv'),
                        help='filename for input .csv file')

    parser.add_argument('--lr_var', type=str, default='lr',
                        help='for inat/dafb use --lr_var base_lr')
    parser.add_argument('--prefix', type=str,
                        default='./scripts/run.sh --serial 1 --run_seed',
                        help='prefix for the file in each line')
    parser.add_argument('--suffix', type=str,
                        default=' --selector cal --project_name CALBackbones --cpu_workers 8',
                        help='suffix for the file in each line')
    parser.add_argument('--output_file', type=str, default='lr_script',
                        help='output file name')
    parser.add_argument('--results_dir', default='results', type=str,
                        help='The directory where results will be stored')

    args= parser.parse_args()

    print(args)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    return args


def main():
    args = parse_args()

    make_lr_script(args)

    return 0


if __name__ == '__main__':
    main()
