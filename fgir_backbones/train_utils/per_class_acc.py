import os
import statistics

import pandas as pd
import torch

from .save_vis_images import vis_images


def calc_per_class_acc(args, curr_img, output, targets,
                       class_correct, class_total, images, dic_preds):
    _, predicted = torch.max(output.data, 1)
    c = (predicted == targets)
    for i, target in enumerate(targets):
        class_correct[target] += c[i].item()
        class_total[target] += 1

        prob = torch.softmax(output, -1)[i, predicted[i]].item() * 100

        dic_preds.update({curr_img: {
            'class_id': target.item(), 'pred_id': predicted[i].item(), 'prob': round(prob, 3)}})

        # c[i].item() is True is correct if not then means False == wrong pred
        if args.vis_errors and not c[i].item():
            title = f'Current image: {curr_img}\
                        Prediction: {predicted[i].item()} ({prob:.2f}%)\
                        Correct: {target.item()}'
            print(title)
            vis_images(args, curr_img, images, title=title)

        curr_img += 1

    return curr_img

def calc_class_deviation(args, class_correct, class_total, dic_preds):
    # dic to hold class correct and class totals for further analysis
    dic_correct_total = {}

    # per class accuracy
    per_class_accuracy = []
    for i in range(args.num_classes):
        correct = class_correct[i]
        total = class_total[i]

        dic_correct_total.update({i: {'correct': correct, 'total': total}})
        per_class_accuracy.append(100 * correct / total)

    df_per_class = pd.DataFrame.from_dict(dic_correct_total, orient='index')
    df_per_class['class_id'] = df_per_class.index
    save_fp = os.path.join(args.results_dir, args.per_class_acc_results)
    df_per_class.to_csv(save_fp, sep=',', header=True, index=False,
                        columns=['class_id', 'correct', 'total'])

    df_preds = pd.DataFrame.from_dict(dic_preds, orient='index')
    df_preds['image_id'] = df_preds.index
    save_fp = os.path.join(args.results_dir, args.ind_preds_results)
    df_preds.to_csv(save_fp, sep=',', header=True, index=False,
                    columns=['image_id', 'class_id', 'pred_id', 'prob'])

    class_mean = round(statistics.mean(per_class_accuracy), 2)
    class_deviation = round(statistics.stdev(per_class_accuracy), 2)
    print(f'Per-class mean accuracy: {class_mean}%\nClass deviation: {class_deviation}%')
    return class_deviation
