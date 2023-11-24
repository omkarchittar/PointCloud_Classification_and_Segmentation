import numpy as np
import argparse

import torch
from models import seg_model
from utils import create_dir, viz_seg
from data_loader import get_data_loader


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')  # model_epoch_0
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output_seg_numpoints')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="seg", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model().to(args.device)

    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("successfully loaded checkpoint from {}".format(model_path))

    index = [20, 168, 96]

    for j in index:
        n_points = [10, 50, 100, 500, 1000, 10000]

        for n in n_points:
            # Sample Points per Object
            ind = np.random.choice(10000, n, replace=False)

            # ------ TO DO: Make Prediction ------
            test_dataloader = get_data_loader(args=args, train=False)

            correct_point = 0
            num_point = 0
            predictions = []
            for batch in test_dataloader:
                point_clouds, labels = batch
                point_clouds = point_clouds[:, ind].to(args.device)
                labels = labels[:, ind].to(args.device).to(torch.long)

                with torch.no_grad():
                    pred_labels = torch.argmax(model(point_clouds), dim=-1, keepdim=False)
                correct_point += pred_labels.eq(labels.data).cpu().sum().item()
                num_point += labels.view([-1, 1]).size()[0]

                predictions.append(pred_labels)

            test_accuracy = correct_point / num_point
            print(f"test accuracy for num of points {n} : {test_accuracy}")
            predictions = torch.cat(predictions).detach().cpu()

            verts = test_dataloader.dataset.data[j, ind].detach().cpu()
            labels = test_dataloader.dataset.label[j, ind].to(torch.long).detach().cpu()

            viz_seg(verts, labels, "{}/3.2. vis_{}_numpoints_{}_gt_{}.gif".format(args.output_dir, j, n, args.exp_name), args.device)
            viz_seg(verts, predictions[j], "{}/3.2. vis_{}_numpoints_{}_pred_{}_acc_{}.gif".format(args.output_dir, j, n, args.exp_name, test_accuracy), args.device)
