import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_cls
from data_loader import get_data_loader

import pytorch3d

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model') #model_epoch_0
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output_cls_rotated')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--task', type=str, default="cls", help='The task: cls or seg')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)

    # ------ TO DO: Make Prediction ------
    test_dataloader = get_data_loader(args=args, train=False)

    index = [94, 702, 870]

    for j in index:
        for theta in range(0, 91, 10):
            # Rotation
            rot = torch.tensor([theta,0,0])
            R = pytorch3d.transforms.euler_angles_to_matrix(rot, 'XYZ')
            test_dataloader.dataset.data = (R @ test_dataloader.dataset.data.transpose(1, 2)).transpose(1, 2)
            rad = torch.Tensor([theta * np.pi / 180.])[0]

            # # rotation around x-axis
            # R_x = torch.Tensor([[1, 0, 0],
            #                     [0, torch.cos(rad), - torch.sin(rad)],
            #                     [0, torch.sin(rad), torch.cos(rad)]])
            R_x = torch.Tensor([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
            
            # rotation around y-axis
            R_y = torch.Tensor([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
            # R_y = torch.Tensor([[torch.cos(rad), 0, torch.sin(rad)],
            #                     [0, 1, 0],
            #                     [- torch.sin(rad), 0, torch.cos(rad)]])

            # # rotation around z-axis
            # R_z = torch.Tensor([[1, 0, 0],
            #                     [0, 1, 0],
            #                     [0, 0, 1]])
            R_z = torch.Tensor([[torch.cos(rad), - torch.sin(rad), 0],
                                [torch.sin(rad), torch.cos(rad), 0],
                                [0, 0, 1]])

            test_dataloader.dataset.data = ((R_x @ R_y @ R_z) @ test_dataloader.dataset.data.transpose(1, 2)).transpose(1, 2)


            correct_obj = 0
            num_obj = 0
            predictions = []
            for batch in test_dataloader:
                point_clouds, labels = batch
                point_clouds = point_clouds[:, ind].to(args.device)
                labels = labels.to(args.device).to(torch.long)

                with torch.no_grad():
                    pred_labels = torch.argmax(model(point_clouds), dim=-1, keepdim=False)
                correct_obj += pred_labels.eq(labels.data).cpu().sum().item()
                num_obj += labels.size()[0]

                predictions.append(pred_labels)

                accuracy = correct_obj / num_obj
            print(f"test accuracy for angle {theta} : {accuracy}")
            predictions = torch.cat(predictions).detach().cpu()

            verts = test_dataloader.dataset.data[j, ind]       # change j to args.i for a particular index visualization
            gt_cls = test_dataloader.dataset.label[j].to(torch.long).detach().cpu().data
            pred_cls = predictions[j].detach().cpu().data

            path = f"output_cls_rotated/3.1. vis_{j}_angle_{theta}_with_gt_{gt_cls}_pred_{pred_cls}_acc_{accuracy}.gif"
            viz_cls(verts, path, "cuda")