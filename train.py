import argparse
import torch
from tqdm import tqdm
import os
from pct_cls import *
from dataloader import get_train_and_test_loaders


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    train_ds, train_dl, test_ds, test_dl, n_classes\
        = get_train_and_test_loaders(args.batch_size, args.num_workers, True)
    print('data loading done')

    if args.model == 'npct':
        model = NPCT(args.d_i, args.d_e, args.d_a, n_classes, args.p)
    elif args.model == 'spct':
        model = SPCT(args.d_i, args.d_e, args.d_a, n_classes, args.p)
    else:
        pass
        # TODO: Implement full PCT
        #model = PCT(args.d_i, args.d_e, args.d_a, n_classes, args.p)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.beta1, args.beta2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    train_losses = []
    valid_losses = []

    for epoch in range(args.epochs):
        train_loss = []
        valid_loss = []
        pbar = tqdm(train_dl)

        model.train()
        for x, y in pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            pbar.set_description(f'E:{epoch + 1:2d} L:{loss.item():.4f}, {torch.isfinite(loss)}')

        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = F.cross_entropy(pred, y)
                valid_loss.append(loss.item())

            valid_loss = sum(valid_loss) / len(valid_loss)
            print(f'validation loss: {valid_loss:.4f}')
            valid_losses.append(valid_loss)

        if (epoch + 1) % 10 == 0:
            # Save the model.
            model_file = os.path.join(
                args.out_dir, f'{args.model}_{epoch+1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses
            }, model_file)
            print("Saved '{}'.".format(model_file))

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loading workers')

    parser.add_argument('--task', type=str, choices=['cls', 'seg'], default='cls',
                        help='classification or segmentation task')
    parser.add_argument('--model', type=str, choices=['npct', 'spct', 'pct'], default='spct',
                        help='naivePCT, simplePCT, (full)PCT')

    parser.add_argument('--d_i', type=int, default=3, help='dim of input')
    parser.add_argument('--d_e', type=int, default=128, help='dim of embedding')
    parser.add_argument('--d_a', type=int, default=32, help='d_a = d_e / 4')
    parser.add_argument('--p', type=float, default=0.5, help='dropout ratio')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta 1')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta 2')
    parser.add_argument('--step_size', type=int, default=20, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

    parser.add_argument('--in_data_file', type=str,
                        default='data/ModelNet/modelnet_classification.h5',
                        help="data directory")
    parser.add_argument('--out_dir', type=str,
                        default='checkpoint', help='checkpoint directory')
    args = parser.parse_args()

    main(args)

