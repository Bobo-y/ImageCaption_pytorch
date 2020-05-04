import torch.backends.cudnn as cudnn
import torch
import math
import time
import datetime
import argparse
import os
import torch.nn as nn
from vision.model import ImageCaption
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tools.data_loader import Flickr8kData
from tools.utils import custom_collate_fn, decode_str


parser = argparse.ArgumentParser("--------Train--------")
parser.add_argument('--weights_save_folder', default='./weights', type=str, help='Dir to save weights')
parser.add_argument('--dataset_root', default='/Users/linyang/Desktop/dataset/Flicker8k/', help='Flicker8k')
parser.add_argument('--net_w', default=224, type=int)
parser.add_argument('--net_h', default=224, type=int)
parser.add_argument('--batch_size', default=8, type=int, help="batch size")
parser.add_argument('--enc_embed_dim', default=256, type=int, help="encoder embedding dim")
parser.add_argument('--dec_embed_dim', default=256, type=int, help="decoder embedding dim")
parser.add_argument('--hidden', default=256, type=int, help="GRU hidden units")
parser.add_argument('--drop_out', default=0.5, type=int, help="drop_out p")
parser.add_argument('--max_epoch', default=50, type=int, help="max training epoch")
parser.add_argument('--initial_lr', default=1e-3, type=float, help="initial learning rate")
parser.add_argument('--gamma', default=0.1, type=float, help="gamma for adjust lr")
parser.add_argument('--weight_decay', default=5e-4, type=float, help="weights decay")
parser.add_argument('--num_workers', default=0, type=int, help="numbers of workers")
parser.add_argument('--num_gpu', default=0, type=int, help="gpu number")
parser.add_argument('--pre_train', default=True, type=bool, help="whether use pre-train weights")
args = parser.parse_args()


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            imgs, targs = batch
            output = model(imgs, targs, 0)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            targs = targs[1:].view(-1)

            loss = criterion(output, targs)

            epoch_loss += loss.item()

    print("evaluate loss: {}" .format(epoch_loss / len(iterator)))


def train(net, optimizer, dataset, criterion, use_gpu):

    net.train()
    epoch = 0
    epoch_size = math.ceil(len(dataset) / args.batch_size)
    max_iter = args.max_epoch * epoch_size

    start_iter = 0
    CLIP = 1
    print("Total training images number: {}".format(len(dataset)))
    print("Begin training...")
    for iteration in range(start_iter, max_iter):

        if iteration % epoch_size == 0:
            epoch += 1

            train_iterator = iter(DataLoader(dataset, args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             collate_fn=custom_collate_fn))
            if epoch % 10 == 0 and epoch > 0:
                if args.num_gpu > 1:
                    torch.save(net.module.state_dict(), os.path.join(args.weights_save_folder, 'epoch_' + str(epoch) + '.pth'))
                else:
                    torch.save(net.state_dict(), os.path.join(args.weights_save_folder, 'epoch_' + str(epoch) + '.pth'))

                # dataset.train = False
                # val_iterator = iter(DataLoader(dataset, args.batch_size, num_workers=args.num_workers, collate_fn=custom_collate_fn))
                # evaluate(net, val_iterator, criterion)
                # net.train()
                # dataset.train = True

        load_t0 = time.time()
        images, labels = next(train_iterator)
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        output = net(images, labels)
        out_str = decode_str(list(output[:, 0, :].argmax(dim=1).numpy()), dataset.itos)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        lab_str = decode_str(list(labels[:, 0].numpy()), dataset.itos)

        print("lab_str: {}, \n out_str:{}".format(lab_str, out_str))
        labels = labels[1:].view(-1)
        loss = criterion(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)

        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loss: {:.4f}|| Batchtime: {:.4f} s || ETA: {}'.format
              (epoch, args.max_epoch, (iteration % epoch_size) + 1, epoch_size, iteration + 1, max_iter, loss,
               batch_time, str(datetime.timedelta(seconds=eta))))
        iteration += 1
    if args.num_gpu > 1:
        torch.save(net.module.state_dict(), os.path.join(args.weights_save_folder, 'Final.pth'))
    else:
        torch.save(net.state_dict(), os.path.join(args.weights_save_folder, 'Final.pth'))
    print('Finished Training')


if __name__ == '__main__':
    cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset = Flickr8kData(root=args.dataset_root, net_w=args.net_w, net_h=args.net_h, train=True, transform=transform)
    net = ImageCaption(vocab_size=dataset.vocab_size,
                       enc_embed_dim=args.enc_embed_dim,
                       dec_embed_dim=args.dec_embed_dim,
                       hidden=args.hidden,
                       drop=args.drop_out,
                       device=device)

    if args.pre_train:
        print("loading pretrained weights!!!")
        pretrained_dict = torch.load(os.path.join(args.weights_save_folder, "Final.pth"))
        net.load_state_dict(pretrained_dict)

    if args.num_gpu > 1 and use_gpu:
        net = torch.nn.DataParallel(net).cuda()
    elif use_gpu:
        net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token)
    train(net, optimizer, dataset, criterion, use_gpu)
