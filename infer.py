import torch.backends.cudnn as cudnn
import torch
import time
import argparse
import os
import cv2
from vision.model import ImageCaption
from tools.data_loader import Flickr8kData
from torchvision.transforms import transforms
from tools.utils import process_img, decode_str


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--dataset_root', default='/Users/linyang/Desktop/dataset/Flicker8k/', help='Flicker8k')
parser.add_argument('--save_folder', default='./result', type=str, help='Dir to save img')
parser.add_argument('--cpu', default=True, help='Use cpu inference')
parser.add_argument('--enc_embed_dim', default=256, type=int, help="encoder embedding dim")
parser.add_argument('--dec_embed_dim', default=256, type=int, help="decoder embedding dim")
parser.add_argument('--hidden', default=256, type=int, help="GRU hidden units")
parser.add_argument('--drop_out', default=0.5, type=int, help="drop_out p")
parser.add_argument('--net_w', default=224, type=int)
parser.add_argument('--net_h', default=224, type=int)
parser.add_argument('--input_path', default='2513260012_03d33305cf.jpg', type=str, help="image or images dir")
args = parser.parse_args()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    net_w = args.net_w
    net_h = args.net_h

    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    cpu = args.cpu
    device = torch.device("cpu" if cpu else "cuda")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    dataset = Flickr8kData(root=args.dataset_root, net_w=args.net_w, net_h=args.net_h, train=False, transform=transform)

    net = ImageCaption(vocab_size=dataset.vocab_size,
                       enc_embed_dim=args.enc_embed_dim,
                       dec_embed_dim=args.dec_embed_dim,
                       hidden=args.hidden,
                       drop=args.drop_out,
                       device=device)

    net.load_state_dict(torch.load(args.trained_model, map_location=torch.device(device)))
    net.eval()

    cudnn.benchmark = True
    net = net.to(device)

    input_path = args.input_path
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

    for img_path in image_paths:
        begin = time.time()
        print("Process {}".format(img_path))
        image = cv2.imread(img_path)
        img = process_img(img=image, height=net_h, width=net_w, transform=transform)
        output = net.infer(img, max_seq_len=20, sos_token=dataset.sos_token, eos_token=dataset.eos_token)
        str_re = decode_str(output, dataset.itos)
        cv2.putText(image, str_re, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0))
        cv2.imwrite(os.path.join(save_folder, img_path.split('/')[-1]), image)
        end = time.time()
        print("per image tiem: {}".format(end - begin))

    print("Done!!!")