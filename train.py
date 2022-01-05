import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from nets.yolo4 import APAN
from nets.yolo_training import LossHistory, YOLOLoss, weights_init
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.dataloader import write_classes
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val')]

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

        
def fit_one_epoch(net,yolo_loss,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    if args.Tensorboard:
        global train_tensorboard_step, val_tensorboard_step
    total_loss = 0
    val_loss = 0

    net.train()
    print('Start Train')
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

            optimizer.zero_grad()

            outputs = net(images)
            losses = []
            num_pos_all = 0

            for i in range(4):
                loss_item, num_pos = yolo_loss(outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            total_loss += loss.item()


            loss.backward()
            optimizer.step()

            if args.Tensorboard:
                writer.add_scalar('Train_loss', loss, train_tensorboard_step)
                train_tensorboard_step += 1

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    # if Tensorboard:
    #     writer.add_scalar('Train_loss', total_loss/(iteration+1), epoch)
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                else:
                    images_val  = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                optimizer.zero_grad()

                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(4):
                    loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()

            # if Tensorboard:
            #     writer.add_scalar('Val_loss', loss, val_tensorboard_step)
            #     val_tensorboard_step += 1
            pbar.set_postfix(**{'Total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            

    if args.Tensorboard:
        writer.add_scalar('Val_loss',val_loss / (epoch_size_val+1), epoch)
    loss_history.append_loss(total_loss/(epoch_size+1), val_loss/(epoch_size_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Total_Loss%.4f-Val_Loss%.4f.pth'%(total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--Tensorboard',default=False, help="Opening tensorboard", action="store_true")
    parser.add_argument('--Cuda',default=True, help="Using CUDA", action="store_true")
    parser.add_argument('--mosaic',default=False, help="Using mosaic data augment", action="store_true")
    parser.add_argument('--Cosine_lr',default=False, help="Using annealing cosine strategy", action="store_true")
    parser.add_argument('--smoooth_label',default=0, help="Using smoooth label", action="store_true")

    parser.add_argument('--classes_path',default='model_data/classes.txt', action="store_true")
    parser.add_argument('--model_path',default='logs\Epoch102-Total_Loss10.9053-Val_Loss15.2265.pth', action="store_true")

    parser.add_argument('--lr_f',default=1e-3, help="the learning rate is in epochs which freeze backbone network", action="store_true")
    parser.add_argument('--lr_u',default=1e-4, help="the learning rate is in epochs which unfreeze backbone network", action="store_true")
    parser.add_argument('--batch_size',default=8, help="Batchsize", action="store_true")
    parser.add_argument('--Init_Epoch',default=0, help="the initial epoch", action="store_true")
    parser.add_argument('--Freeze_Epoch',default=0, help="the epochs which freeze backbone network", action="store_true")
    parser.add_argument('--Unfreeze_Epoch',default=150, help="the epochs which freeze backbone network", action="store_true")
    
    args = parser.parse_args()
    input_shape = (416,416)
    anchors_path = 'model_data/yolo_anchors.txt'
    

    class_names = get_classes(args.classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)

    wd = getcwd()
    write_classes(sets,class_names,wd)



    model = APAN(len(anchors[0]), num_classes)
    weights_init(model)
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_path != '':
        model_path = args.model_path
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    
    print('Finished!')

    net = model.train()

    if args.Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    yolo_loss    = YOLOLoss(np.reshape(anchors,[-1,2]), num_classes, (input_shape[1], input_shape[0]), args.smoooth_label, args.Cuda, normalize =False)
    loss_history = LossHistory("logs/")


    annotation_path = '2007_train.txt'

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    if args.Tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir='logs',flush_secs=60)
        if args.Cuda:
            graph_inputs = torch.randn(1,3,input_shape[0],input_shape[1]).type(torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.randn(1,3,input_shape[0],input_shape[1]).type(torch.FloatTensor)
        writer.add_graph(model, graph_inputs)
        train_tensorboard_step  = 1
        val_tensorboard_step    = 1

 
    if True:
        lr_f            = args.lr_f
        Batch_size      = args.batch_size
        Init_Epoch      = args.Init_Epoch
        Freeze_Epoch    = args.Freeze_Epoch

        optimizer       = optim.Adam(net.parameters(),lr_f)
        if args.Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=args.mosaic, is_train=True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size
        
        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(net,yolo_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,args.Cuda)
            lr_scheduler.step()

    if True:
        lr_u            = args.lr_f
        Batch_size      = args.batch_size
        Freeze_Epoch    = args.Freeze_Epoch
        Unfreeze_Epoch  = args.Unfreeze_Epoch

        optimizer       = optim.Adam(net.parameters(),lr_u)
        if args.Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset   = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=args.mosaic, is_train=True)
        val_dataset     = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size      = num_train // Batch_size
        epoch_size_val  = num_val // Batch_size
        
        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_one_epoch(net,yolo_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,args.Cuda)
            lr_scheduler.step()
