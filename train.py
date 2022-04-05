import argparse
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import videotransforms
from dataset import IVBSSDataset, collate_fn
from model import TemporalActionLocalization
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_video_dir', type=str, help='face video directory')
    parser.add_argument('--cabin_video_dir', type=str, help='cabin video directory')
    parser.add_argument('--train_data_path', type=str, help='path to the training data')
    parser.add_argument('--val_data_path', type=str, help='path to the validation data')
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--val_batch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--display_steps', default=25, type=int)
    parser.add_argument('--ckp_dir', type=str, help='checkpoint directory')
    parser.add_argument('--ckp_path', type=str, help='path to the loaded checkpoint')
    parser.add_argument('--pretrained_I3D_model', type=str, help='path to the pretrained I3D model')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    args = parser.parse_args()
    return args


def save_ckp(checkpoint, ckp_dir, ckp_name):
    ckp_path = os.path.join(ckp_dir, ckp_name)
    torch.save(checkpoint, ckp_path)


def load_ckp(ckp_path, model, optimizer, scheduler):
    checkpoint = torch.load(ckp_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    return start_epoch, model, optimizer, scheduler


def train():
    args = get_parse()
    face_video_dir = args.face_video_dir
    cabin_video_dir = args.cabin_video_dir
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    display_steps = args.display_steps
    ckp_dir = args.ckp_dir
    pretrained_I3D_model = args.pretrained_I3D_model
    num_classes = args.num_classes
    
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
        
    print('Start to load data')
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           videotransforms.ToTensor()
                                           ])
    val_transforms = transforms.Compose([videotransforms.CenterCrop(224),
                                         videotransforms.ToTensor()
                                         ])
    train_dataset = IVBSSDataset(face_video_dir,
                                 cabin_video_dir,
                                 train_data_path,
                                 train_transforms
                                 )
    val_dataset = IVBSSDataset(face_video_dir,
                               cabin_video_dir,
                               val_data_path,
                               val_transforms
                               )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  sampler=RandomSampler(train_dataset, replacement=True),
                                  collate_fn=collate_fn,
                                  drop_last=True
                                  )
    total_steps = num_epochs * len(train_dataloader)
    print('Total number of training samples is {0}'.format(len(train_dataset)))
    print('Total number of validation samples is {0}'.format(len(val_dataset)))
    print('Total number of training steps is {0}'.format(total_steps))

    model = TemporalActionLocalization(num_classes, pretrained_I3D_model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    start_epoch = 0
    
    if args.ckp_path is not None:
        print('Load checkpoint')
        start_epoch, model, optimizer, scheduler = load_ckp(args.ckp_path, model, optimizer, scheduler)
 
    model.cuda()
    model.train()

    print('Start to train')
    num_step = 0
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        class_running_loss = 0.0
        for i, (face_imgs, cabin_imgs, labels) in enumerate(train_dataloader):
            face_imgs = face_imgs.cuda()
            cabin_imgs = cabin_imgs.cuda()
            for k, v in labels.items():
                labels[k] = v.cuda()
            optimizer.zero_grad()
            loss, class_loss = model(face_imgs, cabin_imgs, labels)[:2]           
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            class_running_loss += class_loss.item()
            if (i + 1) % display_steps == 0:
                print('epoch:{0}/{1}, step:{2}/{3}, loss:{4:.4f}, class_loss:{5:.4f}'.format
                      (epoch + 1, num_epochs, i + 1, len(train_dataloader), running_loss/display_steps, class_running_loss/display_steps))
                running_loss = 0.0
                class_running_loss = 0.0
            num_step += 1
            writer.add_scalar('Loss/train', loss.item(), num_step)
            writer.add_scalar('Class_Loss/train', class_loss.item(), num_step)
        
        scheduler.step()
        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        ckp_name = 'epoch_'+str(epoch+1)+'.pt'
        save_ckp(checkpoint, ckp_dir, ckp_name)
        print('Save the checkpoint after {} epochs'.format(epoch+1))
        eval_loss, eval_class_loss, class_accuracy = eval(val_dataset, val_batch_size, model)
        writer.add_scalar('Loss/valid', eval_loss, epoch)
        writer.add_scalar('Class_Loss/valid', eval_class_loss, epoch)
        writer.add_scalar('Accuracy/valid', class_accuracy, epoch)
        
        print('Loss on validation dataset: {0:.4f}, class accuracy on validation dataset: {1:.4f}'.format(eval_loss, class_accuracy))
    writer.close()
    

def eval(val_dataset, batch_size, model):
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                sampler=SequentialSampler(val_dataset),
                                collate_fn=collate_fn,
                                drop_last=True
                                )
    model.eval()
    eval_loss = 0.0
    eval_class_loss = 0.0
    class_accuracy = 0.0
    eval_steps = 0
    for i, (face_imgs, cabin_imgs, labels) in enumerate(val_dataloader):
        face_imgs = face_imgs.cuda()
        cabin_imgs = cabin_imgs.cuda()
        for k, v in labels.items():
            labels[k] = v.cuda()
        with torch.no_grad():
            total_loss, class_loss, class_scores = model(face_imgs, cabin_imgs, labels)[:3]
            class_labels = labels['event_id']
            eval_loss += total_loss.item()
            eval_class_loss += class_loss.item()
            class_pred = torch.argmax(class_scores, dim=1)
            class_accuracy += torch.sum((class_pred==class_labels).float())/class_labels.shape[0]
            eval_steps += 1
    avg_eval_loss = eval_loss/eval_steps
    avg_eval_class_loss = eval_class_loss/eval_steps
    avg_class_accuracy = class_accuracy/eval_steps
    return avg_eval_loss, avg_eval_class_loss, avg_class_accuracy


if __name__ == '__main__':
    train()

