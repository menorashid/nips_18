from train_test_caps import *
from torchvision import datasets, transforms
import models

import os
from helpers import util,visualize,augmenters
import random
import dataset
import numpy as np
import torch
# from analysis import getting_accuracy
from helpers import util,visualize,augmenters


def test_dropout(wdecay,lr,route_iter,model_name='dynamic_capsules',epoch_stuff=[30,60], reconstruct = False, loss_weights = None, exp = False, model_to_test = None, res = False, dropout = 0.5):

    

    out_dirs = []
    out_dir_meta = '../experiments/'+model_name+'_'+str(route_iter)
    num_epochs = epoch_stuff[1]
    if model_to_test is None:
        model_to_test = num_epochs -1

    epoch_start = 0
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr
    
    criterion = 'margin'
    criterion_str = criterion
    n_classes = 10
    save_after = 10
    init = False
    pre_pend = 'mnist'
    strs_append_list = ['reconstruct',reconstruct,'shift',criterion_str,init,'wdecay',wdecay,num_epochs]+dec_after+lr+[dropout]
    if loss_weights is not None:
        strs_append_list = strs_append_list+['lossweights']+loss_weights
    strs_append = '_'+'_'.join([str(val) for val in strs_append_list])
    

    out_dir_train =  os.path.join(out_dir_meta,pre_pend+strs_append)
    final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
    print out_dir_train
    
    if os.path.exists(final_model_file):
        print 'skipping',final_model_file
        raw_input()
    
    model_file = None
    batch_size = 256
    batch_size_val = 256
    num_workers = 0

    data_transforms = {}
    data_transforms['train']= transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_transforms['val']= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = dataset.get('mnist',dict(dir_data = '../data/mnist_downloaded', train=True, transform = data_transforms['train']))
    test_data = dataset.get('mnist',dict(dir_data = '../data/mnist_downloaded', train=False, transform = data_transforms['val']))

    train_dataloader = torch.utils.data.DataLoader(train_data, 
                        batch_size=batch_size,
                        shuffle=True, 
                        num_workers=num_workers)
    
    test_dataloader = torch.utils.data.DataLoader(test_data, 
                        batch_size=batch_size_val,
                        shuffle=False, 
                        num_workers=num_workers)

    network_params = dict(n_classes=n_classes, 
                        r=route_iter,
                        init=init,
                        reconstruct = reconstruct,
                        loss_weights = loss_weights,
                        dropout = dropout)
    
    util.makedirs(out_dir_train)
        
    train_params = dict(out_dir_train = out_dir_train,
                train_data = train_data,
                test_data = test_data,
                batch_size = batch_size,
                batch_size_val = batch_size_val,
                num_epochs = num_epochs,
                save_after = save_after,
                disp_after = 1,
                plot_after = 100,
                test_after = 1,
                lr = lr,
                dec_after = dec_after, 
                model_name = model_name,
                criterion = criterion,
                gpu_id = 0,
                num_workers = 0,
                model_file = model_file,
                epoch_start = epoch_start,
                network_params = network_params,
                weight_decay=wdecay)
    test_params = dict(out_dir_train = out_dir_train,
                model_num = model_to_test,
                train_data = train_data,
                test_data = test_data,
                gpu_id = 0,
                model_name = model_name,
                batch_size_val = batch_size_val,
                criterion = criterion,
                network_params = network_params)
    
    print train_params
    param_file = os.path.join(out_dir_train,'params.txt')
    all_lines = []
    for k in train_params.keys():
        str_print = '%s: %s' % (k,train_params[k])
        print str_print
        all_lines.append(str_print)
    util.writeFile(param_file,all_lines)

    train_model_recon(**train_params)
    # test_model_recon(**test_params)   
    # util.makedirs(out_dir_train)
    
    # network = models.get(model_name,network_params)
    # train_dataloader = torch.utils.data.DataLoader(train_data, 
    #                     batch_size=batch_size,
    #                     shuffle=True, 
    #                     num_workers=num_workers)
    
    # test_dataloader = torch.utils.data.DataLoader(test_data, 
    #                     batch_size=batch_size_val,
    #                     shuffle=False, 
    #                     num_workers=num_workers)
    # gpu_id = 0    
    # torch.cuda.device(gpu_id)
    # model = network.model
    # model = model.cuda()
    # model.train(True)
    # # model.eval()
    # print model
    # optimizer = torch.optim.Adam(network.get_lr_list(lr),weight_decay=wdecay)

    # for num_iter_train,batch in enumerate(train_dataloader):

    #         data = Variable(batch['image'].cuda())
            
    #         if criterion=='marginmulti':
    #             labels = Variable(batch['label'].float().cuda())
    #         else:
    #             labels = Variable(torch.LongTensor(batch['label']).cuda())
    #         optimizer.zero_grad()

    #         output = model(data,labels)
    #         # x = model.features(data)
    #         # output, routes = model.caps.forward_intrusive(x)

    #         print output.size()

    #         break
    #         # loss, margin_loss, recon_loss = model.margin_loss(model(data,labels), labels) 





def main():
    epoch_stuff = [936,500]
    reconstruct = False
    exp = True
    model_name = 'dynamic_capsules_with_dropout'
    lr = [0.001,0.001,0.001]
    route_iter = 3
    dropout = 0.5
    test_dropout(wdecay =0, lr = lr ,route_iter=   route_iter,model_name=model_name,epoch_stuff=epoch_stuff, reconstruct = reconstruct,  exp = exp, dropout = dropout)


if __name__=='__main__':
    main()