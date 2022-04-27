from cgi import print_form
from turtle import shape
from cv2 import split
from sklearn.random_projection import SparseRandomProjection
from openood.postprocessors.sampling_methods.kcenter_greedy import kCenterGreedy
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import faiss
import torch
import cv2
from tqdm import tqdm
from torch import nn
import os
from sklearn.random_projection import SparseRandomProjection
from scipy.ndimage import gaussian_filter
from .base_postprocessor import BasePostprocessor
from typing import Any

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

class PatchcorePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(PatchcorePostprocessor, self).__init__(config)
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.n_neighbors = config.postprocessor.postprocessor_args.n_neighbors
        self.feature_mean, self.feature_prec = None, None
        self.alpha_list = None
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []   
        self.features = []
    
    def setup(self, net: nn.Module, id_loader_dict):
        # step 1:
        self.model = net
        #imagenet
        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])
        # on train start
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_list = []

        # load index
        self.index = faiss.read_index(os.path.join('./results/','index.faiss'))
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0 ,self.index)
        self.init_results_list()


        # #training step
        # train_dataiter = iter(id_loader_dict['patch'])

        # for train_step in tqdm(range(1,
        #                              len(train_dataiter) + 1),
        #                        position=0,
        #                        leave=True):
        #     batch = next(train_dataiter)
        #     target = batch['label'].cuda()
        #     x = batch['data'].cuda()
        #     features = self.model.forward(x, return_feature=True)
        #     embeddings = []
        #     for feature in features:
        #         m = torch.nn.AvgPool2d(3, 1, 1)
        #         embeddings.append(m(feature))
        #     embedding = embedding_concat(embeddings[0], embeddings[1])
        #     self.embedding_list.extend(reshape_embedding(np.array(embedding)))
            
        # #training end
        # total_embeddings = np.array(self.embedding_list)
        
        # # Random projection
        # print("Random projection")
        # self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        # self.randomprojector.fit(total_embeddings)
        # # Coreset Subsampling
        # print("Coreset Subsampling")
        # selector = kCenterGreedy(total_embeddings,0,0)
        # selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*self.postprocessor_args.coreset_sampling_ratio))
        # self.embedding_coreset = total_embeddings[selected_idx]
        
        # print('initial embedding size : ', total_embeddings.shape)
        # print('final embedding size : ', self.embedding_coreset.shape)
        # #faiss
        # print("faiss indexing")
        # self.index = faiss.IndexFlatL2(self.embedding_coreset.shape[1])
        # self.index.add(self.embedding_coreset) 
        # faiss.write_index(self.index,  os.path.join('./results/','index.faiss'))


    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []    


    def postprocess(self, net: nn.Module, data):
        
        self.init_results_list()
        score_patch=[]
        # extract embedding
        for x in data.split(1,dim=0):
            features = self.model.forward(x, return_feature = True)
            embeddings = []
            for feature in features:
                m = torch.nn.AvgPool2d(3, 1, 1)
                embeddings.append(m(feature))
            embedding_ = embedding_concat(embeddings[0], embeddings[1])
            embedding_test = np.array(reshape_embedding(np.array(embedding_)))
            score_patches, _ = self.index.search(embedding_test , k=self.n_neighbors)

            score_patch.append(score_patches)

            anomaly_map = score_patches[:,0].reshape((28,28))
            N_b = score_patches[np.argmax(score_patches[:,0])]
            w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
            score = w*max(score_patches[:,0]) # Image-level score

            anomaly_map_resized = cv2.resize(anomaly_map, (self.config.dataset.image_size, self.config.dataset.image_size))
            anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
            self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
            self.pred_list_img_lvl.append(score)
        
        

        pred = []
        for i in self.pred_list_img_lvl:
            # 6.3 is the value that 
            if(i>6.3): 
                pred.append(torch.tensor(1))
            else:
                pred.append(torch.tensor(-1))
        conf = []
        for i in score_patch:
            conf.append(i)
        conf = torch.tensor(conf, dtype = torch.float32)
        conf = conf.cuda()
        
        return pred, conf