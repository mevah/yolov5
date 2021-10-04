import h5py
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
filename = '/cluster/home/himeva/fastmri-plus/Annotations/brain.csv'
file_list = '/cluster/home/himeva/fastmri-plus/Annotations/brain_file_list.csv'
anns= pd.read_csv(filename)
flist= np.array(pd.read_csv(file_list, header=None))
mods= []
files= np.array(anns['file'])
for file in flist:
    mods.append(str(file).split('_')[2])
flair = []
for i in np.unique(files):
    if 'FLAIR' in i:
        flair.append(i)

global_labels = ['Normal for age', 'Global Ischemia', 'Global label: Small vessel chronic white matter ischemic change', 'Motion artifact', 'Extra-axial collection', 'Global white matter disease']
#number of other classes is 21
#class corresponding numbers
class_dictionary ={'Nonspecific white matter lesion': 0, 'Likely cysts': 1, 'Mass': 2, 'Resection cavity': 3, 'Craniotomy': 4, 'Lacunar infarct': 5, 'Dural thickening': 6, 'Encephalomalacia': 7, 'Craniectomy with Cranioplasty': 8, 'Extra-axial mass': 9, 'Edema': 10, 'Nonspecific lesion': 11, 'Posttreatment change': 12, 'Normal variant': 13, 'Paranasal sinus opacification': 14, 'Possible artifact': 15, 'Enlarged ventricles': 16, 'Craniectomy': 17}
np.save('/cluster/work/cvl/himeva/datasets/fastmri_yolo/class_dict.npy', class_dictionary)

for file in flair:
    filename_train = '/cluster/work/cvl/himeva/flair_data/multicoil_train/'+ file + '.h5'
    filename_val = '/cluster/work/cvl/himeva/flair_data/multicoil_val/'+ file+ '.h5'
    
    if os.path.exists(filename_train):
        img = h5py.File(filename_train , 'r+')
        rec= img['reconstruction_rss']
        n_s , i_w , i_h = rec.shape
        for slc in range(rec.shape[0]):
            plt.imsave('/cluster/work/cvl/himeva/datasets/fastmri_yolo/train/images/'+file+'_'+ str(slc) + '.png', rec[slc])
            labels_for_file = anns.loc[anns['file'] == file]
            labels_for_slice = labels_for_file.loc[labels_for_file['slice'] == slc].values.tolist()
            print_labels =[]
            for label in labels_for_slice:
                _, slc, _, x0, y0, w, h, label_txt = label
                if label_txt not in global_labels:
                    print_labels.append([class_dictionary[label_txt], x0/i_w, y0/i_h, w/i_w, h/i_h])
            if print_labels:
                np.savetxt('/cluster/work/cvl/himeva/datasets/fastmri_yolo/train/labels/'+file+'_'+ str(slc) + '.txt',np.array(print_labels))
    elif os.path.exists(filename_val):
        img = h5py.File(filename_val, 'r+')
        rec= img['reconstruction_rss']
        n_s , i_w , i_h = rec.shape

        for slc in range(rec.shape[0]):
            plt.imsave('/cluster/work/cvl/himeva/datasets/fastmri_yolo/val/images/'+file+'_'+ str(slc) + '.png', rec[slc])
            labels_for_file = anns.loc[anns['file'] == file]
            labels_for_slice = labels_for_file.loc[labels_for_file['slice'] == slc].values.tolist()
            print_labels =[]
            for label in labels_for_slice:
                _, slc, _, x0, y0, w, h, label_txt = label

                if label_txt not in global_labels:
                    print_labels.append([class_dictionary[label_txt], x0/i_w, y0/i_h, w/i_w, h/i_h])
            if print_labels:
                np.savetxt('/cluster/work/cvl/himeva/datasets/fastmri_yolo/val/labels/'+file+'_'+ str(slc) + '.txt',np.array(print_labels))
    else:
        print(file , 'is not found in train or val sets')