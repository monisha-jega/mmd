# public_jsons_dir = "../../raw_catalog/"
# data_dump_dir = "data_dump/"
# model_dump_dir = "model_dump/"
public_jsons_dir = "/scratch/scratch2/monishaj/public_jsons/"
data_dump_dir = "/scratch/scratch2/monishaj/image_data_dump/"
model_dump_dir = "/scratch/scratch2/monishaj/image_model_dump_classif/"
#Must contain ImageUrlToIndex.pkl and annoy.ann
annoy_dir = '/scratch/scratch2/monishaj/image_annoy_index/'


image_size = 4096
batch_size = 500		    			#best value - 64
learning_rate = 0.01    				#best value - 0.0004
epochs = 10							#best value - Early stopping

use_images = True					#If 0, will use 0s for image instead of loading from annoy file
restore_trained = False 			#If true, will restore latest model from checkpoint


hidden_units = [1000, 500, 1000]
num_gender_classes = 4
num_color_classes = 5290
num_mat_classes = 4141


