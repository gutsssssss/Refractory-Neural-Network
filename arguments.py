import math_functions as mf

# define the size, classes number and data address
img_height = 28
img_width = 28
size_input = img_width * img_height
num_classes = 10
batch_size: int = 16
address1 = 'generated_images/*'
address2 = 'generated_images2/*'
initial_lr = 0.00005
decay_rate = 0.05
epochs = 20
lamda = 0.5
dropout_rate = 0.5
g = 0.5
normalization = 'L2'
