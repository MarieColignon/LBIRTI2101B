import torch
import os
import wespeaker

model = wespeaker.load_model('english')
model.set_device('cuda:0')



