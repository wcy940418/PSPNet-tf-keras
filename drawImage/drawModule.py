from PIL import Image, ImageDraw
import scipy.io
import numpy as np
import time
import copy
import os.path


class BaseDraw:
	def __init__(self, color150, objectNames, colorAnnotations, img, pred_size, predicted_classes):
		self.class_colors = scipy.io.loadmat(color150)
		self.class_names = scipy.io.loadmat(objectNames, struct_as_record=False)
		self.color_ann = colorAnnotations
		self.im = img
		self.pred_size = pred_size
		self.predicted_classes = copy.deepcopy(predicted_classes)
		self.original_W = self.im.size[0]
		self.original_H = self.im.size[1]


	def drawSimpleSegment(self, blended=False):

		#Drawing module
		im_Width, im_Height = self.pred_size
		prediction_image = Image.new("RGB", (im_Width, im_Height) ,(0,0,0))
		prediction_imageDraw = ImageDraw.Draw(prediction_image)
		cats = {}
		cats_list = []
		#BASE all image segmentation
		for i in range(im_Width):
			for j in range(im_Height):
				#get matrix element class(0-149)
				px_Class = self.predicted_classes[j][i]
				px_ClassName = self.class_names['objectNames'][px_Class][0][0].encode('ascii')
				if px_ClassName not in cats:
					cats[px_ClassName] = 1
				else:
					cats[px_ClassName] += 1
				

				#assign color from .mat list
				put_Px_Color = tuple(self.class_colors['colors'][px_Class])

				#drawing
				prediction_imageDraw.point((i,j), fill=put_Px_Color)
		for k,v in cats.iteritems():
			cats_list.append((v, k))

		cats_list.sort(reverse=True, key=lambda x:x[0])

		annotiation_image = Image.new("RGB", (150, len(cats_list * 30)), (0,0,0))
		for i in range(len(cats_list)):
			ann_path = os.path.join(self.color_ann, cats_list[i][1] + ".jpg")
			ann = Image.open(ann_path)
			annotiation_image.paste(ann,(0, 30 * i))

		#Resize to original size and save
		OutputImage = prediction_image.resize((self.original_W, self.original_H), resample=Image.BILINEAR)
		if blended:
			OutputImage = Image.blend(OutputImage, self.im, 0.5)

		return OutputImage, annotiation_image
