def load_and_test(frame):
	from keras.models import load_model

	model = load_model('face.h5')
	#{'closed': 0, 'open': 1}


	from keras.preprocessing import image

	from haar import faces

	faces(frame)

	test_image = image.load_img('img.jpg', target_size = (64, 64))
	test_image = image.img_to_array(test_image)

	import numpy as np

	test_image = np.expand_dims(test_image, axis = 0)

	result = model.predict(test_image)

	if(result==1):
		print("open eyes")
	else:
		print("closed eyes")
