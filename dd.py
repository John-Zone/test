import os

imgs_dir = './4000scan'
des_dir = './split'
image_paths = []
i = 0
for root, dirs, files in os.walk(imgs_dir, topdown=False):
    for name in files:
        if name.endswith('.jpg') or name.endswith('.png'):
            image_path = os.path.join(root, name)
            image_paths.append(image_path)

for i, image_path in enumerate(image_paths):
    shutil.move(image_path, os.path.join(des_dir, str(i) + '.jpg'))
