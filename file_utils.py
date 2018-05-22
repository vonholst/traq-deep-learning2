from PIL import Image
import os
TARGET_W = 600
TARGET_H = 800

if __name__ == "__main__":
    path = './training/images/'
    destination_folder = './modified'
    dirs = os.listdir(path)

    for item in dirs:
        file_path = path + item
        if os.path.isfile(file_path) and item.endswith('jpg'):
            im = Image.open(file_path)
            if im.height < im.width:
                im = im.rotate(270, expand=True)
            new_image = im.resize(size=(TARGET_W, TARGET_H), resample=Image.ANTIALIAS)
            if new_image:
                save_file_name = '{}/{}'.format(destination_folder, item)
                new_image.save(save_file_name, 'JPEG', quality=90)
                new_image.close()
                print(u"Resized: {}".format(file_path))
            im.close()




