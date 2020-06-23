import imgaug as ia
import imageio
import imgaug.augmenters as iaa

nGrupos=8;


for i in range(1,nGrupos+1):
    print(i)
    for j in range(0, 20):
        image = imageio.imread('Dataset/images/G%d_%d.jpg'%(i,j))
        imageio.imwrite('Dataset/images/G%d_%d_0.jpg' % (i, j), image)
        if i==8 and j==15:
            # 1- Rotação anti relógio
            rotate = iaa.Affine(rotate=(-5))
            rotated_image = rotate.augment_image(image)
            imageio.imwrite('Dataset/images/G%d_%d_1.jpg' % (i, j), rotated_image)
        else:
            # 1- Rotação anti relógio
            rotate = iaa.Affine(rotate=(-20))
            rotated_image = rotate.augment_image(image)
            imageio.imwrite('Dataset/images/G%d_%d_1.jpg'%(i,j), rotated_image)

        # 2- Rotação relógio
        rotate = iaa.Affine(rotate=(20))
        rotated_image2 = rotate.augment_image(image)
        imageio.imwrite('Dataset/images/G%d_%d_2.jpg' % (i, j), rotated_image2)

        # 3- Adição de ruído gaussiano
        gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
        noise_image = gaussian_noise.augment_image(image)
        imageio.imwrite('Dataset/images/G%d_%d_3.jpg' % (i, j), noise_image)

        # 4- flipping image horizontally
        flip_hr = iaa.Fliplr(p=1.0)
        flip_hr_image = flip_hr.augment_image(image)
        imageio.imwrite('Dataset/images/G%d_%d_4.jpg' % (i, j), flip_hr_image)

        # 5- Escurecer Imagem (Change Brilho)
        contrast = iaa.GammaContrast(gamma=1.5)
        contrast_image = contrast.augment_image(image)
        imageio.imwrite('Dataset/images/G%d_%d_5.jpg' % (i, j), contrast_image)

        # 6 - Enclarecer imagem
        contrast = iaa.GammaContrast(gamma=0.5)
        contrast_image = contrast.augment_image(image)
        imageio.imwrite('Dataset/images/G%d_%d_6.jpg' % (i, j), contrast_image)


'''
# 6 - Scalling da imagem
scale_im=iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})
scale_image =scale_im.augment_image(image)
ia.imshow(scale_image)
'''