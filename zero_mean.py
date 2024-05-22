
def zero_mean(image):
    column_means, row_means = image.mean(0), image.mean(1)
    res_image = image - column_means
    res_image -= row_means
    return res_image

