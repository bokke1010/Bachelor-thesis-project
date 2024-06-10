
def zero_mean(image):
    """Subtracts the column average from each column, then the row average from each row."""
    column_means = image.mean(0)
    intermediate = (image.transpose(1,0,2) - column_means).transpose(1,0,2)
    row_means = image.mean(1)
    return intermediate - row_means

