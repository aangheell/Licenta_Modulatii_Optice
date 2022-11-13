import matplotlib.pyplot as plt

def get_figure(figure_name=None):
        
    if figure_name:
        figure = plt.figure(figure_name)
        plt.title(figure_name)
    else:
        figure = plt.figure()
    
    return figure