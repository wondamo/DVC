import matplotlib.pyplot as plt


def plot_model_accuracy(history):
    plt.figure()
    # Use the history metrics
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    # Make it pretty
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    return plt.gcf()


def plot_model_loss(history):
    plt.figure()
    # Use the history metrics
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # Make it pretty
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    return plt.gcf()