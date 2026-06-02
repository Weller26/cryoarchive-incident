import matplotlib.pyplot as plt

def visualize_spectrogram(spectogram, name=None):
    plt.figure(figsize=(10, 4))
    plt.imshow(spectogram.squeeze().numpy(), origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(format='%+2.0f')
    if name is not None:
        plt.title(f'Спектрограмма {name}')
        
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()